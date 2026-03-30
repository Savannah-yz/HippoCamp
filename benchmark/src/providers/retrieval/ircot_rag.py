"""
IRCoT RAG Provider - Interleaving Retrieval with Chain-of-Thought.

Faithful implementation of IRCoT from:
    Trivedi et al., "Interleaving Retrieval with Chain-of-Thought Reasoning
    for Knowledge-Intensive Multi-Step Questions" (ACL 2023)
    Paper: https://arxiv.org/abs/2212.10509
    Official repo: https://github.com/StonyBrookNLP/ircot

Core algorithm from the paper (commaqa/inference/ircot.py):
    Loop (state machine with 3 participants):
    1. BM25 Retriever: retrieves using original question (first step) or
       last generated CoT sentence (subsequent steps).
       Paragraphs ACCUMULATE across iterations (cumulate_titles=true).
    2. CoT Generator: given ALL accumulated paragraphs + generated sentences so far,
       generates the NEXT SINGLE SENTENCE of chain-of-thought reasoning.
       Uses spaCy sentence segmentation to extract only the first sentence.
    3. Exit Controller: checks termination conditions:
       a. Answer regex: "the answer is:? (.*)[.]" found in last sentence
       b. Max sentences reached (default 10)
       c. Empty generation

    Key detail: reasoning sentences (starting with "thus", "so", "therefore"
    or containing arithmetic) are FILTERED OUT and NOT used as retrieval queries.
    Query source: "question_or_last_generated_sentence".

Adaptations:
    - BM25: replaced with dense+sparse vector search (project's existing embedder)
    - spaCy sentence segmentation: replaced with regex-based extraction
    - Codex/FLAN-T5: uses project's generator (LLM)
    - Max paragraphs: 15 (from paper's global_max_num_paras)
    - Max sentences: 10 (from paper's max_num_sentences)

Usage:
    provider = IRCoTRAGProvider(
        config=ProviderConfig(name="ircot_rag", params={
            "top_k_per_step": 6,
            "max_sentences": 10,
            "max_paragraphs": 15,
            "rerank_top_k": 5,
        }),
        embedder=embedder, vector_store=vector_store,
        reranker=reranker, generator=generator,
    )
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ..utils import format_query_for_embedding
from ..base import (
    RetrievalProvider,
    ProviderConfig,
    RetrievalResult,
    ChunkResult,
    RetrievalError,
)

logger = logging.getLogger(__name__)


class IRCoTRAGProvider(RetrievalProvider):
    """
    IRCoT RAG provider that interleaves retrieval with chain-of-thought.

    Faithfully implements the 3-participant state machine loop from the paper:
    Retriever -> CoT Generator -> Exit Controller -> (loop or exit)
    """

    def __init__(
        self,
        config: ProviderConfig,
        embedder=None,
        vector_store=None,
        reranker=None,
        generator=None,
    ):
        super().__init__(config)
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.generator = generator

        params = config.params
        # Paper: 6 docs per retrieval step
        self.top_k_per_step = params.get("top_k_per_step", 6)
        # Paper: max_num_sentences = 10
        self.max_sentences = params.get("max_sentences", 10)
        # Paper: global_max_num_paras = 15
        self.max_paragraphs = params.get("max_paragraphs", 15)
        self.rerank_top_k = params.get("rerank_top_k", 5)
        self.use_reranker = params.get("use_reranker", True)
        # Paper: max_para_num_words = 350
        self.max_para_words = params.get("max_para_num_words", 350)
        # Paper: answer_extractor_regex = ".* answer is:? (.*)\."
        self.answer_regex = params.get(
            "answer_regex", r".*answer is:?\s*(.*?)\."
        )

    async def setup(self):
        if self._initialized:
            return
        if self.embedder is None:
            raise RetrievalError("Embedder is required", provider=self.get_name())
        if self.vector_store is None:
            raise RetrievalError("Vector store is required", provider=self.get_name())
        if self.generator is None:
            raise RetrievalError(
                "Generator is required for CoT reasoning",
                provider=self.get_name(),
            )
        logger.info(
            f"IRCoTRAGProvider initialized "
            f"(max_sentences={self.max_sentences}, max_paras={self.max_paragraphs})"
        )
        self._initialized = True

    # =========================================================================
    # Helper: embed + search
    # =========================================================================

    async def _embed_and_search(
        self, query: str, top_k: int, filter=None
    ) -> List[ChunkResult]:
        query_for_embed = format_query_for_embedding(query)
        query_embedding = await self.embedder.embed([query_for_embed])
        dense = (
            query_embedding.dense_embeddings[0]
            if query_embedding.dense_embeddings is not None
            and len(query_embedding.dense_embeddings) > 0
            else None
        )
        sparse = (
            query_embedding.sparse_embeddings[0]
            if query_embedding.sparse_embeddings is not None
            and len(query_embedding.sparse_embeddings) > 0
            else None
        )
        results = await self.vector_store.query(
            dense_embedding=dense, sparse_embedding=sparse,
            top_k=top_k, filter=filter,
        )
        return [
            ChunkResult(
                id=r.get("id", ""),
                content=r.get("chunk", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    async def _rerank_chunks(
        self, query: str, chunks: List[ChunkResult], top_k: int
    ) -> List[ChunkResult]:
        if not self.reranker or not chunks:
            return chunks[:top_k]
        candidates = [
            {"id": c.id, "chunk": c.content, "score": c.score, "metadata": c.metadata}
            for c in chunks
        ]
        reranked = await self.reranker.rerank(
            query=query, candidates=candidates, top_k=top_k
        )
        return [
            ChunkResult(
                id=r.get("id", ""),
                content=r.get("chunk", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in reranked
        ]

    # =========================================================================
    # Paper's helper functions from ircot.py
    # =========================================================================

    def _is_reasoning_sentence(self, sentence: str) -> bool:
        """
        From paper (ircot.py): sentences starting with conjunctions or
        containing arithmetic are reasoning sentences and should NOT be
        used as retrieval queries.

        Paper checks: "thus", "so", "therefore" at start, arithmetic patterns.
        """
        reasoning_starters = [
            "thus", "so", "therefore", "hence", "consequently",
            "as a result", "in conclusion", "this means", "that means",
        ]
        lower = sentence.lower().strip()
        for starter in reasoning_starters:
            if lower.startswith(starter):
                return True
        # Contains arithmetic: "3 + 4", "10 * 2", etc.
        if re.search(r"\d+\s*[+\-*/]\s*\d+", sentence):
            return True
        return False

    def _remove_reasoning_sentences(self, sentences: List[str]) -> List[str]:
        """From paper: filter out reasoning sentences from generated CoT."""
        return [s for s in sentences if not self._is_reasoning_sentence(s)]

    def _extract_first_sentence(self, text: str) -> str:
        """
        Extract the first sentence from text.
        Paper uses spaCy; we use regex-based extraction.
        """
        text = text.strip()
        # Match first sentence ending with . ! or ?
        match = re.match(r"([^.!?]+[.!?])", text)
        if match:
            return match.group(1).strip()
        # If no sentence boundary found, return whole text (truncated)
        return text[:200] if text else ""

    def _truncate_paragraph(self, text: str) -> str:
        """Paper: max_para_num_words = 350."""
        words = text.split()
        if len(words) > self.max_para_words:
            return " ".join(words[:self.max_para_words]) + "..."
        return text

    def _check_answer_found(self, sentence: str) -> Optional[str]:
        """
        Paper's exit condition: check if sentence matches answer regex.
        Pattern: ".* answer is:? (.*)\\."
        """
        match = re.search(self.answer_regex, sentence, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    # =========================================================================
    # Participant 2: CoT Generator
    # Paper: generates ONE sentence given accumulated context + prior CoT
    # =========================================================================

    async def _generate_cot_sentence(
        self,
        question: str,
        accumulated_paragraphs: List[ChunkResult],
        generated_sentences: List[str],
    ) -> str:
        """
        Generate the next single CoT reasoning sentence.

        Paper prompt format:
            [paragraph 1]
            [paragraph 2]
            ...
            Q: {question}
            A: {prior sentences so far}
        """
        # Build context from accumulated paragraphs
        context_parts = []
        for c in accumulated_paragraphs:
            truncated = self._truncate_paragraph(c.content)
            # Paper format: "Wikipedia Title: {title}\n{text}"
            # We adapt: use file path as title
            file_path = c.metadata.get("file_info", {}).get("file_path", "")
            if file_path:
                context_parts.append(f"Document: {file_path}\n{truncated}")
            else:
                context_parts.append(truncated)

        context_text = "\n\n".join(context_parts)

        # Build the prompt (paper format)
        prior = " ".join(generated_sentences)
        prompt = f"""{context_text}

Q: {question}
A: {prior}"""

        try:
            response = await self.generator.generate(
                query=prompt, context=None,
                max_tokens=100,  # Just one sentence
                temperature=0.3,
            )
            # Extract first sentence only (paper: spaCy segmentation)
            sentence = self._extract_first_sentence(response.answer)
            return sentence
        except Exception as e:
            logger.warning(f"CoT generation failed: {e}")
            return ""

    # =========================================================================
    # Main retrieve: the 3-participant state machine loop
    # =========================================================================

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve using IRCoT pipeline (faithful to paper's state machine).

        Loop:
        1. Retriever: search with question (first) or last non-reasoning CoT sentence
        2. CoT Generator: generate next single sentence given accumulated context
        3. Exit Controller: check termination (answer found, max sentences, empty)
        """
        if not self._initialized:
            await self.setup()

        logger.debug(f"IRCoT: Processing query: {query[:50]}...")

        latency_breakdown = {}
        thinking_steps = []
        step_counts = {"search_count": 0, "cot_count": 0, "rerank_count": 0}
        total_start = time.time()
        step_counter = 0

        accumulated_chunks = {}  # id -> ChunkResult (cumulate_titles=true in paper)
        generated_sentences = []
        # Paper: first retrieval uses the question; subsequent use last CoT sentence
        current_retrieval_query = query
        answer_found = None

        try:
            while len(generated_sentences) < self.max_sentences:
                # ── Participant 1: Retriever ───────────────────────────────
                step_counter += 1
                t0 = time.time()
                thinking_steps.append({
                    "step_id": f"retrieve_{step_counter}",
                    "step_type": "retrieve",
                    "title": f"Retrieval (sentence {len(generated_sentences) + 1})",
                    "status": "running",
                    "summary": f"Query: {current_retrieval_query[:60]}...",
                    "timestamp_ms": (t0 - total_start) * 1000,
                    "metadata": {
                        "query": current_retrieval_query,
                        "query_type": "question" if current_retrieval_query == query else "cot_sentence",
                    },
                })

                new_chunks = await self._embed_and_search(
                    current_retrieval_query, self.top_k_per_step, filter
                )
                step_counts["search_count"] += 1

                # Accumulate (paper: cumulate_titles=true)
                new_added = 0
                for c in new_chunks:
                    if c.id not in accumulated_chunks:
                        accumulated_chunks[c.id] = c
                        new_added += 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = (
                    f"Found {len(new_chunks)} ({new_added} new), "
                    f"total: {len(accumulated_chunks)}"
                )

                # Check paragraph limit (paper: global_max_num_paras = 15)
                if len(accumulated_chunks) >= self.max_paragraphs:
                    logger.debug(f"IRCoT: Max paragraphs ({self.max_paragraphs}) reached")
                    break

                # ── Participant 2: CoT Generator ──────────────────────────
                step_counter += 1
                t0 = time.time()
                thinking_steps.append({
                    "step_id": f"cot_{step_counter}",
                    "step_type": "reason",
                    "title": f"CoT Sentence {len(generated_sentences) + 1}",
                    "status": "running",
                    "summary": "Generating next reasoning sentence...",
                    "timestamp_ms": (t0 - total_start) * 1000,
                })

                sentence = await self._generate_cot_sentence(
                    query,
                    list(accumulated_chunks.values()),
                    generated_sentences,
                )
                step_counts["cot_count"] += 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"CoT: {sentence[:80]}..." if sentence else "Empty"

                # ── Participant 3: Exit Controller ─────────────────────────
                # Exit condition 1: empty generation
                if not sentence:
                    logger.debug("IRCoT: Empty generation, exiting")
                    break

                generated_sentences.append(sentence)

                # Exit condition 2: answer regex match
                answer_found = self._check_answer_found(sentence)
                if answer_found is not None:
                    logger.debug(f"IRCoT: Answer found: {answer_found}")
                    break

                # Exit condition 3: max sentences (checked by while loop)

                # Prepare next retrieval query:
                # Paper: "question_or_last_generated_sentence"
                # Use last non-reasoning sentence as query; if reasoning, keep question
                non_reasoning = self._remove_reasoning_sentences([sentence])
                if non_reasoning:
                    current_retrieval_query = non_reasoning[-1]
                else:
                    # Reasoning sentence: keep using original question
                    current_retrieval_query = query

            # ── Final: rerank accumulated paragraphs against original query ──
            final_chunks = sorted(
                accumulated_chunks.values(), key=lambda x: x.score, reverse=True
            )

            if self.use_reranker and self.reranker and final_chunks:
                step_counter += 1
                thinking_steps.append({
                    "step_id": f"final_rerank_{step_counter}",
                    "step_type": "rerank",
                    "title": "Final Reranking",
                    "status": "running",
                    "summary": f"Reranking {len(final_chunks)} accumulated paragraphs...",
                    "timestamp_ms": (time.time() - total_start) * 1000,
                })

                final_chunks = await self._rerank_chunks(
                    query, final_chunks, self.rerank_top_k
                )
                step_counts["rerank_count"] += 1

                thinking_steps[-1]["status"] = "complete"
                thinking_steps[-1]["summary"] = f"Final {len(final_chunks)} paragraphs"
            else:
                final_chunks = final_chunks[:self.rerank_top_k]

            latency_breakdown["total_ms"] = (time.time() - total_start) * 1000

            exit_reason = "max_sentences"
            if answer_found is not None:
                exit_reason = "answer_found"
            elif not generated_sentences or not generated_sentences[-1]:
                exit_reason = "empty_generation"
            elif len(accumulated_chunks) >= self.max_paragraphs:
                exit_reason = "max_paragraphs"

            return RetrievalResult(
                chunks=final_chunks,
                query=query,
                iterations=step_counts["search_count"],
                metadata={
                    "provider": self.get_name(),
                    "cot_sentences": generated_sentences,
                    "num_cot_sentences": len(generated_sentences),
                    "total_paragraphs_accumulated": len(accumulated_chunks),
                    "exit_reason": exit_reason,
                    "answer_found": answer_found,
                },
                latency_breakdown=latency_breakdown,
                thinking_steps=thinking_steps,
                step_counts=step_counts,
            )

        except Exception as e:
            logger.error(f"IRCoT retrieval failed: {e}")
            raise RetrievalError(
                f"Retrieval failed: {e}",
                provider=self.get_name(),
                details={"query": query, "error": str(e)},
            )

    def get_name(self) -> str:
        return "ircot_rag"

    async def aclose(self):
        if self.embedder:
            await self.embedder.aclose()
        if self.reranker:
            await self.reranker.aclose()
        if self.generator:
            await self.generator.aclose()
