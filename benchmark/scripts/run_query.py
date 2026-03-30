#!/usr/bin/env python3
"""
Run the full RAG query pipeline with optional evaluation.

Supports multiple retrieval strategies (vector_search, standard_rag, self_rag, graded_rag,
corrective_rag, adaptive_rag, hyde_rag, ircot_rag, decomposition_rag)
and generation methods (gemini, search_r1) through the Provider system.

=== THREE OPERATIONAL MODES ===

Mode 1: Query Only (no evaluation)
    python scripts/run_query.py -q "What is ML?" -e exp_001
    python scripts/run_query.py --batch benchmark.json -e exp_001

Mode 2: Query + Evaluate
    python scripts/run_query.py -q "What is ML?" -e exp_001 --ground-truth "ML is..." --evaluate
    python scripts/run_query.py --batch benchmark.json -e exp_001 --evaluate

Mode 3: Evaluate Only (on existing results file, no querying)
    python scripts/run_query.py --eval-only ./results/summary_20250115.json

=== PROVIDER OPTIONS ===

Retrieval Methods:
    --retrieval vector_search     : Simple vector similarity search
    --retrieval standard_rag      : Retrieve → Rerank → Return (default)
    --retrieval self_rag          : Retrieve → Grade → Filter → Iterate
    --retrieval graded_rag        : Route → Retrieve → Grade → Rewrite → Return
    --retrieval corrective_rag    : Retrieve → Evaluate (3-class) → Refine/Re-retrieve (CRAG)
    --retrieval adaptive_rag      : Classify complexity → Route to Simple/Moderate/Complex
    --retrieval hyde_rag           : Generate hypothetical docs → Average embeddings → Retrieve (HyDE)
    --retrieval ircot_rag          : Interleave retrieval with chain-of-thought (IRCoT)
    --retrieval decomposition_rag : Decompose query → Sequential solving (Least-to-Most)
    --retrieval none              : No retrieval (for end-to-end generators)

Generator Methods:
    --generator gemini         : Google Gemini API (default)
    --generator gemini_react   : Gemini ReAct (end-to-end, Gemini controls search loop)
    --generator search_r1      : End-to-end Search-R1
    --generator qwen_react     : Qwen ReAct (end-to-end)

End-to-End Generator Options:
    --search-url URL           : Retriever server URL (default: http://127.0.0.1:8000/retrieve)
    --mode hotpotqa|custom     : Prompt style for ReAct generators

=== USAGE EXAMPLES ===

Single Query:
    # Default (standard_rag + gemini)
    python scripts/run_query.py -q "What is machine learning?" -e my_experiment

    # Specify generator explicitly
    python scripts/run_query.py -q "What is ML?" -e exp_001 --generator gemini

    # With Self-RAG retrieval + Gemini generator
    python scripts/run_query.py -q "What is ML?" -e exp_001 --retrieval self_rag --generator gemini

    # With Agentic RAG retrieval + Gemini generator
    python scripts/run_query.py -q "What is ML?" -e exp_001 --retrieval graded_rag --generator gemini

    # End-to-end Search-R1 (no separate retrieval needed)
    python scripts/run_query.py -q "What is ML?" -e exp_001 --retrieval none --generator search_r1

Batch Query:
    python scripts/run_query.py --batch benchmark.json -e exp_001 --generator gemini --output-dir ./results/
    python scripts/run_query.py --batch benchmark.json -e exp_001 --retrieval self_rag --generator gemini --evaluate

=== EVALUATION MODES ===

--eval-mode simple   : ROUGE, BLEU, Semantic Similarity, Retrieval metrics (no LLM API calls)
--eval-mode llm      : LLM Judge only (requires Azure OpenAI API)
--eval-mode all      : All metrics (simple + LLM Judge)
--metrics rouge bleu : Custom metrics (overrides --eval-mode)

Available metrics: rouge, bleu, semantic_similarity, retrieval_precision, retrieval_recall, retrieval_f1, llm_judge
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.shared.env import load_release_env
load_release_env(project_root)

from src.shared.service_factory import SharedServiceFactory
from src.providers import ProviderFactory
from src.providers.base import ChunkResult, GeneratorProvider
from src.rag.models import QueryResultRecord, BenchmarkTestCase, get_timestamp
from src.rag.evaluation.runner import EvaluationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


class QueryPipeline:
    """
    Configurable RAG query pipeline with Provider support.

    Supports multiple retrieval strategies:
    - vector_search: Simple vector similarity search
    - standard_rag: Retrieve → Rerank → Return
    - self_rag: Retrieve → Grade → Filter → Iterate
    - graded_rag: Route → Retrieve → Grade → Rewrite → Return

    And multiple generator methods:
    - gemini: Google Gemini API
    - search_r1: End-to-end Search-R1
    """

    def __init__(
        self,
        service_factory: SharedServiceFactory,
        provider_config: Dict[str, Any],
        experiment_id: str = None
    ):
        """
        Initialize pipeline from service factory and provider config.

        Args:
            service_factory: SharedServiceFactory for creating services
            provider_config: Provider configuration (from providers.yaml)
            experiment_id: Experiment ID for vector store collection
        """
        self.service_factory = service_factory
        self.provider_config = provider_config
        self.experiment_id = experiment_id or "default"

        self.provider_factory = None
        self.retrieval_provider = None
        self.generator_provider = None

    def _get_active_gen_params(self) -> dict:
        """Return the params dict of the active generator."""
        pc = self.provider_config
        if "generators" in pc:
            active = pc.get("active_generator", "gemini")
            gen_cfg = pc["generators"].get(active, {})
        else:
            gen_cfg = pc.get("generator", {})
        return gen_cfg.get("params", {})

    async def setup(self):
        """Initialize all pipeline components using Provider system."""
        logger.info(f"Using service profile: {self.service_factory.profile_name}")

        # Create provider factory
        self.provider_factory = ProviderFactory.from_dict(
            config=self.provider_config,
            service_factory=self.service_factory,
            experiment_id=self.experiment_id,
        )

        # Create providers
        self.retrieval_provider, self.generator_provider = (
            await self.provider_factory.create_providers()
        )

        # Log provider info
        retrieval_name = self.retrieval_provider.get_name() if self.retrieval_provider else "none"
        generator_name = self.generator_provider.get_name()

        logger.info(f"Retrieval provider: {retrieval_name}")
        logger.info(f"Generator provider: {generator_name}")

        if self.retrieval_provider:
            collection_name = f"{self.service_factory.collections.get('rag', 'research_rag')}_{self.experiment_id}"
            logger.info(f"Collection: {collection_name} (experiment: {self.experiment_id})")

    async def cleanup(self):
        """Close all components."""
        if self.retrieval_provider:
            await self.retrieval_provider.aclose()
        if self.generator_provider:
            await self.generator_provider.aclose()
        if self.provider_factory:
            await self.provider_factory.cleanup()

    async def run(
        self,
        query: str,
        *,
        skip_rerank: bool = False,
        retrieve_only: bool = False,
        user_profile_context: str = None,
    ) -> Dict[str, Any]:
        """
        Run the full query pipeline.

        Args:
            query: User query
            skip_rerank: Skip reranking (only for vector_search/standard_rag)
            retrieve_only: Skip generation, return retrieved chunks only
            user_profile_context: Optional user profile context

        Returns:
            Dict with 'answer', 'sources', 'query', 'rewritten_query', etc.
        """
        retrieval_name = self.retrieval_provider.get_name() if self.retrieval_provider else "none"
        generator_name = self.generator_provider.get_name()

        result = {
            "original_query": query,
            "query": query,
            "rewritten_query": None,
            "retrieved": [],
            "reranked": [],
            "answer": None,
            "stages": [],
            "user_profile": user_profile_context is not None,
            "retrieval_provider": retrieval_name,
            "generator_provider": generator_name,
            # New fields for latency and step tracking
            "latency_breakdown": {},
            "thinking_steps": [],
            "step_counts": {},
        }

        pipeline_start = time.time()

        # Check if using end-to-end generator (e.g., Search-R1)
        if not self.generator_provider.requires_retrieval:
            logger.info(f"Using end-to-end generator: {generator_name}")
            result["stages"].append("end_to_end")

            gen_result = await self.generator_provider.generate(
                query=query,
                context=None,
            )

            result["answer"] = gen_result.answer
            result["reasoning_trace"] = gen_result.reasoning_trace
            result["search_queries"] = gen_result.search_queries

            # Extract retrieved chunks from metadata (for Search-R1)
            if "retrieved_chunks" in gen_result.metadata:
                retrieved_chunks = gen_result.metadata["retrieved_chunks"]
                result["retrieved"] = retrieved_chunks
                result["reranked"] = retrieved_chunks  # Use same chunks for both

            return result

        # Standard flow: Retrieval + Generation
        if self.retrieval_provider:
            # Stage 1: Retrieve
            logger.info(f"Stage 1: Retrieving with {retrieval_name}...")
            result["stages"].append("retrieve")

            provider_params = self.provider_config.get("retrieval", {}).get("params", {})
            top_k = provider_params.get("top_k", 20)

            retrieval_result = await self.retrieval_provider.retrieve(
                query=query,
                top_k=top_k,
                skip_rerank=skip_rerank,
            )

            # Convert chunks to dict format for compatibility
            for chunk in retrieval_result.chunks:
                result["retrieved"].append({
                    "id": chunk.id,
                    "chunk": chunk.content,
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                    "is_relevant": chunk.is_relevant,
                    "relevance_reason": chunk.relevance_reason,
                })

            result["rewritten_query"] = retrieval_result.rewritten_query
            if retrieval_result.rewritten_query:
                result["query"] = retrieval_result.rewritten_query

            # Use relevant chunks (or all if no grading)
            context_chunks = retrieval_result.relevant_chunks
            result["reranked"] = [
                {
                    "id": c.id,
                    "chunk": c.content,
                    "score": c.score,
                    "metadata": c.metadata,
                }
                for c in context_chunks
            ]

            logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks, using {len(context_chunks)} for context")

            # Store retrieval metadata
            result["retrieval_metadata"] = retrieval_result.metadata
            result["retrieval_iterations"] = retrieval_result.iterations

            # Store latency, thinking steps, and step counts from retrieval
            if retrieval_result.latency_breakdown:
                result["latency_breakdown"].update(retrieval_result.latency_breakdown)
            if retrieval_result.thinking_steps:
                result["thinking_steps"].extend(retrieval_result.thinking_steps)
            if retrieval_result.step_counts:
                result["step_counts"].update(retrieval_result.step_counts)

            # Check if retrieval provider already generated an answer (e.g., Search-R1)
            if "answer" in retrieval_result.metadata:
                logger.info("Retrieval provider generated answer (Search-R1 RAG)")
                result["answer"] = retrieval_result.metadata["answer"]
                result["reasoning_trace"] = retrieval_result.metadata.get("reasoning_trace")
                result["stages"].append("generate")
                return result

            if not context_chunks:
                logger.warning("No relevant chunks found")
                result["answer"] = "I couldn't find any relevant information to answer your question."
                return result

            # Parent-child expansion: expand chunks to full-page context
            parent_child_config = self.provider_config.get("parent_child", {})
            if parent_child_config.get("enabled", False) and context_chunks:
                from src.providers.retrieval.parent_child import ParentChildExpander

                pc_start = time.time()
                result["thinking_steps"].append({
                    "step_id": f"parent_child_{len(result['thinking_steps']) + 1}",
                    "step_type": "parent_child",
                    "title": "Expanding to Page-Level Context",
                    "status": "running",
                    "summary": f"Expanding {len(context_chunks)} chunks to page-level...",
                    "timestamp_ms": (pc_start - pipeline_start) * 1000,
                })

                expander = ParentChildExpander(
                    vector_store=self.provider_factory._vector_store,
                    max_pages=parent_child_config.get("max_pages", 10),
                )

                context_chunks = await expander.expand(context_chunks)

                pc_latency = (time.time() - pc_start) * 1000
                result["latency_breakdown"]["parent_child_ms"] = pc_latency
                result["thinking_steps"][-1]["status"] = "complete"
                result["thinking_steps"][-1]["summary"] = (
                    f"Expanded to {len(context_chunks)} pages in {pc_latency:.0f}ms"
                )

                # Update reranked with expanded page-level results
                result["reranked"] = [
                    {
                        "id": c.id,
                        "chunk": c.content,
                        "score": c.score,
                        "metadata": c.metadata,
                    }
                    for c in context_chunks
                ]
                result["stages"].append("parent_child")

                logger.info(f"Parent-child expansion: {len(context_chunks)} page-level results")

        if retrieve_only:
            logger.info("Retrieve-only mode, skipping generation")
            return result

        # Stage 2: Generate
        logger.info(f"Stage 2: Generating with {generator_name}...")
        result["stages"].append("generate")

        gen_params = self._get_active_gen_params()
        max_tokens = gen_params.get("max_tokens", 512)
        temperature = gen_params.get("temperature")

        # Prepare context chunks for generator
        gen_context = [
            ChunkResult(
                id=c.get("id", ""),
                content=c.get("chunk", ""),
                score=c.get("score", 0),
                metadata=c.get("metadata", {}),
            )
            for c in result["reranked"]
        ]

        # Add user profile to context if provided
        if user_profile_context:
            logger.info("Including user profile context in generation")
            profile_chunk = ChunkResult(
                id="user_profile",
                content=user_profile_context,
                score=1.0,
                metadata={"type": "user_profile"},
            )
            gen_context = [profile_chunk] + gen_context

        # Track generation latency
        gen_start = time.time()
        result["thinking_steps"].append({
            "step_id": f"generate_{len(result['thinking_steps']) + 1}",
            "step_type": "generate",
            "title": "Generating Answer",
            "status": "running",
            "summary": f"Generating answer using {generator_name}...",
            "timestamp_ms": (gen_start - pipeline_start) * 1000,
        })

        gen_result = await self.generator_provider.generate(
            query=query,
            context=gen_context,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        gen_latency = (time.time() - gen_start) * 1000
        result["latency_breakdown"]["generate_ms"] = gen_latency
        result["thinking_steps"][-1]["status"] = "complete"
        result["thinking_steps"][-1]["summary"] = f"Generated {len(gen_result.answer)} chars"

        result["answer"] = gen_result.answer
        logger.info(f"Generated answer ({len(gen_result.answer)} chars)")

        # Finalize total latency
        result["latency_breakdown"]["total_pipeline_ms"] = (time.time() - pipeline_start) * 1000

        return result


def format_result_record(
    result: Dict[str, Any],
    query_id: str,
    ground_truth: Optional[str] = None,
    expected_chunks: Optional[List[Dict[str, Any]]] = None,
    file_list: Optional[List[str]] = None,
    bench_name: str = "default",
    provider: str = "research_core",
    execution_time_ms: Optional[int] = None,
    latency_breakdown: Optional[Dict[str, float]] = None,
    thinking_steps: Optional[List[Dict[str, Any]]] = None,
    step_counts: Optional[Dict[str, int]] = None,
) -> QueryResultRecord:
    """
    Format pipeline result into a QueryResultRecord for evaluation.

    Args:
        result: Raw result from QueryPipeline.run()
        query_id: Unique identifier for this query
        ground_truth: Expected answer for evaluation
        expected_chunks: List of expected chunk dicts for retrieval evaluation
        file_list: List of required file paths from benchmark
        bench_name: Name of the benchmark being run
        provider: Name of the RAG provider
        execution_time_ms: Execution time in milliseconds
        latency_breakdown: Step-by-step latency in milliseconds
        thinking_steps: Detailed thinking/reasoning trace
        step_counts: Counts of various operations

    Returns:
        QueryResultRecord ready for JSON serialization

    Output format for retrieved_chunks (unified across all providers):
    {
        "rank": 1,
        "content": "...",
        "score": 0.7992077,
        "id": "d71d2dbf54592bfbbb09ce98c7052b7a",
        "metadata": {
            "file_info": {
                "file_id": "...",
                "file_path": "...",
                "file_type": "...",
                "file_name": "..."
            },
            "type": "content",
            "start": "00:00",
            "end": "01:05:15",
            "chunk_index": 0
        }
    }
    """
    # Format retrieved chunks for evaluation
    retrieved_chunks = []
    source_chunks = result.get("reranked") or result.get("retrieved", [])

    for rank, chunk in enumerate(source_chunks, 1):
        metadata = chunk.get("metadata", {})

        # Unified format - metadata should already have file_info, segment_info, chunk_meta
        # Just ensure required fields have defaults
        file_info = metadata.get("file_info", {})
        file_info.setdefault("file_id", "unknown")
        file_info.setdefault("file_path", "")
        file_info.setdefault("file_type", "text")
        file_info.setdefault("file_name", "")
        metadata["file_info"] = file_info

        # Ensure segment_info exists
        if "segment_info" not in metadata:
            metadata["segment_info"] = {
                "segment_indices": [],
                "page_numbers": [],
                "time_ranges": []
            }

        # Ensure chunk_meta exists
        if "chunk_meta" not in metadata:
            metadata["chunk_meta"] = {
                "type": "content",
                "chunk_index": 0,
                "char_count": len(chunk.get("content", chunk.get("chunk", ""))),
                "token_count": 0
            }

        # Unified format: use "id" and "content"
        retrieved_chunks.append({
            "rank": rank,
            "content": chunk.get("content", chunk.get("chunk", "")),
            "score": float(chunk.get("score", 0)),
            "id": chunk.get("id", ""),
            "metadata": metadata,
        })

    # Extract unique file paths from retrieved chunks
    # metadata 已在上面统一为新格式 (file_info 嵌套)
    retrieved_file_list = []
    seen_files = set()
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        file_info = metadata.get("file_info", {})
        file_path = (
            file_info.get("file_path") or
            file_info.get("file_name") or
            file_info.get("file_id") or
            ""
        )
        if file_path and file_path not in seen_files:
            retrieved_file_list.append(file_path)
            seen_files.add(file_path)

    return QueryResultRecord(
        timestamp=get_timestamp(),
        provider=provider,
        bench=bench_name,
        query_id=query_id,
        query=result.get("original_query", ""),
        ground_truth=ground_truth,
        answer=result.get("answer"),
        retrieved_chunks=retrieved_chunks,
        expected_chunks=expected_chunks,
        file_list=file_list,
        retrieved_file_list=retrieved_file_list,
        stages=result.get("stages", []),
        execution_time_ms=execution_time_ms,
        rewritten_query=result.get("rewritten_query"),
        user_profile_included=result.get("user_profile", False),
        latency_breakdown=latency_breakdown or result.get("latency_breakdown"),
        thinking_steps=thinking_steps or result.get("thinking_steps"),
        step_counts=step_counts or result.get("step_counts"),
        reasoning_trace=result.get("reasoning_trace"),
        search_queries=result.get("search_queries"),
        retrieval_metadata=result.get("retrieval_metadata"),
    )


def save_result_record(
    record: QueryResultRecord,
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    """Save a QueryResultRecord to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{record.query_id}_{record.timestamp}.json"

    file_path = output_path / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved result to: {file_path}")
    return str(file_path)


def load_benchmark_file(file_path: str) -> List[BenchmarkTestCase]:
    """Load benchmark test cases from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [BenchmarkTestCase.from_dict(item) for item in data]
    elif isinstance(data, dict):
        return [BenchmarkTestCase.from_dict(data)]
    else:
        raise ValueError(f"Invalid benchmark file format: expected list or dict")


def display_results(result: Dict[str, Any]):
    """Display pipeline results."""
    show_sources = True
    max_source_chars = 200

    print("\n" + "=" * 80)
    print("QUERY PIPELINE RESULTS")
    print("=" * 80)

    print(f"\nQuery: {result['original_query']}")
    if result.get("rewritten_query"):
        print(f"Rewritten: {result['rewritten_query']}")
    if result.get("user_profile"):
        print(f"User Profile: Included")

    # Show provider info
    print(f"\nProviders:")
    print(f"  Retrieval: {result.get('retrieval_provider', 'N/A')}")
    print(f"  Generator: {result.get('generator_provider', 'N/A')}")

    print(f"\nStages: {' → '.join(result['stages'])}")
    print(f"Retrieved: {len(result['retrieved'])} chunks")
    print(f"Context: {len(result['reranked'])} chunks")

    # Show Search-R1 specific info
    if result.get("search_queries"):
        print(f"Search queries performed: {len(result['search_queries'])}")

    print("\n" + "-" * 80)
    print("ANSWER")
    print("-" * 80)
    print(result.get("answer", "No answer generated"))

    if show_sources and result["reranked"]:
        print("\n" + "-" * 80)
        print("SOURCES")
        print("-" * 80)

        for i, chunk in enumerate(result["reranked"], 1):
            text = chunk.get("chunk", chunk.get("content", ""))
            if len(text) > max_source_chars:
                text = text[:max_source_chars] + "..."

            score = chunk.get("score", 0)
            chunk_id = chunk.get("id", "unknown")

            if abs(score) < 0.0001 and score != 0:
                score_str = f"{score:.4e}"
            else:
                score_str = f"{score:.4f}"

            print(f"\n[{i}] Score: {score_str} | ID: {chunk_id}")
            print(f"    {text}")

    print("\n" + "=" * 80)


async def run_single_query(
    pipeline: QueryPipeline,
    query: str,
    query_id: str = "q1",
    ground_truth: Optional[str] = None,
    expected_chunks: Optional[List[Dict[str, Any]]] = None,
    file_list: Optional[List[str]] = None,
    skip_rerank: bool = False,
    retrieve_only: bool = False,
    user_profile_context: Optional[str] = None,
    bench_name: str = "default",
    provider: str = "research_core",
) -> QueryResultRecord:
    """Run a single query and return a formatted QueryResultRecord."""
    start_time = time.time()

    result = await pipeline.run(
        query=query,
        skip_rerank=skip_rerank,
        retrieve_only=retrieve_only,
        user_profile_context=user_profile_context,
    )

    execution_time_ms = int((time.time() - start_time) * 1000)

    return format_result_record(
        result=result,
        query_id=query_id,
        ground_truth=ground_truth,
        expected_chunks=expected_chunks,
        file_list=file_list,
        bench_name=bench_name,
        provider=provider,
        execution_time_ms=execution_time_ms,
    )


def find_completed_query_ids(output_dir: str) -> Dict[str, Optional[str]]:
    """Scan output directory for already-completed query IDs.

    Output files are named ``{query_id}_{timestamp}.json``.

    Returns:
        Dict mapping query_id (str) → query text (str or None).
        The query text is used for sanity-checking against the current batch.
    """
    completed: Dict[str, Optional[str]] = {}
    output_path = Path(output_dir)
    if not output_path.exists():
        return completed

    for f in output_path.glob("*.json"):
        name = f.stem  # e.g. "1_20260210_115824"
        if name.startswith("summary_") or name.startswith("evaluation_"):
            continue
        # Try to load and read query_id + query from the file
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            qid = data.get("query_id")
            if qid is not None:
                completed[str(qid)] = data.get("query")
        except Exception:
            # Fallback: parse filename — strip trailing _YYYYMMDD_HHMMSS
            parts = name.rsplit("_", 2)
            if len(parts) >= 3:
                completed[parts[0]] = None
    return completed


def load_completed_records(output_dir: str) -> Dict[str, QueryResultRecord]:
    """Load all completed query result records from an output directory.

    Returns:
        Dict mapping query_id (str) → QueryResultRecord.
    """
    records: Dict[str, QueryResultRecord] = {}
    output_path = Path(output_dir)
    if not output_path.exists():
        return records

    for f in output_path.glob("*.json"):
        name = f.stem
        if name.startswith("summary_") or name.startswith("evaluation_"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            qid = data.get("query_id")
            if qid is not None:
                records[str(qid)] = QueryResultRecord.from_dict(data)
        except Exception:
            pass
    return records


def validate_resume_consistency(
    completed: Dict[str, Optional[str]],
    test_cases: List[BenchmarkTestCase],
) -> None:
    """Spot-check that completed results match the current batch file.

    Compares the first and last completed query IDs against the test cases
    to make sure the resume directory corresponds to the same benchmark.
    Raises ValueError on mismatch.
    """
    if not completed:
        return

    # Build a lookup from current test cases
    tc_lookup = {str(tc.id): tc.question for tc in test_cases}

    # Pick the first and last completed IDs (sorted for determinism)
    sorted_ids = sorted(completed.keys(), key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))
    samples = [sorted_ids[0]]
    if len(sorted_ids) > 1:
        samples.append(sorted_ids[-1])

    for qid in samples:
        saved_query = completed[qid]
        if saved_query is None:
            continue  # couldn't read query from file, skip check
        expected_query = tc_lookup.get(qid)
        if expected_query is None:
            raise ValueError(
                f"Resume sanity check failed: completed query_id '{qid}' "
                f"not found in current batch file. Wrong --batch or --resume dir?"
            )
        # Compare first 150 chars (enough to catch mismatches, tolerant of minor edits)
        if saved_query[:150] != expected_query[:150]:
            raise ValueError(
                f"Resume sanity check failed for query_id '{qid}':\n"
                f"  Saved  : {saved_query[:100]}...\n"
                f"  Current: {expected_query[:100]}...\n"
                f"The resume directory doesn't match the current batch file."
            )


async def run_batch_queries(
    pipeline: QueryPipeline,
    test_cases: List[BenchmarkTestCase],
    skip_rerank: bool = False,
    retrieve_only: bool = False,
    bench_name: str = "default",
    provider: str = "research_core",
    output_dir: Optional[str] = None,
    resume_dir: Optional[str] = None,
) -> List[QueryResultRecord]:
    """Run batch queries from test cases.

    Args:
        resume_dir: Path to a previous output directory.  Any query whose ID
            already has a result file in *resume_dir* will be skipped.  If
            *output_dir* equals *resume_dir* (or *resume_dir* is ``True``-ish
            and *output_dir* is set), existing results in *output_dir* are
            reused automatically.
    """
    results = []
    total = len(test_cases)

    # ── Resume: detect already-completed queries ──────────────────────
    completed_ids: set = set()
    completed_records: Dict[str, QueryResultRecord] = {}
    scan_dir = resume_dir or output_dir  # when --resume without explicit dir, use output_dir

    if resume_dir is not None and scan_dir:
        completed_map = find_completed_query_ids(scan_dir)
        if completed_map:
            # Sanity check: verify saved queries match current batch
            validate_resume_consistency(completed_map, test_cases)
            completed_ids = set(completed_map.keys())
            completed_records = load_completed_records(scan_dir)
            logger.info(
                f"Resume: found {len(completed_ids)} completed query ID(s) in {scan_dir} — "
                f"will skip: {sorted(completed_ids)}"
            )

    for i, test_case in enumerate(test_cases, 1):
        # Load previously completed results when resuming
        if str(test_case.id) in completed_ids:
            logger.info(f"[{i}/{total}] Skipping already-completed query: {test_case.id}")
            if str(test_case.id) in completed_records:
                results.append(completed_records[str(test_case.id)])
            continue

        logger.info(f"\n[{i}/{total}] Processing query: {test_case.id}")
        logger.info(f"Question: {test_case.question[:100]}...")

        record = await run_single_query(
            pipeline=pipeline,
            query=test_case.question,
            query_id=test_case.id,
            ground_truth=test_case.ground_truth,
            expected_chunks=test_case.get_expected_chunks(),
            file_list=test_case.get_file_list(),
            skip_rerank=skip_rerank,
            retrieve_only=retrieve_only,
            bench_name=bench_name,
            provider=provider,
        )

        results.append(record)

        if output_dir:
            save_result_record(record, output_dir)

        logger.info(f"[{i}/{total}] Completed query {test_case.id}")

    return results


def get_metrics_for_mode(eval_mode: str, custom_metrics: list = None) -> list:
    """Get metrics list based on eval mode or custom metrics."""
    if custom_metrics:
        return custom_metrics
    if eval_mode == "simple":
        return [
            "rouge", "bleu", "semantic_similarity",
            "retrieval_precision", "retrieval_recall", "retrieval_f1",
        ]
    elif eval_mode == "llm":
        return ["llm_judge"]
    elif eval_mode == "all":
        return [
            "rouge", "bleu", "semantic_similarity",
            "retrieval_precision", "retrieval_recall", "retrieval_f1",
            "llm_judge",
        ]
    return ["rouge", "bleu"]


async def main():
    parser = argparse.ArgumentParser(
        description="Run the full RAG query pipeline with optional evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ========== Input Mode (3 options) ==========
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--query", "-q", type=str, help="Single query to process")
    input_group.add_argument("--batch", "-b", type=str, help="Path to benchmark JSON file")
    input_group.add_argument("--eval-only", type=str, metavar="RESULTS_FILE",
                            help="Only evaluate: path to existing results JSON file")

    # ========== Provider Options (NEW) ==========
    parser.add_argument(
        "--retrieval",
        type=str,
        choices=["vector_search", "standard_rag", "self_rag", "graded_rag", "hybrid_rag",
                 "corrective_rag", "adaptive_rag", "hyde_rag", "ircot_rag",
                 "decomposition_rag", "none"],
        help="Retrieval provider type (overrides providers.yaml)"
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=["gemini", "gemini_react", "search_r1", "qwen_react"],
        help="Generator provider type (overrides providers.yaml)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hotpotqa", "custom"],
        help="Prompt mode for ReAct generators: 'hotpotqa' (short-answer) or 'custom' (detailed)"
    )
    parser.add_argument(
        "--search-url",
        type=str,
        help="URL for retriever server (for end-to-end generators like gemini_react, search_r1, qwen_react)"
    )
    parser.add_argument(
        "--initial-search-original",
        action="store_true",
        default=None,
        help="Use original query for first search in ReAct loop (overrides providers.yaml)"
    )
    parser.add_argument(
        "--providers-config",
        type=str,
        default="configs/providers.yaml",
        help="Path to providers config (default: configs/providers.yaml)"
    )

    # ========== Query Options ==========
    parser.add_argument("--experiment", "-e", type=str,
                       help="Experiment ID for vector database collection (REQUIRED for --query and --batch)")
    parser.add_argument("--ground-truth", "-gt", type=str,
                       help="Ground truth answer for single query (for evaluation)")
    parser.add_argument("--output-dir", "-o", type=str,
                       help="Directory to save results as JSON files")
    parser.add_argument("--bench-name", type=str, default="default",
                       help="Benchmark name for result tagging (default: default)")
    parser.add_argument("--provider-name", type=str, default=None,
                       help="Provider name for result tagging (default: auto-generated from retrieval_generator)")
    parser.add_argument("--no-rerank", action="store_true",
                       help="Skip reranking even if enabled")
    parser.add_argument("--retrieve-only", action="store_true",
                       help="Only retrieve, skip generation")
    parser.add_argument("--resume", nargs="?", const=True, default=None,
                       metavar="RESUME_DIR",
                       help="Resume batch run, skipping already-completed queries. "
                            "Optionally specify a directory to scan for completed results "
                            "(default: uses --output-dir)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Only run the first N test cases from batch (e.g. --limit 5)")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")
    parser.add_argument("--user", "-u", type=str,
                       help="User identifier to include profile context from MongoDB")
    parser.add_argument("--services-config", default="configs/services.yaml",
                       help="Services config file (default: configs/services.yaml)")

    # ========== Evaluation Options ==========
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation metrics on query results")
    parser.add_argument("--eval-config", type=str, default="configs/evaluation.yaml",
                       help="Path to evaluation config (default: configs/evaluation.yaml)")
    parser.add_argument("--eval-mode", type=str, choices=["simple", "llm", "all"],
                       default=None, help="Evaluation mode (overrides eval config)")
    parser.add_argument("--metrics", type=str, nargs="+",
                       help="Custom metrics to compute (overrides --eval-mode)")
    parser.add_argument("--judge-template", type=str, choices=["simple", "detailed"],
                       default=None,
                       help="LLM Judge prompt template (overrides eval config)")

    args = parser.parse_args()

    # ========== Load Evaluation Config ==========
    eval_config = load_config(args.eval_config)
    if eval_config:
        logger.info(f"Loaded evaluation config from: {args.eval_config}")
    else:
        eval_config = {}

    # Resolve eval settings: CLI args override eval_config, which overrides defaults
    resolved_eval_mode = args.eval_mode or eval_config.get("eval_mode", "simple")
    resolved_metrics = args.metrics or eval_config.get("metrics")

    llm_judge_config = eval_config.get("llm_judge", {})
    resolved_judge_template = args.judge_template or llm_judge_config.get("prompt_template", "simple")

    # Build metric_kwargs from eval config + CLI overrides
    llm_judge_kwargs = {"prompt_template": resolved_judge_template}
    for key in ("api_url", "api_key", "timeout", "max_retries"):
        if key in llm_judge_config:
            llm_judge_kwargs[key] = llm_judge_config[key]
    # Pass prompts from YAML so LLMJudgeMetric uses config instead of built-in fallback
    if "prompts" in llm_judge_config:
        llm_judge_kwargs["prompts"] = llm_judge_config["prompts"]
    resolved_metric_kwargs = {"llm_judge": llm_judge_kwargs}

    # Validate arguments
    if (args.query or args.batch) and not args.experiment:
        parser.error("--experiment (-e) is required when using --query or --batch mode")

    # ========== Mode 3: Evaluate Only ==========
    if args.eval_only:
        logger.info(f"Evaluate-only mode: loading results from {args.eval_only}")

        with open(args.eval_only, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            results_data = [data]
        else:
            results_data = data

        logger.info(f"Loaded {len(results_data)} query result(s) for evaluation")

        metrics_to_use = get_metrics_for_mode(resolved_eval_mode, resolved_metrics)
        logger.info(f"Using metrics: {metrics_to_use}")

        eval_runner = EvaluationRunner(
                metrics=metrics_to_use,
                metric_kwargs=resolved_metric_kwargs,
            )
        evaluations = await eval_runner.evaluate_results(results_data)

        summary = eval_runner.get_summary(evaluations)
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total queries evaluated: {summary['total_queries']}")
        print(f"Metrics: {', '.join(metrics_to_use)}")
        for metric_name, stats in summary.get("metrics", {}).items():
            print(f"\n{metric_name}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")

        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            eval_file = Path(args.output_dir) / f"evaluation_{get_timestamp()}.json"
            eval_data = {
                "summary": summary,
                "eval_mode": args.eval_mode,
                "metrics_used": metrics_to_use,
                "source_file": args.eval_only,
                "results": [e.to_dict() for e in evaluations]
            }
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation to: {eval_file}")

        if args.json:
            output = {"summary": summary, "results": [e.to_dict() for e in evaluations]}
            print(json.dumps(output, indent=2, ensure_ascii=False))

        return

    # ========== Load Configs ==========
    logger.info(f"Loading services config from: {args.services_config}")
    service_factory = SharedServiceFactory.from_yaml(args.services_config)

    # Load provider config
    logger.info(f"Loading providers config from: {args.providers_config}")
    provider_config = load_config(args.providers_config)

    # Default provider config if not found
    if not provider_config:
        provider_config = {
            "retrieval": {"type": "standard_rag", "params": {}},
            "generator": {"type": "gemini", "params": {}},
        }

    # Override with command-line arguments
    if args.retrieval:
        if "retrieval" not in provider_config:
            provider_config["retrieval"] = {"params": {}}
        provider_config["retrieval"]["type"] = args.retrieval
        logger.info(f"Retrieval provider overridden to: {args.retrieval}")

    # ── Resolve active generator (supports new & legacy config format) ────
    # New format: generators map + active_generator selector
    # Legacy format: single generator block
    uses_new_format = "generators" in provider_config

    if args.generator:
        if uses_new_format:
            provider_config["active_generator"] = args.generator
        else:
            if "generator" not in provider_config:
                provider_config["generator"] = {"params": {}}
            provider_config["generator"]["type"] = args.generator
        logger.info(f"Generator provider overridden to: {args.generator}")

    # Get a mutable reference to the active generator's params dict
    def _get_active_gen_params() -> dict:
        """Return the params dict of the active generator, creating it if needed."""
        if uses_new_format:
            active = provider_config.get("active_generator", "gemini")
            gen_map = provider_config["generators"]
            if active not in gen_map:
                gen_map[active] = {"type": active, "params": {}}
            gen_cfg = gen_map[active]
        else:
            if "generator" not in provider_config:
                provider_config["generator"] = {"params": {}}
            gen_cfg = provider_config["generator"]
        if "params" not in gen_cfg:
            gen_cfg["params"] = {}
        return gen_cfg["params"]

    # Pass --mode to generator provider params
    if args.mode:
        _get_active_gen_params()["mode"] = args.mode
        logger.info(f"Generator mode set to: {args.mode}")

    # Pass --search-url to generator provider params
    if args.search_url:
        _get_active_gen_params()["search_url"] = args.search_url
        logger.info(f"Search URL set to: {args.search_url}")

    # Pass --initial-search-original to generator provider params
    if args.initial_search_original is not None:
        _get_active_gen_params()["initial_search_original"] = args.initial_search_original
        logger.info(f"Initial search with original query: {args.initial_search_original}")

    # Auto-set retrieval to none for end-to-end generators
    end_to_end_generators = {"search_r1", "qwen_react", "gemini_react"}
    if uses_new_format:
        gen_type = provider_config.get("active_generator")
    else:
        gen_type = provider_config.get("generator", {}).get("type")
    if gen_type in end_to_end_generators:
        if provider_config.get("retrieval", {}).get("type") not in ["none", None]:
            logger.info(f"Automatically setting retrieval to 'none' for {gen_type} generator")
            provider_config["retrieval"]["type"] = "none"

    # Fetch user profile if specified
    user_profile_context = None
    if args.user:
        mongodb_uri = os.getenv("MONGODB_URI")
        if mongodb_uri:
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
                async_mongo_client = AsyncIOMotorClient(mongodb_uri)
                collection = async_mongo_client.profiling.profiles
                profile_doc = await collection.find_one({"user": args.user})

                if profile_doc and profile_doc.get("profile"):
                    profile_parts = []
                    for topic, info in profile_doc["profile"].get("topics", {}).items():
                        for subtopic, subtopic_info in info.get("sub_topics", {}).items():
                            value = subtopic_info.get("value")
                            if value:
                                profile_parts.append(f"{topic}.{subtopic}: {value}")

                    if profile_parts:
                        user_profile_context = "User Profile:\n" + "\n".join(profile_parts[:15])
                        logger.info(f"Loaded user profile for '{args.user}' with {len(profile_parts)} attributes")
            except Exception as e:
                logger.warning(f"Could not load user profile: {e}")
        else:
            logger.warning("--user specified but MONGODB_URI not set")

    # Propagate --no-rerank into provider params so all providers skip reranker
    if args.no_rerank:
        provider_config.setdefault("retrieval", {}).setdefault("params", {})["use_reranker"] = False

    # Create and run pipeline
    pipeline = QueryPipeline(
        service_factory=service_factory,
        provider_config=provider_config,
        experiment_id=args.experiment,
    )

    try:
        await pipeline.setup()

        # Auto-generate provider name if not specified
        provider_name = args.provider_name
        if provider_name is None:
            retrieval_type = provider_config.get("retrieval", {}).get("type", "none")
            if uses_new_format:
                generator_type = provider_config.get("active_generator", "unknown")
            else:
                generator_type = provider_config.get("generator", {}).get("type", "unknown")
            if retrieval_type in ["none", None]:
                provider_name = generator_type
            else:
                provider_name = f"{retrieval_type}_{generator_type}"
            logger.info(f"Auto-generated provider name: {provider_name}")

        # Mode 1: Batch query
        if args.batch:
            logger.info(f"Loading benchmark file: {args.batch}")
            test_cases = load_benchmark_file(args.batch)
            if args.limit:
                test_cases = test_cases[:args.limit]
            logger.info(f"Loaded {len(test_cases)} test cases")

            bench_name = args.bench_name
            if bench_name == "default":
                bench_name = Path(args.batch).stem

            # Determine resume directory
            resume_dir = None
            if args.resume is not None:
                if args.resume is True:
                    # --resume without a path: reuse --output-dir
                    resume_dir = args.output_dir
                    if not resume_dir:
                        parser.error("--resume requires --output-dir (or pass an explicit directory)")
                else:
                    resume_dir = args.resume

            results = await run_batch_queries(
                pipeline=pipeline,
                test_cases=test_cases,
                skip_rerank=args.no_rerank,
                retrieve_only=args.retrieve_only,
                bench_name=bench_name,
                provider=provider_name,
                output_dir=args.output_dir,
                resume_dir=resume_dir,
            )

            if args.json:
                print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
            else:
                print("\n" + "=" * 80)
                print(f"BATCH QUERY COMPLETED: {len(results)} queries processed")
                print("=" * 80)
                for record in results:
                    print(f"\n[{record.query_id}] {record.query[:50]}...")
                    print(f"  Answer: {(record.answer or 'N/A')[:100]}...")
                    print(f"  Retrieved chunks: {len(record.retrieved_chunks)}")
                    print(f"  Execution time: {record.execution_time_ms}ms")

            if args.output_dir:
                summary_file = Path(args.output_dir) / f"summary_{get_timestamp()}.json"
                with open(summary_file, "w", encoding="utf-8") as f:
                    json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
                logger.info(f"Saved summary to: {summary_file}")

            # Run evaluation if requested
            if args.evaluate:
                metrics_to_use = get_metrics_for_mode(resolved_eval_mode, resolved_metrics)
                logger.info(f"Running evaluation with metrics: {metrics_to_use}")
                eval_runner = EvaluationRunner(
                metrics=metrics_to_use,
                metric_kwargs=resolved_metric_kwargs,
            )
                evaluations = await eval_runner.evaluate_results([r.to_dict() for r in results])

                summary = eval_runner.get_summary(evaluations)
                print("\n" + "=" * 80)
                print("EVALUATION SUMMARY")
                print("=" * 80)
                print(f"Total queries evaluated: {summary['total_queries']}")
                for metric_name, stats in summary.get("metrics", {}).items():
                    print(f"\n{metric_name}:")
                    print(f"  Mean: {stats['mean']:.4f}")
                    print(f"  Min:  {stats['min']:.4f}")
                    print(f"  Max:  {stats['max']:.4f}")

                if args.output_dir:
                    eval_file = Path(args.output_dir) / f"evaluation_{get_timestamp()}.json"
                    eval_data = {"summary": summary, "results": [e.to_dict() for e in evaluations]}
                    with open(eval_file, "w", encoding="utf-8") as f:
                        json.dump(eval_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved evaluation to: {eval_file}")

        # Mode 2: Single query
        else:
            record = await run_single_query(
                pipeline=pipeline,
                query=args.query,
                query_id="q1",
                ground_truth=args.ground_truth,
                skip_rerank=args.no_rerank,
                retrieve_only=args.retrieve_only,
                user_profile_context=user_profile_context,
                bench_name=args.bench_name,
                provider=provider_name,
            )

            if args.output_dir:
                save_result_record(record, args.output_dir)

            if args.json:
                print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
            else:
                # Display results
                result = {
                    "original_query": record.query,
                    "rewritten_query": record.rewritten_query,
                    "user_profile": record.user_profile_included,
                    "stages": record.stages,
                    "retrieved": record.retrieved_chunks,
                    "reranked": [
                        {
                            "id": c.get("chunk_id") or c.get("id"),
                            "chunk": c.get("content") or c.get("chunk"),
                            "score": c.get("score"),
                        }
                        for c in record.retrieved_chunks
                    ],
                    "answer": record.answer,
                    "retrieval_provider": provider_config.get("retrieval", {}).get("type", "unknown"),
                    "generator_provider": provider_config.get("generator", {}).get("type", "unknown"),
                }
                display_results(result)

                if record.ground_truth:
                    print("\n" + "-" * 80)
                    print("EVALUATION INFO")
                    print("-" * 80)
                    print(f"Ground Truth: {record.ground_truth[:200]}...")

            # Run evaluation if requested
            if args.evaluate and record.ground_truth:
                metrics_to_use = get_metrics_for_mode(resolved_eval_mode, resolved_metrics)
                logger.info(f"Running evaluation with metrics: {metrics_to_use}")
                eval_runner = EvaluationRunner(
                metrics=metrics_to_use,
                metric_kwargs=resolved_metric_kwargs,
            )
                eval_result = await eval_runner.evaluate_single(
                    query_id=record.query_id,
                    query=record.query,
                    answer=record.answer,
                    ground_truth=record.ground_truth,
                    retrieved_chunks=record.retrieved_chunks,
                    expected_chunks=record.expected_chunks,
                )

                print("\n" + "=" * 80)
                print("EVALUATION METRICS")
                print("=" * 80)
                for metric in eval_result.metrics:
                    print(f"\n{metric.name}: {metric.score:.4f}")
                    if metric.details and "error" not in metric.details:
                        for key, value in metric.details.items():
                            if isinstance(value, float):
                                print(f"  {key}: {value:.4f}")

                if args.output_dir:
                    eval_file = Path(args.output_dir) / f"evaluation_{record.query_id}_{get_timestamp()}.json"
                    with open(eval_file, "w", encoding="utf-8") as f:
                        json.dump(eval_result.to_dict(), f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved evaluation to: {eval_file}")

        logger.info("Pipeline completed successfully")

    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
