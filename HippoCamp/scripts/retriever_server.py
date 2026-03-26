#!/usr/bin/env python3
"""
Research Core Retriever Server

A FastAPI server that exposes the RAG retrieval pipeline via HTTP API.
Designed to be compatible with Search-R1 agent format.

Uses its own config (configs/retriever_server.yaml), independent from
the query pipeline's providers.yaml.

================================================================================
Quick Start / Stop Guide
================================================================================

1. START SERVER (foreground):
   cd research_core
   source research_score/bin/activate
   source .env
   python scripts/retriever_server.py -e walmart_test -p 18000

2. START SERVER (background):
   cd research_core
   source research_score/bin/activate
   source .env
   nohup python scripts/retriever_server.py -e walmart_test -p 18000 > retriever.log 2>&1 &

3. CHECK STATUS:
   curl http://localhost:18000/health

4. STOP SERVER:
   # Find and kill the process
   pkill -f "retriever_server.py"
   
   # Or find PID first
   lsof -i :18000
   kill <PID>

5. VIEW LOGS (if running in background):
   tail -f retriever.log

================================================================================
Usage Examples
================================================================================

    # Start server with default settings (cloud profile, default experiment)
    python scripts/retriever_server.py

    # With specific experiment
    python scripts/retriever_server.py --experiment walmart_test

    # With custom port
    python scripts/retriever_server.py --port 18000 --experiment walmart_test

    # Disable reranking for faster retrieval
    python scripts/retriever_server.py --experiment walmart_test --no-rerank

================================================================================
API Reference
================================================================================

Endpoints:
    GET  /           - Server info
    GET  /health     - Health check
    POST /retrieve   - Retrieve chunks for a query
    POST /search     - Alias for /retrieve (Search-R1 compatible)

Request Format (supports both):
    1. Simple: {"query": "your question", "top_k": 5}
    2. Search-R1: {"queries": ["your question"], "topk": 5}

Response Format:
    {
        "results": [
            {"id": "...", "text": "chunk content...", "score": 0.95, "metadata": {...}},
            ...
        ]
    }

Test with curl:
    curl -X POST http://localhost:18000/retrieve \\
      -H "Content-Type: application/json" \\
      -d '{"query": "What is the revenue?", "top_k": 3}'
"""

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.shared.env import load_release_env
load_release_env(project_root)

from src.shared.service_factory import SharedServiceFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ========== Configuration ==========
DEFAULT_TOP_K = 5
DEFAULT_EXPERIMENT = "default"


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ========== Request/Response Models ==========
class RetrieveRequest(BaseModel):
    """Request model for retrieval endpoint."""
    # Our format
    query: Optional[str] = None
    top_k: int = DEFAULT_TOP_K
    # Search-R1 compatible format
    queries: Optional[List[str]] = None
    topk: Optional[int] = None
    # Options
    rerank: Optional[bool] = None  # Override server default


class RetrieveResult(BaseModel):
    """Single retrieval result."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class RetrieveResponse(BaseModel):
    """Response model for retrieval endpoint."""
    results: List[RetrieveResult]
    query: str
    reranked: bool


# ========== Global State ==========
retrieval_provider = None
provider_factory = None
server_config: Dict[str, Any] = {}


# ========== FastAPI App ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage retrieval provider lifecycle."""
    global retrieval_provider, provider_factory

    # Startup
    logger.info("Starting retriever server...")
    experiment_id = server_config.get("experiment_id", DEFAULT_EXPERIMENT)
    enable_rerank = server_config.get("enable_rerank", True)
    services_config = server_config.get("services_config", "configs/services.yaml")
    config_path = server_config.get("config_path", "configs/retriever_server.yaml")

    # Load retriever server config (self-contained, independent from providers.yaml)
    retriever_config = load_config(config_path)

    # Override retrieval type if specified via CLI
    retrieval_override = server_config.get("retrieval_override")
    if retrieval_override:
        if "retrieval" not in retriever_config:
            retriever_config["retrieval"] = {"params": {}}
        retriever_config["retrieval"]["type"] = retrieval_override
        logger.info(f"Retrieval type overridden to: {retrieval_override}")

    # Override rerank setting if specified
    if not enable_rerank:
        if "retrieval" not in retriever_config:
            retriever_config["retrieval"] = {"params": {}}
        if "params" not in retriever_config["retrieval"]:
            retriever_config["retrieval"]["params"] = {}
        retriever_config["retrieval"]["params"]["use_reranker"] = False

    # Initialize service factory
    service_factory = SharedServiceFactory.from_yaml(services_config)

    # Import ProviderFactory here to avoid circular imports
    from src.providers import ProviderFactory

    # Create provider factory and retrieval provider ONLY (no generator needed!)
    provider_factory = ProviderFactory.from_dict(
        config=retriever_config,
        service_factory=service_factory,
        experiment_id=experiment_id,
    )
    retrieval_provider = await provider_factory.create_retrieval_provider()

    if retrieval_provider is None:
        raise RuntimeError("Retrieval provider could not be created. Check configs/retriever_server.yaml retrieval config.")

    retrieval_name = retrieval_provider.get_name()
    collection_name = f"{service_factory.collections.get('rag', 'research_rag')}_{experiment_id}"

    logger.info(f"Retriever server ready:")
    logger.info(f"  Retrieval provider: {retrieval_name}")
    logger.info(f"  Collection: {collection_name}")
    logger.info(f"  Rerank enabled: {enable_rerank}")

    yield

    # Shutdown
    logger.info("Shutting down retriever server...")
    if retrieval_provider:
        await retrieval_provider.aclose()
    if provider_factory:
        await provider_factory.cleanup()


app = FastAPI(
    title="Research Core Retriever Server",
    description="RAG retrieval API compatible with Search-R1",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Server info."""
    return {
        "status": "ok",
        "service": "Research Core Retriever Server",
        "experiment": server_config.get("experiment_id", DEFAULT_EXPERIMENT),
        "rerank_enabled": server_config.get("enable_rerank", True),
        "profile": server_config.get("profile", "cloud"),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    is_ready = retrieval_provider is not None
    return {
        "status": "healthy" if is_ready else "initializing",
        "experiment": server_config.get("experiment_id", DEFAULT_EXPERIMENT),
        "retrieval_provider": retrieval_provider.get_name() if retrieval_provider else None,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(request: RetrieveRequest):
    """
    Retrieve relevant chunks for a query.

    Supports two request formats:
    1. Simple: {"query": "...", "top_k": 5}
    2. Search-R1: {"queries": ["..."], "topk": 5}
    """
    if not retrieval_provider:
        raise HTTPException(status_code=503, detail="Retrieval provider not initialized")

    # Parse request (support both formats)
    if request.queries and len(request.queries) > 0:
        query = request.queries[0]
    elif request.query:
        query = request.query
    else:
        raise HTTPException(status_code=400, detail="No query provided")

    top_k = request.topk if request.topk else request.top_k

    # Determine rerank setting
    skip_rerank = False
    if request.rerank is not None:
        skip_rerank = not request.rerank
    elif not server_config.get("enable_rerank", True):
        skip_rerank = True

    # Call retrieval provider directly (no generator, no pipeline!)
    try:
        retrieval_result = await retrieval_provider.retrieve(
            query=query,
            top_k=top_k,
            skip_rerank=skip_rerank,
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Get chunks from retrieval result
    chunks = retrieval_result.relevant_chunks if retrieval_result.relevant_chunks else retrieval_result.chunks
    was_reranked = retrieval_result.metadata.get("reranked", False) if retrieval_result.metadata else False

    # Format response
    formatted_results = []
    for idx, chunk in enumerate(chunks[:top_k]):
        result_item = RetrieveResult(
            id=str(chunk.id),
            text=chunk.content,
            score=float(chunk.score),
            metadata=chunk.metadata,
        )
        formatted_results.append(result_item)

        # Debug: Print first 3 chunks
        if idx < 3:
            logger.info(f"\n{'='*80}\nChunk {idx + 1} being sent:\n{'='*80}")
            logger.info(f"ID: {result_item.id}")
            logger.info(f"Score: {result_item.score:.4f}")
            text_preview = result_item.text[:200] if result_item.text else ""
            logger.info(f"Text preview: {text_preview}...")
            logger.info(f"{'='*80}\n")

    logger.info(f"Query: '{query[:50]}...' | Returned: {len(formatted_results)} | Reranked: {was_reranked}")

    return RetrieveResponse(
        results=formatted_results,
        query=query,
        reranked=was_reranked,
    )


@app.post("/search", response_model=RetrieveResponse)
async def search_endpoint(request: RetrieveRequest):
    """Alias for /retrieve (Search-R1 compatible)."""
    return await retrieve_endpoint(request)


def main():
    """Main entry point."""
    global server_config
    
    parser = argparse.ArgumentParser(
        description="Research Core Retriever Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default=DEFAULT_EXPERIMENT,
        help=f"Experiment ID for collection (default: {DEFAULT_EXPERIMENT})"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=18000,
        help="Port to run server on (default: 18000)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking for faster retrieval"
    )
    
    parser.add_argument(
        "--services-config",
        type=str,
        default="configs/services.yaml",
        help="Path to services config (default: configs/services.yaml)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/retriever_server.yaml",
        help="Path to retriever server config (default: configs/retriever_server.yaml)"
    )

    parser.add_argument(
        "--retrieval",
        type=str,
        choices=["vector_search", "standard_rag", "self_rag", "graded_rag", "hybrid_rag"],
        default=None,
        help="Retrieval provider type (overrides retriever_server.yaml)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Store config for lifespan handler
    server_config = {
        "experiment_id": args.experiment,
        "enable_rerank": not args.no_rerank,
        "services_config": args.services_config,
        "config_path": args.config,
        "retrieval_override": args.retrieval,
    }
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Rerank enabled: {not args.no_rerank}")
    
    uvicorn.run(
        "retriever_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
