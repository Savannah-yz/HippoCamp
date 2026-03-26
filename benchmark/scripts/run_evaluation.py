#!/usr/bin/env python3
"""
Evaluation Script for Research Core
====================================

Run evaluation metrics on query results.

Usage:
    # Simplest: point to a repo directory (auto-finds latest summary, auto-saves)
    python scripts/run_evaluation.py data_local/searchr1_standardrag_victoria_top20

    # Specify metrics explicitly
    python scripts/run_evaluation.py data_local/searchr1_standardrag_victoria_top20 --metrics llm_judge

    # Use a specific summary file (backward-compatible)
    python scripts/run_evaluation.py data_local/repo/summary_20260210_175336.json --metrics rouge bleu

    # Override output path
    python scripts/run_evaluation.py data_local/repo -o custom_output.json

    # Disable file-list metrics (enabled by default)
    python scripts/run_evaluation.py data_local/repo --no-file-list

    # Use detailed LLM judge prompt template
    python scripts/run_evaluation.py data_local/repo --judge-template detailed

    # Dry run (don't save output files)
    python scripts/run_evaluation.py data_local/repo --no-save

    # Show only summary statistics
    python scripts/run_evaluation.py data_local/repo --summary-only

Available metrics (use --list-metrics to see all):
    - rouge, bleu, exact_match, covered_exact_match
    - semantic_similarity, bert_score
    - retrieval_precision, retrieval_recall, retrieval_f1
    - llm_judge (requires Azure OpenAI config)

Requires environment variables (for llm_judge):
    AZURE_OPENAI_ENDPOINT - Full Azure OpenAI endpoint URL
    AZURE_OPENAI_API_KEY  - Azure OpenAI API key
"""

import sys
import os
import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from collections import defaultdict

from src.shared.env import load_release_env
load_release_env(project_root)

from src.rag.evaluation.runner import EvaluationRunner, AVAILABLE_METRICS
from src.rag.evaluation.file_list_metrics import (
    calculate_file_list_metrics,
    calculate_batch_file_list_metrics,
    add_file_list_metrics_to_result,
)

# ── QA type annotation files ────────────────────────────────────────────────
QA_ANNOTATION_FILES = {
    "adam": project_root / "analysis/data/Adam.json",
    "bei": project_root / "analysis/data/Bei.json",
    "victoria": project_root / "analysis/data/Victoria.json",
}

# Metric descriptions for --list-metrics
METRIC_DESCRIPTIONS = {
    "rouge": "ROUGE score - lexical similarity using n-gram overlap",
    "bleu": "BLEU score - n-gram precision for machine translation style evaluation",
    "exact_match": "Exact Match (EM) - 1.0 if normalized answer exactly matches ground truth",
    "covered_exact_match": "Covered Exact Match - 1.0 if ground truth is contained in answer",
    "semantic_similarity": "Semantic Similarity - cosine similarity using embeddings",
    "bert_score": "BERTScore - semantic evaluation using BERT embeddings",
    "retrieval_precision": "Retrieval Precision - precision of retrieved chunks",
    "retrieval_recall": "Retrieval Recall - recall of retrieved chunks",
    "retrieval_f1": "Retrieval F1 - F1 score of chunk retrieval",
    "llm_judge": "LLM Judge - uses LLM to evaluate answer quality (requires Azure OpenAI)",
}


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_input_path(input_arg: str) -> Path:
    """Resolve the input argument to a concrete summary JSON file path.

    If input_arg is a directory, find the latest summary_*.json file in it.
    If input_arg is a file, return it directly.
    """
    p = Path(input_arg)

    if not p.exists():
        print(f"Error: Path not found: {input_arg}")
        sys.exit(1)

    if p.is_file():
        return p

    if p.is_dir():
        summary_files = sorted(p.glob("summary_*.json"))
        if not summary_files:
            print(f"Error: No summary_*.json files found in: {input_arg}")
            sys.exit(1)
        latest = summary_files[-1]
        print(f"Auto-selected latest summary: {latest.name}")
        return latest

    print(f"Error: {input_arg} is neither a file nor a directory")
    sys.exit(1)


def print_available_metrics():
    """Print all available metrics with descriptions."""
    print("\n" + "=" * 70)
    print("AVAILABLE METRICS")
    print("=" * 70)
    print("\nAnswer Quality Metrics:")
    print("-" * 70)
    for name in ["rouge", "bleu", "exact_match", "covered_exact_match",
                 "semantic_similarity", "bert_score", "llm_judge"]:
        if name in METRIC_DESCRIPTIONS:
            print(f"  {name:25} {METRIC_DESCRIPTIONS[name]}")

    print("\nRetrieval Metrics:")
    print("-" * 70)
    for name in ["retrieval_precision", "retrieval_recall", "retrieval_f1"]:
        if name in METRIC_DESCRIPTIONS:
            print(f"  {name:25} {METRIC_DESCRIPTIONS[name]}")

    print("\nFile List Metrics (enabled by default, disable with --no-file-list):")
    print("-" * 70)
    print("  f1_score                 F1 score for file retrieval")
    print("  recall                   Recall for file retrieval")
    print("  precision                Precision for file retrieval")

    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("-" * 70)
    print("  # Simplest: just a directory (auto-finds summary, defaults to llm_judge + file-list)")
    print("  python scripts/run_evaluation.py data_local/searchr1_standardrag_victoria_top20")
    print("")
    print("  # Specify metrics explicitly")
    print("  python scripts/run_evaluation.py data_local/repo --metrics rouge bleu llm_judge")
    print("")
    print("  # Disable file list metrics")
    print("  python scripts/run_evaluation.py data_local/repo --no-file-list")
    print("=" * 70 + "\n")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def compute_latency_stats(input_data: list) -> dict | None:
    """Compute execution_time_ms statistics from input data."""
    times = [
        item["execution_time_ms"]
        for item in input_data
        if isinstance(item, dict) and "execution_time_ms" in item and item["execution_time_ms"] is not None
    ]
    if not times:
        return None
    return {
        "count": len(times),
        "total_ms": sum(times),
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def detect_bench_name(input_data: list) -> str | None:
    """Detect bench/collection name from input data's 'bench' field."""
    for item in input_data:
        bench = item.get("bench", "")
        if bench:
            return bench.lower()
    return None


def load_qa_type_map(bench_name: str) -> dict | None:
    """Load QA_type mapping for the given bench. Returns {query_id: qa_type} or None."""
    qa_file = QA_ANNOTATION_FILES.get(bench_name)
    if not qa_file or not qa_file.exists():
        return None
    with open(qa_file) as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        qid = str(item["id"])
        qa_type = item.get("QA_type")
        if qa_type:
            mapping[qid] = qa_type
    return mapping if mapping else None


def compute_qa_type_breakdown(
    output_results: list,
    input_data: list,
    qa_type_map: dict,
) -> dict:
    """Compute per-QA_type breakdown from evaluation results.

    Returns dict like: {
        "factual_retention": {"total": 103, "yes_count": 39, "accuracy": 0.3786, ...},
        "profiling": {...},
    }
    """
    # Build latency map from input data
    latency_map = {}
    for item in input_data:
        qid = str(item.get("query_id", ""))
        if "execution_time_ms" in item:
            latency_map[qid] = item["execution_time_ms"]

    groups = defaultdict(lambda: {
        "yes_count": 0, "total": 0, "score_sum": 0,
        "f1_sum": 0.0, "recall_sum": 0.0, "precision_sum": 0.0,
        "file_metric_count": 0, "latency_ms_list": [],
    })

    for item in output_results:
        qid = str(item.get("query_id", ""))
        qa_type = qa_type_map.get(qid, "unknown")
        judge = item.get("judge", {})
        pred = judge.get("pred", "no")
        score = judge.get("score_0_5", 0)
        flm = item.get("file_list_metrics") or {}
        f1 = flm.get("f1_score", 0.0)
        recall = flm.get("recall", 0.0)
        precision = flm.get("precision", 0.0)

        g = groups[qa_type]
        g["total"] += 1
        g["score_sum"] += score
        if item.get("file_list_metrics") is not None:
            g["file_metric_count"] += 1
            g["f1_sum"] += f1
            g["recall_sum"] += recall
            g["precision_sum"] += precision
        if pred == "yes":
            g["yes_count"] += 1
        if qid in latency_map:
            g["latency_ms_list"].append(latency_map[qid])

    breakdown = {}
    for qa_type in sorted(groups.keys()):
        g = groups[qa_type]
        n = g["total"]
        fn = g["file_metric_count"]
        accuracy = g["yes_count"] / n if n else 0
        mean_score = g["score_sum"] / n if n else 0
        qa_out = {
            "total": n,
            "yes_count": g["yes_count"],
            "accuracy": round(accuracy, 4),
            "mean_score": round(mean_score, 4),
        }
        if fn:
            qa_out["file_f1"] = round(g["f1_sum"] / fn, 4)
            qa_out["file_recall"] = round(g["recall_sum"] / fn, 4)
            qa_out["file_precision"] = round(g["precision_sum"] / fn, 4)
        lats = g["latency_ms_list"]
        if lats:
            qa_out["latency"] = {
                "total_ms": sum(lats),
                "mean_ms": round(sum(lats) / len(lats), 2),
                "min_ms": min(lats),
                "max_ms": max(lats),
            }
        breakdown[qa_type] = qa_out
    return breakdown


def print_qa_type_breakdown(breakdown: dict):
    """Pretty print QA type breakdown."""
    print("\n" + "-" * 60)
    print("QA TYPE BREAKDOWN")
    print("-" * 60)
    for qa_type, stats in breakdown.items():
        n = stats["total"]
        yes = stats["yes_count"]
        acc = stats["accuracy"] * 100
        score = stats["mean_score"]
        line = f"  {qa_type:22s}  n={n:3d}  yes={yes:3d}  acc={acc:5.1f}%  score={score:.2f}"
        if "file_f1" in stats:
            line += f"  f1={stats['file_f1']:.4f}"
        if "latency" in stats:
            line += f"  lat={stats['latency']['mean_ms']:.0f}ms"
        print(line)


def print_summary(summary: dict, file_list_summary: dict = None, latency_stats: dict = None):
    """Pretty print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total queries evaluated: {summary.get('total_queries', 0)}")
    print("-" * 60)

    metrics = summary.get("metrics", {})
    for metric_name, stats in metrics.items():
        print(f"\n{metric_name}:")
        print(f"  Mean:  {stats.get('mean', 0):.4f}")
        if "yes_count" in stats:
            yes_count = stats.get("yes_count", 0)
            accuracy = stats.get("accuracy", 0) * 100
            print(f"  Yes Count: {yes_count}")
            print(f"  Accuracy:  {accuracy:.1f}%")

    # Print execution time stats
    if latency_stats:
        print("\n" + "-" * 60)
        print("EXECUTION TIME")
        print("-" * 60)
        print(f"  Count:   {latency_stats['count']}")
        print(f"  Total:   {latency_stats['total_ms']} ms ({latency_stats['total_ms'] / 1000:.1f} s)")
        print(f"  Mean:    {latency_stats['mean_ms']:.1f} ms ({latency_stats['mean_ms'] / 1000:.1f} s)")
        print(f"  Min:     {latency_stats['min_ms']} ms")
        print(f"  Max:     {latency_stats['max_ms']} ms")

    # Print latency breakdown if available
    latency = summary.get("latency_breakdown")
    if latency:
        print("\n" + "-" * 60)
        print("LATENCY BREAKDOWN (mean, ms)")
        print("-" * 60)
        for key, stats in latency.items():
            mean_val = stats.get("mean", 0) if isinstance(stats, dict) else stats
            print(f"  {key:25} {mean_val:>10.1f}")

    # Print step counts if available
    step_counts = summary.get("step_counts")
    if step_counts:
        print("\n" + "-" * 60)
        print("STEP COUNTS (mean)")
        print("-" * 60)
        for key, stats in step_counts.items():
            mean_val = stats.get("mean", 0) if isinstance(stats, dict) else stats
            print(f"  {key:25} {mean_val:>10.2f}")

    # Print file list metrics summary if available
    if file_list_summary:
        print("\n" + "-" * 60)
        print("FILE LIST METRICS")
        print("-" * 60)
        print(f"  Total evaluated:    {file_list_summary.get('total_evaluated', 0)}")
        print(f"  Average F1 Score:   {file_list_summary.get('average_f1_score', 0):.4f}")
        print(f"  Average Recall:     {file_list_summary.get('average_recall', 0):.4f}")
        print(f"  Average Precision:  {file_list_summary.get('average_precision', 0):.4f}")

    print("\n" + "=" * 60)


def print_detailed_results(evaluations: list, structured: bool = True, show_file_list: bool = False):
    """Print detailed results for each query."""
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)

    for eval_result in evaluations:
        if structured:
            result_dict = eval_result.to_structured_dict()
        else:
            result_dict = eval_result.to_dict()

        print(f"\nQuery ID: {result_dict['query_id']}")
        query_preview = result_dict['query'][:100] + "..." if len(result_dict['query']) > 100 else result_dict['query']
        print(f"Query: {query_preview}")
        print("-" * 40)

        if structured:
            # Print judge results
            judge = result_dict.get("judge")
            if judge:
                print("\n  [LLM Judge]")
                print(f"    Score (0-5): {judge.get('score_0_5', 0)}")
                print(f"    Prediction: {judge.get('pred', 'no')}")
                print(f"    Normalized: {judge.get('score_normalized', 0):.4f}")
                print(f"    API Status: {judge.get('api_status', 'unknown')}")
                print(f"    Latency: {judge.get('eval_api_latency_seconds', 0):.3f}s")

            # Print simple metrics
            simple_metrics = result_dict.get("simple_metrics", {})
            if simple_metrics:
                print("\n  [Simple Metrics]")
                for name, data in simple_metrics.items():
                    score = data.get("score", 0)
                    print(f"    {name}: {score:.4f}")

            # Print file list metrics
            if show_file_list:
                file_list_metrics = result_dict.get("file_list_metrics")
                if file_list_metrics:
                    print("\n  [File List Metrics]")
                    print(f"    F1 Score:  {file_list_metrics.get('f1_score', 0):.4f}")
                    print(f"    Recall:    {file_list_metrics.get('recall', 0):.4f}")
                    print(f"    Precision: {file_list_metrics.get('precision', 0):.4f}")
        else:
            # Legacy format
            for metric in result_dict.get("metrics", []):
                score = metric.get("score", 0)
                name = metric.get("name", "unknown")
                print(f"  {name}: {score:.4f}")

        print()


async def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation metrics on query results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplest: just point to a directory
  python scripts/run_evaluation.py data_local/searchr1_standardrag_victoria_top20

  # Specify metrics
  python scripts/run_evaluation.py data_local/repo --metrics rouge bleu llm_judge

  # Use a specific file (backward-compatible)
  python scripts/run_evaluation.py data_local/repo/summary_20260210_175336.json --metrics llm_judge

  # Override output
  python scripts/run_evaluation.py data_local/repo -o custom_output.json

  # Disable file-list (enabled by default)
  python scripts/run_evaluation.py data_local/repo --no-file-list
"""
    )

    parser.add_argument(
        "input",
        nargs="?",  # Make optional for --list-metrics
        help="Path to a directory (auto-finds latest summary_*.json) or a specific JSON file"
    )
    parser.add_argument(
        "--metrics", "-m",
        nargs="+",
        choices=list(AVAILABLE_METRICS.keys()),
        help="Metrics to compute (default: llm_judge, configurable in evaluation.yaml)"
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Show all available metrics with descriptions and exit"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (default: auto-save to input directory)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save evaluation results to file (print only)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary statistics, not detailed results"
    )
    parser.add_argument(
        "--legacy-format",
        action="store_true",
        help="Use legacy output format (metrics as list instead of structured)"
    )
    parser.add_argument(
        "--file-list",
        action="store_true",
        default=None,
        help="Compute file list metrics (enabled by default)"
    )
    parser.add_argument(
        "--no-file-list",
        action="store_true",
        help="Disable file list metrics"
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="configs/evaluation.yaml",
        help="Path to evaluation config (default: configs/evaluation.yaml)"
    )
    parser.add_argument(
        "--judge-template",
        type=str,
        choices=["simple", "detailed"],
        default=None,
        help="LLM Judge prompt template (overrides eval config)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N queries (e.g. --limit 1 for a quick dry run)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Handle --list-metrics
    if args.list_metrics:
        print_available_metrics()
        sys.exit(0)

    # Require input if not listing metrics
    if not args.input:
        parser.error("input is required (use --list-metrics to see available metrics)")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ========== Load Evaluation Config ==========
    eval_config = load_config(args.eval_config)
    if eval_config:
        logger.info(f"Loaded evaluation config from: {args.eval_config}")

    # ========== Resolve Input Path ==========
    input_path = resolve_input_path(args.input)
    input_dir = input_path.parent

    # ========== Resolve Metrics ==========
    if args.metrics:
        metrics_to_use = args.metrics
    elif eval_config.get("metrics"):
        metrics_to_use = eval_config["metrics"]
        print(f"Using metrics from config: {metrics_to_use}")
    else:
        metrics_to_use = ["llm_judge"]
        print(f"Using default metrics: {metrics_to_use}")

    # ========== Resolve File List ==========
    if args.no_file_list:
        use_file_list = False
    elif args.file_list:
        use_file_list = True
    else:
        # Default from config, or True
        use_file_list = eval_config.get("file_list", True)

    # ========== Build metric_kwargs from eval config ==========
    llm_judge_config = eval_config.get("llm_judge", {})
    resolved_judge_template = args.judge_template or llm_judge_config.get("prompt_template", "simple")

    llm_judge_kwargs = {"prompt_template": resolved_judge_template}
    for key in ("api_url", "api_key", "timeout", "max_retries"):
        if key in llm_judge_config:
            llm_judge_kwargs[key] = llm_judge_config[key]
    if "prompts" in llm_judge_config:
        llm_judge_kwargs["prompts"] = llm_judge_config["prompts"]
    metric_kwargs = {"llm_judge": llm_judge_kwargs}

    # ========== Resolve Output Path ==========
    if args.no_save:
        output_file = None
    elif args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(input_dir / f"evaluation_{timestamp}.json")

    # ========== Print Config Summary ==========
    print(f"\n{'=' * 60}")
    print(f"Evaluation Configuration")
    print(f"{'=' * 60}")
    print(f"  Input:          {input_path}")
    if output_file:
        print(f"  Output:         {output_file}")
    else:
        print(f"  Output:         (dry run, no save)")
    print(f"  Metrics:        {metrics_to_use}")
    print(f"  File list:      {use_file_list}")
    print(f"  Judge template: {resolved_judge_template}")
    if args.limit:
        print(f"  Limit:          {args.limit}")
    print(f"{'=' * 60}\n")

    # Check Azure OpenAI config if llm_judge is requested
    if "llm_judge" in metrics_to_use:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_url = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not api_url:
            print("Warning: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set in .env")
            print("LLM judge metric will return 0 scores.")
        else:
            print(f"Azure OpenAI configured: {api_url[:50]}...")

    # Initialize runner with resolved metrics and config
    runner = EvaluationRunner(metrics=metrics_to_use, metric_kwargs=metric_kwargs)

    # Load input data
    print(f"Loading results from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    if isinstance(input_data, dict):
        input_data = [input_data]

    # Apply --limit
    if args.limit:
        total_available = len(input_data)
        input_data = input_data[:args.limit]
        print(f"Limited to first {args.limit} of {total_available} queries")

    if output_file:
        output_path_obj = Path(output_file)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Evaluate
    structured = not args.legacy_format
    evaluations = await runner.evaluate_results(input_data)

    # Add file list metrics to each evaluation result if enabled
    file_list_summary = None
    if use_file_list:
        print("\nComputing file list metrics...")
        # Create a map from query_id to input data for file_list lookup
        input_map = {item.get("query_id", ""): item for item in input_data}

        for eval_result in evaluations:
            query_id = eval_result.query_id
            if query_id in input_map:
                source_data = input_map[query_id]
                file_list = source_data.get("file_list")
                retrieved_file_list = source_data.get("retrieved_file_list")
                if file_list:
                    metrics = calculate_file_list_metrics(file_list, retrieved_file_list)
                    eval_result.file_list_metrics = metrics
                else:
                    eval_result.file_list_metrics = None
            else:
                eval_result.file_list_metrics = None

        # Calculate batch file list metrics
        file_list_summary = calculate_batch_file_list_metrics(input_data)

    # Build output_results list (needed for both saving and qa_type breakdown)
    output_results = []
    for eval_result in evaluations:
        if structured:
            result_dict = eval_result.to_structured_dict()
        else:
            result_dict = eval_result.to_dict()
        if use_file_list and hasattr(eval_result, 'file_list_metrics'):
            result_dict["file_list_metrics"] = eval_result.file_list_metrics
        output_results.append(result_dict)

    # Save evaluation results
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=2, ensure_ascii=False)
        print(f"Evaluation results saved to: {output_file}")

    # Print results
    if not args.summary_only:
        print_detailed_results(evaluations, structured=structured, show_file_list=use_file_list)

    # Compute latency stats from input data
    latency_stats = compute_latency_stats(input_data)

    # Print summary
    summary = runner.get_summary(evaluations, input_data=input_data)
    print_summary(summary, file_list_summary=file_list_summary, latency_stats=latency_stats)

    # ── QA type breakdown ────────────────────────────────────────
    qa_type_breakdown = None
    bench_name = detect_bench_name(input_data)
    if bench_name:
        qa_type_map = load_qa_type_map(bench_name)
        if qa_type_map:
            qa_type_breakdown = compute_qa_type_breakdown(
                output_results, input_data, qa_type_map
            )
            print_qa_type_breakdown(qa_type_breakdown)
        else:
            print(f"\nNo QA_type annotations found for bench '{bench_name}', skipping breakdown.")
    else:
        print("\nNo 'bench' field in input data, skipping QA type breakdown.")

    # Save summary to file
    if output_file:
        summary_file = str(Path(output_file).with_suffix(".summary.json"))
        summary_data = summary.copy()
        if file_list_summary:
            summary_data["file_list_metrics"] = file_list_summary
        if latency_stats:
            summary_data["execution_time"] = latency_stats
        if qa_type_breakdown:
            summary_data["qa_type_breakdown"] = qa_type_breakdown
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary saved to: {summary_file}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
