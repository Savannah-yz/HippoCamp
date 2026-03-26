#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = ROOT.parent                     # analysis/
RESULT_ROOT = ANALYSIS_ROOT / "result"
DIFF_SCRIPT = ANALYSIS_ROOT / "difficulty" / "generate_difficulty_reports.py"
OUTPUT_DIR = ANALYSIS_ROOT / "figs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_LABELS = {
    "college": "(a) Bei",
    "law": "(b) Adam",
    "finance": "(c) Victoria",
}
DOMAIN_ORDER = ["(a) Bei", "(b) Adam", "(c) Victoria", "Benchmark"]

METHOD_FILES = {
    "college": {
        "Standard RAG": RESULT_ROOT / "college_standardrag" / "evaluation_results.json",
        "Self RAG": RESULT_ROOT / "college_selfrag_evaluation.json",
        "Search-R1": RESULT_ROOT / "college_searchr1_evaluation.json",
        "ReAct (Qwen3-VL-8B-Instruct)": RESULT_ROOT / "college_qwen_react_evaluation.json",
        "ReAct (Gemini-2.5-flash)": RESULT_ROOT / "college_gemini_react_selfrag_evaluation.json",
        "ChatGPT Agent Mode": RESULT_ROOT / "college_agent_mode_evaluate.json",
        "Terminal Agent (Qwen3-VL-8B-Instruct)": RESULT_ROOT / "college_docker_qwen" / "judge_results.json",
        "Terminal Agent (Gemini-2.5-flash)": RESULT_ROOT / "college_docker_gemini" / "judge_results.json",
        "Terminal Agent (ChatGPT-5.2)": RESULT_ROOT / "college_docker_chatgpt" / "judge_results.json",
    },
    "law": {
        "Standard RAG": RESULT_ROOT / "law_standardrag" / "evaluation_results.json",
        "Self RAG": RESULT_ROOT / "law_selfrag_evaluation.json",
        "Search-R1": RESULT_ROOT / "law_searchr1_evaluation.json",
        "ReAct (Qwen3-VL-8B-Instruct)": RESULT_ROOT / "law_qwen_react_evaluation.json",
        "ReAct (Gemini-2.5-flash)": RESULT_ROOT / "law_gemini_react_selfrag_evaluation.json",
        "ChatGPT Agent Mode": RESULT_ROOT / "law_agent_mode_evaluate.json",
        "Terminal Agent (Qwen3-VL-8B-Instruct)": RESULT_ROOT / "law_docker_qwen" / "judge_results.json",
        "Terminal Agent (Gemini-2.5-flash)": RESULT_ROOT / "law_docker_gemini" / "judge_results.json",
        "Terminal Agent (ChatGPT-5.2)": RESULT_ROOT / "law_docker_chatgpt" / "judge_results.json",
    },
    "finance": {
        "Standard RAG": RESULT_ROOT / "finance_standardrag" / "evaluation_results.json",
        "Self RAG": RESULT_ROOT / "finance_selfrag_evaluation.json",
        "Search-R1": RESULT_ROOT / "finance_searchr1_evaluation.json",
        "ReAct (Qwen3-VL-8B-Instruct)": RESULT_ROOT / "finance_qwen_react_evaluation.json",
        "ReAct (Gemini-2.5-flash)": RESULT_ROOT / "finance_gemini_react_selfrag_evaluation.json",
        "ChatGPT Agent Mode": RESULT_ROOT / "finance_agent_mode_evaluate.json",
        "Terminal Agent (Qwen3-VL-8B-Instruct)": RESULT_ROOT / "finance_docker_qwen" / "judge_results.json",
        "Terminal Agent (Gemini-2.5-flash)": RESULT_ROOT / "finance_docker_gemini" / "judge_results.json",
        "Terminal Agent (ChatGPT-5.2)": RESULT_ROOT / "finance_docker_chatgpt" / "judge_results.json",
    },
}

METHOD_ORDER = [
    "Terminal Agent (Qwen3-VL-8B-Instruct)",
    "Terminal Agent (ChatGPT-5.2)",
    "ChatGPT Agent Mode",
    "Terminal Agent (Gemini-2.5-flash)",
    "Self RAG",
    "Standard RAG",
    "ReAct (Gemini-2.5-flash)",
    "ReAct (Qwen3-VL-8B-Instruct)",
    "Search-R1",
]

METHOD_LABELS = {
    "Terminal Agent (Qwen3-VL-8B-Instruct)": "Terminal Agent (Qwen3-VL-8B-Instruct)",
    "Terminal Agent (ChatGPT-5.2)": "Terminal Agent (ChatGPT-5.2)",
    "ChatGPT Agent Mode": "ChatGPT Agent Mode",
    "Terminal Agent (Gemini-2.5-flash)": "Terminal Agent (Gemini-2.5-flash)",
    "Self RAG": "Self RAG",
    "Standard RAG": "Standard RAG",
    "ReAct (Gemini-2.5-flash)": "ReAct (Gemini-2.5-flash)",
    "ReAct (Qwen3-VL-8B-Instruct)": "ReAct (Qwen3-VL-8B-Instruct)",
    "Search-R1": "Search-R1",
}

OUTPUT_SINGLE = {
    "(a) Bei": OUTPUT_DIR / "difficulty_vs_performance_bei.png",
    "(b) Adam": OUTPUT_DIR / "difficulty_vs_performance_adam.png",
    "(c) Victoria": OUTPUT_DIR / "difficulty_vs_performance_victoria.png",
}
OUTPUT_2X2 = OUTPUT_DIR / "difficulty_vs_performance_all.png"
OUTPUT_1X4 = OUTPUT_DIR / "difficulty_vs_performance_all_1x4.png"

OUTPUT_SINGLE_PDF = {
    "(a) Bei": OUTPUT_DIR / "difficulty_vs_performance_bei.pdf",
    "(b) Adam": OUTPUT_DIR / "difficulty_vs_performance_adam.pdf",
    "(c) Victoria": OUTPUT_DIR / "difficulty_vs_performance_victoria.pdf",
}
OUTPUT_2X2_PDF = OUTPUT_DIR / "difficulty_vs_performance_all.pdf"
OUTPUT_1X4_PDF = OUTPUT_DIR / "difficulty_vs_performance_all_1x4.pdf"


def load_difficulty_module():
    spec = importlib.util.spec_from_file_location("difficulty_core", DIFF_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {DIFF_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_qid(value) -> str | None:
    if value is None:
        return None
    try:
        return str(int(value))
    except Exception:
        s = str(value).strip()
        return s if s else None


def compute_difficulty_maps(diff_mod):
    # Build comparable difficulty scale across three domains.
    rows_by_domain = {}
    all_rows = []
    for domain in ["(a) Bei", "(b) Adam", "(c) Victoria"]:
        cfg = diff_mod.PROFILE_CONFIG[domain]
        records = diff_mod.load_json(cfg["json"])
        t_index = diff_mod.build_time_index(cfg["xlsx"])
        rows = []
        for item in records:
            feats = diff_mod.extract_features(item, t_index)
            row = {
                "id": normalize_qid(item.get("id")),
                "question": str(item.get("question", "")),
                "features": feats,
            }
            rows.append(row)
            all_rows.append(feats)
        rows_by_domain[domain] = rows

    stats = diff_mod.fit_feature_stats(all_rows)

    by_qid = {}
    by_query = {}
    for domain, rows in rows_by_domain.items():
        by_qid[domain] = {}
        by_query[domain] = {}
        for row in rows:
            score = float(diff_mod.compute_difficulty(row["features"], stats))
            if row["id"] is not None:
                by_qid[domain][row["id"]] = score
            if row["question"]:
                by_query[domain][row["question"]] = score
    return by_qid, by_query


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_judge_score(item: dict) -> float | None:
    judge = item.get("judge", {})
    if not isinstance(judge, dict):
        return None
    val = judge.get("llm_as_a_judge_score")
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def build_domain_method_points(by_qid, by_query):
    domain_points = {
        "(a) Bei": {m: {"difficulty": [], "score": []} for m in METHOD_ORDER},
        "(b) Adam": {m: {"difficulty": [], "score": []} for m in METHOD_ORDER},
        "(c) Victoria": {m: {"difficulty": [], "score": []} for m in METHOD_ORDER},
    }

    slug_to_label = DOMAIN_LABELS
    for slug, methods in METHOD_FILES.items():
        domain = slug_to_label[slug]
        for method, path in methods.items():
            if not path.exists():
                print(f"[WARN] Missing {path}")
                continue
            data = load_json(path)
            points = domain_points[domain][method]
            for item in data:
                score = extract_judge_score(item)
                if score is None:
                    continue
                qid = normalize_qid(item.get("query_id"))
                diff = by_qid[domain].get(qid) if qid is not None else None
                if diff is None:
                    query = str(item.get("query", ""))
                    diff = by_query[domain].get(query)
                if diff is None:
                    continue
                points["difficulty"].append(float(diff))
                points["score"].append(float(score))

    # Benchmark aggregation
    bench = {m: {"difficulty": [], "score": []} for m in METHOD_ORDER}
    for method in METHOD_ORDER:
        for d in ["(a) Bei", "(b) Adam", "(c) Victoria"]:
            bench[method]["difficulty"].extend(domain_points[d][method]["difficulty"])
            bench[method]["score"].extend(domain_points[d][method]["score"])
    domain_points["Benchmark"] = bench
    return domain_points


def aggregate_by_difficulty_bins(difficulties, scores, step=5):
    if not difficulties:
        return np.array([]), np.array([]), np.array([])
    x = np.array(difficulties, dtype=float)
    y = np.array(scores, dtype=float)
    bins = np.arange(0, 105, step, dtype=float)
    centers = (bins[:-1] + bins[1:]) / 2
    means = []
    counts = []
    valid_centers = []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        valid_centers.append(centers[i])
        means.append(float(y[mask].mean()))
        counts.append(n)
    return np.array(valid_centers), np.array(means), np.array(counts)


def build_method_colors():
    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(METHOD_ORDER)}


def plot_domain_panel(ax, domain, method_points, colors):
    for method in METHOD_ORDER:
        d = method_points[method]["difficulty"]
        s = method_points[method]["score"]
        xs, ys, counts = aggregate_by_difficulty_bins(d, s, step=5)
        if len(xs) == 0:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=3.8,
            linewidth=1.8,
            alpha=0.95,
            color=colors[method],
            zorder=3,
        )

    ax.set_title(domain, fontsize=14, fontweight="bold")
    ax.set_xlabel("Difficulty Score (binned)", fontsize=11)
    ax.set_ylabel("Avg LLM Judge Score (0-5)", fontsize=11)
    ax.set_xlim(25, 100)
    ax.set_ylim(0, 5)
    ax.set_yticks(np.arange(0, 5.1, 0.5))
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.grid(axis="x", linestyle=":", alpha=0.12)
    ax.set_axisbelow(True)


def method_legend_handles(colors):
    return [
        Line2D(
            [0],
            [0],
            color=colors[m],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=METHOD_LABELS[m],
        )
        for m in METHOD_ORDER
    ]


def save_single_figures(domain_points, colors):
    for domain in ["(a) Bei", "(b) Adam", "(c) Victoria"]:
        fig, ax = plt.subplots(figsize=(12.0, 7.1), dpi=220)
        plot_domain_panel(ax, domain, domain_points[domain], colors)
        ax.legend(handles=method_legend_handles(colors), loc="upper left", fontsize=8.8, ncol=2, framealpha=0.95)
        ax.text(
            0.99,
            0.02,
            "Each point = mean score in one difficulty bin",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#475569",
        )
        fig.tight_layout()
        out = OUTPUT_SINGLE[domain]
        out_pdf = OUTPUT_SINGLE_PDF[domain]
        fig.savefig(out, dpi=320, bbox_inches="tight")
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
        print(f"Saved: {out_pdf}")


def save_2x2(domain_points, colors):
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.8), dpi=220)
    axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, domain in zip(axes_list, DOMAIN_ORDER):
        plot_domain_panel(ax, domain, domain_points[domain], colors)

    fig.legend(
        handles=method_legend_handles(colors),
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Difficulty vs Performance (9 Methods, Question-level)", fontsize=18, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.94])
    fig.savefig(OUTPUT_2X2, dpi=320, bbox_inches="tight")
    fig.savefig(OUTPUT_2X2_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_2X2}")
    print(f"Saved: {OUTPUT_2X2_PDF}")


def save_1x4(domain_points, colors):
    fig, axes = plt.subplots(1, 4, figsize=(25.5, 6.3), dpi=220)
    for ax, domain in zip(axes, DOMAIN_ORDER):
        plot_domain_panel(ax, domain, domain_points[domain], colors)

    fig.legend(
        handles=method_legend_handles(colors),
        loc="lower center",
        ncol=5,
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle("Difficulty vs Performance (9 Methods, Question-level)", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0.01, 0.06, 0.995, 0.94])
    fig.savefig(OUTPUT_1X4, dpi=320, bbox_inches="tight")
    fig.savefig(OUTPUT_1X4_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_1X4}")
    print(f"Saved: {OUTPUT_1X4_PDF}")


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    diff_mod = load_difficulty_module()
    by_qid, by_query = compute_difficulty_maps(diff_mod)
    domain_points = build_domain_method_points(by_qid, by_query)

    colors = build_method_colors()
    save_single_figures(domain_points, colors)
    save_2x2(domain_points, colors)
    save_1x4(domain_points, colors)
    print("Done.")


if __name__ == "__main__":
    main()
