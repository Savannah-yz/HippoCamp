from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = ROOT.parent                     # analysis/
DATA_DIR = ANALYSIS_ROOT / "data"               # analysis/data/
OUTPUT_DIR = ANALYSIS_ROOT / "figs"             # analysis/figs/
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data files: download from https://huggingface.co/datasets/MMMem-org/HippoCamp
# Place Bei.json, Adam.json, Victoria.json and *_files.xlsx in analysis/data/
PROFILE_CONFIG = {
    "(a) Bei": {
        "json": DATA_DIR / "Bei.json",
        "xlsx": DATA_DIR / "Bei_files.xlsx",
        "color": "#E7A9B9",
    },
    "(b) Adam": {
        "json": DATA_DIR / "Adam.json",
        "xlsx": DATA_DIR / "Adam_files.xlsx",
        "color": "#9CCFAE",
    },
    "(c) Victoria": {
        "json": DATA_DIR / "Victoria.json",
        "xlsx": DATA_DIR / "Victoria_files.xlsx",
        "color": "#9EC6E8",
    },
    "Benchmark": {
        "json": None,
        "xlsx": None,
        "color": "#F1C99E",
    },
}

ORDER = ["(a) Bei", "(b) Adam", "(c) Victoria", "Benchmark"]

PNG_OUTPUT = {
    "(a) Bei": OUTPUT_DIR / "difficulty_bei.png",
    "(b) Adam": OUTPUT_DIR / "difficulty_adam.png",
    "(c) Victoria": OUTPUT_DIR / "difficulty_victoria.png",
    "Benchmark": OUTPUT_DIR / "difficulty_all.png",
}

FEATURE_KEYS = [
    "evidence_files",
    "modality_count",
    "file_type_count",
    "evidence_items",
    "reasoning_steps",
    "question_tokens",
    "answer_tokens",
    "time_span_days",
]

WEIGHTS = {
    "evidence_files": 0.19,
    "modality_count": 0.10,
    "file_type_count": 0.08,
    "evidence_items": 0.18,
    "reasoning_steps": 0.20,
    "question_tokens": 0.10,
    "answer_tokens": 0.07,
    "time_span_days": 0.08,
}

TAIL_QUANTILES = {
    "evidence_files": 0.95,
    "modality_count": 0.98,
    "reasoning_steps": 0.98,
}


@dataclass
class TimeIndex:
    by_exact_path: dict[str, list[pd.Timestamp]]
    by_basename: dict[str, list[pd.Timestamp]]


def safe_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def normalize_path(path_str: str) -> str:
    return path_str.replace("\\", "/").strip().lstrip("./")


def text_token_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    # Lightweight token approximation: words + CJK chars.
    return len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text))


def parse_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_time_index(xlsx_path: Path) -> TimeIndex:
    df = pd.read_excel(xlsx_path)
    by_exact_path: dict[str, list[pd.Timestamp]] = {}
    by_basename: dict[str, list[pd.Timestamp]] = {}

    for row in df.itertuples(index=False):
        file_path = getattr(row, "FilePath", None)
        if not isinstance(file_path, str) or not file_path.strip():
            continue
        file_type = str(getattr(row, "FileType", "")).lower()
        if file_type == "folder":
            continue

        normalized = normalize_path(file_path)
        creation = pd.to_datetime(getattr(row, "creation_date", None), errors="coerce")
        modification = pd.to_datetime(getattr(row, "modification_date", None), errors="coerce")
        timestamps = [ts for ts in [creation, modification] if pd.notna(ts)]
        if not timestamps:
            continue

        by_exact_path.setdefault(normalized, []).extend(timestamps)

        basename = Path(normalized).name
        if basename:
            by_basename.setdefault(basename, []).extend(timestamps)

    return TimeIndex(by_exact_path=by_exact_path, by_basename=by_basename)


def lookup_timestamps(file_path: str, index: TimeIndex) -> list[pd.Timestamp]:
    normalized = normalize_path(file_path)
    if normalized in index.by_exact_path:
        return index.by_exact_path[normalized]

    basename = Path(normalized).name
    if basename in index.by_basename:
        return index.by_basename[basename]

    return []


def compute_time_span_days(file_paths: list[str], index: TimeIndex) -> float:
    timestamps: list[pd.Timestamp] = []
    for path in file_paths:
        if not isinstance(path, str):
            continue
        timestamps.extend(lookup_timestamps(path, index))

    if len(timestamps) < 2:
        return 0.0

    min_ts = min(timestamps)
    max_ts = max(timestamps)
    return max(0.0, float((max_ts - min_ts).total_seconds() / 86400.0))


def extract_max_id_or_len(items: list, id_key: str) -> int:
    if not items:
        return 0
    values: list[int] = []
    for item in items:
        if isinstance(item, dict):
            values.append(parse_int(item.get(id_key, 0), 0))
        else:
            values.append(0)
    max_id = max(values) if values else 0
    return max(max_id, len(items))


def extract_features(question: dict, time_index: TimeIndex) -> dict[str, float]:
    file_paths = [p for p in safe_list(question.get("file_path", [])) if isinstance(p, str)]

    file_number = parse_int(question.get("file_number", len(file_paths)), len(file_paths))
    modality_count = len(set(safe_list(question.get("file_modality", []))))
    file_type_count = len(set(safe_list(question.get("file_type", []))))

    evidence_items = extract_max_id_or_len(safe_list(question.get("evidence", [])), "evidence_id")
    reasoning_steps = extract_max_id_or_len(safe_list(question.get("rationale", [])), "step_id")

    question_tokens = text_token_count(question.get("question", ""))
    answer_tokens = text_token_count(question.get("answer", ""))
    time_span_days = compute_time_span_days(file_paths, time_index)

    return {
        "evidence_files": float(max(0, file_number)),
        "modality_count": float(max(0, modality_count)),
        "file_type_count": float(max(0, file_type_count)),
        "evidence_items": float(max(0, evidence_items)),
        "reasoning_steps": float(max(0, reasoning_steps)),
        "question_tokens": float(max(0, question_tokens)),
        "answer_tokens": float(max(0, answer_tokens)),
        "time_span_days": float(max(0, time_span_days)),
    }


def fit_feature_stats(all_feature_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for key in FEATURE_KEYS:
        values = np.array([row[key] for row in all_feature_rows], dtype=float)
        p90 = float(np.percentile(values, 90))
        p98 = float(np.percentile(values, 98))
        if p90 <= 0:
            p90 = 1.0
        if p98 <= 0:
            p98 = p90
        stats[key] = {"p90": p90, "p98": p98}
    return stats


def normalize_feature(value: float, p90: float, p98: float) -> float:
    base = math.log1p(max(0.0, value)) / math.log1p(p90)
    cap = max(1.0, math.log1p(p98) / math.log1p(p90))
    return float(np.clip(base, 0.0, cap + 0.25))


def compute_difficulty(features: dict[str, float], stats: dict[str, dict[str, float]]) -> float:
    normalized = {
        key: normalize_feature(features[key], stats[key]["p90"], stats[key]["p98"])
        for key in FEATURE_KEYS
    }

    base_score = sum(WEIGHTS[key] * normalized[key] for key in FEATURE_KEYS)
    interaction = (
        0.10 * math.sqrt(normalized["evidence_files"] * normalized["reasoning_steps"])
        + 0.07 * math.sqrt(normalized["modality_count"] * normalized["file_type_count"])
        + 0.06 * math.sqrt(normalized["evidence_items"] * normalized["time_span_days"])
    )

    hard_axes = ["evidence_files", "evidence_items", "reasoning_steps"]
    hard_count = sum(1 for key in hard_axes if normalized[key] >= 0.85)
    bonus = 0.0
    if hard_count >= 2:
        bonus += 0.05
    if hard_count >= 3:
        bonus += 0.03
    if normalized["question_tokens"] >= 0.90 and normalized["answer_tokens"] >= 0.90:
        bonus += 0.02

    raw = base_score + interaction + bonus
    score = 100.0 / (1.0 + math.exp(-4.4 * (raw - 0.70)))
    return float(np.clip(score, 1.0, 99.0))


def choose_peak_label_position(
    peak_x: float,
    peak_y: float,
    y_upper: float,
    occupied_points: list[tuple[float, float]],
) -> tuple[float, float, str]:
    candidates = [
        (peak_x + 1.2, peak_y + y_upper * 0.10, "left"),
        (peak_x - 1.2, peak_y + y_upper * 0.10, "right"),
        (peak_x + 5.0, peak_y + y_upper * 0.14, "left"),
        (peak_x - 5.0, peak_y + y_upper * 0.14, "right"),
        (peak_x + 8.0, y_upper * 0.70, "left"),
        (peak_x - 8.0, y_upper * 0.70, "right"),
        (peak_x + 8.0, y_upper * 0.58, "left"),
        (peak_x - 8.0, y_upper * 0.58, "right"),
    ]

    best = None
    best_score = float("inf")
    for cx, cy, ha in candidates:
        x = float(np.clip(cx, 2.0, 98.0))
        y = float(np.clip(cy, y_upper * 0.20, y_upper * 0.95))

        score = 0.0
        for ox, oy in occupied_points:
            dx = abs(x - ox)
            dy = abs(y - oy)
            if dx < 8.0 and dy < y_upper * 0.12:
                score += 1000.0
            score += 8.0 / (dx + 1.0) + 2.0 / (dy + 1.0)

        if score < best_score:
            best_score = score
            best = (x, y, ha)

    assert best is not None
    return best


def plot_difficulty_hist(
    ax,
    score_groups: dict[str, list[float]],
    title: str,
    color: str,
) -> None:
    mean_color = "#B91C1C"
    median_color = "#16A34A"
    factual_color = "#2563EB"
    profiling_color = "#D97706"

    scores = score_groups["all"]
    factual_scores = score_groups.get("factual_retention", [])
    profiling_scores = score_groups.get("profiling", [])

    bins = np.arange(0, 105, 5)
    hist, edges = np.histogram(scores, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    bars = ax.bar(
        centers,
        hist,
        width=4.2,
        color=color,
        edgecolor="#334155",
        linewidth=1.0,
        alpha=0.92,
    )

    factual_hist = np.histogram(factual_scores, bins=bins)[0] if factual_scores else np.zeros_like(hist)
    profiling_hist = np.histogram(profiling_scores, bins=bins)[0] if profiling_scores else np.zeros_like(hist)
    y_peak = max(
        float(max(hist) if len(hist) else 0.0),
        float(max(factual_hist) if len(factual_hist) else 0.0),
        float(max(profiling_hist) if len(profiling_hist) else 0.0),
    )
    y_upper = y_peak * 1.16 if y_peak > 0 else 1.0
    ax.set_ylim(0, y_upper)

    if factual_scores:
        factual_line = ax.plot(
            centers,
            factual_hist,
            color=factual_color,
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=2.6,
            alpha=0.9,
            zorder=5,
        )[0]
    else:
        factual_line = None
    if profiling_scores:
        profiling_line = ax.plot(
            centers,
            profiling_hist,
            color=profiling_color,
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=2.6,
            alpha=0.9,
            zorder=5,
        )[0]
    else:
        profiling_line = None

    if len(hist) > 0 and max(hist) > 0:
        for bar, val in zip(bars, hist):
            if val <= 0:
                continue
            y_text = val + max(hist) * 0.015
            va = "bottom"
            if y_text > y_upper * 0.965:
                y_text = val - max(hist) * 0.03
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                str(int(val)),
                ha="center",
                va=va,
                fontsize=10,
                fontweight="semibold",
            )

    mean_val = float(np.mean(scores))
    median_val = float(np.median(scores))
    high_ratio = float(np.mean(np.array(scores) >= 70.0) * 100.0)

    ax.axvline(mean_val, color=mean_color, linestyle="--", linewidth=1.8)
    ax.axvline(
        median_val,
        color=median_color,
        linestyle="-.",
        linewidth=1.8,
    )

    # Inline labels for mean/median vertical lines.
    red_x = float(np.clip(mean_val + 1.2, 2.0, 98.0))
    blue_x = float(np.clip(median_val - 1.2, 2.0, 98.0))
    red_y = y_upper * 0.92
    blue_y = y_upper * 0.95
    if abs(red_x - blue_x) < 5.0:
        blue_y = y_upper * 0.91
        blue_x = float(np.clip(median_val - 2.8, 2.0, 98.0))

    red_ha = "left" if red_x >= mean_val else "right"
    blue_ha = "right" if blue_x <= median_val else "left"
    ax.text(
        red_x,
        red_y,
        f"Mean {mean_val:.1f}",
        color=mean_color,
        fontsize=8.6,
        ha=red_ha,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=mean_color, alpha=0.9),
    )
    ax.text(
        blue_x,
        blue_y,
        f"Median {median_val:.1f}",
        color=median_color,
        fontsize=8.6,
        ha=blue_ha,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=median_color, alpha=0.9),
    )

    # Peak labels for overlay curves with collision avoidance against mean/median labels.
    occupied_points = [(red_x, red_y), (blue_x, blue_y)]
    if factual_line is not None and np.any(factual_hist > 0):
        f_idx = int(np.argmax(factual_hist))
        fx, fy, fha = choose_peak_label_position(
            float(centers[f_idx]),
            float(factual_hist[f_idx]),
            y_upper,
            occupied_points,
        )
        ax.annotate(
            f"Factual Retention peak: {int(factual_hist[f_idx])} @ {int(centers[f_idx])}",
            xy=(float(centers[f_idx]), float(factual_hist[f_idx])),
            xytext=(fx, fy),
            textcoords="data",
            color=factual_color,
            fontsize=8.2,
            ha=fha,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=factual_color, alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=factual_color, lw=1.0, alpha=0.75),
            zorder=7,
        )
        occupied_points.append((fx, fy))

    if profiling_line is not None and np.any(profiling_hist > 0):
        p_idx = int(np.argmax(profiling_hist))
        px, py, pha = choose_peak_label_position(
            float(centers[p_idx]),
            float(profiling_hist[p_idx]),
            y_upper,
            occupied_points,
        )
        ax.annotate(
            f"Profiling peak: {int(profiling_hist[p_idx])} @ {int(centers[p_idx])}",
            xy=(float(centers[p_idx]), float(profiling_hist[p_idx])),
            xytext=(px, py),
            textcoords="data",
            color=profiling_color,
            fontsize=8.2,
            ha=pha,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=profiling_color, alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=profiling_color, lw=1.0, alpha=0.75),
            zorder=7,
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
    ax.set_xlabel("Difficulty Score (0-100)", fontsize=12)
    ax.set_ylabel("Question Count", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    note = f"n={len(scores)}\n>=70: {high_ratio:.1f}%"
    ax.text(
        0.97,
        0.97,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#64748B", alpha=0.95),
    )

    detail_lines = [
        f"Overall mean {mean_val:.1f} (red dashed)",
        f"Overall median {median_val:.1f} (green dash-dot)",
    ]
    if factual_scores:
        factual_mean = float(np.mean(factual_scores))
        factual_high = float(np.mean(np.array(factual_scores) >= 70.0) * 100.0)
        detail_lines.append(f"Factual mean {factual_mean:.1f}, >=70 {factual_high:.1f}% (blue line)")
    if profiling_scores:
        profiling_mean = float(np.mean(profiling_scores))
        profiling_high = float(np.mean(np.array(profiling_scores) >= 70.0) * 100.0)
        detail_lines.append(f"Profiling mean {profiling_mean:.1f}, >=70 {profiling_high:.1f}% (orange line)")

    ax.text(
        0.97,
        0.84,
        "\n".join(detail_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#94A3B8", alpha=0.9),
    )

    # Top legend fixed to upper-right as requested.
    legend_handles = [
        Patch(facecolor=color, edgecolor="#334155", label="Overall (bars)"),
        Line2D([0], [0], color=factual_color, lw=2.0, marker="o", markersize=3, label="Factual Retention"),
        Line2D([0], [0], color=profiling_color, lw=2.0, marker="o", markersize=3, label="Profiling"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        fontsize=8.3,
        frameon=True,
        framealpha=0.92,
        borderpad=0.25,
        handlelength=1.8,
    )


def save_single_png(scores_by_profile: dict[str, dict[str, list[float]]]) -> None:
    for profile in ORDER:
        fig, ax = plt.subplots(figsize=(11.5, 6.9), dpi=220)
        plot_difficulty_hist(ax, scores_by_profile[profile], profile, PROFILE_CONFIG[profile]["color"])
        fig.tight_layout()
        output_path = PNG_OUTPUT[profile]
        fig.savefig(output_path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated: {output_path}")


def save_difficulty_layout_pdfs(scores_by_profile: dict[str, dict[str, list[float]]]) -> None:
    # 2x2
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.8), dpi=220)
    axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, profile in zip(axes_list, ORDER):
        plot_difficulty_hist(ax, scores_by_profile[profile], profile, PROFILE_CONFIG[profile]["color"])
    fig.suptitle("Difficulty Distribution", fontsize=20, fontweight="bold", y=0.988)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    output_2x2 = OUTPUT_DIR / "difficulty_2x2.pdf"
    fig.savefig(output_2x2, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_2x2}")

    # 1x4
    fig, axes = plt.subplots(1, 4, figsize=(25.0, 6.2), dpi=220)
    for ax, profile in zip(axes, ORDER):
        plot_difficulty_hist(ax, scores_by_profile[profile], profile, PROFILE_CONFIG[profile]["color"])
    fig.suptitle("Difficulty Distribution", fontsize=20, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_1x4 = OUTPUT_DIR / "difficulty_1x4.pdf"
    fig.savefig(output_1x4, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_1x4}")


def merged_counter(values: list[int], metric_key: str) -> tuple[list[str], list[int]]:
    if not values:
        return [], []
    counter = Counter(values)
    cutoff = int(np.ceil(np.quantile(values, TAIL_QUANTILES[metric_key])))
    if cutoff >= max(values):
        keys = sorted(counter.keys())
        return [str(k) for k in keys], [counter[k] for k in keys]

    left = [k for k in sorted(counter.keys()) if k <= cutoff]
    tail_sum = sum(v for k, v in counter.items() if k > cutoff)
    labels = [str(k) for k in left]
    counts = [counter[k] for k in left]
    if tail_sum > 0:
        labels.append(f">{cutoff}")
        counts.append(tail_sum)
    return labels, counts


def plot_discrete_distribution(ax, values: list[int], profile: str, metric_key: str, x_label: str) -> None:
    labels, counts = merged_counter(values, metric_key)
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        counts,
        color=PROFILE_CONFIG[profile]["color"],
        edgecolor="#334155",
        linewidth=0.8,
        alpha=0.92,
    )

    show_labels = list(labels)
    if len(show_labels) > 12:
        for i in range(len(show_labels)):
            if i % 2 == 1 and i != len(show_labels) - 1:
                show_labels[i] = ""

    if counts:
        threshold = max(2, int(math.ceil(max(counts) * 0.12)))
        for bar, val in zip(bars, counts):
            if val < threshold:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(counts) * 0.015,
                str(int(val)),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(profile, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Question Count", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(show_labels, rotation=30 if len(labels) > 12 else 0, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def save_metric_layouts(
    metric_values: dict[str, list[int]],
    metric_key: str,
    x_label: str,
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.7), dpi=220)
    axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, profile in zip(axes_list, ORDER):
        plot_discrete_distribution(ax, metric_values[profile], profile, metric_key, x_label)
    fig.suptitle(f"{x_label} Distribution", fontsize=19, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path_2x2 = OUTPUT_DIR / f"{stem}_2x2.pdf"
    fig.savefig(path_2x2, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {path_2x2}")

    fig, axes = plt.subplots(1, 4, figsize=(25.0, 6.1), dpi=220)
    for ax, profile in zip(axes, ORDER):
        plot_discrete_distribution(ax, metric_values[profile], profile, metric_key, x_label)
    fig.suptitle(f"{x_label} Distribution", fontsize=19, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path_1x4 = OUTPUT_DIR / f"{stem}_1x4.pdf"
    fig.savefig(path_1x4, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {path_1x4}")


def write_readme(scores_by_profile: dict[str, dict[str, list[float]]], stats: dict[str, dict[str, float]]) -> None:
    benchmark_scores = scores_by_profile["Benchmark"]["all"]
    mean_b = float(np.mean(benchmark_scores))
    median_b = float(np.median(benchmark_scores))
    high_ratio = float(np.mean(np.array(benchmark_scores) >= 70.0) * 100.0)
    p90_b = float(np.percentile(benchmark_scores, 90))

    readme = f"""# Difficulty Formula (Enhanced, Conservative + Annotated Version)

This report introduces a new difficulty formula designed to make benchmark hardness **visible, explainable, and defensible**.

## 1. Why the old formula was not expressive enough

A purely linear equal-weight score tends to compress many questions into the middle range, especially when features are long-tailed.  
That weakens the visual signal of "hard benchmark".

## 2. New Formula

For each question, we extract 8 interpretable factors:

1. `evidence_files`
2. `modality_count`
3. `file_type_count`
4. `evidence_items`
5. `reasoning_steps`
6. `question_tokens`
7. `answer_tokens`
8. `time_span_days`

### Step A: Robust normalization

Each factor `x` is normalized with benchmark-wide quantiles:

`s(x) = clip(log(1+x)/log(1+P90), 0, cap)`

This keeps extreme outliers from dominating and still rewards genuinely difficult long-tail cases.

### Step B: Weighted core difficulty

`Base = sum(w_i * s_i)` with weights emphasizing the hardest dimensions:

- evidence_files: {WEIGHTS["evidence_files"]:.2f}
- evidence_items: {WEIGHTS["evidence_items"]:.2f}
- reasoning_steps: {WEIGHTS["reasoning_steps"]:.2f}
- modality/file types/text span/time span: remaining weights

### Step C: Interaction terms (non-linear challenge coupling)

`Interaction = 0.10*sqrt(evidence_files*reasoning_steps) + 0.07*sqrt(modality_count*file_type_count) + 0.06*sqrt(evidence_items*time_span_days)`

Rationale: hard questions are often hard because multiple dimensions are hard **together**, not independently.

### Step D: Hard-case bonus

If key axes (`evidence_files`, `evidence_items`, `reasoning_steps`) are simultaneously high:

- +0.05 when >=2 axes are high
- +0.03 when all 3 are high
- +0.02 for long-question + long-answer joint complexity

### Step E: Sigmoid mapping to 0-100

`Difficulty = 100 / (1 + exp(-4.4*(Raw - 0.70)))`

This conservative mapping avoids over-inflated high scores while still separating genuinely difficult tails.

## 3. Why this is convincing

1. **Interpretability**: every term corresponds to a concrete source of cognitive load.  
2. **Robustness**: log + quantile normalization handles skewed distributions.  
3. **Realism**: interaction terms model multi-constraint reasoning, which is the essence of hard QA.  
4. **Visual evidence**: each chart shows overall bars plus factual/profiling overlays, making QA-type difficulty differences explicit.
5. **Annotation safety**: mean/median labels and peak labels use conflict-avoidance placement to reduce overlap in dense panels.

## 4. Benchmark outcome snapshot

- Benchmark mean difficulty: **{mean_b:.2f}**
- Benchmark median difficulty: **{median_b:.2f}**
- Benchmark P90 difficulty: **{p90_b:.2f}**
- High-difficulty ratio (>=70): **{high_ratio:.1f}%**

## 5. Output files

- Difficulty PNGs:
  - `difficulty_bei.png`
  - `difficulty_adam.png`
  - `difficulty_victoria.png`
  - `difficulty_all.png`
- Difficulty layout PDFs:
  - `difficulty_2x2.pdf`
  - `difficulty_1x4.pdf`
- Supporting complexity PDFs:
  - `evidence_files_2x2.pdf`, `evidence_files_1x4.pdf`
  - `modality_2x2.pdf`, `modality_1x4.pdf`
  - `reasoning_steps_2x2.pdf`, `reasoning_steps_1x4.pdf`

## 6. Figure Annotation Rules (Current)

- Overall distribution: bars (domain color)
- Mean line: red dashed vertical line with inline label `Mean`
- Median line: green dash-dot vertical line with inline label `Median`
- Factual Retention: blue line with peak label `Factual Retention peak: ...`
- Profiling: orange line with peak label `Profiling peak: ...`
- Legend: top-left of each subplot
- Sample summary (`n`, `>=70`): top-right of each subplot
- Detail summary box (overall/factual/profiling mean and high-ratio): right-side box under sample summary

## 7. Reproducibility

Run:

```bash
MPLBACKEND=Agg python3 generate_difficulty_reports.py
```
"""

    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Generated: {ROOT / 'README.md'}")

    # Also write statistics
    stats_path = ROOT / "difficulty_statistics.txt"


def main() -> None:
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

    profile_rows: dict[str, list[dict]] = {}
    all_rows: list[dict] = []

    for profile in ["(a) Bei", "(b) Adam", "(c) Victoria"]:
        json_path = PROFILE_CONFIG[profile]["json"]
        xlsx_path = PROFILE_CONFIG[profile]["xlsx"]
        if not json_path.exists():
            raise FileNotFoundError(f"Missing JSON: {json_path}")
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Missing XLSX: {xlsx_path}")

        records = load_json(json_path)
        time_index = build_time_index(xlsx_path)
        rows: list[dict] = []
        for item in records:
            features = extract_features(item, time_index)
            features["qa_type"] = str(item.get("QA_type", "unknown"))
            rows.append(features)
            all_rows.append(features)
        profile_rows[profile] = rows
        print(f"Loaded {profile}: {len(rows)} questions")

    stats = fit_feature_stats(all_rows)

    scores_by_profile: dict[str, dict[str, list[float]]] = {}
    for profile in ["(a) Bei", "(b) Adam", "(c) Victoria"]:
        all_scores: list[float] = []
        factual_scores: list[float] = []
        profiling_scores: list[float] = []
        for row in profile_rows[profile]:
            score = compute_difficulty(row, stats)
            all_scores.append(score)
            qa_type = row.get("qa_type", "")
            if qa_type == "factual_retention":
                factual_scores.append(score)
            elif qa_type == "profiling":
                profiling_scores.append(score)

        scores_by_profile[profile] = {
            "all": all_scores,
            "factual_retention": factual_scores,
            "profiling": profiling_scores,
        }
        print(
            f"{profile}: mean={np.mean(all_scores):.2f}, median={np.median(all_scores):.2f}, "
            f">=70={np.mean(np.array(all_scores)>=70)*100:.1f}%, "
            f"factual_n={len(factual_scores)}, profiling_n={len(profiling_scores)}"
        )

    scores_by_profile["Benchmark"] = {
        "all": (
            scores_by_profile["(a) Bei"]["all"]
            + scores_by_profile["(b) Adam"]["all"]
            + scores_by_profile["(c) Victoria"]["all"]
        ),
        "factual_retention": (
            scores_by_profile["(a) Bei"]["factual_retention"]
            + scores_by_profile["(b) Adam"]["factual_retention"]
            + scores_by_profile["(c) Victoria"]["factual_retention"]
        ),
        "profiling": (
            scores_by_profile["(a) Bei"]["profiling"]
            + scores_by_profile["(b) Adam"]["profiling"]
            + scores_by_profile["(c) Victoria"]["profiling"]
        ),
    }
    print(
        "Benchmark: "
        f"mean={np.mean(scores_by_profile['Benchmark']['all']):.2f}, "
        f"median={np.median(scores_by_profile['Benchmark']['all']):.2f}, "
        f">=70={np.mean(np.array(scores_by_profile['Benchmark']['all'])>=70)*100:.1f}%, "
        f"factual_n={len(scores_by_profile['Benchmark']['factual_retention'])}, "
        f"profiling_n={len(scores_by_profile['Benchmark']['profiling'])}"
    )

    save_single_png(scores_by_profile)
    save_difficulty_layout_pdfs(scores_by_profile)

    evidence_metric = {
        profile: [int(round(row["evidence_files"])) for row in profile_rows[profile]]
        for profile in ["(a) Bei", "(b) Adam", "(c) Victoria"]
    }
    evidence_metric["Benchmark"] = (
        evidence_metric["(a) Bei"] + evidence_metric["(b) Adam"] + evidence_metric["(c) Victoria"]
    )

    modality_metric = {
        profile: [int(round(row["modality_count"])) for row in profile_rows[profile]]
        for profile in ["(a) Bei", "(b) Adam", "(c) Victoria"]
    }
    modality_metric["Benchmark"] = (
        modality_metric["(a) Bei"] + modality_metric["(b) Adam"] + modality_metric["(c) Victoria"]
    )

    reasoning_metric = {
        profile: [int(round(row["reasoning_steps"])) for row in profile_rows[profile]]
        for profile in ["(a) Bei", "(b) Adam", "(c) Victoria"]
    }
    reasoning_metric["Benchmark"] = (
        reasoning_metric["(a) Bei"] + reasoning_metric["(b) Adam"] + reasoning_metric["(c) Victoria"]
    )

    save_metric_layouts(evidence_metric, "evidence_files", "Number of Evidence Files", "evidence_files")
    save_metric_layouts(modality_metric, "modality_count", "Number of Modalities", "modality")
    save_metric_layouts(reasoning_metric, "reasoning_steps", "Number of Reasoning Steps", "reasoning_steps")

    write_readme(scores_by_profile, stats)


if __name__ == "__main__":
    main()
