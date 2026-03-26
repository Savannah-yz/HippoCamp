from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

# Publication-style defaults (CVPR-friendly visual density and typography)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "figs"
DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = [
    ("Bei_files.xlsx", "(a) Bei", "#ffcbd4"),
    ("Victoria_files.xlsx", "(c) Victoria", "#c5e0b5"),
    ("Adam_files.xlsx", "(b) Adam", "#bfe3f5"),
]

MIN_DATE = pd.Timestamp("2010-01-01")
SPLIT_DATE = pd.Timestamp("2025-01-01")
MAX_DATE = pd.Timestamp("2026-12-31")
EARLY_SHARE = 0.30


@dataclass
class DatasetTimestamps:
    label: str
    color: str
    creation_scaled: np.ndarray
    modification_scaled: np.ndarray
    creation_count: int
    modification_count: int
    activity_timestamps: pd.Series


def blend_with_white(hex_color: str, alpha: float) -> tuple[float, float, float]:
    """Return a lighter variant by blending a color with white."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * alpha)


def datetime_series_to_num(series: pd.Series) -> np.ndarray:
    """Convert pandas timestamps to matplotlib date numbers."""
    return np.array([mdates.date2num(ts.to_pydatetime()) for ts in series], dtype=float)


def piecewise_scale(date_numbers: np.ndarray) -> np.ndarray:
    """Piecewise linear scale: compress 2010-2024 and expand 2025-2026."""
    min_num = mdates.date2num(MIN_DATE.to_pydatetime())
    split_num = mdates.date2num(SPLIT_DATE.to_pydatetime())
    max_num = mdates.date2num(MAX_DATE.to_pydatetime())

    clipped = np.clip(date_numbers, min_num, max_num)
    scaled = np.empty_like(clipped, dtype=float)

    early_mask = clipped <= split_num
    scaled[early_mask] = (
        (clipped[early_mask] - min_num) / (split_num - min_num) * EARLY_SHARE
    )
    scaled[~early_mask] = EARLY_SHARE + (
        (clipped[~early_mask] - split_num) / (max_num - split_num) * (1.0 - EARLY_SHARE)
    )
    return scaled


def scale_one_date(ts: pd.Timestamp) -> float:
    date_num = mdates.date2num(ts.to_pydatetime())
    return float(piecewise_scale(np.array([date_num]))[0])


def load_dataset(path: Path, label: str, color: str) -> DatasetTimestamps | None:
    if not path.exists():
        print(f"[Skip] Missing file: {path}")
        return None

    df = pd.read_excel(path)
    if "FileType" in df.columns:
        df = df[df["FileType"].astype(str).str.lower() != "folder"]

    creation = pd.to_datetime(df.get("creation_date"), errors="coerce").dropna()
    modification = pd.to_datetime(df.get("modification_date"), errors="coerce").dropna()

    if creation.empty or modification.empty:
        print(f"[Skip] Not enough timestamp data in: {path.name}")
        return None

    creation_scaled = piecewise_scale(datetime_series_to_num(creation))
    modification_scaled = piecewise_scale(datetime_series_to_num(modification))

    print(
        f"{label:>8} | creation={len(creation):4d}, modification={len(modification):4d}, file={path.name}"
    )

    return DatasetTimestamps(
        label=label,
        color=color,
        creation_scaled=creation_scaled,
        modification_scaled=modification_scaled,
        creation_count=len(creation),
        modification_count=len(modification),
        activity_timestamps=pd.concat([creation, modification], ignore_index=True),
    )


def build_plot(datasets: list[DatasetTimestamps]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 8.0), dpi=180)

    positions = np.arange(1, len(datasets) + 1, dtype=float)
    width = 0.34

    creation_data = [d.creation_scaled for d in datasets]
    modification_data = [d.modification_scaled for d in datasets]

    creation_bp = ax.boxplot(
        creation_data,
        positions=positions - width / 2.0,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=2.3),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        boxprops=dict(linewidth=1.8),
    )
    modification_bp = ax.boxplot(
        modification_data,
        positions=positions + width / 2.0,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=2.0),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        boxprops=dict(linewidth=1.8),
    )

    for box, dataset in zip(creation_bp["boxes"], datasets):
        box.set_facecolor(dataset.color)
        box.set_edgecolor(dataset.color)
        box.set_alpha(0.88)

    for box, dataset in zip(modification_bp["boxes"], datasets):
        box.set_facecolor(blend_with_white(dataset.color, alpha=0.72))
        box.set_edgecolor(dataset.color)
        box.set_hatch("///")
        box.set_alpha(0.95)

    tick_specs = [
        (pd.Timestamp("2012-01-01"), "2012"),
        (pd.Timestamp("2015-01-01"), "2015"),
        (pd.Timestamp("2018-01-01"), "2018"),
        (pd.Timestamp("2021-01-01"), "2021"),
        (pd.Timestamp("2024-01-01"), "2024"),
        (pd.Timestamp("2025-01-01"), "2025-01"),
        (pd.Timestamp("2025-04-01"), "2025-04"),
        (pd.Timestamp("2025-07-01"), "2025-07"),
        (pd.Timestamp("2025-10-01"), "2025-10"),
        (pd.Timestamp("2026-01-01"), "2026-01"),
        (pd.Timestamp("2026-04-01"), "2026-04"),
        (pd.Timestamp("2026-07-01"), "2026-07"),
    ]
    y_ticks = [scale_one_date(ts) for ts, _ in tick_specs]
    y_labels = [label for _, label in tick_specs]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlim(0.4, len(datasets) + 0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([d.label for d in datasets], fontweight="semibold")

    ax.set_ylabel("Timestamp", fontweight="bold")
    ax.set_xlabel("Profile", fontweight="bold")

    ax.grid(axis="y", color="#94A3B8", linewidth=0.8, alpha=0.30)
    ax.grid(axis="x", visible=False)
    ax.set_facecolor("#F8FAFC")

    legend_handles = [
        Patch(facecolor="#6B7280", edgecolor="#6B7280", alpha=0.88, label="Creation Time"),
        Patch(
            facecolor="#D1D5DB",
            edgecolor="#6B7280",
            hatch="///",
            alpha=0.95,
            label="Modification Time",
        ),
    ]
    legend_handles.extend(
        Patch(facecolor=d.color, edgecolor=d.color, label=d.label) for d in datasets
    )
    ax.legend(
        handles=legend_handles,
        ncol=2,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        borderpad=0.6,
    )

    fig.subplots_adjust(top=0.96, left=0.10, right=0.98, bottom=0.10)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HippoCamp timestamp distribution figure")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing Adam_files.xlsx, Bei_files.xlsx, and Victoria_files.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    global DATA_DIR, OUTPUT_DIR
    args = parse_args()
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Excel files from:", DATA_DIR)
    datasets: list[DatasetTimestamps] = []

    for file_name, label, color in DATASETS:
        loaded = load_dataset(DATA_DIR / file_name, label=label, color=color)
        if loaded is not None:
            datasets.append(loaded)

    if not datasets:
        raise SystemExit("No valid data found. Please check the Excel files in analysis/data/.")

    fig = build_plot(datasets)

    output_pdf = OUTPUT_DIR / "file_combined_boxplot_cvpr.pdf"
    output_png = OUTPUT_DIR / "file_combined_boxplot_cvpr.png"

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved PDF: {output_pdf}")
    print(f"Saved PNG: {output_png}")


if __name__ == "__main__":
    main()
