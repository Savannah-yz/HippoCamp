import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

modality_map = {
    "Text": ["bin", "ipynb", "log", "npy", "pkl", "pt", "pth", "py", "txt", "md", "pyc",
             "sh", "xml", "json", "sqlite", "yaml", "yml"],
    "Documents": ["docx", "eml", "pdf", "pptx", "ics", "csv", "doc", "rtf", "xlsx"],
    "Images": ["gif", "jpg", "png", "jpeg", "webp", "tiff", "svg", "heic", "bmp"],
    "Audio": ["mp3"],
    "Video": ["mp4", "mkv"],
}

ext_to_modality = {}
for modality, extensions in modality_map.items():
    for ext in extensions:
        ext_to_modality[ext.lower()] = modality


def get_file_stats(folder_path):
    """Compute size totals for all supported file extensions under a folder."""
    file_sizes = defaultdict(int)

    if not os.path.exists(folder_path):
        print(f"Warning: folder does not exist: {folder_path}")
        return file_sizes

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.split(".")[-1].lower() if "." in file else ""

            if ext in ext_to_modality:
                try:
                    size = os.path.getsize(file_path)
                    file_sizes[ext] += size
                except OSError:
                    pass

    return file_sizes


def format_size(size_bytes):
    """Format bytes into a readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def create_csv(file_sizes, output_path):
    """Write one CSV summary file."""
    rows = []

    for modality, extensions in modality_map.items():
        for ext in extensions:
            if ext in file_sizes and file_sizes[ext] > 0:
                rows.append(
                    {
                        "extension": ext,
                        "modality": modality,
                        "total_file_size": format_size(file_sizes[ext]),
                        "size_bytes": file_sizes[ext],
                    }
                )

    rows.sort(key=lambda x: (x["modality"], x["extension"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["extension", "modality", "total_file_size"])
        for row in rows:
            writer.writerow([row["extension"], row["modality"], row["total_file_size"]])

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-profile file-size summaries")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the raw profile folders downloaded from Hugging Face (Adam/, Bei/, Victoria/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "outputs" / "file_size"),
        help="Directory to write generated CSV summaries.",
    )
    args = parser.parse_args()

    base_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    folders = ["Bei", "Victoria", "Adam"]

    for folder in folders:
        folder_path = base_path / folder
        output_csv = output_dir / f"{folder}.csv"

        print(f"\nProcessing {folder} ...")
        file_sizes = get_file_stats(str(folder_path))
        count = create_csv(file_sizes, output_csv)
        print(f"Saved {output_csv} with {count} extension rows")

        if file_sizes:
            print(f"{folder} summary:")
            for modality in modality_map.keys():
                total = sum(file_sizes[ext] for ext in modality_map[modality] if ext in file_sizes)
                if total > 0:
                    print(f"  {modality}: {format_size(total)}")


if __name__ == "__main__":
    main()
