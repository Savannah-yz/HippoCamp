import os
import csv
from collections import defaultdict

# 定义文件类型分类
modality_map = {
    'Text': ['bin', 'ipynb', 'log', 'npy', 'pkl', 'pt', 'pth', 'py', 'txt', 'md', 'pyc',
             'sh', 'xml', 'json', 'sqlite', 'yaml', 'yml'],
    'Documents': ['docx', 'eml', 'pdf', 'pptx', 'ics', 'csv', 'doc', 'rtf', 'xlsx'],
    'Images': ['gif', 'jpg', 'png', 'jpeg', 'webp', 'tiff', 'svg', 'heic', 'bmp'],
    'Audio': ['mp3'],
    'Video': ['mp4', 'mkv']
}

# 创建后缀到modality的映射
ext_to_modality = {}
for modality, extensions in modality_map.items():
    for ext in extensions:
        ext_to_modality[ext.lower()] = modality

def get_file_stats(folder_path):
    """统计文件夹中各类文件的大小"""
    file_sizes = defaultdict(int)

    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist")
        return file_sizes

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.split('.')[-1].lower() if '.' in file else ''

            if ext in ext_to_modality:
                try:
                    size = os.path.getsize(file_path)
                    file_sizes[ext] += size
                except OSError:
                    pass

    return file_sizes

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def create_csv(folder_name, file_sizes, output_path):
    """生成CSV文件"""
    rows = []

    # 按modality分组收集数据
    for modality, extensions in modality_map.items():
        for ext in extensions:
            if ext in file_sizes and file_sizes[ext] > 0:
                rows.append({
                    'extension': ext,
                    'modality': modality,
                    'total_file_size': format_size(file_sizes[ext]),
                    'size_bytes': file_sizes[ext]
                })

    # 按modality和extension排序
    rows.sort(key=lambda x: (x['modality'], x['extension']))

    # 写入CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['extension', 'modality', 'total_file_size'])
        for row in rows:
            writer.writerow([row['extension'], row['modality'], row['total_file_size']])

    return len(rows)

if __name__ == "__main__":
    # Base path: the raw source files for each profile.
    # Download from HuggingFace: https://huggingface.co/datasets/MMMem-org/HippoCamp
    # and set this to the directory containing Bei/, Victoria/, Adam/ folders.
    import argparse
    parser = argparse.ArgumentParser(description="Compute file size statistics per profile")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to directory containing Bei/, Victoria/, Adam/ raw file folders")
    args = parser.parse_args()
    base_path = args.data_dir

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 处理三个文件夹
    folders = ['Bei', 'Victoria', 'Adam']

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        output_csv = os.path.join(SCRIPT_DIR, f"{folder}.csv")

        print(f"\n处理 {folder} 文件夹...")
        file_sizes = get_file_stats(folder_path)
        count = create_csv(folder, file_sizes, output_csv)
        print(f"已生成 {output_csv}，包含 {count} 种文件类型")

        # 显示统计摘要
        if file_sizes:
            print(f"{folder} 统计摘要:")
            for modality in modality_map.keys():
                total = sum(file_sizes[ext] for ext in modality_map[modality] if ext in file_sizes)
                if total > 0:
                    print(f"  {modality}: {format_size(total)}")