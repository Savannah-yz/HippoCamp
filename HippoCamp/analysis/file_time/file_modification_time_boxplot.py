import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_csv_date(date_str):
    """解析CSV中的日期格式，支持多种格式"""
    date_str = str(date_str).strip()

    # 尝试多种日期格式
    # 注意：优先尝试DD/MM/YYYY格式，因为如果先尝试YYYY/MM/DD，
    # 像"7/11/2025"这样的日期会被错误解析为2007年
    formats = [
        "%d/%m/%Y %H:%M",   # 21/12/2022 15:58
        "%d/%m/%Y",         # 21/12/2022
        "%Y/%m/%d %H:%M",   # 2025/10/19 0:04
        "%Y/%m/%d",         # 2025/10/19
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue

    return None

def get_modification_times_from_csv(csv_path, folder_name):
    """直接从CSV文件读取所有文件的修改时间"""
    modification_times = []
    file_details = []

    if not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} does not exist")
        return modification_times, file_details

    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            file_path = row['FilePath']
            modification_date_str = row['modification_date']

            # 跳过文件夹类型
            if row.get('FileType') == 'folder':
                continue

            mod_date = parse_csv_date(modification_date_str)
            if mod_date:
                modification_times.append(mod_date)
                file_details.append({
                    'file': file_path,
                    'date': mod_date,
                    'source': 'CSV'
                })
    except Exception as e:
        print(f"Warning: Failed to load CSV {csv_path}: {e}")

    return modification_times, file_details

def transform_date(date_num):
    """非线性变换：2025年之前压缩，2025-2026年8月拉伸"""
    threshold = mdates.date2num(datetime(2025, 1, 1))
    min_date = mdates.date2num(datetime(2010, 1, 1))
    max_date = mdates.date2num(datetime(2026, 8, 1))

    if date_num < min_date:
        # 早于2010年的数据放在最底部
        return 0.0
    elif date_num < threshold:
        # 2010-2024年的数据压缩到 0-25% 的空间
        ratio = (date_num - min_date) / (threshold - min_date)
        return ratio * 0.25
    else:
        # 2025年及之后的数据拉伸到 25%-100% 的空间
        ratio = (date_num - threshold) / (max_date - threshold)
        return 0.25 + ratio * 0.75

# 基础路径
csv_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# 获取三个文件夹的修改时间数据
folders = ['Bei', 'Adam', 'Victoria']
folder_labels = ['College', 'Law', 'Finance']
colors = ['#E74C3C', '#2ECC71', '#3498DB']  # 红、绿、蓝

data = []
data_transformed = []
labels = []
all_details = {}  # 保存所有文件夹的详细信息

for i, folder in enumerate(folders):
    # 直接从CSV文件读取数据
    csv_path = os.path.join(csv_base_path, f"{folder}_files.csv")
    times, file_details = get_modification_times_from_csv(csv_path, folder)
    all_details[folder] = file_details

    if times:
        # 原始数值
        raw_data = [mdates.date2num(t) for t in times]
        data.append(raw_data)
        # 变换后的数值用于绘图
        transformed = [transform_date(d) for d in raw_data]
        data_transformed.append(transformed)
        labels.append(folder_labels[i])
        print(f"{folder_labels[i]} ({folder}): {len(times)} files from {folder}_files.csv")

# 生成详细报告
print("\n" + "="*80)
print("DETAILED MODIFICATION TIME REPORT")
print("="*80)

for folder in folders:
    if folder not in all_details:
        continue

    details = all_details[folder]
    # 按日期排序
    details_sorted = sorted(details, key=lambda x: x['date'])

    # 统计
    csv_count = sum(1 for d in details if d['source'] == 'CSV')
    meta_count = sum(1 for d in details if d['source'] == 'Metadata')

    # 按月份分组统计
    month_counts = {}
    for d in details:
        month_key = d['date'].strftime('%Y-%m')
        if month_key not in month_counts:
            month_counts[month_key] = []
        month_counts[month_key].append(d)

    print(f"\n{'='*40}")
    print(f"Folder: {folder}")
    print(f"{'='*40}")
    print(f"Total files: {len(details)}")
    print(f"From CSV: {csv_count}")
    print(f"From Metadata: {meta_count}")

    print(f"\n--- Files by Month ---")
    for month in sorted(month_counts.keys()):
        files_in_month = month_counts[month]
        print(f"\n{month}: {len(files_in_month)} files")
        for f in files_in_month:
            print(f"  [{f['source']:8}] {f['date'].strftime('%Y-%m-%d %H:%M')} - {f['file']}")

    # 特别标注2025年1月的文件
    jan_2025_files = [d for d in details if d['date'].year == 2025 and d['date'].month == 1]
    if jan_2025_files:
        print(f"\n*** 2025年1月的文件 ({len(jan_2025_files)} files) ***")
        for f in jan_2025_files:
            print(f"  [{f['source']:8}] {f['date'].strftime('%Y-%m-%d %H:%M')} - {f['file']}")
    else:
        print(f"\n*** 没有2025年1月的文件 ***")

print("\n" + "="*80)

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制箱式图（使用变换后的数据）
bp = ax.boxplot(data_transformed, labels=labels, patch_artist=True,
                widths=0.6,
                boxprops=dict(linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                medianprops=dict(linewidth=2, color='white'),
                flierprops=dict(marker='o', markersize=4, alpha=0.5))

# 设置颜色
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 设置y轴范围
ax.set_ylim(0, 1)

# 创建自定义刻度标签
# 2025年之前的几个关键年份 + 2025年每月
tick_positions = []
tick_labels_list = []

# 添加一些早期年份的刻度（压缩区域）
for year in [2012, 2015, 2018, 2021, 2024]:
    pos = transform_date(mdates.date2num(datetime(year, 1, 1)))
    tick_positions.append(pos)
    tick_labels_list.append(f'{year}')

# 添加2025年每月的刻度（拉伸区域）
for month in range(1, 13):
    pos = transform_date(mdates.date2num(datetime(2025, month, 1)))
    tick_positions.append(pos)
    tick_labels_list.append(f'2025-{month:02d}')

# 添加2026年1-8月
for month in range(1, 9):
    pos = transform_date(mdates.date2num(datetime(2026, month, 1)))
    tick_positions.append(pos)
    tick_labels_list.append(f'2026-{month:02d}')

ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_labels_list)

# 添加分隔线标记2025年的开始
threshold_pos = 0.25
ax.axhline(y=threshold_pos, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# 美化图表
ax.set_ylabel('Modification Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Folder', fontsize=14, fontweight='bold')
ax.set_title('File Modification Time Distribution by Folder', fontsize=16, fontweight='bold', pad=20)

# 添加网格
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# 设置背景色
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 调整边距
plt.tight_layout()

# 保存图片
output_path = os.path.join(csv_base_path, "file_modification_time_boxplot.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图片已保存到: {output_path}")

plt.show()
