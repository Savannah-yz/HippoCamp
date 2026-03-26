import csv
import random
from datetime import datetime, timedelta
import os

def generate_random_datetime(start_year=2022, end_year=2025):
    """生成2022-2025年间的随机日期时间，时间在8:00-23:00之间"""
    # 随机年月日
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)

    # 根据月份确定最大天数
    if month in [1, 3, 5, 7, 8, 10, 12]:
        max_day = 31
    elif month in [4, 6, 9, 11]:
        max_day = 30
    else:  # 2月
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            max_day = 29
        else:
            max_day = 28

    day = random.randint(1, max_day)

    # 时间在8:00-23:00之间
    hour = random.randint(8, 22)
    minute = random.randint(0, 59)

    return datetime(year, month, day, hour, minute)

def format_datetime(dt):
    """格式化日期时间为CSV中的格式"""
    return dt.strftime('%Y/%m/%d %H:%M')

def is_folder(file_path):
    """判断是否是文件夹（路径中包含/但不是以/开头的路径分隔符）"""
    # 文件夹通常没有扩展名，或者路径以/结尾
    if file_path.endswith('/'):
        return True
    # 检查是否有扩展名
    base_name = os.path.basename(file_path.rstrip('/'))
    return '.' not in base_name

def get_folder_name(file_path):
    """获取文件所属的文件夹名称"""
    parts = file_path.split('/')
    if len(parts) > 1:
        return parts[0]
    return None

def process_csv(input_file):
    """处理CSV文件，填充日期"""
    rows = []

    # 读取CSV文件
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows.append(header)

        for row in reader:
            rows.append(row)

    # 找到creation_date和modification_date的列索引
    try:
        creation_idx = header.index('creation_date')
        modification_idx = header.index('modification_date')
        filepath_idx = header.index('FilePath')
    except ValueError as e:
        print(f"找不到必需的列: {e}")
        return

    # 第一遍：为文件夹生成日期并记录
    folder_dates = {}

    for i, row in enumerate(rows[1:], 1):
        if len(row) <= max(creation_idx, modification_idx, filepath_idx):
            continue

        file_path = row[filepath_idx]

        # 检查是否是文件夹
        if is_folder(file_path):
            folder_name = file_path.rstrip('/')
            # 为文件夹生成日期
            if not row[creation_idx] or row[creation_idx].strip() == '':
                creation_dt = generate_random_datetime()
                # modification_date 要晚于 creation_date
                days_diff = random.randint(1, 30)
                modification_dt = creation_dt + timedelta(days=days_diff,
                                                         hours=random.randint(0, 5),
                                                         minutes=random.randint(0, 59))
                # 确保时间仍在8:00-23:00
                if modification_dt.hour < 8:
                    modification_dt = modification_dt.replace(hour=8)
                elif modification_dt.hour > 22:
                    modification_dt = modification_dt.replace(hour=22)

                row[creation_idx] = format_datetime(creation_dt)
                row[modification_idx] = format_datetime(modification_dt)
                folder_dates[folder_name] = creation_dt
            else:
                # 解析已有的日期
                try:
                    creation_dt = datetime.strptime(row[creation_idx], '%Y/%m/%d %H:%M')
                    folder_dates[folder_name] = creation_dt
                except:
                    pass

    # 第二遍：为文件生成日期，确保晚于所属文件夹
    for i, row in enumerate(rows[1:], 1):
        if len(row) <= max(creation_idx, modification_idx, filepath_idx):
            continue

        file_path = row[filepath_idx]

        # 跳过文件夹
        if is_folder(file_path):
            continue

        # 只处理空日期
        if row[creation_idx] and row[creation_idx].strip() != '':
            continue

        # 获取所属文件夹
        folder_name = get_folder_name(file_path)

        # 确定creation_date的最早时间
        if folder_name and folder_name in folder_dates:
            # 文件日期要晚于文件夹日期
            min_date = folder_dates[folder_name] + timedelta(days=random.randint(1, 30))

            # 生成creation_date
            creation_dt = min_date + timedelta(days=random.randint(0, 365),
                                               hours=random.randint(0, 14))
            # 确保在2022-2025年范围内
            if creation_dt.year > 2025:
                creation_dt = creation_dt.replace(year=2025)

            # 确保时间在8:00-23:00
            if creation_dt.hour < 8:
                creation_dt = creation_dt.replace(hour=random.randint(8, 22))
            elif creation_dt.hour > 22:
                creation_dt = creation_dt.replace(hour=22)
        else:
            # 没有文件夹，直接生成随机日期
            creation_dt = generate_random_datetime()

        # modification_date 要晚于 creation_date
        days_diff = random.randint(0, 60)
        hours_diff = random.randint(1, 5) if days_diff == 0 else random.randint(0, 5)
        modification_dt = creation_dt + timedelta(days=days_diff,
                                                  hours=hours_diff,
                                                  minutes=random.randint(0, 59))

        # 确保时间在8:00-23:00
        if modification_dt.hour < 8:
            modification_dt = modification_dt.replace(hour=8)
        elif modification_dt.hour > 22:
            modification_dt = modification_dt.replace(hour=22)

        # 确保年份不超过2025
        if modification_dt.year > 2025:
            modification_dt = modification_dt.replace(year=2025)

        row[creation_idx] = format_datetime(creation_dt)
        row[modification_idx] = format_datetime(modification_dt)

    # 写回CSV文件
    with open(input_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'处理完成: {input_file}')

def main():
    csv_files = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Victoria_files.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Bei_files.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Adam_files.csv')
    ]

    for csv_file in csv_files:
        try:
            process_csv(csv_file)
        except FileNotFoundError:
            print(f'文件未找到: {csv_file}')
        except Exception as e:
            print(f'处理{csv_file}时出错: {e}')
            import traceback
            traceback.print_exc()

    print('\n所有文件处理完成！')

if __name__ == '__main__':
    main()
