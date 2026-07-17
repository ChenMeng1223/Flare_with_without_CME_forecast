#!/usr/bin/env python3
"""
单独下载CHASE Hα数据脚本

基于 a_download_sdo_data.py 的架构，专门下载 CHASE Hα 波段数据。
支持智能时间匹配、缺失记录和并发控制。
"""
import sys
import os
import socket
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
import argparse
import yaml
import pandas as pd

logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_logging
from data.download_Halpha_fits import download_halpha_fits


class HAlphaDownloader:
    """Hα数据下载器"""

    def __init__(self, config: Dict):
        """
        初始化下载器

        Args:
            config: 配置字典
        """
        self.config = config
        self.modalities = config['data']['modalities']

        # 创建下载目录
        self.download_base_dir = Path(config['data']['raw_data_dir']) / 'downloaded'
        self.download_base_dir.mkdir(parents=True, exist_ok=True)

    def load_events_metadata(self, events_file: str) -> pd.DataFrame:
        """加载事件元数据"""
        if events_file.endswith('.xlsx'):
            df = pd.read_excel(events_file)
        elif events_file.endswith('.csv'):
            df = pd.read_csv(events_file)
        else:
            raise ValueError(f"不支持的文件格式: {events_file}")

        # 检查必要的列
        required_columns = ['DATE', 'start_time', 'end_time']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"文件缺少必要列: {col}")

        # 将 DATE 和时间列组合成完整的 datetime
        df = self._combine_date_time(df)

        # 如果缺少 event_id 列，根据时间自动生成
        if 'event_id' not in df.columns:
            logger.info("文件缺少 event_id 列，将根据时间自动生成")
            df['event_id'] = self._generate_event_ids(df)

        logger.info(f"加载了 {len(df)} 个事件")
        return df

    def _combine_date_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """将 DATE 和时间列组合成完整的 datetime"""
        df = df.copy()

        for idx, row in df.iterrows():
            # 检查整行是否为空
            if self._is_row_empty(row):
                logger.warning(f"跳过第 {idx+1} 行：整行数据为空")
                continue

            date_str = str(row['DATE'])

            # 检查 DATE 是否有效
            if date_str.lower() == 'nan' or not date_str or date_str == 'None':
                logger.warning(f"跳过第 {idx+1} 行：DATE 列无效 ({date_str})")
                continue

            # 解析 DATE 字符串 (格式: 2024.12.23)
            try:
                date_parts = date_str.split('.')
                if len(date_parts) != 3:
                    logger.warning(f"跳过第 {idx+1} 行：DATE 格式无效 ({date_str})")
                    continue
                year, month, day = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
                base_date = datetime(year, month, day).date()
            except (ValueError, IndexError) as e:
                logger.warning(f"跳过第 {idx+1} 行：DATE 解析失败 ({date_str}): {e}")
                continue

            # 处理 start_time
            try:
                start_time = self._parse_time_column(row['start_time'], base_date)
            except Exception as e:
                logger.warning(f"跳过第 {idx+1} 行：start_time 解析失败: {e}")
                continue

            # 处理 end_time
            try:
                end_time = self._parse_time_column(row['end_time'], base_date)
            except Exception as e:
                logger.warning(f"跳过第 {idx+1} 行：end_time 解析失败: {e}")
                continue

            # 更新 DataFrame
            df.at[idx, 'start_time'] = start_time.isoformat()
            df.at[idx, 'end_time'] = end_time.isoformat()

        return df

    def _is_row_empty(self, row: pd.Series) -> bool:
        """检查一行是否完全为空"""
        for value in row.values:
            if pd.notna(value) and str(value).strip():
                return False
        return True

    def _parse_time_column(self, time_value, base_date: datetime.date) -> datetime:
        """解析时间列，支持 datetime.time 对象和特殊字符串格式"""
        # 检查是否为 NaN 或无效值
        if pd.isna(time_value) or str(time_value).lower() == 'nan' or time_value is None:
            raise ValueError(f"时间值无效: {time_value}")

        if isinstance(time_value, str):
            # 处理特殊格式，如 "00:53(+1day)"
            if '(+1day)' in time_value:
                time_str = time_value.replace('(+1day)', '').strip()
                time_obj = datetime.strptime(time_str, '%H:%M').time()
                actual_date = base_date + timedelta(days=1)
                return datetime.combine(actual_date, time_obj)
            elif '(+2day)' in time_value:
                time_str = time_value.replace('(+2day)', '').strip()
                time_obj = datetime.strptime(time_str, '%H:%M').time()
                actual_date = base_date + timedelta(days=2)
                return datetime.combine(actual_date, time_obj)
            else:
                # 尝试其他字符串格式
                try:
                    return pd.to_datetime(time_value)
                except:
                    # 如果无法解析，假设是 HH:MM 格式
                    time_obj = datetime.strptime(time_value, '%H:%M').time()
                    return datetime.combine(base_date, time_obj)
        else:
            # datetime.time 对象
            return datetime.combine(base_date, time_value)

    def _generate_event_ids(self, df: pd.DataFrame) -> pd.Series:
        """根据时间生成唯一的 event_id"""
        event_ids = []
        time_count = {}

        for idx, row in df.iterrows():
            # 使用 start_time 字符串作为键（现在已经是完整的 datetime 字符串）
            start_time_str = str(row['start_time'])
            end_time_str = str(row['end_time'])

            # 使用完整的时间作为键
            time_key = f"{start_time_str}_{end_time_str}"

            # 如果这个时间组合已经出现过，增加计数器
            if time_key in time_count:
                time_count[time_key] += 1
            else:
                time_count[time_key] = 1

            # 生成 event_id: EVT_YYYYMMDD_HHMMSS_计数器
            try:
                start_dt = pd.to_datetime(start_time_str)
                event_id = f"EVT_{start_dt.strftime('%Y%m%d_%H%M%S')}_{time_count[time_key]}"
                event_id = f"EVT_{start_dt.strftime('%Y%m%d_%H%M%S')}_{time_count[time_key]}"
            except:
                # 如果解析失败，使用索引
                event_id = f"EVT_{idx+1:04d}_{time_count[time_key]}"

            event_ids.append(event_id)

        return pd.Series(event_ids)

    def get_download_time_range(self, start_time: str, end_time: str) -> Tuple[datetime, datetime]:
        """获取下载的时间范围（扩展前后时间）"""
        time_config = self.config['data']['time']

        start_dt = datetime.fromisoformat(start_time.replace('Z', ''))
        end_dt = datetime.fromisoformat(end_time.replace('Z', ''))

        # 扩展时间范围
        pre_hours = time_config['pre_event_hours']
        post_hours = time_config['post_event_hours']

        download_start = start_dt - timedelta(hours=pre_hours)
        download_end = start_dt + timedelta(hours=post_hours)

        return download_start, download_end

    def download_halpha_data(self, start_time: datetime, end_time: datetime, event_id: str) -> bool:
        """下载单个事件的时间范围内的 Hα 数据"""
        modality = 'halpha'
        modality_config = self.modalities.get(modality, {})

        if not modality_config:
            logger.warning(f"配置文件中缺少 {modality} 模态配置")
            return False

        # 获取数据集的大时间步长（分钟，用于事件级"采样点"控制）
        global_cadence_minutes = self.config['data']['time']['cadence_minutes']

        # 创建下载目录
        download_dir = self.download_base_dir / event_id / modality
        download_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"开始下载模态 {modality}（{event_id}）")
            # 事件级时间范围（按 pre/post_event_hours 扩展后）
            download_start = start_time
            download_end = end_time

            # 生成按数据集采样间隔划分的"代表时间点"（例如每 60min 一个）
            sample_times = self._generate_sample_times(
                download_start, download_end, global_cadence_minutes
            )
            if not sample_times:
                logger.warning(f"未生成有效的采样时间点: {modality}, 事件 {event_id}")
                return False

            logger.info(f"{modality}: 计划在 {len(sample_times)} 个采样时间点附近抓取图像")

            total_downloaded = 0

            # 根据 cadence 计算每个窗口大小
            cadence_str = modality_config.get('cadence', '60s')
            try:
                cadence_sec = int(cadence_str.rstrip('s'))
                window_seconds = max(1, cadence_sec // 2)
            except Exception:
                window_seconds = 1800  # 30分钟窗口

            for idx, sample_time in enumerate(sample_times, start=1):
                window_start = sample_time - timedelta(seconds=window_seconds)
                window_end = sample_time + timedelta(seconds=window_seconds)

                logger.info(
                    f"{modality}: 采样点 {idx}/{len(sample_times)} 窗口 [{window_start} ~ {window_end}] - 下载"
                )

                # 下载 Hα 数据
                url_file = modality_config.get('url_file')
                if not url_file:
                    url_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Halpha_download.txt')

                files = download_halpha_fits(
                    window_start,
                    window_end,
                    url_file=url_file,
                    save_dir=str(download_dir),
                    dry_run=self.config.get('dry_run', False),
                )

                total_downloaded += len(files)
                logger.info(f"{modality}: 采样点 {idx}/{len(sample_times)} 下载 {len(files)} 个文件")

            logger.info(f"完成模态 {modality}（{event_id}）- 总共获取 {total_downloaded} 个文件")
            return total_downloaded > 0

        except Exception as e:
            logger.error(f"下载 {modality} 数据失败（整体异常）: {type(e).__name__}: {e}")
            return False

    def _generate_sample_times(self, start_time: datetime, end_time: datetime,
                              cadence_minutes: int) -> List[datetime]:
        """生成采样时间点"""
        sample_times = []
        current_time = start_time

        while current_time <= end_time:
            sample_times.append(current_time)
            current_time += timedelta(minutes=cadence_minutes)

        return sample_times

    def download_event_halpha(self, event_row: pd.Series) -> bool:
        """下载单个事件的 Hα 数据"""
        event_id = event_row['event_id']
        start_time = event_row['start_time']
        end_time = event_row['end_time']

        # 显眼标题
        header = '#' * 10 + f" 开始下载事件 {event_id} 的 Hα 数据 " + '#' * 10
        logger.info(header)

        # 获取下载时间范围
        download_start, download_end = self.get_download_time_range(start_time, end_time)

        # 生成事件级的采样时间点
        global_cadence_minutes = self.config['data']['time']['cadence_minutes']
        sample_times = self._generate_sample_times(download_start, download_end, global_cadence_minutes)

        logger.info(f"事件 {event_id} 时间范围: {download_start} -> {download_end}")
        logger.info(f"事件 {event_id} 将使用 {len(sample_times)} 个采样点（每 {global_cadence_minutes} 分钟）")

        # 清空旧的缺失记录文件
        halpha_dir = self.download_base_dir / event_id / 'halpha'
        missing_file = halpha_dir / 'missing_halpha.txt'
        try:
            if missing_file.exists():
                missing_file.unlink()
        except Exception as e:
            logger.warning(f"无法清除旧的 missing_halpha.txt: {e}")

        # 下载 Hα 数据
        success = self.download_halpha_data(download_start, download_end, event_id)

        logger.info(f"事件 {event_id} Hα 下载完成: {'成功' if success else '失败'}")
        return success

    def download_all_events_halpha(self, events_df: pd.DataFrame, max_workers: int = 1) -> None:
        """批量下载所有事件的 Hα 数据"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 限制并发数为1，避免过多请求
        max_workers = max(1, min(int(max_workers), 1))

        logger.info(f"开始批量下载 {len(events_df)} 个事件的 Hα 数据（max_workers={max_workers}）")

        successful_events = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_event = {
                executor.submit(self.download_event_halpha, row): row['event_id']
                for _, row in events_df.iterrows()
            }

            # 收集结果
            for future in as_completed(future_to_event):
                event_id = future_to_event[future]
                try:
                    success = future.result()
                    if success:
                        successful_events += 1
                        logger.info(f"事件 {event_id} Hα 下载成功")
                    else:
                        logger.warning(f"事件 {event_id} Hα 下载失败")
                except Exception as e:
                    logger.error(f"事件 {event_id} Hα 下载异常: {e}")

        logger.info(f"Hα 批量下载完成: {successful_events}/{len(events_df)} 个事件成功")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='单独下载CHASE Hα数据')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='最大并发下载数（建议保持为1）')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅显示将要下载的内容，不实际下载')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(args.log_dir, 'download_halpha_only', debug=False)
    logger = logging.getLogger(__name__)

    # 设置全局网络超时
    socket.setdefaulttimeout(1800)

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建下载器
    downloader = HAlphaDownloader(config)

    # 从配置中获取事件文件路径
    events_file = config['data']['events_file']
    logger.info(f"使用事件文件: {events_file}")

    # 加载事件元数据
    events_df = downloader.load_events_metadata(events_file)

    if args.dry_run:
        logger.info("干运行模式 - 显示将要下载的内容:")
        for _, row in events_df.iterrows():
            event_id = row['event_id']
            start_time = row['start_time']
            end_time = row['end_time']
            download_start, download_end = downloader.get_download_time_range(start_time, end_time)

            logger.info(f"事件 {event_id}: {download_start} 到 {download_end} (Hα数据)")
        return

    # 开始下载
    downloader.download_all_events_halpha(events_df, max_workers=args.max_workers)


if __name__ == "__main__":
    main()