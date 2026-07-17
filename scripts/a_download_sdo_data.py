#!/usr/bin/env python3
"""
SDO数据批量下载脚本

根据events.xlsx中的时间范围，自动下载对应时间段的各波段数据：
- HMI磁图 (magnetogram)
- AIA 94Å (euv_94)
- AIA 171Å (euv_171)
- AIA 193Å (euv_193)
- AIA 304Å (euv_304)
- CHASE Hα (halpha) - 如果可用

下载完成后保存为FITS文件，供后续批量绘制使用。
"""
import sys
import os
import socket
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta, date
import logging
import argparse
import yaml
import pandas as pd

logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_logging
from utils.active_region_utils import attach_active_region_columns, raise_if_unknown_active_regions
from utils.event_id_utils import ensure_event_ids_like_downloader
from data.download_AIA_fits import download_aia_fits
from data.download_HMI_magnetogram import download_hmi_magnetogram
from data.download_Halpha_fits import download_halpha_fits


class SDODataDownloader:
    """SDO数据下载器"""

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

        # SDO仪器映射
        self.instrument_map = {
            'magnetogram': ('HMI', 'HMI_M_45S'),
            'euv_94': ('AIA', 'AIA_94A_12S'),
            'euv_171': ('AIA', 'AIA_171A_12S'),
            'euv_193': ('AIA', 'AIA_193A_12S'),
            'euv_304': ('AIA', 'AIA_304A_12S'),
            'halpha': ('CHASE', 'CHASE_HA_LINEWIDTH')  # CHASE Hα数据
        }

        # 检查sunpy是否可用
        try:
            import sunpy
            from sunpy.net import Fido, attrs as a
            from sunpy.time import parse_time
            import astropy.units as u
            self.sunpy_available = True
            self.Fido = Fido
            self.attrs = a
            self.parse_time = parse_time
            self.units = u  # 添加astropy单位
            logger.info("SunPy库可用")
        except ImportError:
            logger.warning("SunPy库不可用，请安装: pip install sunpy")
            self.sunpy_available = False

    def load_events_metadata(
        self,
        events_file: str,
        *,
        allow_unknown_active_regions: bool = False,
    ) -> pd.DataFrame:
        """加载事件元数据"""
        if events_file.endswith('.xlsx'):
            df = pd.read_excel(events_file)
        elif events_file.endswith('.csv'):
            df = pd.read_csv(events_file)
        else:
            raise ValueError(f"不支持的文件格式: {events_file}")

        original_count = len(df)

        # 检查必要的列
        required_columns = ['DATE', 'start_time', 'end_time']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"文件缺少必要列: {col}")

        # 将 DATE 和时间列组合成完整的 datetime，并过滤无效行
        df = self._combine_date_time(df)

        # 如果缺少 event_id 列，根据时间自动生成
        if 'event_id' not in df.columns:
            logger.info("文件缺少 event_id 列，将根据时间自动生成")
            df['event_id'] = self._generate_event_ids(df)
        else:
            missing_mask = (
                df['event_id'].isna()
                | (df['event_id'].astype(str).str.strip() == '')
                | (df['event_id'].astype(str).str.lower() == 'nan')
            )
            missing_count = int(missing_mask.sum())
            if missing_count > 0:
                logger.info(f"event_id 列存在 {missing_count} 个缺失值，将根据时间自动补齐")
                df = ensure_event_ids_like_downloader(df)

        logger.info(f"加载事件完成：原始 {original_count} 行，有效 {len(df)} 个事件")
        df = attach_active_region_columns(df)
        if not allow_unknown_active_regions:
            raise_if_unknown_active_regions(df, context="批量下载前检查：")
        return df

    def _combine_date_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """将 DATE 和时间列组合成完整的 datetime，并过滤无效行"""
        df = df.copy()
        valid_indices = []

        for idx, row in df.iterrows():
            # 检查整行是否为空
            if self._is_row_empty(row):
                logger.warning(f"跳过第 {idx+1} 行：整行数据为空")
                continue

            date_str = str(row['DATE']).strip()

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

            # 更新 DataFrame，仅保留成功解析的行
            df.at[idx, 'start_time'] = start_time.isoformat()
            df.at[idx, 'end_time'] = end_time.isoformat()
            valid_indices.append(idx)

        filtered_df = df.loc[valid_indices].copy()
        filtered_df.reset_index(drop=True, inplace=True)
        invalid_count = len(df) - len(filtered_df)
        if invalid_count > 0:
            logger.warning(f"事件表中共有 {invalid_count} 行无效数据，已跳过")

        return filtered_df

    def _is_row_empty(self, row: pd.Series) -> bool:
        """检查一行是否完全为空"""
        for value in row.values:
            if pd.notna(value) and str(value).strip():
                return False
        return True

    def _parse_time_column(self, time_value, base_date: date) -> datetime:
        """解析时间列，支持字符串、datetime.time、datetime、Timestamp 等格式"""
        # 检查是否为 NaN 或无效值
        if pd.isna(time_value) or str(time_value).lower() == 'nan' or time_value is None:
            raise ValueError(f"时间值无效: {time_value}")

        if isinstance(time_value, pd.Timestamp):
            if pd.isna(time_value):
                raise ValueError(f"时间值无效: {time_value}")
            dt = time_value.to_pydatetime()
            if dt.year == 1900:
                return datetime.combine(base_date, dt.time())
            return dt

        if isinstance(time_value, datetime):
            if time_value.year == 1900:
                return datetime.combine(base_date, time_value.time())
            return time_value

        if hasattr(time_value, 'hour') and hasattr(time_value, 'minute') and not isinstance(time_value, str):
            try:
                return datetime.combine(base_date, time_value)
            except TypeError:
                pass

        if isinstance(time_value, str):
            time_value = time_value.strip()
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
                # 先尝试纯时间字符串，避免像 11:20:00 被解析成带默认日期的 Timestamp
                for fmt in ('%H:%M:%S', '%H:%M'):
                    try:
                        time_obj = datetime.strptime(time_value, fmt).time()
                        return datetime.combine(base_date, time_obj)
                    except ValueError:
                        continue

                # 再尝试完整日期时间字符串
                try:
                    dt = pd.to_datetime(time_value).to_pydatetime()
                    if dt.year == 1900:
                        return datetime.combine(base_date, dt.time())
                    return dt
                except Exception:
                    pass

        raise ValueError(f"不支持的时间格式: {time_value} ({type(time_value).__name__})")



    def get_download_time_range(self, start_time, end_time) -> Tuple[datetime, datetime]:
        """获取下载的时间范围（扩展前后时间）"""
        time_config = self.config['data']['time']

        start_dt = self._normalize_datetime_value(start_time)
        end_dt = self._normalize_datetime_value(end_time)

        # 扩展时间范围
        pre_hours = time_config['pre_event_hours']
        post_hours = time_config['post_event_hours']

        download_start = start_dt - timedelta(hours=pre_hours)
        download_end = start_dt + timedelta(hours=post_hours)

        return download_start, download_end

    def _normalize_datetime_value(self, value) -> datetime:
        """将各种日期时间表示统一转为原生 Python datetime。"""
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            cleaned = value.strip().replace('Z', '').replace('+00:00', '')
            return pd.to_datetime(cleaned).to_pydatetime()
        raise ValueError(f"不支持的日期时间类型: {type(value).__name__}: {value}")

    def _print_search_results(self, records, modality: str, instrument: str, window_start: datetime, window_end: datetime) -> None:
        """打印搜索到的文件的观测时间信息
        
        Args:
            records: SunPy Records 对象或其他记录列表
            modality: 模态名称
            instrument: 仪器名称
            window_start: 查询窗口起始时间
            window_end: 查询窗口结束时间
        """
        if not records or len(records) == 0:
            logger.info(f"{modality}({instrument}): 在时间窗口 [{window_start} ~ {window_end}] 中未搜索到文件")
            return
        
        logger.info(f"{modality}({instrument}): 在时间窗口 [{window_start} ~ {window_end}] 中搜索到 {len(records)} 个文件:")
        
        # 尝试打印每个记录的观测时间
        try:
            for idx, record in enumerate(records, start=1):
                obs_time = "N/A"
                # 尝试从不同属性获取观测时间
                if hasattr(record, 'time_start'):
                    obs_time = str(record.time_start)
                elif hasattr(record, 'T_OBS'):
                    obs_time = str(record.T_OBS)
                elif hasattr(record, 'date__obs'):
                    obs_time = str(record.date__obs)
                elif hasattr(record, 'DATE-OBS'):
                    obs_time = str(record.__dict__.get('DATE-OBS', 'N/A'))
                else:
                    # 尝试访问索引（SunPy Row 对象）
                    try:
                        if hasattr(record, '__getitem__'):
                            # 尝试通过列名访问
                            obs_time = str(record['DATE-OBS'] if 'DATE-OBS' in record.colnames else record[0])
                    except:
                        pass
                
                logger.info(f"  [{idx:2d}] 观测时间: {obs_time}")
        except Exception as e:
            logger.warning(f"打印观测时间时出现异常: {type(e).__name__}: {e}")
            # 至少可以显示找到了多少个文件
            logger.info(f"  搜索到 {len(records)} 个文件（详细时间信息解析失败，可能为网络临时问题）")

    def download_modality_data(self, modality: str, start_time: datetime,
                               end_time: datetime, event_id: str) -> bool:
        """下载单个模态的数据"""
        if not self.sunpy_available:
            logger.error("SunPy不可用，无法下载数据")
            return False

        if modality not in self.instrument_map:
            logger.warning(f"不支持的模态: {modality}")
            return False

        instrument, product = self.instrument_map[modality]
        modality_config = self.modalities[modality]
        # 仪器原始采样间隔（例如 '45s' 或 '720s'），用于估算每个采样点的查询窗口
        cadence_str = modality_config.get('cadence', '45s')

        # 获取数据集的大时间步长（分钟，用于事件级“采样点”控制）
        global_cadence_minutes = self.config['data']['time']['cadence_minutes']

        # 创建下载目录
        download_dir = self.download_base_dir / event_id / modality
        download_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 明确打印当前事件、当前波段的下载目录，便于人工检查
            logger.info(
                f"开始下载模态 {modality}（事件 {event_id}），"
                f"下载目录: {download_dir}"
            )
            # 事件级时间范围（按 pre/post_event_hours 扩展后）
            download_start = start_time
            download_end = end_time

            # 生成按数据集采样间隔划分的“代表时间点”（例如每 60min 一个）
            sample_times = self._generate_sample_times(
                download_start, download_end, global_cadence_minutes
            )
            if not sample_times:
                logger.warning(f"未生成有效的采样时间点: {modality}, 事件 {event_id}")
                return False

            total_samples = len(sample_times)
            logger.info(f"{modality}: 计划在 {total_samples} 个采样时间点附近抓取图像")

            total_downloaded = 0

            # 进度条相关变量（按采样窗口维度，每分钟在原位置刷新一次）
            progress_start_ts = time.time()
            last_progress_update_ts = 0.0

            # 根据 cadence 计算每个窗口大小
            try:
                cadence_sec = int(cadence_str.rstrip('s'))
                window_seconds = max(1, cadence_sec // 2)
            except Exception:
                window_seconds = 360

            for idx, sample_time in enumerate(sample_times, start=1):
                window_start = sample_time - timedelta(seconds=window_seconds)
                window_end = sample_time + timedelta(seconds=window_seconds)

                logger.info(
                    f"{modality}: 采样点 {idx}/{len(sample_times)} 窗口 [{window_start} ~ {window_end}] - 逐窗口下载"
                )

                try:
                    if instrument == 'HMI':
                        series = modality_config.get('series')
                        if not series:
                            series = 'hmi.M_720s' if '720' in cadence_str else 'hmi.M_45s'
                        use_max_files = '45s' in series
                        max_files = 2 if use_max_files else None

                        # 先搜索查询文件，打印观测时间
                        try:
                            records = download_hmi_magnetogram(
                                window_start,
                                window_end,
                                series=series,
                                save_dir=str(download_dir),
                                dry_run=self.config.get('dry_run', False),
                                max_files=max_files,
                                fetch=False,  # 仅查询，不下载
                            )
                            self._print_search_results(records, modality, instrument, window_start, window_end)
                        except Exception as e:
                            logger.warning(f"查询 {modality} 文件失败: {type(e).__name__}: {e}")
                        
                        # 执行实际下载
                        files = download_hmi_magnetogram(
                            window_start,
                            window_end,
                            series=series,
                            save_dir=str(download_dir),
                            dry_run=self.config.get('dry_run', False),
                            max_files=max_files,
                        )
                        # 如果没有找到文件，记录到缺失列表
                        if not files:
                            missing_file = Path(download_dir) / f'missing_{modality}.txt'
                            # 记录缺失“采样点”而非采样区间，便于后续直接按点补下载
                            entry = sample_time.isoformat()
                            try:
                                # 如果文件已存在，读取记录并避免重复条目
                                existing = set()
                                if missing_file.exists():
                                    with open(missing_file, 'r') as mf:
                                        for line in mf:
                                            existing.add(line.strip())
                                existing.add(entry)
                                # 将所有条目覆盖写回文件（保持唯一）
                                with open(missing_file, 'w') as mf:
                                    for line in sorted(existing):
                                        mf.write(line + "\n")
                            except Exception as e:
                                logger.warning(f"写入缺失文件失败: {e}")

                    elif instrument == 'AIA':
                        wavelength = modality_config.get('wavelength')
                        series = modality_config.get('series')
                        
                        # 先搜索查询文件，打印观测时间
                        try:
                            records = download_aia_fits(
                                window_start,
                                window_end,
                                float(wavelength),
                                series=series,
                                save_dir=str(download_dir),
                                dry_run=self.config.get('dry_run', False),
                                fetch=False,  # 仅查询，不下载
                            )
                            self._print_search_results(records, modality, instrument, window_start, window_end)
                        except Exception as e:
                            logger.warning(f"查询 {modality} 文件失败: {type(e).__name__}: {e}")
                        
                        # 执行实际下载
                        files = download_aia_fits(
                            window_start,
                            window_end,
                            float(wavelength),
                            series=series,
                            save_dir=str(download_dir),
                            dry_run=self.config.get('dry_run', False),
                        )
                        # 如果没有找到文件，记录到缺失列表
                        if not files:
                            missing_file = Path(download_dir) / f'missing_{modality}.txt'
                            # 记录缺失“采样点”而非采样区间，便于后续直接按点补下载
                            entry = sample_time.isoformat()
                            try:
                                # 如果文件已存在，读取记录并避免重复条目
                                existing = set()
                                if missing_file.exists():
                                    with open(missing_file, 'r') as mf:
                                        for line in mf:
                                            existing.add(line.strip())
                                existing.add(entry)
                                # 将所有条目覆盖写回文件（保持唯一）
                                with open(missing_file, 'w') as mf:
                                    for line in sorted(existing):
                                        mf.write(line + "\n")
                            except Exception as e:
                                logger.warning(f"写入缺失文件失败: {e}")

                    elif instrument == 'CHASE':
                        # CHASE Hα数据处理：download_halpha_fits 内部会负责查询并打印信息
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
                        # 如果没有找到文件，记录到缺失列表
                        if not files:
                            missing_file = Path(download_dir) / 'missing_halpha.txt'
                            # 记录缺失“采样点”而非采样区间，便于后续直接按点补下载
                            entry = sample_time.isoformat()
                            try:
                                # 如果文件已存在，读取记录并避免重复条目
                                existing = set()
                                if missing_file.exists():
                                    with open(missing_file, 'r') as mf:
                                        for line in mf:
                                            existing.add(line.strip())
                                existing.add(entry)
                                # 将所有条目覆盖写回文件（保持唯一）
                                with open(missing_file, 'w') as mf:
                                    for line in sorted(existing):
                                        mf.write(line + "\n")
                            except Exception as e:
                                logger.warning(f"写入缺失文件失败: {e}")

                    else:
                        files = []

                    total_downloaded += len(files)
                    logger.info(f"{modality}: 采样点 {idx}/{len(sample_times)} 下载 {len(files)} 个文件")

                except Exception as e:
                    logger.warning(
                        f"{modality}: 采样点 {idx}/{len(sample_times)} "
                        f"窗口 [{window_start} ~ {window_end}] 下载失败: {type(e).__name__}: {e}"
                    )
                    continue

                # 每隔约 60 秒在同一行更新一次采样窗口级进度###############################################################################################
                now_ts = time.time()
                if (now_ts - last_progress_update_ts >= 60) or (idx == 1) or (idx == total_samples):
                    done_samples = idx
                    elapsed = now_ts - progress_start_ts
                    if done_samples > 0 and elapsed > 0:
                        progress = done_samples / total_samples
                        eta_sec = elapsed / progress - elapsed
                        eta_min = int(eta_sec // 60)
                        progress_pct = progress * 100.0
                        msg = (
                            f"\r{modality}: 采样窗口进度 {done_samples}/{total_samples} "
                            f"({progress_pct:5.1f}%)，估计剩余 ~{eta_min:3d} 分钟\n"
                        )
                    else:
                        msg = f"\r{modality}: 采样窗口进度 {done_samples}/{total_samples} (初始化中...)\n"
                    # 直接写到 stdout，在原位置刷进度条
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    last_progress_update_ts = now_ts

            # 进度条换行，避免影响后续日志输出
            sys.stdout.write("\n")
            sys.stdout.flush()

            logger.info(f"完成模态 {modality}（{event_id}）- 总共获取 {total_downloaded} 个文件\n")
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

    def download_event_data(self, event_row: pd.Series) -> bool:
        """下载单个事件的所有模态数据"""
        event_id = event_row['event_id']
        start_time = event_row['start_time']
        end_time = event_row['end_time']

        # 显眼标题 —— 只保留这一条最全面的信息
        header = '#' * 10 + f" 开始下载事件 {event_id} 的数据 " + '#' * 10
        logger.info(header)

        # 获取下载时间范围
        download_start, download_end = self.get_download_time_range(start_time, end_time)

        # 生成事件级的采样时间点（基于 data_config 中的 cadence_minutes）并先输出所有采样窗口
        global_cadence_minutes = self.config['data']['time']['cadence_minutes']
        sample_times = self._generate_sample_times(download_start, download_end, global_cadence_minutes)

        logger.info(f"事件 {event_id} 时间范围: {download_start} -> {download_end}")
        logger.info(f"事件 {event_id} 将使用 {len(sample_times)} 个采样点（每 {global_cadence_minutes} 分钟）; 采样窗口（每点左边界=采样点-6min, 右边界=采样点+6min）:")
        for i, st in enumerate(sample_times, start=1):
            left = st - timedelta(minutes=6)
            right = st + timedelta(minutes=6)
            logger.info(f"  {i:02d}: [{left} -> {right}] (中心: {st})")

        success_count = 0
        total_count = 0

        # 下载每个模态的数据
        # 为了保证各波段的 missing_*.txt 在每次事件运行开始时清空（覆盖上次运行），
        # 先把对应目录下的文件删除
        for modality in self.modalities:
            if self.modalities[modality].get('required', False) or self.modalities[modality].get('download', True):
                modality_dir = self.download_base_dir / event_id / modality
                missing_file = modality_dir / f'missing_{modality}.txt'
                try:
                    if missing_file.exists():
                        missing_file.unlink()
                except Exception as e:
                    logger.warning(f"无法清除旧的 missing_{modality}.txt: {e}")

        # 临时关闭 HMI 磁图下载，用于单独调试 AIA/EUV 等波段##################################################################################################
        # 如果后续需要恢复 HMI 下载，只需删除或注释掉下面对 'magnetogram' 的跳过逻辑。
        for modality, modality_config in self.modalities.items():
            # 跳过 HMI 磁图
            if modality == 'magnetogram':
                logger.info("暂时跳过 HMI magnetogram 下载，仅调试 AIA/EUV 等波段")
                continue
            if modality == 'euv_94':
                logger.info("暂时跳过 euv_94 下载，仅调试 AIA/EUV 等波段")
                continue
            if modality == 'euv_171':
                logger.info("暂时跳过 euv_171 下载，仅调试 AIA/EUV 等波段")
                continue
            if modality == 'euv_193':
                logger.info("暂时跳过 euv_193 下载，仅调试 AIA/EUV 等波段")
                continue
            if modality == 'halpha':
                logger.info("暂时跳过 halpha 下载，仅调试 AIA/EUV 等波段")
                continue
            if modality_config.get('required', False) or modality_config.get('download', True):
                total_count += 1
                if self.download_modality_data(modality, download_start, download_end, event_id):
                    success_count += 1

        logger.info(f"事件 {event_id} 下载完成: {success_count}/{total_count} 个模态成功")
        return success_count > 0

    def download_all_events(self, events_df: pd.DataFrame, max_workers: int = 1) -> None:
        """批量下载所有事件的数据"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 为了配合 JSOC 的“单用户 pending 导出数量限制”，将并发数限制为 1（串行下载），
        # 即使外部传入更大的 max_workers，也做一次约束。
        max_workers = max(1, min(int(max_workers), 1))

        logger.info(f"开始批量下载 {len(events_df)} 个事件的数据（max_workers={max_workers}）")

        successful_events = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_event = {
                executor.submit(self.download_event_data, row): row['event_id']
                for _, row in events_df.iterrows()
            }

            # 收集结果
            for future in as_completed(future_to_event):
                event_id = future_to_event[future]
                try:
                    success = future.result()
                    if success:
                        successful_events += 1
                        logger.info(f"事件 {event_id} 下载成功")
                    else:
                        logger.warning(f"事件 {event_id} 下载失败")
                except Exception as e:
                    logger.error(f"事件 {event_id} 下载异常: {e}")

        logger.info(f"批量下载完成: {successful_events}/{len(events_df)} 个事件成功")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量下载SDO数据')
    parser.add_argument('--config', type=str, default=r'D:/ChenMeng/Graduate_Student/Experiment_and_Data/Flare_with_without_CME_forecast/configs/data_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='最大并发下载数（建议保持为1，避免JSOC导出请求过多）')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅显示将要下载的内容，不实际下载')
    parser.add_argument('--allow_unknown_active_regions', action='store_true',
                        help='允许活动区编号未知的事件继续下载（不推荐）')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(log_dir=args.log_dir)
    logger = logging.getLogger(__name__)

    # 设置全局网络超时，避免某个请求无限卡住（SunPy/JSOC 下载常见问题）
    # 若你希望更激进，可调小（例如 300 秒）；更保守可调大（例如 1200 秒）
    # 之前设置得太短会把“慢下载”误判为卡死；这里放宽到 30 分钟，仍可避免无限期等待。    # 如果仍然遇到"[信号灯超时时间已到]"错误，可以进一步增加到 3600 (1小时)    socket.setdefaulttimeout(1800)

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建下载器
    downloader = SDODataDownloader(config)

    # 优先读取 data/raw 目录下的 events.xlsx
    raw_data_dir = Path(config['data']['raw_data_dir'])
    if not raw_data_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        raw_data_dir = project_root / raw_data_dir
    events_file = raw_data_dir / 'events.xlsx'
    logger.info(f"使用事件文件: {events_file}")

    # 加载事件元数据
    events_df = downloader.load_events_metadata(
        str(events_file),
        allow_unknown_active_regions=args.allow_unknown_active_regions,
    )

    if args.dry_run:
        logger.info("干运行模式 - 显示将要下载的内容:")
        for _, row in events_df.iterrows():
            event_id = row['event_id']
            start_time = row['start_time']
            end_time = row['end_time']
            download_start, download_end = downloader.get_download_time_range(start_time, end_time)

            logger.info(f"事件 {event_id}: {download_start} 到 {download_end}")

            for modality in downloader.modalities:
                if downloader.modalities[modality].get('required', False):
                    logger.info(f"  - {modality}")
        return

    # 开始下载
    downloader.download_all_events(events_df, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
