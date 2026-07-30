"""
HDF5数据创建器 - 将原始数据转换为HDF5格式
"""
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HDF5DatasetCreator:
    """HDF5数据集创建器"""

    def __init__(self, config: Dict):
        """
        初始化创建器

        Args:
            config: 配置字典，包含创建器所需的各种配置参数
        """
        # 将传入的配置字典保存为实例属性，供后续方法使用
        self.config = config
        # 从配置中提取模态信息，可能包含多种数据类型
        self.modalities = config['modalities']
        # 从配置中提取时间相关的设置，如时间窗口、采样频率等
        self.time_config = config['time']

        # 创建输出目录路径对象，用于存储生成的HDF5文件
        self.output_path = Path(config['hdf5_path'])
        # 创建输出目录的父目录（如果不存在）
        # parents=True表示创建所有必要的父目录
        # exist_ok=True表示如果目录已存在则不会抛出异常
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def create_dataset(self, events_metadata: pd.DataFrame,
                       data_collector_func,
                       append: bool = False) -> None:
        """创建完整 HDF5 数据集，或在兼容配置下向现有文件追加事件。"""
        mode = 'a' if append and self.output_path.exists() else 'w'
        logger.info(f"开始创建HDF5数据集: {self.output_path} | mode={mode}")

        with h5py.File(self.output_path, mode) as f:
            if mode == 'w':
                self._set_global_attributes(f)
                self._create_index_table(f, events_metadata)
                self._create_event_groups(f, events_metadata, data_collector_func)
            else:
                self._validate_append_compatibility(f)
                self._append_events(f, events_metadata, data_collector_func)

        logger.info(f"HDF5数据集创建完成: {self.output_path}")

    def _set_global_attributes(self, h5_file: h5py.File) -> None:
        """设置全局属性"""
        h5_file.attrs['dataset_name'] = 'Solar_Flares_CME_dataset'
        h5_file.attrs['creation_date'] = datetime.now().isoformat()
        h5_file.attrs['version'] = '1.0'
        h5_file.attrs['modalities'] = json.dumps(list(self.modalities.keys()))
        h5_file.attrs['time_config'] = json.dumps(self.time_config)

    def _index_dtype(self):
        return np.dtype([
            ('event_id', 'S32'),
            ('start_time', 'S26'),
            ('end_time', 'S26'),
            ('flare_class', 'S4'),
            ('label', 'uint8'),
            ('peak_time', 'S26'),
            ('peak_flux', 'float32'),
            ('duration', 'float32'),
            ('active_region', 'S12'),
            ('data_available', 'bool'),
            ('num_frames', 'uint16')
        ])

    def _event_row_to_index_record(self, row) -> np.void:
        record = np.zeros((), dtype=self._index_dtype())
        record['event_id'] = str(row['event_id']).encode('utf-8')
        record['start_time'] = str(row['start_time']).encode('utf-8')
        record['end_time'] = str(row['end_time']).encode('utf-8')
        record['flare_class'] = str(row['flare_class']).encode('utf-8')
        record['label'] = int(row['label'])
        record['peak_time'] = str(row['peak_time']).encode('utf-8')
        record['peak_flux'] = float(row['peak_flux'])
        record['duration'] = float(row['duration'])
        record['active_region'] = str(row['active_region']).encode('utf-8')
        record['data_available'] = bool(row['data_available'])
        record['num_frames'] = 0
        return record

    def _decode_index_event_id(self, value) -> str:
        if isinstance(value, bytes):
            return value.decode('utf-8').replace(chr(0), '').strip()
        return str(value).strip()

    def _event_group_has_complete_data(self, event_group: h5py.Group) -> bool:
        if 'timestamps' not in event_group:
            return False
        if 'data' not in event_group:
            return False
        if len(event_group['data'].keys()) == 0:
            return False
        num_frames = int(event_group.attrs.get('num_frames', 0) or 0)
        return num_frames > 0

    def _get_index_record_map(self, h5_file: h5py.File) -> Dict[str, np.void]:
        records = {}
        if 'index_table' not in h5_file:
            return records
        for row in h5_file['index_table']:
            event_id = self._decode_index_event_id(row['event_id'])
            if event_id:
                records[event_id] = row
        return records

    def _split_existing_event_ids(self, h5_file: h5py.File) -> Tuple[set, set]:
        complete = set()
        incomplete = set()
        index_records = self._get_index_record_map(h5_file)

        if 'events' in h5_file:
            for event_id, group in h5_file['events'].items():
                has_complete_group = self._event_group_has_complete_data(group)
                row = index_records.get(str(event_id))
                has_complete_index = False
                if row is not None:
                    has_complete_index = bool(row['data_available']) and int(row['num_frames']) > 0

                if has_complete_group and has_complete_index:
                    complete.add(str(event_id))
                else:
                    incomplete.add(str(event_id))

        for event_id, row in index_records.items():
            has_complete_index = bool(row['data_available']) and int(row['num_frames']) > 0
            if event_id in complete:
                continue
            if has_complete_index and event_id not in incomplete:
                complete.add(event_id)
            else:
                incomplete.add(event_id)

        complete.discard('')
        incomplete.discard('')
        incomplete -= complete
        return complete, incomplete

    def _remove_existing_event(self, h5_file: h5py.File, event_id: str) -> None:
        if 'events' in h5_file and event_id in h5_file['events']:
            del h5_file['events'][event_id]

        if 'index_table' in h5_file:
            index_table = self._ensure_index_table_resizable(h5_file)
            keep_rows = []
            for row in index_table:
                if self._decode_index_event_id(row['event_id']) != event_id:
                    keep_rows.append(row)
            new_size = len(keep_rows)
            index_table.resize((new_size,))
            if new_size:
                index_table[:] = keep_rows

    def _validate_append_compatibility(self, h5_file: h5py.File) -> None:
        existing_modalities = json.loads(h5_file.attrs.get('modalities', '[]'))
        if list(existing_modalities) != list(self.modalities.keys()):
            raise ValueError(
                f'追加失败：模态配置不兼容。existing={existing_modalities}, current={list(self.modalities.keys())}'
            )
        existing_time_config = json.loads(h5_file.attrs.get('time_config', '{}'))
        if dict(existing_time_config) != dict(self.time_config):
            raise ValueError(
                f'追加失败：时间配置不兼容。existing={existing_time_config}, current={self.time_config}'
            )

    def _ensure_index_table_resizable(self, h5_file: h5py.File) -> h5py.Dataset:
        index_table = h5_file['index_table']
        if index_table.maxshape == (None,):
            return index_table

        existing = index_table[:]
        del h5_file['index_table']
        return h5_file.create_dataset(
            'index_table',
            data=existing,
            dtype=self._index_dtype(),
            maxshape=(None,),
            compression='gzip'
        )

    def _append_index_records(self, h5_file: h5py.File, rows: List[pd.Series]) -> int:
        index_table = self._ensure_index_table_resizable(h5_file)
        start_idx = len(index_table)
        index_table.resize((start_idx + len(rows),))
        for offset, row in enumerate(rows):
            index_table[start_idx + offset] = self._event_row_to_index_record(row)
        return start_idx

    def _append_events(self, h5_file: h5py.File,
                       events_metadata: pd.DataFrame,
                       data_collector_func) -> None:
        complete_event_ids, incomplete_event_ids = self._split_existing_event_ids(h5_file)
        rows_to_add = []
        for _, row in events_metadata.iterrows():
            event_id = str(row['event_id']).strip()
            if not event_id or event_id.lower() == 'nan':
                logger.warning('跳过缺失 event_id 的事件: %s', row.to_dict())
                continue
            if event_id in incomplete_event_ids:
                logger.warning('检测到 HDF5 中的不完整事件，将重建: %s', event_id)
                self._remove_existing_event(h5_file, event_id)
                incomplete_event_ids.discard(event_id)
            elif event_id in complete_event_ids:
                logger.info('事件已完整存在于 HDF5，跳过: %s', event_id)
                continue
            rows_to_add.append(row)

        if not rows_to_add:
            logger.info('没有需要追加的新事件。')
            return

        if 'events' not in h5_file:
            h5_file.create_group('events')

        start_idx = self._append_index_records(h5_file, rows_to_add)
        self._create_event_groups(
            h5_file,
            pd.DataFrame(rows_to_add),
            data_collector_func,
            start_index=start_idx,
        )

    def _create_index_table(self, h5_file: h5py.File,
                            events_metadata: pd.DataFrame) -> None:
        """创建索引表。"""
        num_events = len(events_metadata)
        index_table = h5_file.create_dataset(
            'index_table',
            shape=(num_events,),
            dtype=self._index_dtype(),
            maxshape=(None,),
            compression='gzip'
        )

        for i, (_, row) in enumerate(events_metadata.iterrows()):
            index_table[i] = self._event_row_to_index_record(row)

        logger.info(f"索引表创建完成: {num_events} 个事件")

    def _create_event_groups(self, h5_file: h5py.File,
                             events_metadata: pd.DataFrame,
                             data_collector_func,
                             start_index: int = 0) -> None:
        """创建事件组并写入数据。"""
        events_group = h5_file.require_group('events')

        for offset, (_, row) in enumerate(events_metadata.iterrows()):
            i = start_index + offset
            event_id = str(row['event_id']).strip()
            logger.info(f"处理事件 {offset + 1}/{len(events_metadata)}: {event_id}")

            try:
                event_group = events_group.create_group(event_id)
                self._add_event_metadata(event_group, row)
                time_series_data = data_collector_func(row)
                num_frames = self._add_time_series_data(event_group, time_series_data)

                logger.info(f"事件 {event_id} (索引 {i}) 写入完成: {num_frames} 帧")
                index_row = np.array(h5_file['index_table'][i])
                index_row['num_frames'] = np.uint16(num_frames)
                h5_file['index_table'][i] = index_row
                h5_file.flush()

            except Exception as e:
                logger.error(f"处理事件 {event_id} 失败: {e}", exc_info=True)
                h5_file['index_table']['data_available'][i] = False

    def _add_event_metadata(self, event_group: h5py.Group,
                            metadata: pd.Series) -> None:
        """添加事件元数据（支持多个活动）"""
        # 检查是否有多个活动
        if 'activities' in metadata and isinstance(metadata['activities'], list):
            # 多活动模式
            activities_data = []
            for activity in metadata['activities']:
                activity_dict = {
                    'flare_class': activity.get('flare_class', ''),
                    'peak_time': activity.get('peak_time', ''),
                    'start_time': activity.get('start_time', ''),
                    'end_time': activity.get('end_time', ''),
                    'label': int(activity.get('label', 0)),
                    'active_region': activity.get('active_region', ''),
                    'peak_flux': float(activity.get('peak_flux', 0)),
                    'duration_minutes': float(activity.get('duration', 0))
                }
                activities_data.append(activity_dict)

            # 存储为JSON字符串
            import json
            event_group.attrs['activities'] = json.dumps(activities_data)
            event_group.attrs['num_activities'] = len(activities_data)

            # 为了向后兼容，设置主要活动的属性
            primary_activity = activities_data[0] if activities_data else {}
            event_group.attrs['flare_class'] = primary_activity.get('flare_class', '')
            event_group.attrs['peak_time'] = primary_activity.get('peak_time', '')
            event_group.attrs['start_time'] = primary_activity.get('start_time', '')
            event_group.attrs['end_time'] = primary_activity.get('end_time', '')
            event_group.attrs['label'] = int(primary_activity.get('label', 0))
        else:
            # 单活动模式（向后兼容）
            event_group.attrs['flare_class'] = metadata['flare_class']
            event_group.attrs['peak_time'] = metadata['peak_time']
            event_group.attrs['start_time'] = metadata['start_time']
            event_group.attrs['end_time'] = metadata['end_time']
            event_group.attrs['label'] = int(metadata['label'])
            event_group.attrs['active_region'] = metadata['active_region']
            event_group.attrs['peak_flux'] = float(metadata['peak_flux'])
            event_group.attrs['duration_minutes'] = float(metadata['duration'])
            event_group.attrs['num_activities'] = 1

        # 如果 CSV 中包含 bbox（xmin,ymin,xmax,ymax），则保存为事件属性（非破坏性）
        try:
            if not pd.isna(metadata.get('bbox_xmin')):
                event_group.attrs['bbox_xmin'] = float(metadata.get('bbox_xmin'))
            if not pd.isna(metadata.get('bbox_ymin')):
                event_group.attrs['bbox_ymin'] = float(metadata.get('bbox_ymin'))
            if not pd.isna(metadata.get('bbox_xmax')):
                event_group.attrs['bbox_xmax'] = float(metadata.get('bbox_xmax'))
            if not pd.isna(metadata.get('bbox_ymax')):
                event_group.attrs['bbox_ymax'] = float(metadata.get('bbox_ymax'))
        except Exception:
            logger.debug(f"事件 {event_group.name} 的 bbox 字段无法解析，已跳过保存 bbox")

        # 从事件组目录的 bboxes.json 读取多事件 bbox（主事件+非主事件）
        self._add_bboxes_from_json(event_group, metadata)

    def _add_bboxes_from_json(self, event_group: h5py.Group,
                              metadata: pd.Series) -> None:
        """从 bboxes.json 读取 region/activity 结构并写入事件组。"""
        event_id = str(metadata.get('event_id', event_group.name))
        processed_dir = Path(self.config.get('processed_data_dir', 'data/processed'))
        bbox_file = processed_dir / event_id / 'bboxes.json'
        if not bbox_file.exists():
            return
        try:
            with open(bbox_file, 'r', encoding='utf-8') as f:
                bbox_info = json.load(f)
        except Exception as e:
            logger.warning(f"读取 bboxes.json 失败 {event_id}: {e}")
            return

        regions = bbox_info.get('regions', [])
        activities = bbox_info.get('activities', [])
        bbox_resolution = bbox_info.get('bbox_resolution', {}) or {}
        bbox_width = int(bbox_resolution.get('width', bbox_info.get('anno_width', 0)) or 0)
        bbox_height = int(bbox_resolution.get('height', bbox_info.get('anno_height', 0)) or 0)

        if not regions and bbox_info.get('bboxes'):
            # 兼容旧格式：每个 bbox 当成一个 region 与 activity
            legacy = bbox_info.get('bboxes', [])
            regions = []
            activities = []
            for idx, item in enumerate(legacy, start=1):
                region_id = str(item.get('event_id', f'R{idx}'))
                regions.append({
                    'region_id': region_id,
                    'region_position': '',
                    'is_primary_region': bool(item.get('is_primary', False)),
                    'bbox': item.get('bbox', [0, 0, 0, 0]),
                })
                activities.append({
                    'event_id': str(item.get('event_id', '')),
                    'is_primary_activity': bool(item.get('is_primary', False)),
                    'label': int(item.get('label', 0)),
                    'region_id': region_id,
                    'active_region_source': '',
                    'active_region_position': '',
                })

        if regions:
            region_boxes = np.array([r.get('bbox', [0, 0, 0, 0]) for r in regions], dtype=np.float32)
            event_group.create_dataset('regions', data=region_boxes, compression='gzip')
            event_group.attrs['region_ids'] = json.dumps([str(r.get('region_id', '')) for r in regions])
            event_group.attrs['region_positions'] = json.dumps([str(r.get('region_position', '')) for r in regions])
            event_group.attrs['region_is_primary'] = json.dumps([bool(r.get('is_primary_region', False)) for r in regions])
            event_group.attrs['bbox_width'] = int(bbox_width) if bbox_width > 0 else int(np.max(region_boxes[:, [0, 2]]) if region_boxes.size > 0 else 0)
            event_group.attrs['bbox_height'] = int(bbox_height) if bbox_height > 0 else int(np.max(region_boxes[:, [1, 3]]) if region_boxes.size > 0 else 0)
            event_group.attrs['num_regions'] = len(regions)

        if activities:
            event_group.attrs['activity_event_ids'] = json.dumps([str(a.get('event_id', '')) for a in activities])
            event_group.attrs['activity_labels'] = json.dumps([int(a.get('label', 0)) for a in activities])
            event_group.attrs['activity_is_primary'] = json.dumps([bool(a.get('is_primary_activity', False)) for a in activities])
            event_group.attrs['activity_region_ids'] = json.dumps([str(a.get('region_id', '')) for a in activities])
            event_group.attrs['activity_region_positions'] = json.dumps([str(a.get('active_region_position', '')) for a in activities])
            event_group.attrs['num_activities'] = len(activities)

        logger.info(f"事件 {event_id} 已加载 {len(regions)} 个 regions / {len(activities)} 个 activities")

    def _add_time_series_data(self, event_group: h5py.Group,
                              time_series_data: Dict) -> int:
        """添加时间序列数据"""
        # 1. 添加时间戳
        timestamps = time_series_data['timestamps']
        event_group.create_dataset(
            'timestamps',
            data=np.array(timestamps, dtype='S26'),
            compression='gzip'
        )
        num_frames = len(timestamps)
        
        # 将帧数写入事件组属性，供后续读取使用
        event_group.attrs['num_frames'] = num_frames

        # 2. 添加每种模态的数据
        data_group = event_group.create_group('data')

        for modality_name, modality_config in self.modalities.items():
            if modality_name in time_series_data:
                # 获取数据
                modality_data = time_series_data[modality_name]

                # 创建模态组
                modality_group = data_group.create_group(modality_name)

                # 添加模态属性
                modality_group.attrs['wavelength'] = modality_config['wavelength']
                modality_group.attrs['unit'] = modality_config['unit']
                modality_group.attrs['instrument'] = modality_config['instrument']

                # 创建数据集
                # 使用实际数据形状而不是配置中的分辨率，以支持不同分辨率的测试
                data_shape = modality_data.shape

                dset = modality_group.create_dataset(
                    'images',
                    shape=data_shape,
                    dtype='float32',
                    chunks=True,
                    compression='gzip',
                    compression_opts=6,
                    fillvalue=np.nan
                )

                # 填充数据
                dset[:] = modality_data

                # 添加数据质量信息
                if f'{modality_name}_quality' in time_series_data:
                    quality_data = time_series_data[f'{modality_name}_quality']
                    modality_group.create_dataset(
                        'quality_mask',
                        data=quality_data.astype('bool'),
                        compression='gzip'
                    )

                event_group.attrs[f'{modality_name}_available'] = True
            else:
                # 标记模态缺失
                event_group.attrs[f'{modality_name}_available'] = False
                warnings.warn(f"事件 {event_group.name} 缺失模态 {modality_name}")

        # 3. 添加辅助数据
        if 'auxiliary' in time_series_data:
            aux_group = event_group.create_group('auxiliary')
            for key, value in time_series_data['auxiliary'].items():
                if isinstance(value, np.ndarray):
                    aux_group.create_dataset(key, data=value, compression='gzip')

        return num_frames


if __name__ == '__main__':
    import sys
    import argparse
    from pathlib import Path

    # 添加项目路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging

    # ==================== 直接修改这部分配置 ====================
    # 如果想直接运行代码，修改以下参数
    USE_CLI_ARGS = False  # 设置为 False 使用下面的配置，True 使用命令行参数
    
    CONFIG_PATH = r'D:\ChenMeng\Graduate_Student\Experiment_and_Data\Flare_with_without_CME_forecast\configs\data_config.yaml'
    EVENTS_CSV_PATH = r'D:\ChenMeng\Graduate_Student\Experiment_and_Data\Flare_with_without_CME_forecast\data\raw\events_example.csv'  # 必需：修改为你的 CSV 文件路径
    # CONFIG_PATH = r'configs\data_config.yaml'
    # EVENTS_CSV_PATH = r'data\raw\events_example.csv'  # 必需：修改为你的 CSV 文件路径
    OUTPUT_PATH = r'data/Solar_Flares_CME_dataset.h5'
    LOG_DIR = 'logs'
    DEBUG = False
    # ============================================================

    def parse_args():
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description='创建HDF5数据集',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  python data/hdf5_creator.py --config configs/data_config.yaml --events data/raw/events.csv
  python data/hdf5_creator.py --events data/raw/events.csv --output data/my_dataset.h5
  
或者：修改脚本中的 USE_CLI_ARGS = False 并直接运行脚本
            """
        )
        parser.add_argument('--config', type=str, default='configs/data_config.yaml',
                            help='数据配置文件路径')
        parser.add_argument('--events', type=str, default=None,
                            help='事件元数据CSV文件路径')
        parser.add_argument('--output', type=str, default=None,
                            help='输出HDF5文件路径（如不指定则使用配置中的路径）')
        parser.add_argument('--log_dir', type=str, default='logs',
                            help='日志目录')
        parser.add_argument('--debug', action='store_true',
                            help='启用调试模式')
        return parser.parse_args()

    def load_events_metadata(csv_path: str) -> pd.DataFrame:
        """加载事件元数据CSV"""
        df = pd.read_csv(csv_path)

        # 检查必需列
        required_columns = ['event_id', 'start_time', 'end_time', 'flare_class',
                            'cme_associated', 'peak_time', 'peak_flux', 'duration',
                            'active_region']

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV文件缺少必需列: {col}")

        # 添加标签列（如果不存在）
        if 'label' not in df.columns:
            df['label'] = df['cme_associated'].apply(lambda x: 1 if x else 0)

        if 'data_available' not in df.columns:
            df['data_available'] = True

        # 解析可选 bbox 列（xmin, ymin, xmax, ymax），若不存在则填充为 NA
        bbox_cols = ['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']
        for col in bbox_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = pd.NA

        return df

    def collect_event_data(event_row: pd.Series, config: Dict) -> Dict:
        """
        收集事件数据

        注意：这是一个示例实现，使用随机数据。
        在实际使用时，你需要修改这个函数来加载真实的卫星数据。
        """
        event_id = event_row['event_id']
        logger.info(f"收集事件数据: {event_id}")

        start_time = datetime.fromisoformat(event_row['start_time'].replace('Z', '').replace('+00:00', ''))
        end_time = datetime.fromisoformat(event_row['end_time'].replace('Z', '').replace('+00:00', ''))

        # 从配置文件读取时间参数，确保与data_config.yaml中的配置一致
        time_config = config.get('time', {})
        pre_hours = time_config.get('pre_event_hours', 6)
        post_hours = time_config.get('post_event_hours', 3)
        cadence_minutes = time_config.get('cadence_minutes', 30)

        # 统一以 start_time 为基准，确保所有事件的帧数一致
        # 这样可以避免因事件持续时间不同导致的帧数差异
        extended_start = start_time - timedelta(hours=pre_hours)
        extended_end = start_time + timedelta(hours=post_hours)

        # 生成时间戳
        timestamps = []
        current = extended_start
        while current <= extended_end:
            timestamps.append(current.isoformat())
            current += timedelta(minutes=cadence_minutes)

        num_frames = len(timestamps)
        logger.info(f"为事件 {event_id} 生成 {num_frames} 帧的测试数据")

        # 从配置获取模态信息
        modalities = config['modalities']

        # 初始化时间序列数据字典
        time_series_data = {}

        # ========== 这里需要根据你的实际数据源修改 ==========
        # 示例：生成随机数据用于测试（使用缩小分辨率）
        # 生产环境中应改为实际的卫星数据加载，分辨率改为config中定义的值
        test_resolution = (256, 256)  # 测试模式用256x256；生产环境改为(512, 512)等
        
        for modality_name, modality_config in modalities.items():
            channels = modality_config.get('channels', 1)

            if channels > 1:
                data_shape = (num_frames, channels, *test_resolution)
            else:
                data_shape = (num_frames, *test_resolution)

            # 生成随机数据（实际使用时替换为真实数据加载）
            time_series_data[modality_name] = np.random.randn(*data_shape).astype(np.float32)

        # 辅助数据
        time_series_data['auxiliary'] = {
            'ar_mask': np.random.randint(0, 3, (num_frames, *test_resolution), dtype=np.uint8),
        }

        # 添加时间戳
        time_series_data['timestamps'] = timestamps

        return time_series_data

    def main():
        """主函数"""
        try:
            # 支持两种模式：直接配置或命令行参数
            if USE_CLI_ARGS:
                args = parse_args()
                config_path = args.config
                events_csv_path = args.events
                output_path = args.output
                log_dir = args.log_dir
                debug = args.debug
            else:
                config_path = CONFIG_PATH
                events_csv_path = EVENTS_CSV_PATH
                output_path = OUTPUT_PATH
                log_dir = LOG_DIR
                debug = DEBUG

            # 设置日志
            setup_logging(log_dir, 'hdf5_creator', debug=debug)
            global logger
            logger = logging.getLogger(__name__)

            # 打印参数调试信息
            logger.info("=" * 60)
            logger.info("【参数配置信息】")
            logger.info(f"USE_CLI_ARGS: {USE_CLI_ARGS}")
            logger.info(f"config_path: {config_path}")
            logger.info(f"events_csv_path: {events_csv_path}")
            logger.info(f"output_path: {output_path}")
            logger.info(f"log_dir: {log_dir}")
            logger.info(f"debug: {debug}")
            logger.info("=" * 60)

            # 同时打印到终端
            print("=" * 60)
            print("【参数配置信息】")
            print(f"USE_CLI_ARGS: {USE_CLI_ARGS}")
            print(f"config_path: {config_path}")
            print(f"events_csv_path: {events_csv_path}")
            print(f"output_path: {output_path}")
            print(f"log_dir: {log_dir}")
            print(f"debug: {debug}")
            print("=" * 60)

            # 检查 events_csv_path 是否为空
            if not events_csv_path:
                raise ValueError("请在代码中设置 EVENTS_CSV_PATH 或使用 --events 参数指定 CSV 文件路径")

            # 加载配置
            logger.info(f"加载配置: {config_path}")
            full_config = load_config(config_path)
            
            # 打印配置内容
            logger.info("【配置文件内容】")
            logger.info(f"配置顶级键: {list(full_config.keys())}")
            logger.info(f"完整配置: {full_config}")
            
            # 同时打印到终端
            print("\n【配置文件内容】")
            print(f"配置顶级键: {list(full_config.keys())}")
            print(f"完整配置: {full_config}")
            
            # 获取数据配置（data_config.yaml已包含'data'键）
            if 'data' in full_config:
                config = full_config['data']
                logger.info("✓ 成功提取 'data' 键")
                print("✓ 成功提取 'data' 键")
            else:
                config = full_config
                logger.warning("未在配置中找到 'data' 键，使用根配置")
                print("⚠ 未在配置中找到 'data' 键，使用根配置")
            
            logger.info(f"【数据配置】: {config}")
            print(f"【数据配置】: {config}")
            logger.info("=" * 60)
            print("=" * 60)

            # 更新输出路径
            if output_path:
                config['hdf5_path'] = output_path
                logger.info(f"输出路径已覆盖: {output_path}")

            # 加载事件元数据
            logger.info(f"加载事件元数据: {events_csv_path}")
            events_metadata = load_events_metadata(events_csv_path)
            logger.info(f"加载到 {len(events_metadata)} 个事件")

            # 创建HDF5数据集
            creator = HDF5DatasetCreator(config)

            # 使用lambda包装以传递config
            def data_collector(row):
                return collect_event_data(row, config)

            creator.create_dataset(events_metadata, data_collector)

            # 输出成功完成标识
            logger.info("=" * 60)
            logger.info("✓ HDF5数据集创建完成！")
            logger.info(f"✓ 文件路径: {config['hdf5_path']}")
            
            # 获取文件大小
            from pathlib import Path
            file_path = Path(config['hdf5_path'])
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ 文件大小: {file_size_mb:.2f} MB")
                logger.info(f"✓ 事件数量: {len(events_metadata)}")
            logger.info("=" * 60)
        
        except Exception as e:
            logger.error(f"程序执行失败: {e}", exc_info=True)
            raise

    main()
