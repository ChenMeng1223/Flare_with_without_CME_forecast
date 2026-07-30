"""
HDF5数据读取器 - 从HDF5文件读取数据
"""
import json
import os

import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class HDF5DatasetReader:
    """HDF5数据集读取器"""

    def __init__(self, hdf5_path: str, open_kwargs: Optional[Dict[str, Any]] = None):
        """
        初始化读取器

        Args:
            hdf5_path: HDF5文件路径
            open_kwargs: 传递给 h5py.File 的打开参数
        """
        self.hdf5_path = hdf5_path
        self.open_kwargs = dict(open_kwargs or {})

        if os.name == 'nt':
            self.open_kwargs.setdefault('locking', False)

        # 打开文件读取元数据
        with h5py.File(hdf5_path, 'r', **self.open_kwargs) as f:
            # 获取全局属性
            self.dataset_name = f.attrs.get('dataset_name', 'Unknown')
            self.creation_date = f.attrs.get('creation_date', 'Unknown')
            self.version = f.attrs.get('version', '1.0')
            self.modalities = json.loads(f.attrs.get('modalities', '[]'))

            # 加载索引表
            self.index_df = self._load_index_table(f)

            # 获取时间配置
            self.time_config = json.loads(f.attrs.get('time_config', '{}'))

        logger.info(f"加载数据集: {self.dataset_name}")
        logger.info(f"事件数量: {len(self.index_df)}")
        logger.info(f"可用模态: {self.modalities}")

    def _load_index_table(self, h5_file: h5py.File) -> pd.DataFrame:
        """加载索引表为DataFrame"""
        index_data = h5_file['index_table'][:]

        # build base dict
        data_dict = {
            'event_id': [x.decode('utf-8') for x in index_data['event_id']],
            'start_time': [x.decode('utf-8') for x in index_data['start_time']],
            'end_time': [x.decode('utf-8') for x in index_data['end_time']],
            'flare_class': [x.decode('utf-8') for x in index_data['flare_class']],
            'label': index_data['label'],
            'peak_time': [x.decode('utf-8') for x in index_data['peak_time']],
            'peak_flux': index_data['peak_flux'],
            'duration': index_data['duration'],
            'active_region': [x.decode('utf-8') for x in index_data['active_region']],
            'data_available': index_data['data_available'],
            'num_frames': index_data['num_frames']
        }
        if 'cme_associated' in index_data.dtype.names:
            data_dict['cme_associated'] = index_data['cme_associated']
        # optional bbox columns
        for col in ['bbox_xmin','bbox_ymin','bbox_xmax','bbox_ymax']:
            if col in index_data.dtype.names:
                data_dict[col] = index_data[col]
            else:
                data_dict[col] = np.zeros(len(index_data), dtype=np.float32)

        df = pd.DataFrame(data_dict)

        return df

    def get_event_ids(self, available_only: bool = True) -> List[str]:
        """
        获取事件ID列表

        Args:
            available_only: 是否只返回数据可用的事件

        Returns:
            事件ID列表
        """
        if available_only:
            return self.index_df[self.index_df['data_available']]['event_id'].tolist()
        return self.index_df['event_id'].tolist()

    def get_event_metadata(self, event_id: str) -> Dict[str, Any]:
        """
        获取事件元数据

        Args:
            event_id: 事件ID

        Returns:
            事件元数据字典
        """
        with h5py.File(self.hdf5_path, 'r', **self.open_kwargs) as f:
            event_group = f[f'events/{event_id}']

            # load attributes
            metadata = {'event_id': event_id}
            for k, v in event_group.attrs.items():
                # attributes are often stored as numpy types, keep label as plain int
                if k == 'label':
                    metadata[k] = int(v)
                else:
                    metadata[k] = v

            # 解析activities（如果是多活动模式）
            if 'activities' in metadata and isinstance(metadata['activities'], str):
                import json
                metadata['activities'] = json.loads(metadata['activities'])

            # 添加模态可用性信息
            modality_availability = {}
            for modality in self.modalities:
                modality_availability[modality] = event_group.attrs.get(
                    f'{modality}_available', False
                )
            metadata['modality_availability'] = modality_availability

            return metadata

    def get_event_data(self, event_id: str,
                       modalities: Optional[List[str]] = None,
                       time_range: Optional[Tuple[str, str]] = None) -> Dict:
        """
        获取事件数据

        Args:
            event_id: 事件ID
            modalities: 要获取的模态列表，None表示获取所有可用模态
            time_range: 时间范围(start_time, end_time)，None表示获取所有时间

        Returns:
            事件数据字典
        """
        with h5py.File(self.hdf5_path, 'r', **self.open_kwargs) as f:
            event_path = f'events/{event_id}'
            if event_path not in f:
                raise ValueError(f"事件 {event_id} 不存在")

            event_group = f[event_path]

            # 获取时间戳
            timestamps = [ts.decode('utf-8') for ts in event_group['timestamps'][:]]

            # 时间范围筛选
            if time_range:
                start_idx, end_idx = self._filter_by_time(timestamps, time_range)
                timestamps = timestamps[start_idx:end_idx]
            else:
                start_idx, end_idx = 0, len(timestamps)

            # 确定要获取的模态
            if modalities is None:
                modalities = self.modalities

            # 加载数据
            data_dict = {'timestamps': timestamps}

            for modality in modalities:
                modality_path = f'{event_path}/data/{modality}'
                if modality_path in f:
                    modality_group = f[modality_path]

                    # 加载图像数据
                    images = modality_group['images'][start_idx:end_idx]

                    # 处理缺失数据
                    images = self._handle_missing_data(images)

                    data_dict[modality] = images

                    # 加载质量掩码（如果有）
                    if 'quality_mask' in modality_group:
                        quality_mask = modality_group['quality_mask'][start_idx:end_idx]
                        data_dict[f'{modality}_quality'] = quality_mask
                else:
                    # 模态缺失
                    data_dict[modality] = None
                    logger.warning(f"事件 {event_id} 缺失模态 {modality}")

            # 加载辅助数据
            aux_path = f'{event_path}/auxiliary'
            if aux_path in f:
                aux_group = f[aux_path]
                auxiliary_data = {}
                for key in aux_group.keys():
                    aux_data = aux_group[key][start_idx:end_idx]
                    auxiliary_data[key] = aux_data
                data_dict['auxiliary'] = auxiliary_data

            # 加载 region/activity 信息（如果存在）
            if 'regions' in event_group:
                data_dict['regions'] = event_group['regions'][:]
                data_dict['bbox_resolution'] = {
                    'width': int(event_group.attrs.get('bbox_width', 0) or 0),
                    'height': int(event_group.attrs.get('bbox_height', 0) or 0),
                }
                try:
                    data_dict['region_ids'] = json.loads(event_group.attrs.get('region_ids', '[]'))
                except Exception:
                    data_dict['region_ids'] = []
                try:
                    data_dict['region_positions'] = json.loads(event_group.attrs.get('region_positions', '[]'))
                except Exception:
                    data_dict['region_positions'] = []
                try:
                    data_dict['region_is_primary'] = json.loads(event_group.attrs.get('region_is_primary', '[]'))
                except Exception:
                    data_dict['region_is_primary'] = []
                data_dict['num_regions'] = int(event_group.attrs.get('num_regions', 0))

            for attr_name, key in [
                ('activity_event_ids', 'activity_event_ids'),
                ('activity_labels', 'activity_labels'),
                ('activity_is_primary', 'activity_is_primary'),
                ('activity_region_ids', 'activity_region_ids'),
                ('activity_region_positions', 'activity_region_positions'),
            ]:
                try:
                    data_dict[key] = json.loads(event_group.attrs.get(attr_name, '[]'))
                except Exception:
                    data_dict[key] = []
            if 'activity_event_ids' in data_dict:
                data_dict['num_activities'] = len(data_dict['activity_event_ids'])

            # 兼容旧bbox字段
            if 'bboxes' in event_group and 'regions' not in event_group:
                boxes = event_group['bboxes'][:]
                data_dict['bboxes'] = boxes
                try:
                    data_dict['bbox_event_ids'] = json.loads(event_group.attrs.get('bbox_event_ids', '[]'))
                except Exception:
                    data_dict['bbox_event_ids'] = []
                try:
                    data_dict['bbox_is_primary'] = json.loads(event_group.attrs.get('bbox_is_primary', '[]'))
                except Exception:
                    data_dict['bbox_is_primary'] = []
                try:
                    data_dict['bbox_labels'] = json.loads(event_group.attrs.get('bbox_labels', '[]'))
                except Exception:
                    data_dict['bbox_labels'] = []
                data_dict['num_bboxes'] = int(event_group.attrs.get('num_bboxes', 0))

            return data_dict

    def get_event_regions(self, event_id: str) -> Tuple[np.ndarray, List[str], List[str], List[bool]]:
        with h5py.File(self.hdf5_path, 'r', **self.open_kwargs) as f:
            event_path = f'events/{event_id}'
            if event_path not in f:
                raise ValueError(f"事件 {event_id} 不存在")
            event_group = f[event_path]
            if 'regions' not in event_group:
                return np.zeros((0, 4), dtype=np.float32), [], [], []
            boxes = event_group['regions'][:]
            try:
                region_ids = json.loads(event_group.attrs.get('region_ids', '[]'))
            except Exception:
                region_ids = []
            try:
                region_positions = json.loads(event_group.attrs.get('region_positions', '[]'))
            except Exception:
                region_positions = []
            try:
                is_primary = json.loads(event_group.attrs.get('region_is_primary', '[]'))
            except Exception:
                is_primary = []
            return boxes, region_ids, region_positions, is_primary

    def get_event_activities(self, event_id: str) -> Tuple[List[str], List[int], List[bool], List[str]]:
        with h5py.File(self.hdf5_path, 'r', **self.open_kwargs) as f:
            event_path = f'events/{event_id}'
            if event_path not in f:
                raise ValueError(f"事件 {event_id} 不存在")
            event_group = f[event_path]
            try:
                event_ids = json.loads(event_group.attrs.get('activity_event_ids', '[]'))
            except Exception:
                event_ids = []
            try:
                labels = json.loads(event_group.attrs.get('activity_labels', '[]'))
            except Exception:
                labels = []
            try:
                is_primary = json.loads(event_group.attrs.get('activity_is_primary', '[]'))
            except Exception:
                is_primary = []
            try:
                region_ids = json.loads(event_group.attrs.get('activity_region_ids', '[]'))
            except Exception:
                region_ids = []
            return event_ids, labels, is_primary, region_ids

    def get_event_bboxes(self, event_id: str) -> Tuple[np.ndarray, List[str], List[bool], List[int]]:
        """
        获取事件组中预存的bbox信息

        Returns:
            boxes: numpy array shape [N, 4]
            event_ids: list of对应事件ID
            is_primary: list of bool标记是否为主事件
            labels: list of int 对应CME标签（1/2）
        """
        with h5py.File(self.hdf5_path, 'r', **self.open_kwargs) as f:
            event_path = f'events/{event_id}'
            if event_path not in f:
                raise ValueError(f"事件 {event_id} 不存在")
            event_group = f[event_path]
            if 'bboxes' not in event_group:
                return np.zeros((0, 4), dtype=np.float32), [], [], []

            boxes = event_group['bboxes'][:]
            try:
                event_ids = json.loads(event_group.attrs.get('bbox_event_ids', '[]'))
            except Exception:
                event_ids = []
            try:
                is_primary = json.loads(event_group.attrs.get('bbox_is_primary', '[]'))
            except Exception:
                is_primary = []
            try:
                labels = json.loads(event_group.attrs.get('bbox_labels', '[]'))
            except Exception:
                labels = []

            return boxes, event_ids, is_primary, labels

    def _filter_by_time(self, timestamps: List[str],
                        time_range: Tuple[str, str]) -> Tuple[int, int]:
        """根据时间范围筛选数据"""
        start_time, end_time = time_range

        start_idx = 0
        for i, ts in enumerate(timestamps):
            if ts >= start_time:
                start_idx = i
                break

        end_idx = len(timestamps)
        for i, ts in enumerate(timestamps):
            if ts > end_time:
                end_idx = i
                break

        return start_idx, end_idx

    def _handle_missing_data(self, data: np.ndarray) -> np.ndarray:
        """处理缺失数据"""
        if np.any(np.isnan(data)):
            # 时间维度线性插值
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    if data.shape[0] > 1:  # 多时间帧
                        column = data[:, i, j]
                        mask = ~np.isnan(column)
                        if np.any(mask) and np.any(~mask):
                            # 线性插值
                            indices = np.where(mask)[0]
                            values = column[mask]
                            column = np.interp(
                                np.arange(len(column)),
                                indices,
                                values
                            )
                            data[:, i, j] = column
                    else:
                        # 单时间帧，用平均值填充
                        data[np.isnan(data)] = np.nanmean(data)

        return data

    def create_sliding_windows(self, event_id: str,
                               window_size: int = 24,
                               stride: int = 1) -> Tuple[List[Dict], List[Dict]]:
        """
        创建滑动窗口样本

        Args:
            event_id: 事件ID
            window_size: 窗口大小(帧数)
            stride: 滑动步长(帧数)

        Returns:
            (窗口数据列表, 窗口元数据列表)
        """
        # 获取事件数据
        event_data = self.get_event_data(event_id)
        T = len(event_data['timestamps'])

        windows = []
        window_metadata = []

        # 创建滑动窗口
        for start_idx in range(0, T - window_size + 1, stride):
            end_idx = start_idx + window_size

            # 提取窗口数据
            window_data = {}
            for modality in self.modalities:
                if event_data.get(modality) is not None:
                    window_data[modality] = event_data[modality][start_idx:end_idx]
                else:
                    window_data[modality] = None

            # 窗口时间戳
            window_timestamps = event_data['timestamps'][start_idx:end_idx]

            # 窗口元数据
            event_meta = self.get_event_metadata(event_id)
            window_meta = {
                'event_id': event_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': window_timestamps[0],
                'end_time': window_timestamps[-1],
                'timestamps': window_timestamps,
                'label': self._get_window_label(event_meta, window_timestamps),
                'contains_event': self._window_contains_event(event_meta, window_timestamps),
                'peak_time_in_window': event_meta['peak_time'] in window_timestamps
            }

            windows.append(window_data)
            window_metadata.append(window_meta)

        return windows, window_metadata

    def _get_window_label(self, event_meta: Dict,
                          window_timestamps: List[str]) -> int:
        """获取窗口标签"""
        # 检查窗口是否包含峰值时间
        if event_meta['peak_time'] in window_timestamps:
            return event_meta['label']  # 1或2

        # 检查窗口是否在事件期间内
        window_start = window_timestamps[0]
        window_end = window_timestamps[-1]

        event_start = event_meta['start_time']
        event_end = event_meta['end_time']

        if (window_start <= event_end and window_end >= event_start):
            # 窗口与事件有重叠
            return event_meta['label']

        return 0  # 无事件

    def _window_contains_event(self, event_meta: Dict,
                               window_timestamps: List[str]) -> bool:
        """检查窗口是否包含事件"""
        window_start = window_timestamps[0]
        window_end = window_timestamps[-1]

        event_start = event_meta['start_time']
        event_end = event_meta['end_time']

        # 检查是否有重叠
        return (window_start <= event_end and window_end >= event_start)