"""
PyTorch数据集类
"""
from datetime import datetime
import json
import os

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from data.hdf5_reader import HDF5DatasetReader

logger = logging.getLogger(__name__)


def _to_fixed_auxiliary(aux: Optional[np.ndarray]) -> torch.Tensor:
    """将辅助特征转为固定形状 (1,)，便于 DataLoader collate；训练未使用，统一占位"""
    return torch.zeros(1, dtype=torch.float32)


class SolarFlareDataset(Dataset):
    """太阳耀斑数据集类"""

    def __init__(self, reader: HDF5DatasetReader,
                 event_ids: List[str],
                 modalities: Optional[List[str]] = None,
                 sequence_length: int = 48,
                 stride: int = 6,
                 target_size: Tuple[int, int] = (512, 512),
                 handle_missing: str = 'interpolate',
                 fill_value: float = 0.0,
                 transform: Optional[Any] = None,
                 target_transform: Optional[Any] = None,
                 max_activities: int = 5,
                 config: Optional[Dict] = None):
        """
        初始化数据集

        Args:
            reader: HDF5读取器
            event_ids: 事件ID列表
            modalities: 模态列表，None表示使用所有模态
            sequence_length: 序列长度
            stride: 滑动步长
            target_size: 目标图像尺寸
            handle_missing: 缺失数据处理方式
            fill_value: 填充值
            transform: 数据变换
            target_transform: 目标变换
            max_activities: 最大活动数量，用于标签和bbox填充
            config: 可选配置字典，至少包含'max_activities'
        """
        self.reader = reader
        self.event_ids = event_ids
        self.modalities = modalities or reader.modalities
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_size = target_size
        self.handle_missing = handle_missing
        self.fill_value = fill_value
        self.transform = transform
        self.target_transform = target_transform

        # 保持max_activities和config一致
        self.max_activities = max_activities
        if config is None:
            self.config = {'max_activities': max_activities}
        else:
            self.config = config
            # 如果config中也指定了max_activities，将其覆盖参数值
            self.max_activities = config.get('max_activities', max_activities)

        # 预计算所有滑动窗口
        self._h5_open_kwargs = self._build_h5_open_kwargs()
        self.proposal_cache_path = self.config.get('proposal_cache_path')
        self.proposal_cache = self._load_proposal_cache(self.proposal_cache_path)
        self.windows = self._precompute_windows()

        logger.info(f"数据集初始化完成: {len(self.event_ids)}个事件, {len(self.windows)}个窗口")

    def _build_h5_open_kwargs(self) -> Dict[str, Any]:
        open_kwargs: Dict[str, Any] = {}
        open_kwargs.update(getattr(self.reader, 'open_kwargs', {}))
        if os.name == 'nt':
            open_kwargs.setdefault('locking', False)
        return open_kwargs

    def _open_hdf5(self):
        return h5py.File(self.reader.hdf5_path, 'r', **self._h5_open_kwargs)

    def _load_proposal_cache(self, proposal_cache_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
        if not proposal_cache_path:
            return {}
        cache_path = proposal_cache_path
        if not os.path.isabs(cache_path):
            workspace_root = os.path.dirname(os.path.abspath(self.reader.hdf5_path))
            candidate_from_workspace = os.path.abspath(os.path.join(workspace_root, '..', cache_path))
            candidate_from_cwd = os.path.abspath(os.path.join(os.getcwd(), cache_path))
            if os.path.exists(candidate_from_workspace):
                cache_path = candidate_from_workspace
            else:
                cache_path = candidate_from_cwd
        if not os.path.exists(cache_path):
            logger.warning("proposal cache not found: %s", cache_path)
            return {}
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                logger.info("Loaded proposal cache: %s (entries=%d)", cache_path, len(payload))
                return payload
        except Exception as exc:
            logger.warning("Failed to load proposal cache %s: %s", cache_path, exc)
        return {}

    def _get_external_proposals(self, window_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes = torch.zeros((self.max_activities, 4), dtype=torch.float32)
        scores = torch.zeros((self.max_activities, 1), dtype=torch.float32)
        entry = getattr(self, 'proposal_cache', {}).get(window_id)
        if not entry:
            return boxes, scores

        raw_boxes = entry.get('proposal_boxes', entry.get('boxes', [])) or []
        raw_scores = entry.get('proposal_scores', entry.get('scores', [])) or []

        for idx, box in enumerate(raw_boxes[:self.max_activities]):
            try:
                arr = np.asarray(box, dtype=np.float32).reshape(-1)
                if arr.size < 4 or not np.isfinite(arr[:4]).all():
                    continue
                x1, y1, x2, y2 = [float(v) for v in arr[:4]]
                boxes[idx] = torch.tensor([
                    max(0.0, min(1.0, min(x1, x2))),
                    max(0.0, min(1.0, min(y1, y2))),
                    max(0.0, min(1.0, max(x1, x2))),
                    max(0.0, min(1.0, max(y1, y2))),
                ], dtype=torch.float32)
            except Exception:
                continue

        for idx, score in enumerate(raw_scores[:self.max_activities]):
            try:
                value = float(np.asarray(score).reshape(-1)[0])
                if np.isfinite(value):
                    scores[idx, 0] = max(0.0, min(1.0, value))
            except Exception:
                continue

        return boxes, scores

    def _precompute_windows(self) -> List[Dict]:
        """预计算所有滑动窗口"""
        windows = []
        skipped_events = []

        for event_id in self.event_ids:
            # 获取事件数据长度
            with self._open_hdf5() as f:
                event_path = f'events/{event_id}'
                if event_path not in f:
                    skipped_events.append((event_id, "事件不存在"))
                    continue

                event_group = f[event_path]
                num_frames = event_group.attrs.get('num_frames', 0)
                
                # 如果属性中没有num_frames（可能是旧版本文件），从时间戳数组长度获取
                if num_frames == 0 and 'timestamps' in event_group:
                    num_frames = len(event_group['timestamps'])
                    logger.debug(f"事件 {event_id} 从时间戳数组读取帧数: {num_frames}")

                if num_frames < self.sequence_length:
                    skipped_events.append((event_id, f"帧数({num_frames})小于序列长度({self.sequence_length})"))
                    continue

                # 计算滑动窗口
                window_count = 0
                for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                    end_idx = start_idx + self.sequence_length

                    window_info = {
                        'event_id': event_id,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'window_id': f"{event_id}_{start_idx}_{end_idx}"
                    }
                    windows.append(window_info)
                    window_count += 1

        # 如果没有任何窗口，记录警告信息
        if len(windows) == 0:
            logger.warning(f"无法创建任何滑动窗口！序列长度: {self.sequence_length}, 步长: {self.stride}")
            if skipped_events:
                logger.warning(f"跳过了 {len(skipped_events)} 个事件（前5个）:")
                for event_id, reason in skipped_events[:5]:
                    logger.warning(f"  - {event_id}: {reason}")
        elif skipped_events:
            logger.warning(f"跳过了 {len(skipped_events)} 个事件（帧数不足）")

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取样本"""
        window_info = self.windows[idx]
        event_id = window_info['event_id']
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']

        # 获取窗口数据
        event_data = self.reader.get_event_data(event_id, self.modalities)

        # 提取多模态数据
        data_dict = {}
        missing_modalities = []

        for modality in self.modalities:
            if event_data.get(modality) is not None:
                # 提取窗口数据
                modality_data = event_data[modality][start_idx:end_idx]

                # 处理缺失数据
                modality_data = self._process_modality_data(modality_data)

                # 转换为张量
                modality_tensor = torch.FloatTensor(modality_data)

                # 添加通道维度 (T, H, W) -> (T, 1, H, W)
                if modality_tensor.ndim == 3:
                    modality_tensor = modality_tensor.unsqueeze(1)

                data_dict[modality] = modality_tensor
            else:
                # 模态缺失，创建占位张量
                missing_modalities.append(modality)
                placeholder = self._create_placeholder(modality)
                data_dict[modality] = placeholder

        # 警告缺失模态
        if missing_modalities:
            logger.debug(f"窗口 {window_info['window_id']} 缺失模态: {missing_modalities}")

        # 获取标签（现在返回列表）
        labels = self._get_window_label(event_id, start_idx, end_idx)

        # 获取边界框（现在从HDF5的bboxes数据集中读取）
        bboxes = self._get_window_bbox(event_id, start_idx, end_idx)
        activity_region_ids = list(event_data.get('activity_region_ids', [])[:self.max_activities])
        region_bboxes, unique_region_ids = self._build_unique_region_targets(
            bboxes[:self.max_activities],
            activity_region_ids,
        )

        # 获取时间特征（每个活动槽位一份 [start_offset, peak_offset, duration]）
        time_features = self._get_time_features(event_id, start_idx, end_idx)

        # 获取辅助特征
        auxiliary = self._get_auxiliary_features(event_id, start_idx, end_idx)

        # 转换为固定长度张量（填充到最大活动数）
        max_activities = self.max_activities

        # 填充 labels：0=无事件(padding)，1=爆发，2=束缚
        padded_labels = labels + [0] * (max_activities - len(labels))
        padded_labels = padded_labels[:max_activities]  # 截断过长的

        # 填充bboxes
        padded_bboxes = bboxes + [[0, 0, 0, 0]] * (max_activities - len(bboxes))
        padded_bboxes = padded_bboxes[:max_activities]
        padded_region_bboxes = region_bboxes + [[0, 0, 0, 0]] * (max_activities - len(region_bboxes))
        padded_region_bboxes = padded_region_bboxes[:max_activities]

        # 创建mask表示哪些是真实活动
        activity_mask = [1] * len(labels) + [0] * (max_activities - len(labels))
        activity_mask = activity_mask[:max_activities]
        region_mask = [1] * len(region_bboxes) + [0] * (max_activities - len(region_bboxes))
        region_mask = region_mask[:max_activities]

        # 填充时间特征（每个槽位 3 个值）
        padded_time_features = time_features + [[0.0, 0.0, 1.0]] * (max_activities - len(time_features))
        padded_time_features = padded_time_features[:max_activities]

        proposal_boxes, proposal_scores = self._get_external_proposals(window_info['window_id'])

        sample = {
            'data': data_dict,
            'label': torch.tensor(padded_labels, dtype=torch.long),
            'bbox': torch.FloatTensor(padded_bboxes),
            'activity_mask': torch.tensor(activity_mask, dtype=torch.bool),
            'region_bbox': torch.FloatTensor(padded_region_bboxes),
            'region_mask': torch.tensor(region_mask, dtype=torch.bool),
            'time_features': torch.FloatTensor(padded_time_features),
            'auxiliary': _to_fixed_auxiliary(auxiliary),
            'proposal_boxes': proposal_boxes,
            'proposal_scores': proposal_scores,
            'metadata': {
                'window_id': window_info['window_id'],
                'event_id': event_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'missing_modalities': missing_modalities,
                'num_activities': len(labels),
                'num_regions': len(region_bboxes),
                'activity_region_ids': activity_region_ids,
                'unique_activity_region_ids': unique_region_ids[:max_activities],
                'activity_region_positions': list(event_data.get('activity_region_positions', [])[:max_activities]),
                'has_external_proposals': window_info['window_id'] in getattr(self, 'proposal_cache', {}),
            }
        }

        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _process_modality_data(self, data: np.ndarray) -> np.ndarray:
        """处理模态数据"""
        # 1. 处理缺失值
        if np.any(np.isnan(data)):
            if self.handle_missing == 'interpolate' and data.shape[0] > 1:
                data = self._interpolate_missing(data)
            else:
                data[np.isnan(data)] = self.fill_value

        # 2. 调整大小
        if data.shape[-2:] != self.target_size:
            data = self._resize_data(data, self.target_size)

        return data

    def _interpolate_missing(self, data: np.ndarray) -> np.ndarray:
        """插值填充缺失数据"""
        interpolated = data.copy()

        # 对每个空间位置进行时间维度插值
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                column = data[:, i, j]
                mask = ~np.isnan(column)

                if np.any(mask) and np.any(~mask):
                    indices = np.where(mask)[0]
                    values = column[mask]

                    # 线性插值
                    column_interp = np.interp(
                        np.arange(len(column)),
                        indices,
                        values
                    )
                    interpolated[:, i, j] = column_interp

        return interpolated

    def _resize_data(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整数据大小"""
        from scipy.ndimage import zoom

        T = data.shape[0]
        H, W = target_size

        # 计算缩放因子
        zoom_factors = [1]  # 时间维度不缩放
        zoom_factors.append(H / data.shape[1])
        zoom_factors.append(W / data.shape[2])

        # 应用缩放
        resized_data = zoom(data, zoom_factors, order=1)  # 双线性插值

        return resized_data

    def _create_placeholder(self, modality: str) -> torch.Tensor:
        """创建占位张量"""
        # 根据模态配置确定形状
        placeholder_shape = (self.sequence_length, 1, *self.target_size)
        return torch.full(placeholder_shape, self.fill_value, dtype=torch.float32)

    def _get_window_label(self, event_id: str, start_idx: int, end_idx: int) -> List[int]:
        """按 activity 槽位返回窗口标签。"""
        event_data = self.reader.get_event_data(event_id, self.modalities)
        window_timestamps = event_data['timestamps'][start_idx:end_idx]
        labels: List[int] = []

        if event_data.get('activity_labels'):
            labels = [int(x) for x in event_data['activity_labels']]
        elif 'bbox_labels' in event_data and event_data.get('bbox_labels'):
            labels = [int(x) for x in event_data['bbox_labels']]
        else:
            event_meta = self.reader.get_event_metadata(event_id)
            if 'activities' in event_meta:
                for activity in event_meta['activities']:
                    labels.append(int(self.reader._get_window_label(activity, window_timestamps)))
            else:
                labels.append(int(self.reader._get_window_label(event_meta, window_timestamps)))

        return labels

    def get_window_labels(self, window_info: Dict[str, Any]) -> List[int]:
        """公共接口：根据窗口信息返回该窗口的 activity 标签列表。"""
        return self._get_window_label(
            window_info['event_id'],
            window_info['start_idx'],
            window_info['end_idx']
        )

    def _normalize_bbox(self, box: Any, bbox_resolution: Optional[Dict[str, Any]] = None) -> List[float]:
        """将像素坐标 bbox 按其标注分辨率归一化到 [0, 1]。"""
        try:
            coords = [float(coord) for coord in box]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
        if len(coords) < 4 or not np.isfinite(coords[:4]).all():
            return [0.0, 0.0, 0.0, 0.0]

        x1, y1, x2, y2 = coords[:4]
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)

        bbox_resolution = bbox_resolution or {}
        bbox_width = float(bbox_resolution.get('width', 0) or 0)
        bbox_height = float(bbox_resolution.get('height', 0) or 0)

        if not np.isfinite(bbox_width) or bbox_width <= 0:
            bbox_width = float(self.target_size[1])
        if not np.isfinite(bbox_height) or bbox_height <= 0:
            bbox_height = float(self.target_size[0])

        normalized = np.asarray([
            x_min / bbox_width,
            y_min / bbox_height,
            x_max / bbox_width,
            y_max / bbox_height,
        ], dtype=np.float32)
        if not np.isfinite(normalized).all():
            return [0.0, 0.0, 0.0, 0.0]
        normalized = np.clip(normalized, 0.0, 1.0)
        return [float(v) for v in normalized.tolist()]

    def _get_window_bbox(self, event_id: str, start_idx: int, end_idx: int) -> List[List[float]]:
        """按 activity 所属 region 返回 bbox；同一 region 可重复出现在多个 activity 槽位。"""
        event_data = self.reader.get_event_data(event_id, self.modalities)
        bboxes: List[List[float]] = []

        if event_data.get('regions') is not None and event_data.get('activity_region_ids') is not None:
            region_boxes = event_data.get('regions', [])
            region_ids = event_data.get('region_ids', [])
            activity_region_ids = event_data.get('activity_region_ids', [])
            bbox_resolution = event_data.get('bbox_resolution', {})
            region_box_map = {}
            for rid, box in zip(region_ids, region_boxes):
                region_box_map[str(rid)] = self._normalize_bbox(box, bbox_resolution)
            for rid in activity_region_ids:
                bboxes.append(region_box_map.get(str(rid), [0.0, 0.0, 0.0, 0.0]))
            return bboxes

        if 'bboxes' in event_data:
            raw_boxes = event_data['bboxes']
            bbox_resolution = event_data.get('bbox_resolution', {})
            for box in raw_boxes:
                bboxes.append(self._normalize_bbox(box, bbox_resolution))
            return bboxes

        event_meta = self.reader.get_event_metadata(event_id)
        if 'activities' in event_meta:
            for activity in event_meta['activities']:
                x = activity.get('position_x', 0)
                y = activity.get('position_y', 0)
                r = activity.get('position_r', 0)
                if x == 0 and y == 0 and r == 0:
                    continue
                scale = max(self.target_size[0], self.target_size[1])
                bboxes.append([(x - r) / scale, (y - r) / scale, (x + r) / scale, (y + r) / scale])
        return bboxes

    def _build_unique_region_targets(
        self,
        slot_bboxes: List[List[float]],
        activity_region_ids: List[Any],
    ) -> Tuple[List[List[float]], List[str]]:
        """按唯一活动区构建定位监督；若缺少 region id，则退化为按 bbox 去重。"""
        unique_boxes: List[List[float]] = []
        unique_region_ids: List[str] = []
        seen_region_ids = set()
        seen_box_keys = set()

        for idx, box in enumerate(slot_bboxes):
            region_id = ''
            if idx < len(activity_region_ids):
                raw_region_id = activity_region_ids[idx]
                region_id = '' if raw_region_id is None else str(raw_region_id).strip()

            if region_id:
                if region_id in seen_region_ids:
                    continue
                seen_region_ids.add(region_id)
                unique_region_ids.append(region_id)
                unique_boxes.append([float(v) for v in box])
                continue

            try:
                box_key = tuple(np.round(np.asarray(box, dtype=np.float64), 6).tolist())
            except Exception:
                box_key = ()
            if box_key in seen_box_keys:
                continue
            seen_box_keys.add(box_key)
            unique_region_ids.append('')
            unique_boxes.append([float(v) for v in box])

        return unique_boxes, unique_region_ids

    def _get_time_features(self, event_id: str, start_idx: int, end_idx: int) -> List[List[float]]:
        """按 activity 槽位返回时间目标。"""
        event_data = self.reader.get_event_data(event_id, self.modalities)
        event_meta = self.reader.get_event_metadata(event_id)
        timestamps = event_data['timestamps'][start_idx:end_idx]
        if timestamps is None or len(timestamps) == 0:
            return [[0.0, 0.0, 1.0]]

        ref_dt = datetime.fromisoformat(str(timestamps[-1]).replace('Z', ''))

        def _build_time_target(meta: Dict[str, Any]) -> List[float]:
            try:
                event_start = datetime.fromisoformat(str(meta['start_time']).replace('Z', ''))
                peak_time = datetime.fromisoformat(str(meta['peak_time']).replace('Z', ''))
                event_end = datetime.fromisoformat(str(meta['end_time']).replace('Z', ''))
            except Exception:
                return [0.0, 0.0, 0.0]
            start_offset_hours = (event_start - ref_dt).total_seconds() / 3600.0
            peak_offset_hours = (peak_time - ref_dt).total_seconds() / 3600.0
            end_offset_hours = (event_end - ref_dt).total_seconds() / 3600.0
            # 格式改为显式 start / peak / end（单位：天）
            return [
                float(np.clip(start_offset_hours / 24.0, -2.0, 2.0)),
                float(np.clip(peak_offset_hours / 24.0, -2.0, 2.0)),
                float(np.clip(end_offset_hours / 24.0, -2.0, 2.0)),
            ]

        time_targets: List[List[float]] = []
        if event_data.get('activity_event_ids'):
            for other_id in event_data['activity_event_ids']:
                try:
                    other_meta = self.reader.get_event_metadata(other_id)
                except Exception:
                    other_meta = event_meta
                time_targets.append(_build_time_target(other_meta))
        elif 'bbox_event_ids' in event_data and event_data.get('bbox_event_ids'):
            for other_id in event_data['bbox_event_ids']:
                try:
                    other_meta = self.reader.get_event_metadata(other_id)
                except Exception:
                    other_meta = event_meta
                time_targets.append(_build_time_target(other_meta))
        elif 'activities' in event_meta and event_meta['activities']:
            for activity in event_meta['activities']:
                time_targets.append(_build_time_target(activity))
        else:
            time_targets.append(_build_time_target(event_meta))

        return time_targets

    def _get_auxiliary_features(self, event_id: str, start_idx: int, end_idx: int) -> Optional[np.ndarray]:
        """获取辅助特征"""
        try:
            event_data = self.reader.get_event_data(event_id, self.modalities)
            if 'auxiliary' in event_data:
                auxiliary_data = event_data['auxiliary']

                # 提取窗口数据
                auxiliary_window = {}
                for key, data in auxiliary_data.items():
                    if len(data) >= end_idx:
                        auxiliary_window[key] = data[start_idx:end_idx]

                # 简化为特征向量
                features = []
                for key, data in auxiliary_window.items():
                    # 计算统计特征
                    if len(data.shape) == 1:  # 1D数据
                        features.extend([
                            np.mean(data), np.std(data), np.min(data), np.max(data)
                        ])

                return np.array(features) if features else None
        except:
            return None
