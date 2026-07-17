"""
推理预测器
"""
import h5py
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
from pathlib import Path

from models.multimodal_transformer import MultimodalTransformer
from data.hdf5_reader import HDF5DatasetReader
from inference.post_processing import PostProcessor
from inference.uncertainty_estimation import UncertaintyEstimator

logger = logging.getLogger(__name__)


def _to_numpy_cpu(x: Any) -> np.ndarray:
    """Tensor（含 CUDA）/ 标量 / 序列 → CPU numpy，避免 np.asarray(cuda_tensor) 报错。"""
    if x is None:
        return np.asarray([])
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        converted = []
        for item in x:
            if torch.is_tensor(item):
                converted.append(item.detach().cpu().numpy())
            else:
                converted.append(item)
        try:
            return np.asarray(converted)
        except Exception:
            return np.asarray(converted, dtype=object)
    return np.asarray(x)


def _to_serializable(x: Any) -> Any:
    """将 tensor / numpy / 嵌套结构转换为可直接写入 JSON 的 Python 对象。"""
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    return x


def _safe_box_to_list(boxes: Any, idx: int) -> List[float]:
    """安全读取单个 bbox，兼容 None / object-array / 空值。"""
    try:
        if boxes is None or idx < 0 or idx >= len(boxes):
            return []
        box = boxes[idx]
        if box is None:
            return []
        if torch.is_tensor(box):
            return box.detach().cpu().reshape(-1).tolist()
        arr = np.asarray(box, dtype=np.float32).reshape(-1)
        if arr.size < 4 or not np.isfinite(arr[:4]).all():
            return []
        return arr[:4].tolist()
    except Exception:
        return []


class SolarFlarePredictor:
    """太阳耀斑预测器"""

    def _build_detections(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """构造最终保留 bbox 的检测列表，保证 bbox/class/time/prob 一一对应。"""
        slot_classes = _to_numpy_cpu(predictions.get('final_classes', predictions.get('predicted_class', []))).reshape(-1)
        slot_confidences = _to_numpy_cpu(predictions.get('classification_confidences', [])).reshape(-1)
        slot_event_probs = _to_numpy_cpu(predictions.get('processed_event_probs', predictions.get('event_prob_mean', []))).reshape(-1)
        slot_times = None
        if self.time_head_enabled:
            slot_times = _to_numpy_cpu(predictions.get('processed_time_pred', predictions.get('time_pred_mean', [])))
            if slot_times.ndim == 1:
                slot_times = slot_times.reshape(1, -1)
        kept_slot_indices = _to_numpy_cpu(predictions.get('kept_bbox_slot_indices', [])).astype(np.int64).reshape(-1)
        processed_bboxes = _to_numpy_cpu(predictions.get('processed_bboxes', []))
        proposal_boxes = _to_numpy_cpu(predictions.get('slot_proposal_boxes', []))
        context_boxes = _to_numpy_cpu(predictions.get('slot_context_boxes', []))
        proposal_scores = _to_numpy_cpu(predictions.get('slot_proposal_scores', []))
        bboxes = processed_bboxes.reshape(-1, 4) if processed_bboxes.size > 0 else np.empty((0, 4), dtype=np.float32)
        bbox_confidences = _to_numpy_cpu(predictions.get('bbox_confidences', [])).reshape(-1)

        detections: List[Dict[str, Any]] = []
        linked_events = predictions.get('linked_events', []) or []
        for det_idx, slot_idx in enumerate(kept_slot_indices.tolist()):
            class_id = int(slot_classes[det_idx]) if det_idx < len(slot_classes) else -1
            # class_id == 0 表示"无事件"，不应出现在检测结果中
            if class_id == 0:
                continue
            det: Dict[str, Any] = {
                'detection_idx': int(det_idx),
                'slot_id': int(slot_idx),
                'class_id': class_id,
                'class_confidence': float(slot_confidences[det_idx]) if det_idx < len(slot_confidences) else 0.0,
                'event_probability': float(slot_event_probs[det_idx]) if det_idx < len(slot_event_probs) else 0.0,
                'proposal_score': float(proposal_scores[det_idx]) if det_idx < len(proposal_scores) else 0.0,
                'bbox_confidence': float(bbox_confidences[det_idx]) if det_idx < len(bbox_confidences) else 0.0,
                'bbox': bboxes[det_idx].tolist() if det_idx < len(bboxes) else [],
                'proposal_bbox': _safe_box_to_list(proposal_boxes, det_idx),
                'context_bbox': _safe_box_to_list(context_boxes, det_idx),
                'linked_events': linked_events[det_idx] if det_idx < len(linked_events) else [],
            }
            if self.time_head_enabled and slot_times is not None and det_idx < len(slot_times):
                slot_time = np.asarray(slot_times[det_idx]).reshape(-1)
                if slot_time.size >= 3 and np.isfinite(slot_time[:3]).all() and not np.allclose(slot_time[:3], 0.0):
                    det['time_prediction'] = {
                        'start_offset_hours': float(slot_time[0]),
                        'peak_offset_hours': float(slot_time[1]),
                        'end_offset_hours': float(slot_time[2]),
                    }
            detections.append(det)
        return detections

    def __init__(self, model_path: str, config_path: str,
                 device: Optional[str] = None):
        """
        初始化预测器

        Args:
            model_path: 模型路径
            config_path: 配置文件路径
            device: 设备
        """
        # 加载配置
        try:
            from utils.config_utils import load_config
        except ImportError:
            # 如果相对导入失败，尝试绝对导入
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from utils.config_utils import load_config
        self.config = load_config(config_path)

        # 设置设备
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 初始化后处理器
        self.post_processor = PostProcessor(self.config.get('inference', {}))

        # 初始化不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator(
            self.model, self.config.get('uncertainty', {})
        )

        logger.info(f"预测器初始化完成，设备: {self.device}")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 兼容直接保存 state_dict 或完整 checkpoint 两种情况
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

        # 创建模型配置；若推理配置中未完整给出 modalities，则尝试从 data_config.yaml 补齐
        model_config = dict(self.config['model'])
        if 'modalities' not in model_config or not model_config['modalities']:
            try:
                from utils.config_utils import load_config
                data_cfg = load_config(str(Path(__file__).parent.parent / 'configs' / 'data_config.yaml'))
                data_section = data_cfg.get('data', {})
                enabled_modalities = {
                    name: cfg
                    for name, cfg in data_section.get('modalities', {}).items()
                    if bool(cfg.get('enabled', True))
                }
                model_config['modalities'] = enabled_modalities
                model_config.setdefault('max_activities', data_section.get('max_activities', 5))
                model_config.setdefault('sequence_length', data_section.get('sequence_length', 9))
            except Exception as e:
                raise ValueError(f"模型配置缺少 modalities，且无法从 data_config.yaml 补全: {e}")

        head_cfg = dict(model_config.get('prediction_heads', {}))
        time_cfg = dict(head_cfg.get('time', {}))
        has_time_head_weights = any(
            key.startswith('stage_two_predictor.time_sequence_attn.') or key.startswith('stage_two_predictor.time_predictor.')
            for key in state_dict.keys()
        )
        self.time_head_enabled = has_time_head_weights
        time_cfg['enabled'] = has_time_head_weights
        head_cfg['time'] = time_cfg
        model_config['prediction_heads'] = head_cfg
        logger.info("根据 checkpoint 自动设置时间头 enabled=%s", has_time_head_weights)

        if model_config.get('type') == 'multimodal_transformer':
            model = MultimodalTransformer(model_config)
        else:
            raise ValueError(f"未知的模型类型: {model_config.get('type')}")

        model.load_state_dict(state_dict)
        model = model.to(self.device)

        return model

    def predict_batch(self, inputs: Dict[str, torch.Tensor],
                      use_uncertainty: bool = False,
                      num_mc_samples: int = 10) -> Dict[str, Any]:
        """
        批量预测

        Args:
            inputs: 输入数据字典
            use_uncertainty: 是否使用不确定性估计
            num_mc_samples: MC采样次数

        Returns:
            预测结果字典
        """
        # 将数据移到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if use_uncertainty:
                # 带不确定性的预测
                predictions = self.model.predict_with_uncertainty(
                    inputs, num_samples=num_mc_samples
                )
            else:
                # 标准预测
                outputs = self.model(inputs)

                predictions = {
                    'class_probs_mean': outputs['class_probs'],
                    'bbox_pred_mean': outputs.get('bbox_pred'),
                    'event_prob_mean': outputs['event_prob'],
                    'event_gate_mean': outputs.get('event_gate'),
                    'proposal_boxes': outputs.get('proposal_boxes', outputs['bbox_pred']),
                    'context_boxes': outputs.get('context_boxes'),
                    'proposal_scores': outputs.get('proposal_scores', outputs['event_prob']),
                    'proposal_features': outputs.get('proposal_features'),
                    'global_feature': outputs.get('global_feature'),
                    'bbox_size_gated': outputs.get('bbox_size_gated'),
                    'refine_delta': outputs.get('refine_delta'),
                    'predicted_class': outputs['class_probs'].argmax(dim=-1)
                }
                if self.time_head_enabled:
                    predictions['time_pred_mean'] = outputs.get('time_pred')

        # 后处理
        predictions = self.post_processor.process(predictions)

        return predictions

    def predict_single(self, data_dict: Dict[str, np.ndarray],
                       metadata: Optional[Dict] = None,
                       use_uncertainty: bool = False) -> Dict[str, Any]:
        """
        单样本预测

        Args:
            data_dict: 数据字典
            metadata: 元数据
            use_uncertainty: 是否使用不确定性估计

        Returns:
            预测结果
        """
        # 转换为张量
        inputs = {}
        for modality, data in data_dict.items():
            if data is not None:
                # 添加批次维度
                tensor_data = torch.FloatTensor(data).unsqueeze(0)
                inputs[modality] = tensor_data

        # 预测
        predictions = self.predict_batch(inputs, use_uncertainty)

        # 添加元数据
        if metadata:
            predictions['metadata'] = metadata

        return predictions

    def predict_from_hdf5(self, hdf5_path: str, event_id: str,
                          start_idx: int, end_idx: int,
                          modalities: Optional[List[str]] = None,
                          use_uncertainty: bool = False) -> Dict[str, Any]:
        """
        从HDF5文件预测

        Args:
            hdf5_path: HDF5文件路径
            event_id: 事件ID
            start_idx: 开始索引
            end_idx: 结束索引
            modalities: 模态列表
            use_uncertainty: 是否使用不确定性估计

        Returns:
            预测结果
        """
        # 读取数据
        reader = HDF5DatasetReader(hdf5_path)
        if modalities is None:
            modalities_cfg = self.config.get('model', {}).get('modalities', {})
            if modalities_cfg:
                modalities = list(modalities_cfg.keys())
        event_data = reader.get_event_data(event_id, modalities)

        # 提取窗口数据
        data_dict = {}
        active_modalities = modalities or reader.modalities
        for modality in active_modalities:
            if event_data.get(modality) is not None:
                data_dict[modality] = event_data[modality][start_idx:end_idx]
            else:
                data_dict[modality] = None

        # 获取元数据
        metadata = reader.get_event_metadata(event_id)
        metadata.update({
            'window_start_idx': start_idx,
            'window_end_idx': end_idx,
            'window_timestamps': event_data['timestamps'][start_idx:end_idx]
        })

        # 预测
        predictions = self.predict_single(data_dict, metadata, use_uncertainty)

        return predictions

    def predict_time_series(self, hdf5_path: str, event_id: str,
                            window_size: Optional[int] = None,
                            stride: Optional[int] = None,
                            modalities: Optional[List[str]] = None,
                            use_uncertainty: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        预测时间序列

        Args:
            hdf5_path: HDF5文件路径
            event_id: 事件ID
            window_size: 窗口大小；默认为配置中的 sequence_length / windowing.window_size
            stride: 滑动步长；默认为配置中的 windowing.stride
            modalities: 模态列表
            use_uncertainty: 是否使用不确定性估计；默认读取配置

        Returns:
            预测结果列表
        """
        reader = HDF5DatasetReader(hdf5_path)

        inference_cfg = self.config.get('inference', {})
        window_cfg = inference_cfg.get('windowing', {})
        if window_size is None:
            window_size = int(window_cfg.get('window_size', self.config.get('model', {}).get('sequence_length', 9)))
        if stride is None:
            stride = int(window_cfg.get('stride', 6))
        if use_uncertainty is None:
            use_uncertainty = bool(window_cfg.get('use_uncertainty', True))

        # 获取事件有效帧数：时间戳长度与各模态 images 第一维的较小值，避免属性 num_frames 与真实数据不一致
        with h5py.File(hdf5_path, 'r') as f:
            event_path = f'events/{event_id}'
            eg = f[event_path]
            attr_frames = int(eg.attrs.get('num_frames', 0) or 0)
            lens = []
            if 'timestamps' in eg:
                lens.append(len(eg['timestamps']))
            want_modalities = modalities if modalities is not None else reader.modalities
            for mod in want_modalities:
                img_path = f'{event_path}/data/{mod}/images'
                if img_path in f:
                    lens.append(int(f[img_path].shape[0]))
            num_frames = min(lens) if lens else attr_frames
            if not num_frames and attr_frames:
                num_frames = attr_frames
            if lens and attr_frames and num_frames < attr_frames:
                logger.warning(
                    f"事件 {event_id}: HDF5 属性 num_frames={attr_frames} 与各模态实际长度 min={num_frames} 不一致，"
                    f"滑动窗口将按实际长度 {num_frames} 计算"
                )

        if num_frames < window_size:
            logger.warning(
                f"事件 {event_id}: 有效帧数 {num_frames} 小于窗口大小 {window_size}，无法进行滑动预测"
            )
            return []

        # 滑动窗口预测
        predictions = []

        for start_idx in range(0, num_frames - window_size + 1, stride):
            end_idx = start_idx + window_size

            logger.info(f"预测窗口 {start_idx}-{end_idx}")

            try:
                pred_result = self.predict_from_hdf5(
                    hdf5_path, event_id, start_idx, end_idx,
                    modalities, use_uncertainty=use_uncertainty
                )

                # 添加窗口信息
                pred_result['window_info'] = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'window_size': window_size
                }

                predictions.append(pred_result)

            except Exception as e:
                logger.error(f"窗口 {start_idx}-{end_idx} 预测失败: {e}")

        return predictions

    def generate_alert(self, predictions: Dict[str, Any],
                       threshold_class1: float = 0.7,
                       threshold_class2: float = 0.6) -> Dict[str, Any]:
        """
        生成警报

        Args:
            predictions: 预测结果
            threshold_class1: 类别1阈值
            threshold_class2: 类别2阈值

        Returns:
            警报信息
        """
        class_probs = _to_numpy_cpu(predictions['class_probs_mean']).flatten()

        ep = predictions['event_prob_mean']
        event_prob = float(_to_numpy_cpu(ep).reshape(-1).max())

        pred_classes_arr = _to_numpy_cpu(predictions['predicted_class']).reshape(-1)
        pred_class = int(pred_classes_arr[0]) if pred_classes_arr.size > 0 else 0
        class_probs_arr = _to_numpy_cpu(predictions['class_probs_mean'])
        if class_probs_arr.ndim >= 2:
            alert_class_probs = class_probs_arr[0].reshape(-1)
        else:
            alert_class_probs = class_probs_arr.reshape(-1)

        alert = {
            'timestamp': datetime.now().isoformat(),
            'event_probability': event_prob,
            'class_probabilities': alert_class_probs.tolist(),
            'predicted_class': pred_class,
            'alert_level': 'none',
            'message': '',
            'recommendations': []
        }

        # 判断警报级别
        if len(alert_class_probs) > 1 and alert_class_probs[1] >= threshold_class1:  # 爆发耀斑
            alert['alert_level'] = 'high'
            alert['message'] = '高风险：检测到爆发耀斑（可能伴随CME）'
            alert['recommendations'] = [
                '启动CME监测程序',
                '通知空间天气预报中心',
                '准备卫星保护措施'
            ]
        elif len(alert_class_probs) > 2 and alert_class_probs[2] >= threshold_class2:  # 束缚耀斑
            alert['alert_level'] = 'medium'
            alert['message'] = '中风险：检测到束缚耀斑'
            alert['recommendations'] = [
                '持续监测耀斑发展',
                '检查无线电通信影响'
            ]
        elif event_prob >= 0.3:  # 可能发生耀斑
            alert['alert_level'] = 'low'
            alert['message'] = '低风险：24小时内可能发生耀斑'
            alert['recommendations'] = [
                '加强监测频率',
                '准备应急预案'
            ]

        # 添加位置和时间预测
        if 'bbox_pred_mean' in predictions:
            alert['predicted_location'] = _to_numpy_cpu(
                predictions['bbox_pred_mean']
            ).tolist()

        # 时间预测：优先使用 post-processed 的结果（槽位级）
        if 'processed_time_pred' in predictions:
            tp = predictions['processed_time_pred']
        elif 'time_pred_mean' in predictions:
            tp = predictions['time_pred_mean']
        else:
            tp = None

        if tp is not None:
            time_pred = _to_numpy_cpu(tp)
            slot_time_predictions = []
            for idx, slot_time in enumerate(np.asarray(time_pred)):
                slot_time = np.asarray(slot_time).reshape(-1)
                if slot_time.size < 3:
                    continue
                slot_time_predictions.append({
                    'slot_id': idx,
                    'start_offset_hours': float(slot_time[0]),
                    'peak_offset_hours': float(slot_time[1]),
                    'end_offset_hours': float(slot_time[2])
                })
            alert['time_predictions'] = slot_time_predictions

        if 'processed_event_probs' in predictions:
            slot_event_probs = _to_numpy_cpu(predictions['processed_event_probs']).reshape(-1)
            alert['slot_event_probabilities'] = [
                {'slot_id': idx, 'event_probability': float(prob)}
                for idx, prob in enumerate(slot_event_probs)
            ]
            if slot_event_probs.size > 0:
                alert['event_probability'] = float(np.max(slot_event_probs))

        # 添加不确定性信息
        if 'class_probs_std' in predictions:
            std_arr = _to_numpy_cpu(predictions['class_probs_std'])
            uncertainty = std_arr.tolist()
            alert['uncertainty'] = uncertainty
            alert['confidence'] = 1.0 - float(np.mean(std_arr))

        return alert

    def export_predictions(self, predictions: List[Dict[str, Any]],
                           output_path: str, format: str = 'json') -> None:
        """
        导出预测结果

        Args:
            predictions: 预测结果列表
            output_path: 输出路径
            format: 输出格式（json, csv, parquet）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            export_payload = []
            for pred in predictions:
                pred_copy = _to_serializable(pred)
                pred_copy['detections'] = self._build_detections(pred)
                export_payload.append(pred_copy)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_payload, f, indent=2, ensure_ascii=False)

        elif format == 'csv':
            import pandas as pd

            # 转换为DataFrame
            rows = []
            for pred in predictions:
                cp = _to_numpy_cpu(pred['class_probs_mean']).flatten()
                pc = int(_to_numpy_cpu(pred.get('predicted_class', -1)).reshape(-1)[0])
                ev = float(_to_numpy_cpu(pred['event_prob_mean']).reshape(-1)[0])
                row = {
                    'event_id': pred.get('metadata', {}).get('event_id', 'unknown'),
                    'window_start': pred.get('metadata', {}).get('window_start_idx', -1),
                    'window_end': pred.get('metadata', {}).get('window_end_idx', -1),
                    'predicted_class': pc,
                    'class0_prob': float(cp[0]) if len(cp) > 0 else float('nan'),
                    'class1_prob': float(cp[1]) if len(cp) > 1 else float('nan'),
                    'class2_prob': float(cp[2]) if len(cp) > 2 else float('nan'),
                    'event_prob': ev,
                    'timestamp': datetime.now().isoformat()
                }

                # 添加不确定性
                if 'class_probs_std' in pred:
                    cs = _to_numpy_cpu(pred['class_probs_std']).flatten()
                    row.update({
                        'class0_std': float(cs[0]) if len(cs) > 0 else float('nan'),
                        'class1_std': float(cs[1]) if len(cs) > 1 else float('nan'),
                        'class2_std': float(cs[2]) if len(cs) > 2 else float('nan')
                    })

                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

        logger.info(f"预测结果已导出到: {output_path}")
