"""
推理管道脚本

Bbox 坐标约定（与训练 / HDF5 / 可视化一致）:
- 归一化 [x1,y1,x2,y2] ∈ [0,1]²；(0,0) 为 **图像左上角**，x 向右、y 向下（非日心坐标）。
"""
import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 可选导入 matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualizations will be disabled")

import numpy as np
import pandas as pd
import torch

from inference.visualization import PredictionVisualizer
from inference import SolarFlarePredictor
from utils.config_utils import load_config
from utils.metrics_calculation import calculate_metrics
from data.hdf5_reader import HDF5DatasetReader
from data.dataset import SolarFlareDataset

logger = logging.getLogger(__name__)


# =========================
# 直接运行配置（推荐在 VS Code 中直接点“运行 Python 文件”时使用）
# 需要测试时，直接改下面这些值即可，无需命令行。
# - `event_id` 留空时，会自动选择 HDF5 中第一个可用事件。
# - `model_path` 留空时，会自动尝试查找 logs/checkpoints 下的最佳 .pth。
# =========================
USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    'run_mode': 'test_set',  # 可选: 'single_event' | 'test_set' | 'batch_dir'
    'model_path': 'outputs/checkpoints_fitscache_mag_halpha_euv94_euv171_euv304_256/solar_flare_cme_model_epoch_0088.pth',
    'data_path': 'data/Solar_Flares_CME_dataset_fitscache.h5',
    'event_id': '',    # single_event 模式留空则自动选择第一个可用事件
    'output_dir': 'outputs/predictions/fitscache_mag_halpha_euv94_euv171_euv304_256_direct_run',
    'config_path': 'configs/inference_config_fitscache_mag_halpha_euv94_euv171_euv304_256.yaml',
    'data_config_path': 'configs/data_config_fitscache_mag_halpha_euv94_euv171_euv304_256.yaml',  # 权威来源：resolution/stride/modalities
    'window_size': 10,
    'stride': 10,   # 与整事件阶段2训练保持一致：每个事件只产生一个窗口
    'split_name': 'test',
    'split_file': '',  # 留空则默认使用 data/split_data/<split_name>_events.txt
    'max_events': 0,   # 0 表示不限制；调试时可设为 5
    'disable_uncertainty': False,
    'disable_visualization': False,
    'viz_frame': 'last',  # last | first | middle — 底图使用窗口内哪一帧
}


def _find_default_checkpoint() -> str:
    """自动查找可用的 checkpoint 文件。"""
    checkpoint_dir = project_root / 'logs' / 'checkpoints'
    if not checkpoint_dir.exists():
        return ''

    best_candidates = sorted(checkpoint_dir.glob('*best*.pth'))
    if best_candidates:
        return str(best_candidates[-1])

    all_candidates = sorted(checkpoint_dir.glob('*.pth'))
    if all_candidates:
        return str(all_candidates[-1])

    return ''


def _build_direct_run_args() -> List[str]:
    """把文件内配置转换为命令行参数列表，便于复用 main() 逻辑。"""
    cfg = dict(DIRECT_RUN_CONFIG)
    if not cfg.get('model_path'):
        cfg['model_path'] = _find_default_checkpoint()

    args = []
    for key in [
        'run_mode', 'model_path', 'data_path', 'event_id', 'output_dir',
        'config_path', 'data_config_path', 'window_size', 'stride', 'split_name', 'split_file', 'max_events',
        'viz_frame',
    ]:
        value = cfg.get(key)
        if value not in (None, ''):
            args.extend([f'--{key}', str(value)])

    if cfg.get('disable_uncertainty', False):
        args.append('--disable_uncertainty')
    if cfg.get('disable_visualization', False):
        args.append('--disable_visualization')

    return args


def _to_serializable(value: Any) -> Any:
    """将 tensor / ndarray 等对象转换为可写入 JSON/CSV 的形式。"""
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, dict, str, int, float, bool)):
        return value
    return str(value)


def _flatten_numeric_sequence(value: Any) -> List[Any]:
    """把嵌套的 tensor/ndarray/list 展开成一维数值列表。"""
    if value is None:
        return []
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return value.reshape(-1).tolist()
    if isinstance(value, list):
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return []
            return arr.reshape(-1).tolist()
        except Exception:
            flattened = []
            for item in value:
                if isinstance(item, list):
                    flattened.extend(_flatten_numeric_sequence(item))
                else:
                    flattened.append(item)
            return flattened
    return [value]


def _to_numeric_array(value: Any) -> np.ndarray:
    """把任意预测输出转成 numpy 数组，便于兼容槽位级多任务结果。"""
    if value is None:
        return np.asarray([])
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        arr = np.asarray(_flatten_numeric_sequence(value), dtype=np.float32)
    return arr


def _safe_first_numeric(value: Any, default: float = np.nan) -> float:
    """取标量/向量/矩阵中的第一个数值；空时返回 default。"""
    arr = _to_numeric_array(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _safe_to_csv(df: pd.DataFrame, path: Path, *, index: bool = False, encoding: str = 'utf-8-sig') -> Path:
    """稳健写 CSV；若目标文件被占用，则回退到带时间戳的新文件名。"""
    path = Path(path)
    try:
        df.to_csv(path, index=index, encoding=encoding)
        return path
    except PermissionError:
        from datetime import datetime
        fallback = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        logger.warning(f"文件被占用，改为写入: {fallback}")
        df.to_csv(fallback, index=index, encoding=encoding)
        return fallback



def _prediction_to_slot_rows(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把单个窗口预测结果展开为 activity 槽位级多行表格（仅保留 class_id 为 1/2 的检测）。"""
    metadata = pred.get('metadata', {}) or {}
    window_info = pred.get('window_info', {}) or {}
    detections = pred.get('detections', []) or []

    rows: List[Dict[str, Any]] = []
    for det_idx, det in enumerate(detections):
        if int(det.get('class_id', -1)) == 0:
            continue
        slot_idx = int(det.get('slot_id', det_idx))
        bbox = det.get('bbox', []) or []
        proposal_bbox = det.get('proposal_bbox', []) or []
        context_bbox = det.get('context_bbox', []) or []
        linked_events = det.get('linked_events', []) or []
        main_time = det.get('time_prediction', {}) or {}
        row = {
            'event_id': metadata.get('event_id', 'unknown'),
            'window_start_idx': metadata.get('window_start_idx', window_info.get('start_idx', -1)),
            'window_end_idx': metadata.get('window_end_idx', window_info.get('end_idx', -1)),
            'window_timestamps': json.dumps(metadata.get('window_timestamps', []), ensure_ascii=False),
            'detection_idx': det_idx,
            'slot_id': slot_idx,
            'class_id': int(det.get('class_id', -1)),
            'class_confidence': float(det.get('class_confidence', np.nan)),
            'event_probability': float(det.get('event_probability', np.nan)),
            'proposal_score': float(det.get('proposal_score', np.nan)),
            'bbox_confidence': float(det.get('bbox_confidence', np.nan)),
            'bbox': json.dumps(bbox, ensure_ascii=False),
            'proposal_bbox': json.dumps(proposal_bbox, ensure_ascii=False),
            'context_bbox': json.dumps(context_bbox, ensure_ascii=False),
            'linked_events': json.dumps(linked_events, ensure_ascii=False),
        }
        if 'time_prediction' in det:
            row['time_start_offset'] = float(main_time.get('start_offset_hours', np.nan))
            row['time_peak_offset'] = float(main_time.get('peak_offset_hours', np.nan))
            row['time_end_offset'] = float(main_time.get('end_offset_hours', np.nan))
        rows.append(row)
    return rows


def _prediction_to_slot_debug_rows(pred: Dict[str, Any], gt_sample: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """把单个窗口展开为 window×slot 调试表，保留所有槽位与最终 detection 的映射关系。"""
    metadata = pred.get('metadata', {}) or {}
    window_info = pred.get('window_info', {}) or {}

    final_classes = _flatten_numeric_sequence(_to_serializable(pred.get('final_classes', pred.get('predicted_class', []))))
    class_confidences = _flatten_numeric_sequence(_to_serializable(pred.get('classification_confidences', [])))
    processed_event_probs = _flatten_numeric_sequence(_to_serializable(pred.get('processed_event_probs', pred.get('event_prob_mean', []))))
    slot_proposal_scores = _flatten_numeric_sequence(_to_serializable(pred.get('slot_proposal_scores', pred.get('proposal_scores', []))))

    processed_class_probs_arr = _to_numeric_array(pred.get('processed_class_probs', pred.get('class_probs_mean', [])))
    if processed_class_probs_arr.ndim == 1 and processed_class_probs_arr.size > 0:
        processed_class_probs_arr = processed_class_probs_arr.reshape(1, -1)

    processed_time_arr = _to_numeric_array(pred.get('processed_time_pred', pred.get('time_pred_mean', [])))
    if processed_time_arr.ndim == 1 and processed_time_arr.size > 0:
        processed_time_arr = processed_time_arr.reshape(1, -1)

    processed_bboxes_arr = _to_numeric_array(pred.get('processed_bboxes', pred.get('bbox_pred_mean', [])))
    if processed_bboxes_arr.ndim == 1 and processed_bboxes_arr.size >= 4:
        processed_bboxes_arr = processed_bboxes_arr.reshape(1, -1)

    kept_slot_indices = _flatten_numeric_sequence(_to_serializable(pred.get('kept_bbox_slot_indices', [])))
    detections = pred.get('detections', []) or []
    detection_by_slot: Dict[int, Dict[str, Any]] = {}
    for det_idx, det in enumerate(detections):
        slot_id = int(det.get('slot_id', det_idx))
        detection_by_slot[slot_id] = {'detection_idx': det_idx, 'det': det}

    num_slots = max(
        len(final_classes),
        len(class_confidences),
        len(processed_event_probs),
        len(slot_proposal_scores),
        len(kept_slot_indices),
        int(processed_class_probs_arr.shape[0]) if processed_class_probs_arr.ndim >= 2 else 0,
        int(processed_time_arr.shape[0]) if processed_time_arr.ndim >= 2 else 0,
        int(processed_bboxes_arr.shape[0]) if processed_bboxes_arr.ndim >= 2 else 0,
    )

    gt_labels: List[Any] = []
    gt_bboxes: List[Any] = []
    gt_time_features: List[Any] = []
    gt_activity_mask: List[Any] = []
    if gt_sample is not None:
        gt_labels = _flatten_numeric_sequence(_to_serializable(gt_sample.get('label')))
        gt_activity_mask = _flatten_numeric_sequence(_to_serializable(gt_sample.get('activity_mask')))
        gt_bbox_arr = _to_numeric_array(gt_sample.get('bbox'))
        if gt_bbox_arr.ndim == 1 and gt_bbox_arr.size >= 4:
            gt_bbox_arr = gt_bbox_arr.reshape(1, -1)
        gt_time_arr = _to_numeric_array(gt_sample.get('time_features'))
        if gt_time_arr.ndim == 1 and gt_time_arr.size >= 3:
            gt_time_arr = gt_time_arr.reshape(1, -1)
        gt_bboxes = gt_bbox_arr.tolist() if gt_bbox_arr.size > 0 else []
        gt_time_features = gt_time_arr.tolist() if gt_time_arr.size > 0 else []
        num_slots = max(num_slots, len(gt_labels), len(gt_activity_mask), len(gt_bboxes), len(gt_time_features))

    rows: List[Dict[str, Any]] = []
    for slot_idx in range(num_slots):
        det_info = detection_by_slot.get(slot_idx, {})
        det = det_info.get('det', {})
        detection_idx = det_info.get('detection_idx', -1)

        class_probs = processed_class_probs_arr[slot_idx].tolist() if processed_class_probs_arr.ndim >= 2 and slot_idx < processed_class_probs_arr.shape[0] else []
        time_pred = processed_time_arr[slot_idx].tolist() if processed_time_arr.ndim >= 2 and slot_idx < processed_time_arr.shape[0] else []
        bbox_pred = processed_bboxes_arr[slot_idx].tolist() if processed_bboxes_arr.ndim >= 2 and slot_idx < processed_bboxes_arr.shape[0] else []

        gt_label = int(gt_labels[slot_idx]) if slot_idx < len(gt_labels) else -1
        gt_mask = bool(gt_activity_mask[slot_idx]) if slot_idx < len(gt_activity_mask) else False
        gt_bbox = gt_bboxes[slot_idx] if slot_idx < len(gt_bboxes) else []
        gt_time = gt_time_features[slot_idx] if slot_idx < len(gt_time_features) else []

        rows.append({
            'event_id': metadata.get('event_id', 'unknown'),
            'window_start_idx': metadata.get('window_start_idx', window_info.get('start_idx', -1)),
            'window_end_idx': metadata.get('window_end_idx', window_info.get('end_idx', -1)),
            'window_timestamps': json.dumps(metadata.get('window_timestamps', []), ensure_ascii=False),
            'slot_id': slot_idx,
            'gt_label': gt_label,
            'gt_activity_mask': gt_mask,
            'gt_bbox': json.dumps(gt_bbox, ensure_ascii=False),
            'gt_time_features': json.dumps(gt_time, ensure_ascii=False),
            'pred_class_probs': json.dumps(class_probs, ensure_ascii=False),
            'final_class': int(final_classes[slot_idx]) if slot_idx < len(final_classes) else -1,
            'class_confidence': float(class_confidences[slot_idx]) if slot_idx < len(class_confidences) else np.nan,
            'event_probability': float(processed_event_probs[slot_idx]) if slot_idx < len(processed_event_probs) else np.nan,
            'proposal_score': float(slot_proposal_scores[slot_idx]) if slot_idx < len(slot_proposal_scores) else np.nan,
            'processed_bbox': json.dumps(bbox_pred, ensure_ascii=False),
            'kept_after_postprocess': slot_idx in {int(x) for x in kept_slot_indices if x is not None},
            'final_detection_idx': int(detection_idx),
            'final_detection_bbox': json.dumps(det.get('bbox', []) or [], ensure_ascii=False),
            'final_detection_class': int(det.get('class_id', -1)) if det else -1,
            'final_detection_linked_events': json.dumps(det.get('linked_events', []) or [], ensure_ascii=False),
        })
    return rows


def _prediction_to_row(pred: Dict[str, Any]) -> Dict[str, Any]:
    """把单个窗口预测结果展开为一行表格。"""
    metadata = pred.get('metadata', {}) or {}
    window_info = pred.get('window_info', {}) or {}

    class_probs = pred.get('processed_class_probs', pred.get('class_probs_mean', []))
    class_probs = _to_serializable(class_probs)
    class_probs_flat = _flatten_numeric_sequence(class_probs) if isinstance(class_probs, list) else _flatten_numeric_sequence(pred.get('processed_class_probs', pred.get('class_probs_mean', [])))

    time_pred = pred.get('processed_time_pred', pred.get('time_pred_mean', []))
    time_pred = _to_serializable(time_pred)
    time_pred_flat = _flatten_numeric_sequence(time_pred) if isinstance(time_pred, list) else _flatten_numeric_sequence(pred.get('processed_time_pred', pred.get('time_pred_mean', [])))

    bboxes = pred.get('processed_bboxes', pred.get('bbox_pred_mean', []))
    bboxes = _to_serializable(bboxes)
    bbox_count = len(bboxes) if isinstance(bboxes, list) else 0

    final_classes = _flatten_numeric_sequence(_to_serializable(pred.get('final_classes', pred.get('predicted_class', []))))
    class_confidences = _flatten_numeric_sequence(_to_serializable(pred.get('classification_confidences', [])))

    row = {
        'event_id': metadata.get('event_id', 'unknown'),
        'window_start_idx': metadata.get('window_start_idx', window_info.get('start_idx', -1)),
        'window_end_idx': metadata.get('window_end_idx', window_info.get('end_idx', -1)),
        'predicted_class': int(_safe_first_numeric(pred.get('predicted_class', -1), default=-1.0)),
        'final_class': int(_safe_first_numeric(pred.get('final_class', -1), default=-1.0)),
        'classification_confidence': float(pred.get('classification_confidence', 0.0) or 0.0),
        'processed_event_prob': float(pred.get('processed_event_prob', 0.0) or 0.0),
        'bbox_count': bbox_count,
        'bboxes': json.dumps(bboxes, ensure_ascii=False),
        'window_timestamps': json.dumps(metadata.get('window_timestamps', []), ensure_ascii=False),
        'slot_final_classes': json.dumps(final_classes, ensure_ascii=False),
        'slot_classification_confidences': json.dumps(class_confidences, ensure_ascii=False),
    }

    if len(class_probs_flat) >= 3:
        row['class0_prob'] = float(class_probs_flat[0])
        row['class1_prob'] = float(class_probs_flat[1])
        row['class2_prob'] = float(class_probs_flat[2])
    else:
        row['class0_prob'] = row['class1_prob'] = row['class2_prob'] = np.nan

    if len(time_pred_flat) >= 3:
        row['time_start_offset'] = float(time_pred_flat[0])
        row['time_peak_offset'] = float(time_pred_flat[1])
        row['time_end_offset'] = float(time_pred_flat[2])

    return row


def _predictions_to_dataframe(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [_prediction_to_row(pred) for pred in predictions]
    return pd.DataFrame(rows)


def _predictions_to_slot_dataframe(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pred in predictions:
        rows.extend(_prediction_to_slot_rows(pred))
    return pd.DataFrame(rows)


def _build_predictor_inputs_from_dataset_sample(
        sample: Dict[str, Any],
        modalities: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """灏?dataset 鏍锋湰杞垚 predictor.predict_batch() 闇€瑕佺殑 batch=1 杈撳叆銆?"""
    predictor_inputs: Dict[str, torch.Tensor] = {}
    sample_data = sample.get('data', {}) or {}
    active_modalities = modalities or list(sample_data.keys())
    for modality in active_modalities:
        value = sample_data.get(modality)
        if value is None:
            continue
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        predictor_inputs[modality] = value.unsqueeze(0)

    for key in ('proposal_boxes', 'proposal_scores'):
        value = sample.get(key)
        if value is None:
            continue
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        predictor_inputs[key] = value.unsqueeze(0)
    return predictor_inputs


def _build_gt_sample_lookup(dataset: SolarFlareDataset) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for sample in dataset:
        metadata = sample.get('metadata', {}) or {}
        window_id = metadata.get('window_id')
        if window_id:
            lookup[str(window_id)] = sample
    return lookup


def _load_split_event_ids(hdf5_path: str, split_name: str = 'test', split_file: str = '', max_events: int = 0) -> List[str]:
    """读取指定划分中的 event_id 列表；若不存在则退回到 HDF5 中所有可用事件。"""
    event_ids: List[str] = []

    split_path = Path(split_file) if split_file else project_root / 'data' / 'split_data' / f'{split_name}_events.txt'
    if split_path.exists():
        with open(split_path, 'r', encoding='utf-8') as f:
            event_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"从划分文件读取到 {len(event_ids)} 个事件: {split_path}")
    else:
        logger.warning(f"未找到划分文件 {split_path}，将退回到 HDF5 中所有可用事件")
        reader = HDF5DatasetReader(hdf5_path)
        event_ids = reader.get_event_ids(available_only=True)

    if max_events and max_events > 0:
        event_ids = event_ids[:max_events]
        logger.info(f"限制只测试前 {len(event_ids)} 个事件")

    return event_ids


def _infer_pred_class_int(pred: Dict[str, Any]) -> int:
    """兼容旧逻辑：优先 final_class，否则按 1/2 二分类推断窗口类别。"""
    fc = pred.get('final_class', None)
    if fc is not None:
        if hasattr(fc, 'item'):
            return int(fc.item())
        return int(fc)
    pc = pred.get('predicted_class', None)
    if pc is not None:
        if hasattr(pc, 'item'):
            pc = pc.item()
        pc_arr = np.asarray(pc).reshape(-1)
        if pc_arr.size > 0:
            return int(pc_arr[0])
    cp = pred.get('class_probs_mean', pred.get('processed_class_probs', None))
    if cp is not None:
        if torch.is_tensor(cp):
            cp = cp.detach().cpu().numpy()
        cp = np.asarray(cp)
        if cp.ndim == 1:
            cp = cp.reshape(1, -1)
        if cp.size > 0:
            cp12 = cp[0, 1:3] if cp.shape[-1] >= 3 else cp[0, :2]
            if cp12.size > 0:
                return int(np.argmax(cp12)) + 1
    return -1


def _infer_pred_slot_classes(pred: Dict[str, Any], max_activities: int) -> np.ndarray:
    """提取每个 slot 的预测类别，保留 0/负值以评估背景和空预测。"""
    slot_classes = pred.get('final_classes', None)
    if slot_classes is not None:
        if torch.is_tensor(slot_classes):
            slot_classes = slot_classes.detach().cpu().numpy()
        slot_classes = np.asarray(slot_classes, dtype=np.int64).reshape(-1)
    else:
        class_probs = pred.get('processed_class_probs', pred.get('class_probs_mean', None))
        if class_probs is None:
            fallback = _infer_pred_class_int(pred)
            if fallback < 0:
                return np.full((max_activities,), -1, dtype=np.int64)
            slot_classes = np.full((max_activities,), fallback, dtype=np.int64)
        else:
            if torch.is_tensor(class_probs):
                class_probs = class_probs.detach().cpu().numpy()
            class_probs = np.asarray(class_probs)
            if class_probs.ndim == 1:
                if class_probs.size >= 3:
                    fallback = int(np.argmax(class_probs[:3]))
                elif class_probs.size > 0:
                    fallback = int(np.argmax(class_probs))
                else:
                    fallback = -1
                if fallback < 0:
                    return np.full((max_activities,), -1, dtype=np.int64)
                slot_classes = np.full((max_activities,), fallback, dtype=np.int64)
            else:
                usable_probs = class_probs[:, :3] if class_probs.shape[-1] >= 3 else class_probs
                slot_classes = np.argmax(usable_probs, axis=-1).astype(np.int64).reshape(-1)

    if slot_classes.size < max_activities:
        padded = np.full((max_activities,), -1, dtype=np.int64)
        padded[:slot_classes.size] = slot_classes
        slot_classes = padded
    else:
        slot_classes = slot_classes[:max_activities]

    return slot_classes


def _safe_binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    tss = recall - (fp / (fp + tn + 1e-8))
    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'tss': float(tss),
    }


def _compute_activity_binary_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    target_positive = targets > 0
    pred_positive = predictions > 0
    tp = int(np.sum(target_positive & pred_positive))
    fp = int(np.sum(~target_positive & pred_positive))
    fn = int(np.sum(target_positive & ~pred_positive))
    tn = int(np.sum(~target_positive & ~pred_positive))
    return _safe_binary_metrics(tp, fp, fn, tn)


def _compute_positive_class_ovr_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    per_class: Dict[str, Any] = {}
    for cls in (1, 2):
        target_positive = targets == cls
        pred_positive = predictions == cls
        tp = int(np.sum(target_positive & pred_positive))
        fp = int(np.sum(~target_positive & pred_positive))
        fn = int(np.sum(target_positive & ~pred_positive))
        tn = int(np.sum(~target_positive & ~pred_positive))
        per_class[str(cls)] = _safe_binary_metrics(tp, fp, fn, tn)

    return {
        'per_class': per_class,
        'macro_precision': float(np.mean([v['precision'] for v in per_class.values()])),
        'macro_recall': float(np.mean([v['recall'] for v in per_class.values()])),
        'macro_f1': float(np.mean([v['f1'] for v in per_class.values()])),
        'macro_tss': float(np.mean([v['tss'] for v in per_class.values()])),
    }


def _compute_test_set_classification_metrics(
    hdf5_path: str,
    predictions: List[Dict[str, Any]],
    window_size: int,
    stride: int,
    modalities: Optional[List[str]],
    model_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    """
    与训练/验证时一致：按 activity 槽位逐一比较 final_classes[k] 与 HDF5 标签。
    同时返回预测/真实类别分布，便于诊断 test 集为何出现极端指标。
    """
    if not predictions:
        return {}, 0

    # 优先从 model_cfg.modalities 的第一个模态 resolution 读取，与训练时
    # f_train_model.py 中 first_mod.get('resolution', ...) 的路径保持一致。
    # 回退顺序：modalities[first].resolution -> input_size -> 硬编码默认值
    cfg_modalities = model_cfg.get('modalities', {})
    first_mod_cfg = next(iter(cfg_modalities.values()), {}) if cfg_modalities else {}
    resolution = first_mod_cfg.get('resolution', None)
    if resolution and len(resolution) >= 2:
        target_size: Tuple[int, int] = (int(resolution[0]), int(resolution[1]))
    else:
        inp = model_cfg.get('input_size', [256, 256])
        target_size = (int(inp[0]), int(inp[1]))
    max_activities = int(model_cfg.get('max_activities', 5))
    num_classes = int(model_cfg.get('num_classes', 3))

    reader = HDF5DatasetReader(hdf5_path)
    eff_modalities = modalities if modalities else reader.modalities
    from collections import defaultdict

    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        eid = (p.get('metadata') or {}).get('event_id')
        if eid:
            by_event[str(eid)].append(p)

    activity_pred: List[int] = []
    activity_tgt: List[int] = []
    all_slot_pred: List[int] = []
    all_slot_tgt: List[int] = []

    for event_id, preds in by_event.items():
        try:
            ds = SolarFlareDataset(
                reader,
                [event_id],
                modalities=eff_modalities,
                sequence_length=window_size,
                stride=stride,
                target_size=target_size,
                max_activities=max_activities,
                config={'max_activities': max_activities},
            )
            ds_lookup = _build_gt_sample_lookup(ds)
        except Exception as e:
            logger.warning(f"无法创建数据集以计算 GT 指标: {event_id}: {e}")
            continue

        for pred in preds:
            meta = pred.get('metadata') or {}
            window_id = meta.get('window_id')
            if not window_id:
                continue
            sample = ds_lookup.get(str(window_id))
            if sample is None:
                continue

            labels = sample['label'].numpy()
            mask = sample['activity_mask'].numpy().astype(bool)
            pred_slot_classes = _infer_pred_slot_classes(pred, max_activities)

            for k in range(min(len(labels), len(pred_slot_classes))):
                gt_active = bool(mask[k]) and int(labels[k]) > 0
                tgt_cls = int(labels[k]) if gt_active else 0
                pred_cls_raw = int(pred_slot_classes[k])
                pred_cls = pred_cls_raw if pred_cls_raw >= 0 else 0
                pred_cls = pred_cls if pred_cls < num_classes else 0

                all_slot_tgt.append(tgt_cls)
                all_slot_pred.append(pred_cls)
                if gt_active:
                    activity_tgt.append(tgt_cls)
                    activity_pred.append(pred_cls)

    n_all = len(all_slot_pred)
    n_activity = len(activity_pred)
    if n_all == 0:
        return {}, 0

    all_pred_arr = np.array(all_slot_pred, dtype=np.int64)
    all_tgt_arr = np.array(all_slot_tgt, dtype=np.int64)
    all_slot_metrics = calculate_metrics(
        all_pred_arr,
        all_tgt_arr,
        num_classes=num_classes,
    )
    if np.isnan(all_slot_metrics.get('accuracy', 0.0)):
        all_slot_metrics['accuracy'] = 0.0

    if n_activity > 0:
        activity_pred_arr = np.array(activity_pred, dtype=np.int64)
        activity_tgt_arr = np.array(activity_tgt, dtype=np.int64)
        activity_metrics = calculate_metrics(
            activity_pred_arr,
            activity_tgt_arr,
            num_classes=num_classes,
        )
        if np.isnan(activity_metrics.get('accuracy', 0.0)):
            activity_metrics['accuracy'] = 0.0
    else:
        activity_pred_arr = np.zeros((0,), dtype=np.int64)
        activity_tgt_arr = np.zeros((0,), dtype=np.int64)
        activity_metrics = {}

    pred_counts = np.bincount(all_pred_arr, minlength=num_classes)
    tgt_counts = np.bincount(all_tgt_arr, minlength=num_classes)
    activity_pred_counts = np.bincount(activity_pred_arr, minlength=num_classes)
    activity_tgt_counts = np.bincount(activity_tgt_arr, minlength=num_classes)

    m: Dict[str, Any] = {
        'classification_all_slots': all_slot_metrics,
        'classification_activity_only': activity_metrics,
        'activity_binary_all_slots': _compute_activity_binary_metrics(all_tgt_arr, all_pred_arr),
        'positive_class_ovr_all_slots': _compute_positive_class_ovr_metrics(all_tgt_arr, all_pred_arr),
        'positive_class_ovr_activity_only': _compute_positive_class_ovr_metrics(activity_tgt_arr, activity_pred_arr)
        if n_activity > 0 else {},
        'num_eval_all_slots': int(n_all),
        'num_eval_activity_slots': int(n_activity),
        'all_slots_pred_class_counts': {str(i): int(pred_counts[i]) for i in range(num_classes)},
        'all_slots_target_class_counts': {str(i): int(tgt_counts[i]) for i in range(num_classes)},
        'activity_only_pred_class_counts': {str(i): int(activity_pred_counts[i]) for i in range(num_classes)},
        'activity_only_target_class_counts': {str(i): int(activity_tgt_counts[i]) for i in range(num_classes)},
    }

    # Backward-compatible top-level fields keep the old activity-only intent.
    m.update(activity_metrics)
    m['pred_class_counts'] = m['activity_only_pred_class_counts']
    m['target_class_counts'] = m['activity_only_target_class_counts']
    return m, n_activity


def _save_batch_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """为批量测试结果生成基础可视化图表。"""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping visualizations")
        return

    if df.empty:
        logger.warning("没有可视化数据，跳过图表生成")
        return

    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1) 槽位级最终类别分布
    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts = df['class_id'].value_counts().sort_index()
    class_counts.plot(kind='bar', ax=ax, color=['#DD8452', '#55A868'])
    ax.set_title('testset kept bbox class distribution')
    ax.set_xlabel('class_id')
    ax.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(vis_dir / 'class_distribution.png', dpi=200)
    plt.close(fig)

    # 2) 事件概率分布
    fig, ax = plt.subplots(figsize=(8, 5))
    df['event_probability'].dropna().plot(kind='hist', bins=20, ax=ax, color='#4C72B0', alpha=0.8)
    ax.set_title('testset kept bbox probability distribution')
    ax.set_xlabel('event_probability')
    fig.tight_layout()
    fig.savefig(vis_dir / 'event_probability_hist.png', dpi=200)
    plt.close(fig)

    # 3) 每个事件平均风险 Top-N
    event_risk = df.groupby('event_id')['event_probability'].mean().sort_values(ascending=False).head(15)
    if not event_risk.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        event_risk.sort_values().plot(kind='barh', ax=ax, color='#C44E52')
        ax.set_title('testset activity average risk')
        ax.set_xlabel('mean event_probability')
        fig.tight_layout()
        fig.savefig(vis_dir / 'top_risk_events.png', dpi=200)
        plt.close(fig)

    # 4) 时间预测分布
    time_cols = ['time_start_offset', 'time_peak_offset', 'time_end_offset']
    available_time_cols = [c for c in time_cols if c in df.columns and df[c].notna().any()]
    if available_time_cols:
        fig, axes = plt.subplots(len(available_time_cols), 1, figsize=(8, 4 * len(available_time_cols)))
        if len(available_time_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, available_time_cols):
            df[col].dropna().plot(kind='hist', bins=20, ax=ax, alpha=0.8)
            ax.set_title(f'{col} distribution')
        fig.tight_layout()
        fig.savefig(vis_dir / 'time_predictions_hist.png', dpi=200)
        plt.close(fig)

    logger.info(f"批量测试可视化已保存到: {vis_dir}")


def _run_test_set_inference(
    predictor: SolarFlarePredictor,
    hdf5_path: str,
    output_dir: Path,
    split_name: str,
    split_file: str,
    window_size: int,
    stride: int,
    modalities: List[str],
    use_uncertainty: bool,
    save_visualizations: bool,
    max_events: int = 0,
    visualize_predictions: bool = True,
    viz_frame: Union[str, int] = 'last',
    model_cfg: Optional[Dict[str, Any]] = None,
    proposal_cache_path: Optional[str] = None,
    save_window_visualizations: bool = False,
    save_event_overlay_visualization: bool = True,
) -> int:
    """批量测试整个 train/val/test 划分。"""
    event_ids = _load_split_event_ids(hdf5_path, split_name, split_file, max_events)
    if not event_ids:
        logger.error("未找到可用于批量测试的 event_id")
        return 1

    logger.info(f"开始批量测试 {split_name} 集，共 {len(event_ids)} 个事件")
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_cfg is None:
        model_cfg = {}

    all_predictions: List[Dict[str, Any]] = []
    all_slot_debug_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    cfg_modalities = model_cfg.get('modalities', {}) if model_cfg else {}
    first_mod_cfg = next(iter(cfg_modalities.values()), {}) if cfg_modalities else {}
    resolution = first_mod_cfg.get('resolution', None)
    if resolution and len(resolution) >= 2:
        target_size: Tuple[int, int] = (int(resolution[0]), int(resolution[1]))
    else:
        inp = model_cfg.get('input_size', [256, 256]) if model_cfg else [256, 256]
        target_size = (int(inp[0]), int(inp[1]))
    max_activities = int(model_cfg.get('max_activities', 5)) if model_cfg else 5

    # 初始化可视化器
    visualizer = PredictionVisualizer(
        image_size=target_size,
        time_head_enabled=bool(getattr(predictor, 'time_head_enabled', True)),
    ) if visualize_predictions and MATPLOTLIB_AVAILABLE else None

    for idx, event_id in enumerate(event_ids, start=1):
        logger.info(f"[{idx}/{len(event_ids)}] 测试事件: {event_id}")
        try:
            gt_dataset = SolarFlareDataset(
                HDF5DatasetReader(hdf5_path),
                [event_id],
                modalities=modalities,
                sequence_length=window_size,
                stride=stride,
                target_size=target_size,
                max_activities=max_activities,
                config={
                    'max_activities': max_activities,
                    'proposal_cache_path': proposal_cache_path,
                },
            )
            gt_sample_lookup = _build_gt_sample_lookup(gt_dataset)
            predictions = []
            gt_samples_for_event: List[Optional[Dict[str, Any]]] = []
            for sample in gt_dataset:
                predictor_inputs = _build_predictor_inputs_from_dataset_sample(sample, modalities=modalities)
                pred = predictor.predict_batch(
                    predictor_inputs,
                    use_uncertainty=use_uncertainty,
                )
                meta = sample.get('metadata', {}) or {}
                pred['metadata'] = {
                    'event_id': event_id,
                    'window_id': meta.get('window_id'),
                    'window_start_idx': meta.get('start_idx', -1),
                    'window_end_idx': meta.get('end_idx', -1),
                    'has_external_proposals': bool(meta.get('has_external_proposals', False)),
                }
                pred['detections'] = predictor._build_detections(pred)
                predictions.append(pred)
                gt_samples_for_event.append(sample)

            alerts = [predictor.generate_alert(pred) for pred in predictions]
            all_predictions.extend(predictions)

            event_output_dir = output_dir / event_id
            event_output_dir.mkdir(parents=True, exist_ok=True)
            predictor.export_predictions(predictions, str(event_output_dir / 'predictions.json'), format='json')
            with open(event_output_dir / 'alerts.json', 'w', encoding='utf-8') as f:
                json.dump(alerts, f, indent=2, ensure_ascii=False, default=str)

            event_df = _predictions_to_dataframe(predictions)
            event_slots_df = _predictions_to_slot_dataframe(predictions)
            event_slots_df.to_csv(event_output_dir / 'predictions_slots_detailed.csv', index=False, encoding='utf-8-sig')

            event_debug_rows: List[Dict[str, Any]] = []
            reader = None
            try:
                reader = HDF5DatasetReader(hdf5_path)
                for pred in predictions:
                    meta = pred.get('metadata') or {}
                    gt_sample = gt_sample_lookup.get(str(meta.get('window_id')))
                    event_debug_rows.extend(_prediction_to_slot_debug_rows(pred, gt_sample))
            except Exception as e:
                logger.warning(f"事件 {event_id} 调试槽位表生成失败: {e}")

            event_slot_debug_df = pd.DataFrame(event_debug_rows)
            _safe_to_csv(event_slot_debug_df, event_output_dir / 'predictions_slots_debug.csv', index=False)
            all_slot_debug_rows.extend(event_debug_rows)

            # 生成预测可视化（可选）
            if visualizer:
                try:
                    # 加载模态图像用于可视化
                    reader = HDF5DatasetReader(hdf5_path)
                    modality_images = reader.get_event_data(event_id, modalities=modalities)
                    
                    # 为每个预测生成可视化
                    vis_dir = output_dir / 'visualizations'
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    
                    if save_window_visualizations:
                        for pred_idx, pred in enumerate(predictions):
                            try:
                                gt_sample = gt_samples_for_event[pred_idx] if pred_idx < len(gt_samples_for_event) else None
                                visualizer.visualize_prediction(
                                    modality_images=modality_images,
                                    predictions=pred,
                                    event_id=event_id,
                                    gt_sample=gt_sample,
                                    output_dir=vis_dir,
                                    reference_frame=viz_frame,
                                )
                                plt.close('all')
                            except Exception as e:
                                logger.debug(f"Window visualization {pred_idx} failed: {e}")
                    
                    # 生成事件摘要报告
                    if save_event_overlay_visualization:
                        visualizer.visualize_event_prediction_summary(
                            modality_images=modality_images,
                            predictions_list=predictions,
                            event_id=event_id,
                            gt_samples=gt_samples_for_event,
                            output_dir=vis_dir,
                            reference_frame=viz_frame,
                            show_gt_proposal_prediction_overlay=True,
                        )
                        plt.close('all')
                    visualizer.create_summary_report(predictions, vis_dir, event_id)
                    plt.close('all')
                    
                except Exception as e:
                    logger.warning(f"事件 {event_id} 可视化生成失败: {e}")

            summary_rows.append({
                'event_id': event_id,
                'num_windows': len(predictions),
                'num_kept_detections': int(len(event_slots_df)),
                'mean_event_prob': float(event_df['processed_event_prob'].mean()) if not event_df.empty else 0.0,
                'max_event_prob': float(event_df['processed_event_prob'].max()) if not event_df.empty else 0.0,
                'dominant_final_class': int(event_slots_df['class_id'].mode().iloc[0]) if not event_slots_df.empty else -1,
            })

        except Exception as e:
            logger.error(f"事件 {event_id} 批量测试失败: {e}")

    if not all_predictions:
        logger.error("批量测试未产生任何预测结果")
        return 1

    predictor.export_predictions(all_predictions, str(output_dir / f'{split_name}_predictions.json'), format='json')
    detailed_df = _predictions_to_dataframe(all_predictions)
    _safe_to_csv(detailed_df, output_dir / f'{split_name}_predictions_detailed.csv', index=False)
    slot_detailed_df = _predictions_to_slot_dataframe(all_predictions)
    _safe_to_csv(slot_detailed_df, output_dir / f'{split_name}_predictions_slots_detailed.csv', index=False)
    slot_debug_df = pd.DataFrame(all_slot_debug_rows)
    _safe_to_csv(slot_debug_df, output_dir / f'{split_name}_predictions_slots_debug.csv', index=False)
    _safe_to_csv(pd.DataFrame(summary_rows), output_dir / f'{split_name}_event_summary.csv', index=False)

    if save_visualizations:
        _save_batch_visualizations(slot_detailed_df, output_dir)

    # 与训练/验证一致的分类指标（仅 test_set 批量模式）
    try:
        metrics, n_slots = _compute_test_set_classification_metrics(
            hdf5_path,
            all_predictions,
            window_size,
            stride,
            modalities,
            model_cfg,
        )
        if n_slots > 0:
            scalar_metric_keys = {
                'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                'weighted_precision', 'weighted_recall', 'weighted_f1'
            }
            metrics_out = {k: float(metrics[k]) for k in scalar_metric_keys if k in metrics}
            metrics_out['num_eval_activity_slots'] = int(metrics.get('num_eval_activity_slots', n_slots))
            metrics_out['num_eval_all_slots'] = int(metrics.get('num_eval_all_slots', 0))
            metrics_out['pred_class_counts'] = metrics.get('pred_class_counts', {})
            metrics_out['target_class_counts'] = metrics.get('target_class_counts', {})
            metrics_out['classification_activity_only'] = metrics.get('classification_activity_only', {})
            metrics_out['classification_all_slots'] = metrics.get('classification_all_slots', {})
            metrics_out['activity_binary_all_slots'] = metrics.get('activity_binary_all_slots', {})
            metrics_out['positive_class_ovr_all_slots'] = metrics.get('positive_class_ovr_all_slots', {})
            metrics_out['positive_class_ovr_activity_only'] = metrics.get('positive_class_ovr_activity_only', {})
            metrics_out['all_slots_pred_class_counts'] = metrics.get('all_slots_pred_class_counts', {})
            metrics_out['all_slots_target_class_counts'] = metrics.get('all_slots_target_class_counts', {})
            metrics_out['activity_only_pred_class_counts'] = metrics.get('activity_only_pred_class_counts', {})
            metrics_out['activity_only_target_class_counts'] = metrics.get('activity_only_target_class_counts', {})
            metrics_path = output_dir / f'{split_name}_classification_metrics.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_out, f, indent=2, ensure_ascii=False)
            logger.info(
                f"{split_name} 集分类指标（逐槽位评估，{n_slots} 个 activity 槽位）: "
                f"accuracy={metrics.get('accuracy', 0):.4f}, "
                f"macro_f1={metrics.get('macro_f1', 0):.4f}, "
                f"macro_precision={metrics.get('macro_precision', 0):.4f}, "
                f"macro_recall={metrics.get('macro_recall', 0):.4f}"
            )
            all_slot = metrics.get('classification_all_slots', {})
            activity_binary = metrics.get('activity_binary_all_slots', {})
            logger.info(
                f"{split_name} 集新指标: "
                f"all_slots_macro_precision={all_slot.get('macro_precision', 0):.4f}, "
                f"activity_binary_precision={activity_binary.get('precision', 0):.4f}, "
                f"activity_binary_recall={activity_binary.get('recall', 0):.4f}, "
                f"activity_binary_fp={activity_binary.get('fp', 0)}"
            )
            logger.info(
                f"{split_name} 集类别分布: pred={metrics.get('pred_class_counts', {})}, "
                f"gt={metrics.get('target_class_counts', {})}"
            )
            logger.info(f"指标已写入: {metrics_path}")
        else:
            logger.warning("未能聚合任何 GT activity 槽位，跳过分类指标（检查 HDF5 与窗口是否对齐）。")
    except Exception as e:
        logger.warning(f"计算测试集分类指标失败: {e}")

    logger.info(f"{split_name} 集批量测试完成，详细结果已保存到: {output_dir}")
    return 0


def main(args=None):
    """主函数"""
    parser = argparse.ArgumentParser(description='运行太阳耀斑预测')
    parser.add_argument('--run_mode', type=str, default='single_event',
                        choices=['single_event', 'test_set', 'batch_dir'],
                        help='运行模式：单事件 / 整个划分测试集 / 目录批处理')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='输入数据路径（HDF5文件或目录）')
    parser.add_argument('--event_id', type=str,
                        help='事件ID（如果输入是HDF5文件）')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='输出目录')
    parser.add_argument('--config_path', type=str,
                        default='configs/inference_config.yaml',
                        help='推理配置文件路径')
    parser.add_argument('--data_config_path', type=str,
                        default='configs/data_config.yaml',
                        help='数据配置文件路径；resolution/stride 等参数以此为权威来源')
    parser.add_argument('--window_size', type=int, default=None,
                        help='滑动预测窗口大小；默认读取配置中的 inference.windowing.window_size')
    parser.add_argument('--stride', type=int, default=None,
                        help='滑动窗口步长；默认读取配置中的 inference.windowing.stride')
    parser.add_argument('--split_name', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='当 run_mode=test_set 时，要测试的数据划分')
    parser.add_argument('--split_file', type=str, default='',
                        help='可选：自定义 event_id 列表文件，每行一个 event_id')
    parser.add_argument('--max_events', type=int, default=0,
                        help='可选：限制批量测试事件数；0 表示不限制')
    parser.add_argument('--disable_uncertainty', action='store_true',
                        help='禁用 MC Dropout 不确定性估计')
    parser.add_argument('--disable_visualization', action='store_true',
                        help='批量测试时不生成可视化图表')
    parser.add_argument(
        '--viz_frame',
        type=str,
        default='last',
        choices=['last', 'first', 'middle'],
        help='可视化底图使用滑动窗口内哪一帧：last=最后一帧（推荐，与序列末端对齐）；first/middle 可选',
    )

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # 加载推理配置，并统一窗口参数
    config = load_config(args.config_path)
    inference_cfg = config.get('inference', {})
    window_cfg = inference_cfg.get('windowing', {})
    model_cfg = config.get('model', {})

    # 从 data_config.yaml 读取权威参数（resolution / stride / sequence_length），
    # 覆盖 inference_config.yaml 中对应字段，确保推理与训练完全一致。
    data_config: Dict[str, Any] = {}
    data_cfg_path = Path(args.data_config_path)
    if not data_cfg_path.is_absolute():
        data_cfg_path = project_root / data_cfg_path
    try:
        data_config = load_config(str(data_cfg_path)).get('data', {})
        data_modalities_cfg = data_config.get('modalities', {})
        if data_modalities_cfg:
            enabled_data_modalities_cfg = {
                mod_name: mod_cfg
                for mod_name, mod_cfg in data_modalities_cfg.items()
                if bool(mod_cfg.get('enabled', True))
            }
            merged_modalities = {
                mod_name: dict(model_cfg.get('modalities', {}).get(mod_name, {}))
                for mod_name in model_cfg.get('modalities', {}).keys()
            }
            for mod_name, mod_cfg in enabled_data_modalities_cfg.items():
                if mod_name in merged_modalities:
                    merged_modalities[mod_name]['resolution'] = mod_cfg.get(
                        'resolution', merged_modalities[mod_name].get('resolution', [256, 256])
                    )
                elif not merged_modalities:
                    merged_modalities[mod_name] = {'resolution': mod_cfg.get('resolution', [256, 256])}
            if merged_modalities:
                model_cfg = dict(model_cfg)
                model_cfg['modalities'] = merged_modalities
        if 'sequence_length' in data_config:
            model_cfg['sequence_length'] = data_config['sequence_length']
    except Exception as e:
        logger.warning(f"加载 data_config 失败，将使用 inference_config 中的默认值: {e}")

    window_size = args.window_size if args.window_size is not None else int(
        window_cfg.get('window_size', model_cfg.get('sequence_length', 9))
    )
    # stride 优先级：命令行 > inference_config > data_config > 默认值 1
    stride = args.stride if args.stride is not None else int(
        window_cfg.get('stride', data_config.get('stride', 1))
    )
    use_uncertainty = bool(window_cfg.get('use_uncertainty', True))
    if args.disable_uncertainty:
        use_uncertainty = False
    modalities = window_cfg.get('modalities', None)

    # 设置日志
    log_file = Path(args.output_dir) / 'inference.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    logger.info("开始运行预测")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(
        f"运行模式: {args.run_mode} | 窗口配置: window_size={window_size}, stride={stride}, "
        f"use_uncertainty={use_uncertainty}, modalities={modalities}"
    )

    try:
        # 创建预测器
        logger.info("创建预测器...")
        predictor = SolarFlarePredictor(
            model_path=args.model_path,
            config_path=args.config_path
        )

        # 检查数据路径
        data_path = Path(args.data_path)

        if args.run_mode == 'test_set':
            if not (data_path.is_file() and data_path.suffix == '.h5'):
                logger.error("run_mode=test_set 时，data_path 必须是单个 HDF5 文件")
                return 1

            result = _run_test_set_inference(
                predictor=predictor,
                hdf5_path=str(data_path),
                output_dir=Path(args.output_dir) / args.split_name,
                split_name=args.split_name,
                split_file=args.split_file,
                window_size=window_size,
                stride=stride,
                modalities=modalities,
                use_uncertainty=use_uncertainty,
                save_visualizations=not args.disable_visualization and MATPLOTLIB_AVAILABLE,
                max_events=args.max_events,
                visualize_predictions=not args.disable_visualization,
                viz_frame=args.viz_frame,
                model_cfg=model_cfg,
                proposal_cache_path=data_config.get('proposal_cache_path'),
                save_window_visualizations=False,
                save_event_overlay_visualization=True,
            )
            if result != 0:
                return result

        elif data_path.is_file() and data_path.suffix == '.h5':
            # HDF5 单事件模式（保留原来的 event_id 方式）
            if not args.event_id:
                try:
                    reader = HDF5DatasetReader(str(data_path))
                    available_event_ids = reader.get_event_ids(available_only=True)
                    if available_event_ids:
                        args.event_id = available_event_ids[0]
                        logger.warning(f"未指定 event_id，自动使用第一个可用事件: {args.event_id}")
                    else:
                        logger.error("HDF5 文件中没有可用事件，无法自动选择 event_id")
                        return 1
                except Exception as e:
                    logger.error(f"自动选择 event_id 失败: {e}")
                    return 1

            logger.info(f"预测事件: {args.event_id}")

            predictions = predictor.predict_time_series(
                hdf5_path=str(data_path),
                event_id=args.event_id,
                window_size=window_size,
                stride=stride,
                modalities=modalities,
                use_uncertainty=use_uncertainty
            )
            for pred in predictions:
                pred['detections'] = predictor._build_detections(pred)

            alerts = [predictor.generate_alert(pred) for pred in predictions]

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'{args.event_id}_predictions.json'
            predictor.export_predictions(predictions, str(output_path), format='json')

            detailed_df = _predictions_to_dataframe(predictions)
            _safe_to_csv(detailed_df, output_dir / f'{args.event_id}_predictions_detailed.csv', index=False)
            slot_detailed_df = _predictions_to_slot_dataframe(predictions)
            _safe_to_csv(slot_detailed_df, output_dir / f'{args.event_id}_predictions_slots_detailed.csv', index=False)

            slot_debug_rows: List[Dict[str, Any]] = []
            try:
                first_mod_cfg = next(iter(model_cfg.get('modalities', {}).values()), {}) if model_cfg.get('modalities') else {}
                resolution = first_mod_cfg.get('resolution', model_cfg.get('input_size', [256, 256]))
                target_size = (int(resolution[0]), int(resolution[1]))
                max_activities = int(model_cfg.get('max_activities', 5))
                gt_dataset = SolarFlareDataset(
                    reader,
                    [args.event_id],
                    modalities=modalities,
                    sequence_length=window_size,
                    stride=stride,
                    target_size=target_size,
                    max_activities=max_activities,
                    config={'max_activities': max_activities},
                )
                gt_sample_lookup = _build_gt_sample_lookup(gt_dataset)
                for pred in predictions:
                    meta = pred.get('metadata') or {}
                    gt_sample = gt_sample_lookup.get(str(meta.get('window_id')))
                    slot_debug_rows.extend(_prediction_to_slot_debug_rows(pred, gt_sample))
            except Exception as e:
                logger.warning(f"单事件调试槽位表生成失败: {e}")
            _safe_to_csv(pd.DataFrame(slot_debug_rows), output_dir / f'{args.event_id}_predictions_slots_debug.csv', index=False)

            alerts_path = output_dir / f'{args.event_id}_alerts.json'
            with open(alerts_path, 'w', encoding='utf-8') as f:
                json.dump(alerts, f, indent=2, ensure_ascii=False, default=str)

            # 生成预测可视化（可选）
            if not args.disable_visualization and MATPLOTLIB_AVAILABLE:
                try:
                    visualizer = PredictionVisualizer(
                        image_size=target_size,
                        time_head_enabled=bool(getattr(predictor, 'time_head_enabled', True)),
                    )
                    reader = HDF5DatasetReader(str(data_path))
                    modality_images = reader.get_event_data(args.event_id, modalities=modalities)
                    
                    vis_dir = output_dir / 'visualizations'
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    
                    for pred_idx, pred in enumerate(predictions):
                        try:
                            gt_sample = None
                            if 'gt_dataset' in locals() and gt_dataset is not None:
                                start_idx = (pred.get('metadata') or {}).get('window_start_idx')
                                if start_idx is not None:
                                    try:
                                        gt_sample = gt_dataset[int(start_idx)]
                                    except Exception:
                                        gt_sample = None
                            visualizer.visualize_prediction(
                                modality_images=modality_images,
                                predictions=pred,
                                event_id=args.event_id,
                                gt_sample=gt_sample,
                                output_dir=vis_dir,
                                reference_frame=args.viz_frame,
                            )
                            plt.close('all')
                        except Exception as e:
                            logger.debug(f"预测可视化 {pred_idx} 失败: {e}")
                    
                    # 生成事件摘要报告
                    visualizer.create_summary_report(predictions, vis_dir, args.event_id)
                    plt.close('all')
                    logger.info(f"可视化结果已保存到: {vis_dir}")
                    
                except Exception as e:
                    logger.warning(f"可视化生成失败: {e}")

            logger.info(f"预测结果保存到: {output_path}")
            logger.info(f"详细表格保存到: {output_dir / f'{args.event_id}_predictions_detailed.csv'}")
            logger.info(f"警报保存到: {alerts_path}")

        elif data_path.is_dir() or args.run_mode == 'batch_dir':
            # 目录批处理模式
            logger.info("批处理模式")

            # 寻找HDF5文件
            h5_files = list(data_path.glob('*.h5'))

            if not h5_files:
                logger.error(f"目录中没有找到HDF5文件: {data_path}")
                return 1

            all_predictions = []

            for h5_file in h5_files:
                logger.info(f"处理文件: {h5_file.name}")

                # 这里需要读取HDF5文件中的事件ID
                # 简化处理：使用文件名的前部分作为事件ID
                event_id = h5_file.stem

                try:
                    # 预测整个时间序列
                    predictions = predictor.predict_time_series(
                        hdf5_path=str(h5_file),
                        event_id=event_id,
                        window_size=window_size,
                        stride=stride,
                        modalities=modalities,
                        use_uncertainty=use_uncertainty
                    )

                    all_predictions.extend(predictions)

                    # 为每个事件保存单独的结果
                    event_output_dir = Path(args.output_dir) / event_id
                    event_output_dir.mkdir(parents=True, exist_ok=True)

                    output_path = event_output_dir / 'predictions.json'
                    predictor.export_predictions(predictions, str(output_path), format='json')

                    logger.info(f"事件 {event_id} 预测完成")

                except Exception as e:
                    logger.error(f"处理文件 {h5_file.name} 时出错: {e}")

            # 保存汇总结果
            if all_predictions:
                summary_path = Path(args.output_dir) / 'summary_predictions.csv'
                predictor.export_predictions(all_predictions, str(summary_path), format='csv')
                logger.info(f"汇总结果保存到: {summary_path}")

        else:
            logger.error(f"无效的数据路径: {args.data_path}")
            return 1

        logger.info("预测完成!")

    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    # 未传命令行参数时，自动使用文件内 DIRECT_RUN_CONFIG，适合直接点“运行”。
    if len(sys.argv) == 1 and USE_DIRECT_RUN_CONFIG:
        direct_args = _build_direct_run_args()
        print('使用 DIRECT_RUN_CONFIG 直接运行推理：')
        print(' ', ' '.join(direct_args) if direct_args else '(未生成参数)')

        if '--model_path' not in direct_args:
            print('未找到可用的 .pth 模型文件。请先重新训练，或在 DIRECT_RUN_CONFIG["model_path"] 中手动填写路径后再点运行。')
            sys.exit(1)

        sys.exit(main(direct_args))

    sys.exit(main())
