"""
一键评估模型在 train / val / test 上的指标。

当前输出的核心指标：
- accuracy
- precision
- recall
- f1
- tss
- iou
- rss
- rmse

说明：
- 直接复用训练阶段的数据划分与验证逻辑，确保与项目当前 trainer 中的指标定义一致。
- train 集评估会使用“顺序遍历、drop_last=False”的评估 DataLoader，避免训练时 weighted sampler 对结果造成干扰。
"""
import sys
import os
import json
import copy
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.hdf5_reader import HDF5DatasetReader
from data.dataset import SolarFlareDataset
from models.multimodal_transformer import MultimodalTransformer
from training.trainer import SolarFlareTrainer
from utils.config_utils import load_config
from utils.logging_utils import setup_logging
from utils.metrics_calculation import BoundingBoxMetrics
from scripts.f_train_model import custom_collate


logger = logging.getLogger("train_model")


# =========================
# 直接运行配置（推荐直接点运行时使用）
# =========================
USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    'model_path': 'outputs/checkpoints_fitscache_mag_halpha_euv94_euv171_euv304_256/solar_flare_cme_model_epoch_0088.pth',
    'data_config': 'configs/data_config_fitscache_mag_halpha_euv94_euv171_euv304_256.yaml',
    'model_config': 'configs/model_config.yaml',
    'train_config': 'configs/training_config_fitscache_mag_halpha_euv94_euv171_euv304.yaml',
    'hdf5_path': '',
    'output_dir': 'outputs/evaluation_fitscache_mag_halpha_euv94_euv171_euv304_256',
    'log_dir': 'logs',
    'batch_size': 1,
    'num_workers': 4,
    'device': '',
    'debug': False,
}


def _get_enabled_modalities(modalities_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: cfg
        for name, cfg in modalities_cfg.items()
        if bool(cfg.get('enabled', True))
    }


def _build_direct_run_args() -> list[str]:
    cfg = dict(DIRECT_RUN_CONFIG)
    args = []
    for key in [
        'model_path', 'data_config', 'model_config', 'train_config',
        'hdf5_path', 'output_dir', 'log_dir', 'batch_size',
        'num_workers', 'device'
    ]:
        value = cfg.get(key)
        if value not in (None, ''):
            args.extend([f'--{key}', str(value)])

    if cfg.get('debug', False):
        args.append('--debug')
    return args


def parse_args():
    parser = argparse.ArgumentParser(description='一键评估 train/val/test 指标')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml',
                        help='数据配置文件路径')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml',
                        help='模型配置文件路径')
    parser.add_argument('--train_config', type=str, default='configs/training_config.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--hdf5_path', type=str, default='',
                        help='HDF5 路径；为空则优先使用 data_config 中的 hdf5_path')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation_h',
                        help='评估结果输出目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='评估 batch size；默认 1，便于与当前训练设置保持一致')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker 数')
    parser.add_argument('--device', type=str, default='',
                        help='指定设备，如 cpu / cuda；为空则自动选择')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试日志')
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)

    if not torch.cuda.is_available():
        return torch.device('cpu')

    try:
        major, minor = torch.cuda.get_device_capability(0)
        device_sm = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and device_sm not in supported_arches:
            logger.warning(
                "Detected CUDA device capability %s, but current PyTorch build only supports %s. "
                "Falling back to CPU for stable evaluation.",
                device_sm,
                sorted(supported_arches),
            )
            return torch.device('cpu')
    except Exception as exc:
        logger.warning("Failed to validate CUDA compatibility (%s). Falling back to CPU.", exc)
        return torch.device('cpu')

    return torch.device('cuda')


def _load_split_events(reader: HDF5DatasetReader, data_config: Dict[str, Any], split_name: str) -> list[str]:
    splits_dir = Path(data_config.get('split_data_dir', 'data/split_data'))
    split_file = splits_dir / f'{split_name}_events.txt'
    if not split_file.exists():
        raise FileNotFoundError(f'未找到划分文件: {split_file}')

    with open(split_file, 'r', encoding='utf-8') as f:
        event_ids = [line.strip() for line in f if line.strip()]

    available_event_ids = set(reader.get_event_ids(available_only=True))
    filtered = [event_id for event_id in event_ids if event_id in available_event_ids]
    logger.info('%s 划分: 文件中 %d 个事件，HDF5 可用 %d 个事件', split_name, len(event_ids), len(filtered))
    return filtered


def _create_eval_loader(
    reader: HDF5DatasetReader,
    data_config: Dict[str, Any],
    split_name: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    event_ids = _load_split_events(reader, data_config, split_name)

    enabled_modalities = _get_enabled_modalities(data_config['modalities'])
    modalities_list = list(enabled_modalities.keys())
    seq_len = data_config.get('sequence_length', 9)
    first_mod = next(iter(enabled_modalities.values()), {})
    target_size = tuple(first_mod.get('resolution', [256, 256]))
    stride = data_config.get('stride', 1)

    dataset = SolarFlareDataset(
        reader=reader,
        event_ids=event_ids,
        modalities=modalities_list,
        sequence_length=seq_len,
        target_size=target_size,
        stride=stride,
        max_activities=data_config.get('max_activities', 5),
        config={
            'max_activities': data_config.get('max_activities', 5),
            'proposal_cache_path': data_config.get('proposal_cache_path'),
        },
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate,
    )
    return loader


def _build_model_config(data_config: Dict[str, Any], model_config: Dict[str, Any], train_config: Dict[str, Any]) -> Dict[str, Any]:
    enabled_modalities = _get_enabled_modalities(data_config['modalities'])
    first_mod = next(iter(enabled_modalities.values()), {})
    target_size = tuple(first_mod.get('resolution', model_config.get('input_size', [256, 256])))

    model_config_merged = dict(model_config)
    model_config_merged['modalities'] = enabled_modalities
    model_config_merged['max_activities'] = data_config.get('max_activities', model_config.get('max_activities', 5))
    model_config_merged['input_size'] = list(target_size)
    model_config_merged['sequence_length'] = data_config.get('sequence_length', model_config.get('sequence_length', 9))
    model_config_merged['max_sequence_length'] = model_config_merged['sequence_length']
    model_config_merged.setdefault('stage2', {})
    if train_config.get('two_stage_schedule', {}).get('enabled', False):
        model_config_merged['stage2']['roi_source_mix'] = train_config['two_stage_schedule'].get(
            'roi_source_mix', {'predicted': 1.0}
        )
    return model_config_merged


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        model_state_dict = model.state_dict()
        unexpected_keys = [key for key in state_dict.keys() if key not in model_state_dict]
        missing_keys = [key for key in model_state_dict.keys() if key not in state_dict]

        time_head_disabled = not bool(getattr(model, 'enable_time_prediction', True))
        allowed_time_head_prefixes = (
            'stage_two_predictor.time_sequence_attn.',
            'stage_two_predictor.time_sequence_norm.',
            'stage_two_predictor.time_predictor.',
        )
        only_disabled_time_head_keys = (
            time_head_disabled and
            unexpected_keys and
            all(key.startswith(allowed_time_head_prefixes) for key in unexpected_keys)
        )

        if not only_disabled_time_head_keys:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}'. "
                f"Unexpected keys: {unexpected_keys}. Missing keys: {missing_keys}."
            ) from exc

        filtered_state_dict = {
            key: value for key, value in state_dict.items()
            if key not in unexpected_keys
        }
        load_result = model.load_state_dict(filtered_state_dict, strict=False)
        logger.warning(
            "Checkpoint contains time-head weights, but current model has time prediction disabled. "
            "Ignored %d time-head keys: %s",
            len(unexpected_keys),
            unexpected_keys,
        )
        if load_result.missing_keys:
            logger.warning("Missing keys after compatibility load: %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            logger.warning("Unexpected keys after compatibility load: %s", load_result.unexpected_keys)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _extract_core_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    all_slots = metrics.get('classification_all_slots', {}) or {}
    activity_binary_all = metrics.get('activity_binary_all_slots', {}) or {}
    activity_only = metrics.get('classification_activity_only', {}) or {}
    region_iou = metrics.get('region_iou_metrics', {}) or {}
    return {
        'accuracy': float(metrics.get('accuracy', 0.0)),
        'precision': float(metrics.get('precision', 0.0)),
        'recall': float(metrics.get('recall', 0.0)),
        'f1': float(metrics.get('f1', 0.0)),
        'tss': float(metrics.get('tss', 0.0)),
        'classification_activity_only_precision': float(activity_only.get('macro_precision', metrics.get('precision', 0.0))),
        'classification_activity_only_recall': float(activity_only.get('macro_recall', metrics.get('recall', 0.0))),
        'classification_activity_only_f1': float(activity_only.get('macro_f1', metrics.get('f1', 0.0))),
        'classification_activity_only_tss': float(activity_only.get('macro_tss', metrics.get('tss', 0.0))),
        'classification_all_slots_precision': float(all_slots.get('macro_precision', 0.0)),
        'classification_all_slots_recall': float(all_slots.get('macro_recall', 0.0)),
        'classification_all_slots_f1': float(all_slots.get('macro_f1', 0.0)),
        'classification_all_slots_tss': float(all_slots.get('macro_tss', 0.0)),
        'activity_binary_all_slots_precision': float(activity_binary_all.get('precision', 0.0)),
        'activity_binary_all_slots_recall': float(activity_binary_all.get('recall', 0.0)),
        'activity_binary_all_slots_f1': float(activity_binary_all.get('f1', 0.0)),
        'activity_binary_all_slots_tss': float(activity_binary_all.get('tss', 0.0)),
        'activity_binary_all_slots_tp': float(activity_binary_all.get('tp', 0)),
        'activity_binary_all_slots_fp': float(activity_binary_all.get('fp', 0)),
        'activity_binary_all_slots_fn': float(activity_binary_all.get('fn', 0)),
        'activity_binary_all_slots_tn': float(activity_binary_all.get('tn', 0)),
        'region_precision_iou50': float(region_iou.get('precision', 0.0)),
        'region_recall_iou50': float(region_iou.get('recall', 0.0)),
        'region_f1_iou50': float(region_iou.get('f1', 0.0)),
        'iou': float(metrics.get('iou', 0.0)),
        'rss': float(metrics.get('rss', 0.0)),
        'rmse': float(metrics.get('rmse', 0.0)),
    }


def _deduplicate_region_targets(
    target_boxes: np.ndarray,
    region_ids: List[Any],
) -> np.ndarray:
    target_boxes = np.asarray(target_boxes, dtype=np.float32).reshape(-1, 4)
    if target_boxes.size == 0:
        return target_boxes

    unique_boxes: List[np.ndarray] = []
    seen_region_ids = set()
    seen_box_keys = set()

    for box_idx, box in enumerate(target_boxes):
        region_id = ''
        if box_idx < len(region_ids):
            raw_region_id = region_ids[box_idx]
            region_id = '' if raw_region_id is None else str(raw_region_id).strip()

        if region_id:
            if region_id in seen_region_ids:
                continue
            seen_region_ids.add(region_id)
            unique_boxes.append(box)
            continue

        box_key = tuple(np.round(box.astype(np.float64), 6).tolist())
        if box_key in seen_box_keys:
            continue
        seen_box_keys.add(box_key)
        unique_boxes.append(box)

    if not unique_boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(unique_boxes, dtype=np.float32).reshape(-1, 4)


def _extract_raw_slot_targets(
    targets: Dict[str, torch.Tensor],
    sample_idx: int,
) -> np.ndarray:
    gt_mask = (targets['label'][sample_idx] > 0) & targets['activity_mask'][sample_idx]
    if not gt_mask.any():
        return np.zeros((0, 4), dtype=np.float32)
    return targets['bbox'][sample_idx][gt_mask].detach().cpu().numpy()


@torch.no_grad()
def _collect_region_level_iou_details(
    trainer: SolarFlareTrainer,
    loader: DataLoader,
) -> List[Dict[str, Any]]:
    trainer.model.eval()
    details: List[Dict[str, Any]] = []
    dataset_sample_index = 0

    for batch_idx, batch in enumerate(loader):
        inputs = {k: v.to(trainer.device) for k, v in batch['data'].items()}
        if 'proposal_boxes' in batch:
            inputs['proposal_boxes'] = batch['proposal_boxes'].to(trainer.device)
        if 'proposal_scores' in batch:
            inputs['proposal_scores'] = batch['proposal_scores'].to(trainer.device)

        targets = {
            'label': batch['label'].to(trainer.device),
            'bbox': batch['bbox'].to(trainer.device),
            'time_features': batch['time_features'].to(trainer.device),
            'activity_mask': batch.get('activity_mask', torch.ones_like(batch['label'], dtype=torch.bool)).to(trainer.device),
        }

        physics_inputs = batch.get('physics_inputs')
        if physics_inputs is not None:
            physics_inputs = {k: v.to(trainer.device) for k, v in physics_inputs.items()}

        outputs = trainer.model(inputs, physics_inputs, targets=targets)
        _, eval_mask, _, _ = trainer._build_hungarian_supervision_targets(outputs, targets)
        metadata_list = batch.get('metadata', [{} for _ in range(targets['label'].size(0))])

        for sample_idx in range(eval_mask.size(0)):
            sample_mask = eval_mask[sample_idx]
            metadata = metadata_list[sample_idx] if sample_idx < len(metadata_list) else {}

            if sample_mask.any():
                pred_boxes_slot = outputs['bbox_pred'][sample_idx][sample_mask].detach().cpu().numpy()
            else:
                pred_boxes_slot = np.zeros((0, 4), dtype=np.float32)

            target_boxes_slot = _extract_raw_slot_targets(targets, sample_idx)
            region_ids = list(metadata.get('activity_region_ids', []))
            target_boxes_region = _deduplicate_region_targets(target_boxes_slot, region_ids)

            region_metrics = BoundingBoxMetrics.compute_metrics_hungarian(pred_boxes_slot, target_boxes_region)
            slot_metrics = BoundingBoxMetrics.compute_metrics_hungarian(pred_boxes_slot, target_boxes_slot)

            matched_ious_region = [float(v) for v in region_metrics.get('matched_ious', [])]
            target_ious_region = [0.0] * int(region_metrics.get('num_targets', 0))
            for target_idx, matched_iou in enumerate(matched_ious_region[:len(target_ious_region)]):
                target_ious_region[target_idx] = float(matched_iou)

            details.append({
                'dataset_sample_index': int(dataset_sample_index),
                'batch_index': int(batch_idx),
                'batch_sample_index': int(sample_idx),
                'window_id': metadata.get('window_id', ''),
                'event_id': metadata.get('event_id', ''),
                'start_idx': int(metadata.get('start_idx', -1)),
                'end_idx': int(metadata.get('end_idx', -1)),
                'num_gt_boxes_region': int(region_metrics.get('num_targets', 0)),
                'num_gt_boxes_slot_level': int(slot_metrics.get('num_targets', 0)),
                'num_pred_boxes': int(region_metrics.get('num_predictions', 0)),
                'true_positives_iou50': int(region_metrics.get('true_positives', 0)),
                'false_positives_iou50': int(region_metrics.get('false_positives', 0)),
                'false_negatives_iou50': int(region_metrics.get('false_negatives', 0)),
                'sample_average_iou': float(region_metrics.get('average_iou', 0.0)),
                'sample_precision_iou50': float(region_metrics.get('precision', 0.0)),
                'sample_recall_iou50': float(region_metrics.get('recall', 0.0)),
                'sample_f1_iou50': float(region_metrics.get('f1', 0.0)),
                'matched_ious': matched_ious_region,
                'target_ious_for_average': target_ious_region,
                'activity_region_ids_raw': region_ids,
                'pred_boxes': pred_boxes_slot.tolist(),
                'target_boxes_region': target_boxes_region.tolist(),
                'target_boxes_slot_level': target_boxes_slot.tolist(),
                'sample_average_iou_slot_level': float(slot_metrics.get('average_iou', 0.0)),
            })
            dataset_sample_index += 1

    return details


def _build_region_level_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not details:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'average_iou': 0.0,
            'num_predictions': 0,
            'num_targets': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'num_samples': 0,
        }

    total_predictions = int(sum(int(row.get('num_pred_boxes', 0)) for row in details))
    total_targets = int(sum(int(row.get('num_gt_boxes_region', 0)) for row in details))
    total_true_positives = int(sum(int(row.get('true_positives_iou50', 0)) for row in details))
    total_false_positives = int(sum(int(row.get('false_positives_iou50', 0)) for row in details))
    total_false_negatives = int(sum(int(row.get('false_negatives_iou50', 0)) for row in details))
    sample_average_ious = [float(row.get('sample_average_iou', 0.0)) for row in details]

    precision = total_true_positives / (total_true_positives + total_false_positives + 1e-8)
    recall = total_true_positives / (total_true_positives + total_false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    average_iou = float(np.mean(sample_average_ious)) if sample_average_ious else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'average_iou': float(average_iou),
        'num_predictions': total_predictions,
        'num_targets': total_targets,
        'true_positives': total_true_positives,
        'false_positives': total_false_positives,
        'false_negatives': total_false_negatives,
        'num_samples': int(len(details)),
    }


@torch.no_grad()
def _collect_test_per_sample_iou_details(
    trainer: SolarFlareTrainer,
    loader: DataLoader,
) -> List[Dict[str, Any]]:
    return _collect_region_level_iou_details(trainer, loader)


def _save_results(output_dir: Path, results: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / 'evaluation_results_all_splits.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize_for_json(results), f, ensure_ascii=False, indent=2)

    csv_path = output_dir / 'evaluation_summary_all_splits.csv'
    rows = []
    for split_name, split_result in results.items():
        row = {'split': split_name, 'num_samples': split_result.get('num_samples', 0)}
        row.update(split_result.get('core_metrics', {}))
        rows.append(row)

    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    md_path = output_dir / 'evaluation_summary_all_splits.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 模型评估结果汇总\n\n')
        f.write('| split | num_samples | activity_precision | all_slots_precision | activity_binary_precision | activity_binary_recall | region_precision | region_recall | iou | f1 | tss |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for split_name, split_result in results.items():
            m = split_result.get('core_metrics', {})
            f.write(
                f"| {split_name} | {split_result.get('num_samples', 0)} | "
                f"{m.get('classification_activity_only_precision', 0.0):.6f} | "
                f"{m.get('classification_all_slots_precision', 0.0):.6f} | "
                f"{m.get('activity_binary_all_slots_precision', 0.0):.6f} | "
                f"{m.get('activity_binary_all_slots_recall', 0.0):.6f} | "
                f"{m.get('region_precision_iou50', 0.0):.6f} | "
                f"{m.get('region_recall_iou50', 0.0):.6f} | "
                f"{m.get('iou', 0.0):.6f} | {m.get('f1', 0.0):.6f} | "
                f"{m.get('tss', 0.0):.6f} |\n"
            )

    logger.info('评估结果已保存到: %s', output_dir)
    logger.info('JSON: %s', json_path)
    logger.info('CSV : %s', csv_path)
    logger.info('MD  : %s', md_path)


def _save_test_per_sample_iou_details(output_dir: Path, details: List[Dict[str, Any]]) -> None:
    if not details:
        logger.info('No test per-sample IoU details to save.')
        return

    json_path = output_dir / 'test_per_sample_iou_details.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize_for_json(details), f, ensure_ascii=False, indent=2)

    csv_rows = []
    for row in details:
        csv_rows.append({
            'dataset_sample_index': row.get('dataset_sample_index', -1),
            'batch_index': row.get('batch_index', -1),
            'batch_sample_index': row.get('batch_sample_index', -1),
            'window_id': row.get('window_id', ''),
            'event_id': row.get('event_id', ''),
            'start_idx': row.get('start_idx', -1),
            'end_idx': row.get('end_idx', -1),
            'num_gt_boxes_region': row.get('num_gt_boxes_region', 0),
            'num_gt_boxes_slot_level': row.get('num_gt_boxes_slot_level', 0),
            'num_pred_boxes': row.get('num_pred_boxes', 0),
            'true_positives_iou50': row.get('true_positives_iou50', 0),
            'false_positives_iou50': row.get('false_positives_iou50', 0),
            'false_negatives_iou50': row.get('false_negatives_iou50', 0),
            'sample_average_iou': row.get('sample_average_iou', 0.0),
            'sample_average_iou_slot_level': row.get('sample_average_iou_slot_level', 0.0),
            'sample_precision_iou50': row.get('sample_precision_iou50', 0.0),
            'sample_recall_iou50': row.get('sample_recall_iou50', 0.0),
            'sample_f1_iou50': row.get('sample_f1_iou50', 0.0),
            'matched_ious': json.dumps(row.get('matched_ious', []), ensure_ascii=False),
            'target_ious_for_average': json.dumps(row.get('target_ious_for_average', []), ensure_ascii=False),
            'activity_region_ids_raw': json.dumps(row.get('activity_region_ids_raw', []), ensure_ascii=False),
        })

    csv_path = output_dir / 'test_per_sample_iou_details.csv'
    import pandas as pd
    csv_output_path = csv_path
    try:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    except PermissionError:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_output_path = output_dir / f'test_per_sample_iou_details_{timestamp}.csv'
        pd.DataFrame(csv_rows).to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        logger.warning('CSV 文件被占用，已写入备选文件: %s', csv_output_path)

    logger.info('Test per-sample IoU JSON: %s', json_path)
    logger.info('Test per-sample IoU CSV : %s', csv_output_path)


def main(cli_args=None):
    args = parse_args() if cli_args is None else parse_args_from_list(cli_args)

    setup_logging(args.log_dir, 'train_model', debug=args.debug)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_config(args.data_config)['data']
    model_config = load_config(args.model_config)['model']
    train_config = copy.deepcopy(load_config(args.train_config)['training'])
    train_config.setdefault('logging', {})
    train_config['logging']['wandb'] = False

    hdf5_path = args.hdf5_path or data_config.get('hdf5_path', '')
    if not hdf5_path:
        raise ValueError('未提供 hdf5_path，且 data_config 中也没有 hdf5_path')

    device = _resolve_device(args.device)
    logger.info('使用设备: %s', device)
    logger.info('模型路径: %s', args.model_path)
    logger.info('HDF5路径: %s', hdf5_path)

    reader = HDF5DatasetReader(hdf5_path)
    model_cfg = _build_model_config(data_config, model_config, train_config)
    model = MultimodalTransformer(model_cfg)
    _load_model_weights(model, args.model_path, device)

    trainer_config = dict(train_config)
    trainer_config['model'] = model_cfg
    trainer = SolarFlareTrainer(model=model, config=trainer_config, device=device)
    trainer.use_wandb = False

    splits = ['train', 'val', 'test']
    results: Dict[str, Any] = {}
    test_per_sample_iou_details: List[Dict[str, Any]] = []

    for split_name in splits:
        logger.info('开始评估 %s 集...', split_name)
        loader = _create_eval_loader(
            reader=reader,
            data_config=data_config,
            split_name=split_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        metrics = trainer.validate(loader)

        region_iou_details = _collect_region_level_iou_details(trainer, loader)
        region_iou_metrics = _build_region_level_metrics(region_iou_details)
        metrics['iou_slot_level'] = float(metrics.get('iou', 0.0))
        metrics['iou'] = float(region_iou_metrics.get('average_iou', 0.0))
        metrics['region_iou_metrics'] = region_iou_metrics

        core_metrics = _extract_core_metrics(metrics)
        results[split_name] = {
            'num_samples': len(loader.dataset),
            'core_metrics': core_metrics,
            'all_metrics': metrics,
        }

        logger.info(
            '%s | samples=%d | activity_precision=%.4f | all_slots_precision=%.4f | activity_binary_precision=%.4f | activity_binary_recall=%.4f | region_precision=%.4f | iou=%.4f',
            split_name,
            len(loader.dataset),
            core_metrics['classification_activity_only_precision'],
            core_metrics['classification_all_slots_precision'],
            core_metrics['activity_binary_all_slots_precision'],
            core_metrics['activity_binary_all_slots_recall'],
            core_metrics['region_precision_iou50'],
            core_metrics['iou'],
        )

        if split_name == 'test':
            logger.info('Collecting test per-sample region-level IoU details...')
            test_per_sample_iou_details = region_iou_details

    _save_results(output_dir, results)
    _save_test_per_sample_iou_details(output_dir, test_per_sample_iou_details)
    return 0


def parse_args_from_list(cli_args):
    parser = argparse.ArgumentParser(description='一键评估 train/val/test 指标')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--train_config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--hdf5_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation_h')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args(cli_args)


if __name__ == '__main__':
    if USE_DIRECT_RUN_CONFIG:
        sys.exit(main(_build_direct_run_args()))
    sys.exit(main())
