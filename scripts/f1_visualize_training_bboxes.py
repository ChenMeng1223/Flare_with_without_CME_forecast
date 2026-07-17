"""
训练前 bbox 可视化检查脚本（独立于训练环境）

用途：
- 单独加载 train / val 数据集
- 随机抽取事件组或绘制全部事件组
- 将训练阶段实际使用的归一化 bbox 画到对应底图上
- 便于在正式训练前检查 bbox 与图像是否对齐
"""
import sys
import os
import random
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from data.hdf5_reader import HDF5DatasetReader
from data.dataset import SolarFlareDataset
from utils.config_utils import load_config
from utils.logging_utils import setup_logging


logger = logging.getLogger(__name__)
IMAGE_CMAP = 'turbo'
MAGNETOGRAM_MODALITY = 'magnetogram'


# =========================
# 直接运行配置（推荐在 VS Code / Cursor 中直接运行本文件时使用）
# 想切换参数时，直接改这里，无需每次手动输入命令行。
# 当 USE_DIRECT_RUN_CONFIG=True 时，main() 会优先使用这里的配置。
# =========================
USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    'data_config': 'configs/data_config.yaml',
    'train_config': 'configs/training_config.yaml',
    'hdf5_path': '',  # 留空则使用 data_config 中的 hdf5_path
    'output_dir': 'outputs',
    'samples_per_split': 4,
    'seed': 42,
    'splits': ['train'],
    'render_all': True,
    'event_id': 'EVT_20241225_044600_1',
    'render_all_timestamps_for_event': True,
    'log_dir': 'logs',
    'debug': False,
}


def parse_args():
    parser = argparse.ArgumentParser(description='独立运行训练前 bbox 可视化检查')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml',
                        help='数据配置文件路径')
    parser.add_argument('--train_config', type=str, default='configs/training_config.yaml',
                        help='训练配置文件路径（读取 bbox_sanity_check 配置）')
    parser.add_argument('--hdf5_path', type=str, default='',
                        help='HDF5数据集路径；为空时优先使用 data_config 中的 hdf5_path')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录')
    parser.add_argument('--samples_per_split', type=int, default=0,
                        help='每个划分抽样事件组数；0 表示使用 training_config 中的配置')
    parser.add_argument('--seed', type=int, default=-1,
                        help='随机种子；-1 表示使用 training_config 中的配置')
    parser.add_argument('--splits', type=str, nargs='*', default=['train', 'val'],
                        help='要可视化的划分，默认 train val')
    parser.add_argument('--render_all', action='store_true',
                        help='若启用，则绘制指定划分中的全部事件组，而不是随机抽样')
    parser.add_argument('--event_id', type=str, default='',
                        help='指定事件组ID；若提供，则只绘制该事件组')
    parser.add_argument('--render_all_timestamps_for_event', action='store_true',
                        help='配合 --event_id 使用，绘制该事件组内所有时间戳的图像和 bbox')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    return parser.parse_args()


def _build_direct_run_namespace() -> argparse.Namespace:
    cfg = dict(DIRECT_RUN_CONFIG)
    return argparse.Namespace(
        data_config=cfg.get('data_config', 'configs/data_config.yaml'),
        train_config=cfg.get('train_config', 'configs/training_config.yaml'),
        hdf5_path=cfg.get('hdf5_path', ''),
        output_dir=cfg.get('output_dir', 'outputs'),
        samples_per_split=int(cfg.get('samples_per_split', 4) or 4),
        seed=int(cfg.get('seed', 42) or 42),
        splits=list(cfg.get('splits', ['train', 'val']) or ['train', 'val']),
        render_all=bool(cfg.get('render_all', False)),
        event_id=str(cfg.get('event_id', '') or ''),
        render_all_timestamps_for_event=bool(cfg.get('render_all_timestamps_for_event', False)),
        log_dir=cfg.get('log_dir', 'logs'),
        debug=bool(cfg.get('debug', False)),
    )


def _load_split_events(reader: HDF5DatasetReader, data_config: Dict, split_name: str) -> List[str]:
    """从划分文件加载事件ID；若不存在则报错，避免和训练实际划分不一致。"""
    splits_dir = Path(data_config.get('split_data_dir', 'data/split_data'))
    split_file = splits_dir / f'{split_name}_events.txt'
    if not split_file.exists():
        raise FileNotFoundError(f'未找到划分文件: {split_file}')

    with open(split_file, 'r', encoding='utf-8') as f:
        event_ids = [line.strip() for line in f if line.strip()]

    available_event_ids = set(reader.get_event_ids(available_only=True))
    return [event_id for event_id in event_ids if event_id in available_event_ids]


def create_dataset_for_split(hdf5_path: str, data_config: Dict, split_name: str) -> SolarFlareDataset:
    reader = HDF5DatasetReader(hdf5_path)
    split_events = _load_split_events(reader, data_config, split_name)

    logger.info('%s 划分事件组数量: %d', split_name, len(split_events))

    modalities_list = list(data_config['modalities'].keys())
    seq_len = data_config.get('sequence_length', 9)
    first_mod = next(iter(data_config['modalities'].values()), {})
    target_size = tuple(first_mod.get('resolution', [256, 256]))
    stride = data_config.get('stride', 1)

    return SolarFlareDataset(
        reader=reader,
        event_ids=split_events,
        modalities=modalities_list,
        sequence_length=seq_len,
        target_size=target_size,
        stride=stride,
        max_activities=data_config.get('max_activities', 5),
    )


def _find_window_index_for_annotation(dataset: SolarFlareDataset, event_id: str) -> Optional[int]:
    """优先寻找包含标注参考时间戳的窗口；找不到时退回事件首窗口。"""
    event_data = dataset.reader.get_event_data(event_id)
    region_ids = event_data.get('region_ids', [])
    event_dir = Path(dataset.reader.hdf5_path).resolve().parent.parent / 'processed' / event_id / 'bboxes.json'

    annotation_timestamp = ''
    if event_dir.exists():
        try:
            import json
            with open(event_dir, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            regions = raw.get('regions', [])
            target_region_id = region_ids[0] if region_ids else ''
            for region in regions:
                if not target_region_id or str(region.get('region_id', '')) == str(target_region_id):
                    annotation_timestamp = str(region.get('annotation_frame_timestamp', '') or '')
                    if annotation_timestamp:
                        break
        except Exception:
            annotation_timestamp = ''

    fallback_idx = None
    for idx, window_info in enumerate(dataset.windows):
        if str(window_info.get('event_id', '')) != event_id:
            continue
        if fallback_idx is None:
            fallback_idx = idx
        if not annotation_timestamp:
            continue
        timestamps = event_data.get('timestamps', [])[window_info['start_idx']:window_info['end_idx']]
        if annotation_timestamp in timestamps:
            return idx

        if timestamps and annotation_timestamp < timestamps[0]:
            return idx

    return fallback_idx


def _normalize_background_image(background: np.ndarray) -> Optional[np.ndarray]:
    background = np.asarray(background, dtype=np.float32)
    if background.size == 0:
        return None

    bg_min, bg_max = float(np.min(background)), float(np.max(background))
    if bg_max > bg_min:
        background = (background - bg_min) / (bg_max - bg_min)
    return background



def _select_background_from_sample(sample: Dict, dataset: SolarFlareDataset) -> Tuple[Optional[np.ndarray], str]:
    background = None
    background_modality = ''
    for modality in dataset.modalities:
        modality_tensor = sample['data'].get(modality)
        if modality_tensor is None:
            continue
        modality_np = modality_tensor.detach().cpu().numpy()
        if modality_np.ndim == 4 and modality_np.shape[0] > 0:
            background = modality_np[-1, 0]
            background_modality = modality
            break
        if modality_np.ndim == 3 and modality_np.shape[0] > 0:
            background = modality_np[-1]
            background_modality = modality
            break
    if background is None:
        return None, ''

    background = _normalize_background_image(background)
    if background is None:
        return None, background_modality
    return background, background_modality



def _get_display_kwargs(modality: str, background: np.ndarray) -> Dict:
    if modality == MAGNETOGRAM_MODALITY:
        if background.ndim >= 3:
            background = np.asarray(background)
            if background.shape[-1] == 3:
                background = background[..., 0]
            elif background.shape[-1] == 4:
                background = background[..., 0]
        return {'image': background, 'imshow_kwargs': {'cmap': 'gray', 'origin': 'upper'}}
    return {'image': background, 'imshow_kwargs': {'cmap': IMAGE_CMAP, 'origin': 'upper'}}



def _extract_annotations_from_sample(sample: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    metadata = sample.get('metadata', {})
    num_activities = int(metadata.get('num_activities', 0))
    labels = sample['label'].detach().cpu().numpy()
    bboxes = sample['bbox'].detach().cpu().numpy()
    activity_mask = sample.get('activity_mask')
    if activity_mask is not None:
        activity_mask = activity_mask.detach().cpu().numpy().astype(bool)
    else:
        activity_mask = np.ones(len(labels), dtype=bool)
    return labels, bboxes, activity_mask, num_activities



def _draw_bbox_annotations(
        ax,
        background: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        activity_mask: np.ndarray) -> int:
    if background.ndim >= 3:
        h, w = background.shape[0], background.shape[1]
    else:
        h, w = background.shape[-2], background.shape[-1]
    drawn = 0
    for slot_idx, (label, bbox, valid) in enumerate(zip(labels, bboxes, activity_mask)):
        if not valid or int(label) <= 0:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        rect = patches.Rectangle(
            (x1 * w, y1 * h),
            max((x2 - x1) * w, 1.0),
            max((y2 - y1) * h, 1.0),
            linewidth=2,
            edgecolor='red' if int(label) == 1 else 'yellow',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1 * w,
            max(y1 * h - 4, 2),
            f'slot={slot_idx} cls={int(label)}',
            color='white',
            fontsize=8,
            bbox=dict(facecolor='black', alpha=0.65, pad=2)
        )
        drawn += 1
    return drawn



def _prepare_raw_modality_frames(frames: np.ndarray, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if frames is None:
        return None
    frames = np.asarray(frames)
    if frames.size == 0:
        return None
    if frames.ndim != 3:
        return None
    if frames.shape[-2:] != target_size:
        from scipy.ndimage import zoom
        zoom_factors = [1, target_size[0] / frames.shape[1], target_size[1] / frames.shape[2]]
        frames = zoom(frames, zoom_factors, order=1)
    return np.asarray(frames, dtype=np.float32)



def _load_processed_image(image_path: Path) -> Optional[np.ndarray]:
    try:
        image = Image.open(image_path)
        return np.asarray(image)
    except Exception as exc:
        logger.warning('读取图片失败 %s: %s', image_path, exc)
        return None



def _parse_image_time_from_filename(fname: str) -> Optional[datetime]:
    m = re.search(r'(\d{8})[_\-\sT]?(\d{6})', fname)
    if m:
        try:
            return datetime.strptime(f'{m.group(1)}_{m.group(2)}', '%Y%m%d_%H%M%S')
        except ValueError:
            pass

    m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})[T_\s](\d{2})[:_](\d{2})[:_](\d{2})', fname)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6))
            )
        except ValueError:
            pass

    m = re.search(r'(\d{8})[_\-\sT]?(\d{4})', fname)
    if m:
        try:
            return datetime.strptime(f'{m.group(1)}_{m.group(2)}00', '%Y%m%d_%H%M%S')
        except ValueError:
            pass

    m = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_\s](\d{2})[:_](\d{2})[:_]?(\d{4})', fname)
    if m:
        try:
            sec = int(m.group(6)[:2]) if len(m.group(6)) >= 2 else 0
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), sec
            )
        except ValueError:
            pass

    return None



def _parse_dataset_timestamp(timestamp: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(timestamp).replace('Z', ''))
    except Exception:
        return None



def _select_closest_processed_images(image_paths: List[Path], timestamps: List[str]) -> List[Tuple[str, Path]]:
    parsed_images = []
    for image_path in image_paths:
        image_dt = _parse_image_time_from_filename(image_path.name)
        if image_dt is not None:
            parsed_images.append((image_path, image_dt))

    if not parsed_images:
        return []

    selected: List[Tuple[str, Path]] = []
    used_paths = set()
    for timestamp in timestamps:
        target_dt = _parse_dataset_timestamp(timestamp)
        if target_dt is None:
            continue

        closest_path = None
        closest_delta = None
        for image_path, image_dt in parsed_images:
            delta = abs((image_dt - target_dt).total_seconds())
            if closest_delta is None or delta < closest_delta:
                closest_path = image_path
                closest_delta = delta

        if closest_path is None or str(closest_path) in used_paths:
            continue

        used_paths.add(str(closest_path))
        selected.append((str(timestamp), closest_path))

    return selected



def _draw_processed_image_bbox_figure(
        image_path: Path,
        modality: str,
        split_name: str,
        event_id: str,
        timestamp: str,
        labels: np.ndarray,
        bboxes: np.ndarray,
        activity_mask: np.ndarray,
        num_activities: int,
        save_path: Path) -> bool:
    background = _load_processed_image(image_path)
    if background is None or background.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    display = _get_display_kwargs(modality, background)
    ax.imshow(display['image'], **display['imshow_kwargs'])
    drawn = _draw_bbox_annotations(ax, display['image'], labels, bboxes, activity_mask)

    ax.set_title(
        f'{split_name} | {event_id} | mod={modality} | acts={num_activities} | drawn={drawn}\n'
        f'ts={timestamp} | image={image_path.name}'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('已保存 %s processed 图片 bbox 检查图: %s', split_name, save_path)
    return True



def _find_processed_event_dir(data_config: Dict, event_id: str) -> Path:
    return Path(data_config.get('processed_data_dir', 'data/processed')) / event_id



def _list_processed_modality_images(modality_dir: Path) -> List[Path]:
    if not modality_dir.exists() or not modality_dir.is_dir():
        return []
    allowed_suffixes = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
    return sorted(
        [path for path in modality_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed_suffixes],
        key=lambda path: path.name,
    )



def _draw_sample_bbox_figure(
        sample: Dict,
        dataset: SolarFlareDataset,
        split_name: str,
        event_id: str,
        title_suffix: str,
        save_path: Path) -> bool:
    labels, bboxes, activity_mask, num_activities = _extract_annotations_from_sample(sample)

    background, background_modality = _select_background_from_sample(sample, dataset)
    if background is None:
        logger.warning('事件组 %s 无可用背景图，跳过: %s', event_id, title_suffix)
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    display = _get_display_kwargs(background_modality, background)
    ax.imshow(display['image'], **display['imshow_kwargs'])
    drawn = _draw_bbox_annotations(ax, display['image'], labels, bboxes, activity_mask)

    ax.set_title(
        f'{split_name} | {event_id} | mod={background_modality} | acts={num_activities} | drawn={drawn}\n'
        f'{title_suffix}'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('已保存 %s bbox 检查图: %s', split_name, save_path)
    return True



def _draw_raw_frame_bbox_figure(
        background: np.ndarray,
        modality: str,
        split_name: str,
        event_id: str,
        timestamp: str,
        labels: np.ndarray,
        bboxes: np.ndarray,
        activity_mask: np.ndarray,
        num_activities: int,
        save_path: Path) -> bool:
    normalized_background = _normalize_background_image(background)
    if normalized_background is None:
        logger.warning('事件组 %s 在 %s / %s 无可用背景图，跳过', event_id, modality, timestamp)
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    display = _get_display_kwargs(modality, normalized_background)
    ax.imshow(display['image'], **display['imshow_kwargs'])
    drawn = _draw_bbox_annotations(ax, display['image'], labels, bboxes, activity_mask)

    ax.set_title(
        f'{split_name} | {event_id} | mod={modality} | acts={num_activities} | drawn={drawn}\n'
        f'timestamp={timestamp}'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('已保存 %s 原始帧 bbox 检查图: %s', split_name, save_path)
    return True



def visualize_event_all_timestamps(
        dataset: SolarFlareDataset,
        split_name: str,
        output_dir: Path,
        event_id: str,
        data_config: Dict) -> None:
    event_windows = [
        (idx, window_info)
        for idx, window_info in enumerate(dataset.windows)
        if str(window_info.get('event_id', '')) == event_id
    ]
    if not event_windows:
        logger.warning('%s 划分中未找到事件组 %s，跳过逐时间戳绘制', split_name, event_id)
        return

    processed_event_dir = _find_processed_event_dir(data_config, event_id)
    if not processed_event_dir.exists():
        logger.warning('未找到事件组 %s 的 processed 目录: %s', event_id, processed_event_dir)
        return

    reference_sample = dataset[event_windows[0][0]]
    labels, bboxes, activity_mask, num_activities = _extract_annotations_from_sample(reference_sample)
    event_data = dataset.reader.get_event_data(event_id, modalities=[])
    timestamps = list(event_data.get('timestamps', []))
    if not timestamps:
        logger.warning('事件组 %s 没有可用时间戳，跳过 processed 图片 bbox 绘制', event_id)
        return
    preview_dir = output_dir / 'bbox_sanity_check' / split_name / event_id

    rendered_count = 0
    for modality in dataset.modalities:
        modality_dir = processed_event_dir / modality
        image_paths = _list_processed_modality_images(modality_dir)
        if not image_paths:
            logger.warning('事件组 %s 在模态 %s 下没有可用 processed 图片，跳过该模态', event_id, modality)
            continue

        matched_images = _select_closest_processed_images(image_paths, timestamps)
        if not matched_images:
            logger.warning('事件组 %s 在模态 %s 下没有能匹配时间戳的 processed 图片，跳过该模态', event_id, modality)
            continue

        save_modality_dir = preview_dir / modality
        for image_idx, (timestamp, image_path) in enumerate(matched_images, start=1):
            save_path = save_modality_dir / f'{image_idx:03d}_{image_path.name}'
            saved = _draw_processed_image_bbox_figure(
                image_path=image_path,
                modality=modality,
                split_name=split_name,
                event_id=event_id,
                timestamp=timestamp,
                labels=labels,
                bboxes=bboxes,
                activity_mask=activity_mask,
                num_activities=num_activities,
                save_path=save_path,
            )
            if saved:
                rendered_count += 1

    logger.info('事件组 %s 基于 processed 图片绘制完成，共生成 %d 张图', event_id, rendered_count)



def visualize_dataset_bbox_samples(
        dataset: SolarFlareDataset,
        split_name: str,
        output_dir: Path,
        data_config: Dict,
        sample_count: int = 4,
        seed: int = 42,
        render_all: bool = False,
        event_id: str = '',
        render_all_timestamps_for_event: bool = False) -> None:
    """随机抽取事件组或绘制全部事件组，画出训练阶段真实使用的 bbox。"""
    if len(dataset) == 0:
        logger.warning('%s 数据集为空，跳过 bbox 可视化检查', split_name)
        return

    event_to_index: Dict[str, int] = {}
    for current_event_id in sorted({str(window_info.get('event_id', '')) for window_info in dataset.windows if window_info.get('event_id', '')}):
        matched_idx = _find_window_index_for_annotation(dataset, current_event_id)
        if matched_idx is not None:
            event_to_index[current_event_id] = matched_idx

    if not event_to_index:
        logger.warning('%s 数据集未找到可视化事件，跳过 bbox 可视化检查', split_name)
        return

    if event_id:
        if event_id not in event_to_index:
            logger.warning('%s 划分中未找到事件组 %s，跳过', split_name, event_id)
            return
        if render_all_timestamps_for_event:
            visualize_event_all_timestamps(
                dataset=dataset,
                split_name=split_name,
                output_dir=output_dir,
                event_id=event_id,
                data_config=data_config,
            )
            return
        selected_event_ids = [event_id]
    else:
        rng = random.Random(seed)
        event_ids = list(event_to_index.keys())
        if render_all:
            selected_event_ids = sorted(event_ids)
        else:
            selected_event_ids = rng.sample(event_ids, k=min(sample_count, len(event_ids)))

    preview_dir = output_dir / 'bbox_sanity_check' / split_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    for order, current_event_id in enumerate(selected_event_ids, start=1):
        sample_idx = event_to_index[current_event_id]
        sample = dataset[sample_idx]
        window_info = dataset.windows[sample_idx]
        event_data = dataset.reader.get_event_data(current_event_id)
        window_timestamps = event_data.get('timestamps', [])[window_info['start_idx']:window_info['end_idx']]
        ref_timestamp = window_timestamps[-1] if window_timestamps else ''
        title_suffix = f'window=[{window_info["start_idx"]},{window_info["end_idx"]}) | ref={ref_timestamp}'
        save_path = preview_dir / f'{order:02d}_{current_event_id}.png'
        _draw_sample_bbox_figure(
            sample=sample,
            dataset=dataset,
            split_name=split_name,
            event_id=current_event_id,
            title_suffix=title_suffix,
            save_path=save_path,
        )


def main():
    args = _build_direct_run_namespace() if USE_DIRECT_RUN_CONFIG else parse_args()
    setup_logging(args.log_dir, 'bbox_sanity_check', debug=args.debug)

    data_config = load_config(args.data_config)['data']
    train_config = load_config(args.train_config).get('training', {})
    sanity_cfg = train_config.get('bbox_sanity_check', {})

    hdf5_path = args.hdf5_path or data_config.get('hdf5_path', '')
    if not hdf5_path:
        raise ValueError('未提供 hdf5_path，且 data_config 中未配置 hdf5_path')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = args.samples_per_split if args.samples_per_split > 0 else int(sanity_cfg.get('samples_per_split', 4) or 4)
    seed = args.seed if args.seed >= 0 else int(sanity_cfg.get('seed', 42) or 42)

    logger.info('开始独立 bbox 可视化检查')
    logger.info('运行模式: %s', 'direct_run_config' if USE_DIRECT_RUN_CONFIG else 'cli_args')
    logger.info('HDF5: %s', hdf5_path)
    logger.info('输出目录: %s', output_dir)
    logger.info('每个划分抽样事件组数: %d', sample_count)
    logger.info('随机种子: %d', seed)
    logger.info('绘制全部事件组: %s', args.render_all)
    logger.info('指定事件组: %s', args.event_id or '未指定')
    logger.info('指定事件组绘制全部时间戳: %s', args.render_all_timestamps_for_event)
    logger.info('目标划分: %s', args.splits)

    for split_offset, split_name in enumerate(args.splits):
        dataset = create_dataset_for_split(hdf5_path, data_config, split_name)
        visualize_dataset_bbox_samples(
            dataset=dataset,
            split_name=split_name,
            output_dir=output_dir,
            data_config=data_config,
            sample_count=sample_count,
            seed=seed + split_offset,
            render_all=args.render_all,
            event_id=args.event_id,
            render_all_timestamps_for_event=args.render_all_timestamps_for_event,
        )

    logger.info('独立 bbox 可视化检查完成')


if __name__ == '__main__':
    main()
