#!/usr/bin/env python3
"""
为每个事件组生成 bboxes.json 文件（region 级假 bbox，供后续人工标注替换）

- 以主事件的时间窗（pre_event_hours + post_event_hours）为范围
- 找出该时间窗内发生的所有 activity（主事件 + 重叠的其他事件）
- 按 active region id 聚合成多个 region，每个 region 只生成一个假 bbox
- 输出结构为 regions + activities，便于后续 GUI 标注与训练
- 默认只为 events 文件中列出的事件生成，并跳过已存在的 bboxes.json，避免覆盖已标注结果
"""
import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_utils import setup_logging
from utils.active_region_utils import (
    attach_active_region_columns,
    build_region_key,
    raise_if_unknown_active_regions,
)
from utils.event_id_utils import ensure_event_ids_like_downloader, normalize_event_time_columns

logger = logging.getLogger(__name__)

# 图像分辨率，用于生成合法范围内的 bbox
DEFAULT_RESOLUTION = (256, 256)


def _get_cme_column(df: pd.DataFrame) -> str:
    """获取 CME 相关列名，支持多种拼写"""
    for col in ['cme_associated', 'CME_asociated', 'CME_associated', 'cme_asociated']:
        if col in df.columns:
            return col
    raise ValueError("事件表中未找到 CME 相关列 (cme_associated / CME_asociated)")


def _cme_to_label(value) -> int:
    """将 CME 列值映射为 label: yes->1(爆发耀斑), no->2(束缚耀斑)"""
    if pd.isna(value):
        return 2
    s = str(value).strip().lower()
    if s in ('yes', 'true', '1', 'y'):
        return 1
    if s in ('no', 'false', '0', 'n'):
        return 2
    return 2


def _parse_event_times(df: pd.DataFrame) -> pd.DataFrame:
    """解析事件时间，支持 DATE+start_time/end_time 或完整 ISO 格式"""
    df = df.copy()

    if 'DATE' in df.columns and 'start_time' in df.columns and 'end_time' in df.columns:
        for idx, row in df.iterrows():
            date_str = str(row.get('DATE', '')).strip()
            if not date_str or date_str.lower() == 'nan':
                continue
            try:
                parts = date_str.split('.')
                if len(parts) == 3:
                    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                    base = datetime(y, m, d)
                else:
                    base = pd.to_datetime(date_str).to_pydatetime()
            except Exception:
                continue

            for col, out_col in [('start_time', 'start_dt'), ('end_time', 'end_dt')]:
                v = row[col]
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if '(+1day)' in s:
                    base_adj = base + timedelta(days=1)
                    s = s.replace('(+1day)', '').strip()
                elif '(+2day)' in s:
                    base_adj = base + timedelta(days=2)
                    s = s.replace('(+2day)', '').strip()
                else:
                    base_adj = base
                try:
                    t = datetime.strptime(s[:5], '%H:%M').time()
                    df.at[idx, out_col] = datetime.combine(base_adj.date(), t)
                except Exception:
                    try:
                        df.at[idx, out_col] = pd.to_datetime(v).to_pydatetime()
                    except Exception:
                        pass
        if 'start_dt' in df.columns and 'end_dt' in df.columns:
            return df

    if 'start_time' in df.columns and 'end_time' in df.columns:
        def _to_dt(v):
            if pd.isna(v):
                return None
            s = str(v).replace('Z', '').replace('+00:00', '').strip()
            try:
                return pd.to_datetime(s).to_pydatetime()
            except Exception:
                return None

        df['start_dt'] = df['start_time'].apply(_to_dt)
        df['end_dt'] = df['end_time'].apply(_to_dt)
        return df

    raise ValueError("事件表需包含 start_time/end_time 或 DATE+start_time/end_time")


def _generate_fake_bbox(resolution: Tuple[int, int], rng: np.random.Generator) -> List[int]:
    """在图像范围内生成随机假 bbox [xmin, ymin, xmax, ymax]"""
    H, W = resolution
    margin = min(20, W // 8, H // 8)
    x1 = rng.integers(margin, max(margin + 1, W - margin - 30))
    y1 = rng.integers(margin, max(margin + 1, H - margin - 30))
    x2 = rng.integers(x1 + 20, min(W - margin, x1 + 80))
    y2 = rng.integers(y1 + 20, min(H - margin, y1 + 80))
    x2 = min(x2, W - 1)
    y2 = min(y2, H - 1)
    return [int(x1), int(y1), int(x2), int(y2)]




def load_events(
    events_file: str,
    config: Dict,
    *,
    allow_unknown_active_regions: bool = False,
) -> pd.DataFrame:
    """加载并规范化事件表，并生成与 downloaded/processed 一致的 event_id"""
    if events_file.endswith('.xlsx'):
        df = pd.read_excel(events_file)
    elif events_file.endswith('.csv'):
        df = pd.read_csv(events_file)
    else:
        raise ValueError(f"不支持的文件格式: {events_file}")

    df = _parse_event_times(df)
    df = normalize_event_time_columns(df)
    df = ensure_event_ids_like_downloader(df)
    df = attach_active_region_columns(df)
    df = df.dropna(subset=['start_dt', 'end_dt'])
    if not allow_unknown_active_regions:
        raise_if_unknown_active_regions(df, context="生成 bbox.json 前检查：")
    return df


def _build_group_regions_and_activities(
    group_events: pd.DataFrame,
    primary_event_id: str,
    resolution: Tuple[int, int],
    rng: np.random.Generator,
    cme_col: str,
) -> Tuple[List[Dict], List[Dict], str]:
    regions: List[Dict] = []
    activities: List[Dict] = []
    region_index: Dict[str, int] = {}
    primary_region_key = ''

    for _, row_ev in group_events.iterrows():
        ev_id = str(row_ev['event_id'])
        region_key = build_region_key(row_ev.get('active_region_id', 'UNKNOWN'), ev_id)
        region_position = str(row_ev.get('active_region_position', '') or '')
        is_primary_activity = (ev_id == primary_event_id)

        if region_key not in region_index:
            region_index[region_key] = len(regions)
            regions.append({
                'region_id': region_key,
                'region_position': region_position,
                'is_primary_region': False,
                'bbox': _generate_fake_bbox(resolution, rng),
            })

        if is_primary_activity:
            primary_region_key = region_key

        activities.append({
            'event_id': ev_id,
            'is_primary_activity': is_primary_activity,
            'label': _cme_to_label(row_ev[cme_col]),
            'region_id': region_key,
            'active_region_source': str(row_ev.get('active_region_source', '') or ''),
            'active_region_position': region_position,
        })

    for region in regions:
        region['is_primary_region'] = (region['region_id'] == primary_region_key)

    return regions, activities, primary_region_key


def generate_bboxes_for_all_events(
    events_file: str,
    config: Dict,
    image_base_dir: Path,
    resolution: Tuple[int, int],
    seed: int = 42,
    overwrite_existing: bool = False,
    allow_unknown_active_regions: bool = False,
) -> int:
    """为每个事件组生成 region/activity 结构的 bboxes.json"""
    df = load_events(
        events_file,
        config,
        allow_unknown_active_regions=allow_unknown_active_regions,
    )
    cme_col = _get_cme_column(df)

    time_config = config.get('time', {})
    pre_hours = time_config.get('pre_event_hours', 6)
    post_hours = time_config.get('post_event_hours', 3)

    rng = np.random.default_rng(seed)
    count = 0

    for _, row_primary in df.iterrows():
        event_id = str(row_primary['event_id'])
        start_dt = row_primary['start_dt']

        extended_start = start_dt - timedelta(hours=pre_hours)
        extended_end = start_dt + timedelta(hours=post_hours)

        mask = (
            (df['start_dt'] <= extended_end) &
            (df['end_dt'] >= extended_start)
        )
        group_events = df[mask].copy().sort_values('start_dt').reset_index(drop=True)

        regions, activities, primary_region_id = _build_group_regions_and_activities(
            group_events, event_id, resolution, rng, cme_col
        )

        out = {
            'primary_event_id': event_id,
            'primary_region_id': primary_region_id,
            'bbox_resolution': {'width': int(resolution[1]), 'height': int(resolution[0])},
            'annotated': False,
            'regions': regions,
            'activities': activities,
        }

        out_dir = image_base_dir / event_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'bboxes.json'
        if out_path.exists() and not overwrite_existing:
            logger.info(f'跳过已存在的标注文件: {out_path}')
            continue

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        count += 1
        logger.debug(f"已生成 {out_path}")

    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description='为每个事件组生成 region/activity 结构的假 bbox json，便于后续人工标注替换'
    )
    parser.add_argument('--config', type=str, default='configs/data_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--events', type=str, default="data/raw/events_plot.xlsx",
                        help='事件文件路径，不指定则使用配置文件中的 events_file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='图像根目录，不指定则使用 data/processed（每个事件一个子文件夹）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--overwrite_existing', action='store_true',
                        help='覆盖已存在的 bboxes.json；默认跳过，避免覆盖已标注结果')
    parser.add_argument('--allow_unknown_active_regions', action='store_true',
                        help='允许活动区编号未知的事件继续生成 bbox.json（不推荐）')
    parser.add_argument('--log_dir', type=str, default='logs')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_dir=args.log_dir, level='INFO')

    with open(args.config, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    config = full_config.get('data', full_config)

    events_file = args.events or config.get('events_file')
    if not events_file:
        raise ValueError('请通过 --events 或配置文件指定 events_file')

    processed_dir = Path(config.get('processed_data_dir', 'data/processed'))
    image_base_dir = Path(args.output_dir) if args.output_dir else processed_dir

    modalities = config.get('modalities', {})
    resolution = DEFAULT_RESOLUTION
    if modalities:
        first_mod = next(iter(modalities.values()), {})
        res = first_mod.get('resolution', [256, 256])
        resolution = (int(res[0]), int(res[1]))

    logger.info(f'事件文件: {events_file}')
    logger.info(f'输出目录: {image_base_dir}')
    logger.info(f'分辨率: {resolution}')
    logger.info(f'已存在 bbox 处理策略: {"覆盖" if args.overwrite_existing else "跳过"}')

    n = generate_bboxes_for_all_events(
        events_file, config, image_base_dir, resolution,
        seed=args.seed,
        overwrite_existing=args.overwrite_existing,
        allow_unknown_active_regions=args.allow_unknown_active_regions,
    )
    logger.info(f'已为 {n} 个事件组生成 bboxes.json')


if __name__ == '__main__':
    main()
