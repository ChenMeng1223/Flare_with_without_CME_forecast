#!/usr/bin/env python3
"""
批量绘制 Halpha 波段图像脚本

遍历 data/raw/downloaded/ 下所有事件的 halpha 文件夹，
读取 FITS 文件，选择时间切片，绘制伪彩色图像（afmhot），保存到 data/processed/。
基于 b1_save_gray_picture.py 和 b2_format_change_with_colour.py，符合项目文件保存规范。

- 按观测时间排序，输出命名优先使用原始 FITS 文件名里的时间（不使用顺序编号）
- 使用 PIL 保存（避免 Windows 上 matplotlib savefig 崩溃）
- 支持灰度/伪彩色两种模式
"""

import os
os.environ.setdefault('NUMEXPR_MAX_THREADS', '32')

from pathlib import Path
from typing import Optional
from datetime import datetime
import logging
import argparse
import re

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _extract_time_str_from_filename(fits_path: Path) -> Optional[str]:
    """
    从原始 FITS 文件名中提取时间字符串，用于输出命名。

    支持常见形式（示例）：
    - 20120913_123456
    - 2012-09-13T12_34_56(.12)Z
    - 2012-09-13T12:34:56
    - 20120913T123456
    """
    name = fits_path.name

    patterns = [
        re.compile(r'(?P<d>\d{8})[T_\- ]?(?P<t>\d{6})'),
        re.compile(
            r'(?P<y>\d{4})[-_](?P<m>\d{2})[-_](?P<day>\d{2})'
            r'[T _\-](?P<h>\d{2})[:_](?P<min>\d{2})[:_](?P<s>\d{2})'
        ),
        re.compile(
            r'(?P<y>\d{4})[_](?P<m>\d{2})[_](?P<day>\d{2})'
            r'[T _\-](?P<h>\d{2})[_:](?P<min>\d{2})[_:](?P<s>\d{2})'
        ),
        re.compile(r'(?P<d>\d{8})[T_\- ]?(?P<t>\d{4})'),
        re.compile(
            r'(?P<y>\d{4})[-_](?P<m>\d{2})[-_](?P<day>\d{2})'
            r'[T _\-](?P<h>\d{2})[:_](?P<min>\d{2})'
        ),
    ]

    for pat in patterns:
        m = pat.search(name)
        if not m:
            continue
        gd = m.groupdict()
        if 'd' in gd and gd.get('d') and gd.get('t'):
            d = gd['d']
            t = gd['t']
            if len(t) == 6:
                return f"{d}_{t}"
            if len(t) == 4:
                return f"{d}_{t}00"
        if gd.get('y') and gd.get('m') and gd.get('day') and gd.get('h') and gd.get('min'):
            d = f"{gd['y']}{gd['m']}{gd['day']}"
            s = gd.get('s') or "00"
            return f"{d}_{gd['h']}{gd['min']}{s}"

    return None


def _unique_path(path: Path) -> Path:
    """当输出文件名冲突时，自动追加后缀避免覆盖。"""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(2, 10000):
        cand = parent / f"{stem}_v{i}{suffix}"
        if not cand.exists():
            return cand
    return parent / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"


def _get_fits_time(fits_path: Path, hdul=None) -> Optional[datetime]:
    """从 FITS 文件获取观测时间（用于排序）"""
    from astropy.io import fits
    from astropy.time import Time

    def _parse_from_hdul(h):
        for key in ('DATE-OBS', 'T_OBS', 'DATE_OBS', 'TSTART'):
            if key in h:
                try:
                    t = Time(h[key])
                    return t.to_datetime()
                except Exception:
                    pass
        return None

    if hdul is not None:
        for hdu in hdul:
            if hdu.header is not None:
                dt = _parse_from_hdul(hdu.header)
                if dt is not None:
                    return dt

    try:
        with fits.open(str(fits_path), mode='readonly') as h:
            return _parse_from_hdul(h[0].header)
    except Exception:
        return None


def plot_halpha(
    fits_path: Path,
    output_path: Path,
    slice_index: Optional[int] = None,
    use_color: bool = True,
    colormap_name: str = 'afmhot',
) -> bool:
    """
    绘制单个 Halpha FITS 文件

    Args:
        fits_path: FITS 文件路径
        output_path: 输出 PNG 路径
        slice_index: 时间切片索引，None 表示使用中间切片
        use_color: True=伪彩色(afmhot)，False=灰度
        colormap_name: 伪彩色时的配色表

    Returns:
        成功返回 True
    """
    try:
        from astropy.io import fits
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from PIL import Image, ImageDraw, ImageFont

        with fits.open(str(fits_path), mode='readonly') as hdu_list:
            # CHASE/Hα 数据通常在 HDU 1
            if len(hdu_list) < 2 or hdu_list[1].data is None:
                logger.warning(f"FITS {fits_path} HDU[1] 无数据")
                return False

            hdu_data = hdu_list[1].data

            if hdu_data.ndim == 3:
                n_slices = hdu_data.shape[0]
                idx = slice_index if slice_index is not None else n_slices // 2
                idx = max(0, min(idx, n_slices - 1))
                image_data = hdu_data[idx, :, :].astype(np.float32)
            elif hdu_data.ndim == 2:
                image_data = hdu_data.astype(np.float32)
            else:
                logger.warning(f"FITS {fits_path} 数据维度异常: {hdu_data.shape}")
                return False

        # 处理 NaN/Inf
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        image_data = np.maximum(image_data, 0.0)

        # 大图降采样（>2048）
        max_side = 2048
        if image_data.shape[0] > max_side or image_data.shape[1] > max_side:
            step = max(image_data.shape[0], image_data.shape[1]) // max_side
            image_data = image_data[::step, ::step]

        if use_color:
            # 伪彩色：b2 逻辑，norm vmin=0, vmax=mean*4
            vmin = 0.0
            vmax = float(image_data.mean() * 4) if image_data.mean() > 0 else float(image_data.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = colormap_name
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            rgba = sm.to_rgba(image_data)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(rgb, 'RGB')
        else:
            # 灰度：b1 逻辑，min-max 归一化
            if image_data.max() > image_data.min():
                normalized = (image_data - image_data.min()) * 255 / (image_data.max() - image_data.min())
            else:
                normalized = np.zeros_like(image_data)
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            img = Image.fromarray(normalized, 'L')

        # 可选添加标题
        try:
            obs_time = _get_fits_time(fits_path)
            time_str = obs_time.strftime('%Y-%m-%d %H:%M:%S') if obs_time else fits_path.stem
        except Exception:
            time_str = fits_path.stem

        try:
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            title = f'Halpha | {time_str}'
            fill = (0, 0, 0) if use_color else 0
            draw.text((8, 8), title, fill=fill, font=font)
        except Exception:
            pass

        img.save(str(output_path), format='PNG')
        return True

    except Exception as e:
        logger.error(f"绘制失败 {fits_path}: {e}")
        return False


def process_event_halpha(
    event_dir: Path,
    processed_base_dir: Path,
    slice_index: Optional[int] = None,
    use_color: bool = True,
    skip_existing: bool = True,
) -> int:
    """处理单个事件的 Halpha 数据"""
    event_id = event_dir.name
    modality_dir = event_dir / 'halpha'

    if not modality_dir.exists():
        logger.warning(f"事件 {event_id} 没有 halpha 文件夹")
        return 0

    output_dir = processed_base_dir / event_id / 'halpha'
    output_dir.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(modality_dir.glob('*.fits')) + sorted(modality_dir.glob('*.fit'))
    if not fits_files:
        logger.warning(f"事件 {event_id} 的 halpha 文件夹为空")
        return 0

    def sort_key(p: Path):
        t = _get_fits_time(p)
        return t if t is not None else datetime.min

    fits_files = sorted(fits_files, key=sort_key)
    logger.info(f"处理事件 {event_id} 的 halpha: {len(fits_files)} 个 FITS 文件")

    success_count = 0
    time_arr = []

    for idx, fits_file in enumerate(fits_files):
        time_str = _extract_time_str_from_filename(fits_file)
        if time_str is None:
            t_guess = _get_fits_time(fits_file)
            time_str = t_guess.strftime('%Y%m%d_%H%M%S') if t_guess else fits_file.stem

        output_file = output_dir / f"Halpha_full_{time_str}.png"
        logger.info(f"  正在处理第 {idx+1}/{len(fits_files)} 个文件: {fits_file.name}")

        if skip_existing and output_file.exists():
            logger.info(f"  文件已存在，跳过: {output_file.name}")
            success_count += 1
            try:
                t = _get_fits_time(fits_file)
                if t:
                    time_arr.append(t.timestamp())
            except Exception:
                pass
            continue

        output_file = _unique_path(output_file)

        if plot_halpha(fits_file, output_file, slice_index=slice_index, use_color=use_color):
            success_count += 1
            try:
                t = _get_fits_time(fits_file)
                if t:
                    time_arr.append(t.timestamp())
            except Exception:
                pass

    if time_arr:
        meta_path = output_dir / "halpha_time.npz"
        np.savez(meta_path, time_arr=np.array(time_arr))

    return success_count


def batch_plot_halpha(
    downloaded_dir: Path,
    processed_dir: Path,
    slice_index: Optional[int] = None,
    use_color: bool = True,
    skip_existing: bool = True,
) -> None:
    """批量绘制所有事件的 Halpha 图像"""
    if not downloaded_dir.exists():
        logger.error(f"下载目录不存在: {downloaded_dir}")
        return

    event_dirs = sorted([d for d in downloaded_dir.iterdir() if d.is_dir() and d.name.startswith('EVT_')])
    if not event_dirs:
        logger.warning("没有找到事件文件夹")
        return

    logger.info(f"找到 {len(event_dirs)} 个事件文件夹")
    total_success = 0

    for event_dir in event_dirs:
        total_success += process_event_halpha(
            event_dir, processed_dir,
            slice_index=slice_index,
            use_color=use_color,
            skip_existing=skip_existing,
        )

    logger.info("==================== 处理完成 ====================")
    logger.info(f"有效文件数：{total_success}")
    logger.info(f"保存路径：{processed_dir}")


def main():
    parser = argparse.ArgumentParser(description='批量绘制 Halpha 波段图像')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml', help='配置文件路径')
    parser.add_argument('--slice', type=int, default=69,
                        help='时间切片索引，默认使用中间切片。Hα 原代码常用 69')
    parser.add_argument('--gray', action='store_true', help='使用灰度模式（默认伪彩色 afmhot）')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已存在的输出文件')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_data_dir = Path(config['data']['raw_data_dir'])
    downloaded_dir = raw_data_dir / 'downloaded'
    processed_dir = Path(config['data']['processed_data_dir'])

    logger.info("开始批量绘制 Halpha 图像...")
    logger.info(f"模式: {'灰度' if args.gray else '伪彩色(afmhot)'}")
    logger.info(f"下载目录: {downloaded_dir}")
    logger.info(f"处理目录: {processed_dir}")

    batch_plot_halpha(
        downloaded_dir, processed_dir,
        slice_index=args.slice,
        use_color=not args.gray,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
