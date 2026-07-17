#!/usr/bin/env python3
"""
批量绘制EUV波段完整图像脚本（94Å、171Å、193Å）

遍历 data/raw/downloaded/ 下所有事件的 euv_94、euv_171、euv_193 文件夹，
读取FITS文件，绘制完整图像（对数拉伸），并保存到 data/processed/ 下。
基于IDL代码 plot_aia193_fullmap 的 Python 实现。

功能：
- 曝光时间标准化（消除不同曝光时长的亮度差异）
- 按观测时间排序，输出命名优先使用原始 FITS 文件名里的时间（不使用顺序编号）
- 保存 duration/time 元数据（.npz，便于后续分析）
- 可选保存标准化数据（.npy）
"""

import os
# 在导入 numpy/astropy/sunpy 之前设置，避免 NumExpr 线程数警告
os.environ.setdefault('NUMEXPR_MAX_THREADS', '32')

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import yaml
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================== 波段参数（与IDL一致）========================
BAND_CONFIG = {
    94: {
        'dmax': 10000.0,
        'dmin': 1.0,
        'cmap': 'sdoaia94',
        'label': '94Å',
    },
    171: {
        'dmax': 5000.0,
        'dmin': 5.0,
        'cmap': 'sdoaia171',
        'label': '171Å',
    },
    193: {
        'dmax': 8000.0,
        'dmin': 10.0,
        'cmap': 'sdoaia193',
        'label': '193Å',
    },
}


def _get_fits_time(fits_path: Path, hdul=None) -> Optional[datetime]:
    """从FITS文件获取观测时间（用于排序）"""
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
        # 20120913_123456 / 20120913T123456 / 20120913-123456
        re.compile(r'(?P<d>\d{8})[T_\- ]?(?P<t>\d{6})'),
        # 2012-09-13T12_34_56(.12)Z / 2012-09-13 12:34:56
        re.compile(
            r'(?P<y>\d{4})[-_](?P<m>\d{2})[-_](?P<day>\d{2})'
            r'[T _\-](?P<h>\d{2})[:_](?P<min>\d{2})[:_](?P<s>\d{2})'
        ),
        # 2012_09_13T12_34_56
        re.compile(
            r'(?P<y>\d{4})[_](?P<m>\d{2})[_](?P<day>\d{2})'
            r'[T _\-](?P<h>\d{2})[_:](?P<min>\d{2})[_:](?P<s>\d{2})'
        ),
        # 20120913_1234 (无秒)
        re.compile(r'(?P<d>\d{8})[T_\- ]?(?P<t>\d{4})'),
        # 2012-09-13T12_34 (无秒)
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


def _get_exposure_time(fits_path: Path, aia_map=None) -> float:
    """获取曝光时间（秒），若无法获取则返回 1.0"""
    if aia_map is not None and hasattr(aia_map, 'exposure_time') and aia_map.exposure_time is not None:
        try:
            v = float(aia_map.exposure_time.value)
            return v if v > 0 else 1.0
        except Exception:
            pass

    from astropy.io import fits
    try:
        with fits.open(str(fits_path), mode='readonly') as hdul:
            h = hdul[0].header
            for key in ('EXPTIME', 'EXPOSURE', 'EXPOSUR'):
                if key in h:
                    v = float(h[key])
                    return v if v > 0 else 1.0
    except Exception:
        pass
    return 1.0


def plot_euv(
    fits_path: Path,
    output_path: Path,
    wavelength: int,
    save_npy: bool = False,
    npy_path: Optional[Path] = None,
) -> Tuple[bool, Optional[float], Optional[datetime]]:
    """
    绘制单个 EUV FITS 文件（完整图像，对数拉伸）

    Returns:
        (success, original_exposure, obs_time)
    """
    cfg = BAND_CONFIG.get(wavelength, BAND_CONFIG[193])
    dmax = cfg['dmax']
    dmin = cfg['dmin']
    cmap_name = cfg['cmap']
    label = cfg['label']

    sunpy_available = False
    aia_map = None
    obs_time = None
    data = None

    try:
        # 优先使用 astropy（兼容性更好，避免 Path/ostream 相关问题）
        from astropy.io import fits
        try:
            with fits.open(str(fits_path), mode='readonly') as hdul:
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = np.array(hdul[1].data, dtype=float)
                elif hdul[0].data is not None:
                    data = np.array(hdul[0].data, dtype=float)
                if data is not None:
                    obs_time = _get_fits_time(fits_path, hdul)
        except Exception as e:
            logger.warning(f"Astropy 读取失败: {e}")

        # astropy 失败时尝试 sunpy
        if data is None:
            try:
                import sunpy.map
                from sunpy.visualization import colormaps
                sunpy_available = True
                aia_map = sunpy.map.Map(str(fits_path))
                data = np.array(aia_map.data, dtype=float)
                if hasattr(aia_map, 'date') and aia_map.date is not None:
                    obs_time = aia_map.date.to_datetime()
            except Exception as e:
                logger.warning(f"Sunpy 读取失败: {e}")
            if data is not None:
                sunpy_available = True

        if data is None:
            logger.warning(f"FITS 文件 {fits_path} 无法读取")
            return False, None, None

        # sunpy 可用时使用 AIA 配色表
        if not sunpy_available:
            try:
                import sunpy.visualization.colormaps
                sunpy_available = True
            except ImportError:
                pass

        # 获取原始曝光时间
        original_exposure = _get_exposure_time(fits_path, aia_map)
        if original_exposure <= 0:
            logger.warning(f"跳过损坏文件（曝光时长<=0）: {fits_path}")
            return False, None, None

        # 曝光时间标准化（与 IDL 一致）
        data = data / original_exposure
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.maximum(data, dmin)

        # 可选保存 .npy（降采样前保存完整分辨率）
        if save_npy and npy_path is not None:
            np.save(npy_path, data)

        # 大图降采样（>2048 时缩小，避免内存/渲染问题）
        max_side = 2048
        if data.shape[0] > max_side or data.shape[1] > max_side:
            step = max(data.shape[0], data.shape[1]) // max_side
            data = data[::step, ::step]

        # 对数拉伸
        norm = colors.LogNorm(vmin=dmin, vmax=dmax, clip=True)

        # 配色表（sunpy 提供 sdoaia94/171/193，纯 astropy 时用 hot）
        cmap = cmap_name if sunpy_available else 'hot'

        # 使用 PIL 保存（避免 Windows 上 matplotlib savefig 崩溃）
        try:
            from matplotlib.cm import ScalarMappable
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            rgba = sm.to_rgba(data)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            from PIL import Image
            img = Image.fromarray(rgb)
            if obs_time is not None:
                time_str = obs_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = fits_path.stem
            # 可选：在图像顶部留白并添加标题（PIL 文本）
            from PIL import ImageDraw, ImageFont
            try:
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                title = f'AIA {label} | {time_str}'
                draw.text((8, 8), title, fill=(0, 0, 0), font=font)
            except Exception:
                pass
            img.save(str(output_path), format='PNG')
        except Exception as save_err:
            # 备用：尝试 matplotlib savefig
            logger.warning(f"PIL 保存失败，尝试 matplotlib: {save_err}")
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
            ax.imshow(data, cmap=cmap, norm=norm, origin='lower')
            ax.axis('off')
            if obs_time is not None:
                time_str = obs_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = fits_path.stem
            ax.set_title(f'AIA {label} | {time_str}', fontsize=8)
            plt.savefig(str(output_path), dpi=30, bbox_inches='tight', facecolor='white')
            plt.close()

        return True, original_exposure, obs_time

    except Exception as e:
        logger.error(f"绘制失败 {fits_path}: {e}")
        return False, None, None


def process_event_euv(
    event_dir: Path,
    processed_base_dir: Path,
    config: Dict[str, Any],
    save_npy: bool = False,
    skip_existing: bool = True,
    bands: Optional[List[str]] = None,
) -> int:
    """处理单个事件的所有 EUV 波段数据

    Args:
        bands: 指定波段，如 ['euv_171'] 或 ['euv_94','euv_171','euv_193']，None 表示全部
    """
    event_id = event_dir.name
    euv_modalities = ['euv_94', 'euv_171', 'euv_193']
    if bands is not None:
        euv_modalities = [b for b in euv_modalities if b in bands]
    modalities_cfg = config.get('data', {}).get('modalities', {})

    total_success = 0

    for modality in euv_modalities:
        modality_dir = event_dir / modality
        if not modality_dir.exists():
            logger.warning(f"事件 {event_id} 没有 {modality} 文件夹")
            continue

        mod_cfg = modalities_cfg.get(modality, {})
        wavelength = int(mod_cfg.get('wavelength', modality.split('_')[1]))

        output_dir = processed_base_dir / event_id / modality
        output_dir.mkdir(parents=True, exist_ok=True)

        fits_files = sorted(modality_dir.glob('*.fits'))
        if not fits_files:
            logger.warning(f"事件 {event_id} 的 {modality} 文件夹为空")
            continue

        # 按观测时间排序（与 IDL 时间顺序序号对应）
        def sort_key(p: Path) -> datetime:
            t = _get_fits_time(p)
            return t if t is not None else datetime.min

        fits_files = sorted(fits_files, key=sort_key)

        logger.info(f"处理事件 {event_id} 的 {modality}: {len(fits_files)} 个 FITS 文件")

        dur_arr: List[float] = []
        time_arr: List[float] = []  # 存储时间戳（秒，便于后续分析）

        for idx, fits_file in enumerate(fits_files):
            logger.info(f"  正在处理第 {idx+1}/{len(fits_files)} 个文件: {fits_file.name}")

            # 优先使用原始 FITS 文件名里的时间（按用户需求命名），其次用 FITS 头时间，最后用 stem
            time_str = _extract_time_str_from_filename(fits_file)
            if time_str is None:
                obs_time_guess = _get_fits_time(fits_file)
                time_str = obs_time_guess.strftime('%Y%m%d_%H%M%S') if obs_time_guess is not None else fits_file.stem

            output_file = output_dir / f"AIA{int(wavelength)}_full_{time_str}.png"
            npy_file = output_dir / f"AIA{int(wavelength)}_full_{time_str}.npy" if save_npy else None

            if skip_existing and output_file.exists():
                logger.info(f"  文件已存在，跳过: {output_file.name}")
                total_success += 1
                # 仍读取 FITS 头以收集 duration/time 元数据
                try:
                    exp = _get_exposure_time(fits_file)
                    if exp > 0:
                        dur_arr.append(exp)
                    t = _get_fits_time(fits_file)
                    if t is not None:
                        time_arr.append(t.timestamp())
                except Exception:
                    pass
                continue

            # 不跳过时，避免覆盖同名文件（例如同一时间戳有重复文件）
            output_file = _unique_path(output_file)
            if npy_file is not None:
                npy_file = _unique_path(npy_file)

            success, orig_dur, obs_time = plot_euv(
                fits_file, output_file, wavelength,
                save_npy=save_npy, npy_path=npy_file,
            )
            if success:
                total_success += 1
                if orig_dur is not None:
                    dur_arr.append(orig_dur)
                if obs_time is not None:
                    time_arr.append(obs_time.timestamp())
            else:
                logger.error(f"  处理失败: {fits_file.name}")

        # 保存 duration/time 元数据（与 IDL aia193_duration_time.sav 对应）
        if dur_arr or time_arr:
            meta_path = output_dir / f"aia{int(wavelength)}_duration_time.npz"
            np.savez(meta_path, dur_arr=np.array(dur_arr), time_arr=np.array(time_arr))
            logger.info(f"  元数据已保存: {meta_path.name}")

    return total_success


def batch_plot_euv(
    downloaded_dir: Path,
    processed_dir: Path,
    config: Dict[str, Any],
    save_npy: bool = False,
    skip_existing: bool = True,
    bands: Optional[List[str]] = None,
) -> None:
    """批量绘制所有事件的 EUV 图像（三波段）"""
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
        total_success += process_event_euv(
            event_dir, processed_dir, config,
            save_npy=save_npy, skip_existing=skip_existing, bands=bands,
        )

    logger.info(f"==================== 处理完成 ====================")
    logger.info(f"有效文件数：{total_success}")
    logger.info(f"保存路径：{processed_dir}")


def main():
    parser = argparse.ArgumentParser(description='Batch plot EUV 3 bands (94/171/193 Angstrom)')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml', help='Config file path')
    parser.add_argument('--bands', type=str, default=None,
                        help='Bands to process, comma-sep, e.g. 94,171,193. Default: all')
    parser.add_argument('--save-npy', action='store_true', help='Also save normalized .npy data')
    parser.add_argument('--no-skip', action='store_true', help='Do not skip existing output files')
    args = parser.parse_args()

    bands = None
    if args.bands:
        bands = [f"euv_{b.strip()}" for b in args.bands.split(',')]
        bands = [b for b in bands if b in ('euv_94', 'euv_171', 'euv_193')]
        if not bands:
            logger.warning("无效的 --bands，将处理全部三波段")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_data_dir = Path(config['data']['raw_data_dir'])
    downloaded_dir = raw_data_dir / 'downloaded'
    processed_dir = Path(config['data']['processed_data_dir'])

    logger.info("开始批量绘制 EUV 图像...")
    logger.info(f"波段: {bands or '全部 (94, 171, 193)'}")
    logger.info(f"下载目录: {downloaded_dir}")
    logger.info(f"处理目录: {processed_dir}")

    batch_plot_euv(
        downloaded_dir, processed_dir, config,
        save_npy=args.save_npy,
        skip_existing=not args.no_skip,
        bands=bands,
    )


if __name__ == "__main__":
    main()
