"""
创建HDF5数据集脚本

支持：
- 按 data_config.yaml 时间配置精确选择图像（每个采样点选时间最近的图像）
- 缺失处理：zero_fill（填0）或 nearest（用最近文件替代）
"""
import sys
import os
import re
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from PIL import Image

from data.hdf5_creator import HDF5DatasetCreator
from utils.event_id_utils import ensure_event_ids_like_downloader, normalize_event_time_columns
from utils.logging_utils import setup_logging


RAW_ALIGN_SIZE = (2048, 2048)


def _parse_cadence_to_seconds(cadence_str: str) -> int:
    """将 cadence 字符串转为秒数，如 '720s' -> 720, '12s' -> 12"""
    if not cadence_str:
        return 0
    s = str(cadence_str).strip().lower()
    if s.endswith('s'):
        return int(s[:-1]) if s[:-1].isdigit() else 0
    return 0


def _parse_image_time_from_filename(fname: str) -> Optional[datetime]:
    """
    从图像文件名解析观测时间，支持多种常见格式
    """
    # 格式1: YYYYMMDD_HHMMSS
    m = re.search(r'(\d{8})[_\-\sT]?(\d{6})', fname)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    # 格式2: YYYY-MM-DDTHH:MM:SS 或 YYYY_MM_DD_HH_MM_SS
    m = re.search(
        r'(\d{4})[-_](\d{2})[-_](\d{2})[T_\s](\d{2})[:_](\d{2})[:_](\d{2})',
        fname
    )
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6))
            )
        except ValueError:
            pass

    # 格式3: YYYYMMDD_HHMM 或 YYYYMMDDTHHMM
    m = re.search(r'(\d{8})[_\-\sT]?(\d{4})', fname)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}00", "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    # 格式4: 带 Z 后缀的 ISO
    m = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_\s](\d{2})[:_](\d{2})[:_]?(\d{4})', fname)
    if m:
        try:
            y, mo, d, h, mi, s = m.groups()
            sec = int(s[:2]) if len(s) >= 2 else 0
            return datetime(int(y), int(mo), int(d), int(h), int(mi), sec)
        except ValueError:
            pass

    return None


def _fits_suffix_name(path: Path) -> str:
    """Return a stable cache basename for .fits and .fits.fz inputs."""
    name = path.name
    lower = name.lower()
    if lower.endswith('.fits.fz'):
        return name[:-8]
    if lower.endswith('.fits'):
        return name[:-5]
    return path.stem


def _get_fits_module():
    try:
        from astropy.io import fits
        return fits
    except ImportError as exc:
        raise RuntimeError("astropy is required for hdf5_image_source=fits_cache") from exc


def _get_exposure_time(fits_path: Path) -> float:
    fits = _get_fits_module()
    try:
        with fits.open(str(fits_path), mode='readonly') as hdul:
            for hdu in hdul:
                header = getattr(hdu, 'header', {})
                for key in ('EXPTIME', 'EXPOSURE', 'EXPOSUR'):
                    if key in header:
                        value = float(header[key])
                        return value if value > 0 else 1.0
            header = hdul[0].header
            for key in ('EXPTIME', 'EXPOSURE', 'EXPOSUR'):
                if key in header:
                    value = float(header[key])
                    return value if value > 0 else 1.0
    except Exception:
        pass
    return 1.0


def _read_first_fits_image(fits_path: Path) -> np.ndarray:
    fits = _get_fits_module()
    with fits.open(str(fits_path), mode='readonly') as hdul:
        if len(hdul) > 1 and hdul[1].data is not None:
            data = hdul[1].data
        elif hdul[0].data is not None:
            data = hdul[0].data
        else:
            raise ValueError(f"No image data found in {fits_path}")
        return np.asarray(data, dtype=np.float32)


def _read_halpha_fits_image(fits_path: Path, slice_index: int = 69) -> np.ndarray:
    fits = _get_fits_module()
    is_gong = str(fits_path).lower().endswith('.fits.fz')
    with fits.open(str(fits_path), mode='readonly') as hdul:
        if is_gong:
            if len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
            elif hdul[0].data is not None:
                data = hdul[0].data
            else:
                raise ValueError(f"No H-alpha image data found in {fits_path}")
            if data.ndim != 2:
                raise ValueError(f"Unsupported GONG data shape {data.shape} in {fits_path}")
            return np.asarray(data, dtype=np.float32)

        if len(hdul) < 2 or hdul[1].data is None:
            raise ValueError(f"No CHASE H-alpha image data found in {fits_path}")
        data = hdul[1].data
        if data.ndim == 3:
            idx = max(0, min(slice_index, data.shape[0] - 1))
            return np.asarray(data[idx, :, :], dtype=np.float32)
        if data.ndim == 2:
            return np.asarray(data, dtype=np.float32)
        raise ValueError(f"Unsupported H-alpha data shape {data.shape} in {fits_path}")


def _solar_disk_params(fits_path: Path, data_shape: Tuple[int, int]) -> Dict[str, float]:
    fits = _get_fits_module()
    height, width = data_shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0
    radius_pix = min(height, width) / 2.0 * 0.9
    try:
        with fits.open(str(fits_path), mode='readonly') as hdul:
            header = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.size > 0:
                    header = hdu.header
                    break
            if header is None:
                header = hdul[0].header

            if header.get('CRPIX1') is not None and header.get('CRPIX2') is not None:
                center_x = float(header['CRPIX1']) - 1.0
                center_y = float(header['CRPIX2']) - 1.0

            rsun_pix = header.get('RSUN_PIX')
            if rsun_pix is not None and float(rsun_pix) > 0:
                radius_pix = float(rsun_pix)
            else:
                cdelt1 = header.get('CDELT1', header.get('CD1_1', 0.5))
                cdelt2 = header.get('CDELT2', header.get('CD2_2', 0.5))
                pixel_scale = (abs(float(cdelt1)) + abs(float(cdelt2))) / 2.0
                rsun_obs = float(header.get('RSUN_OBS', header.get('SOLAR_R', 479.63)))
                if pixel_scale > 0 and rsun_obs > 0:
                    radius_pix = rsun_obs / pixel_scale
    except Exception:
        pass

    target_radius = min(RAW_ALIGN_SIZE) / 2.0 * 0.9
    zoom_factor = target_radius / radius_pix if radius_pix > 0 else 1.0
    return {
        'center_x': center_x,
        'center_y': center_y,
        'radius_pix': radius_pix,
        'zoom_factor': zoom_factor,
    }


def _resize_float_image(data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if data.shape == (target_h, target_w):
        return data.astype(np.float32, copy=False)
    try:
        from scipy.ndimage import zoom
        factors = (target_h / data.shape[0], target_w / data.shape[1])
        return zoom(data, factors, order=1).astype(np.float32, copy=False)
    except Exception:
        img = Image.fromarray(data.astype(np.float32, copy=False))
        return np.asarray(img.resize((target_w, target_h), Image.Resampling.BILINEAR), dtype=np.float32)


def _align_fits_image(
    data: np.ndarray,
    fits_path: Path,
    alignment_mode: str,
    target_shape: Tuple[int, int] = RAW_ALIGN_SIZE,
) -> np.ndarray:
    params = _solar_disk_params(fits_path, data.shape)
    center_x = params['center_x']
    center_y = params['center_y']
    radius = params['radius_pix']
    zoom_factor = params['zoom_factor']

    target_h, target_w = target_shape
    out = np.zeros((target_h, target_w), dtype=np.float32)
    offset_x = int(round(target_w / 2.0 - center_x * zoom_factor))
    offset_y = int(round(target_h / 2.0 - center_y * zoom_factor))

    src = data.astype(np.float32, copy=False)
    if alignment_mode == 'disk_only':
        h, w = src.shape[:2]
        y, x = np.ogrid[:h, :w]
        disk_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        disk_only = np.zeros_like(src, dtype=np.float32)
        disk_only[disk_mask] = src[disk_mask]
        src = disk_only

    if abs(zoom_factor - 1.0) < 1e-6:
        scaled = src
    else:
        scaled_h = max(1, int(round(src.shape[0] * zoom_factor)))
        scaled_w = max(1, int(round(src.shape[1] * zoom_factor)))
        scaled = _resize_float_image(src, (scaled_h, scaled_w))

    src_h, src_w = scaled.shape[:2]
    dst_x0 = max(0, offset_x)
    dst_y0 = max(0, offset_y)
    dst_x1 = min(target_w, offset_x + src_w)
    dst_y1 = min(target_h, offset_y + src_h)
    src_x0 = max(0, -offset_x)
    src_y0 = max(0, -offset_y)
    src_x1 = src_x0 + max(0, dst_x1 - dst_x0)
    src_y1 = src_y0 + max(0, dst_y1 - dst_y0)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]
    return out


def _target_disk_mask(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    cy, cx = h // 2, w // 2
    radius = int(min(shape) / 2.0 * 0.9)
    y, x = np.ogrid[:h, :w]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def _fits_to_physical_array(fits_path: Path, modality: str, target_shape: Tuple[int, int]) -> np.ndarray:
    if modality == 'halpha':
        data = _read_halpha_fits_image(fits_path)
    else:
        data = _read_first_fits_image(fits_path)
        if data.ndim == 3:
            data = data[0]

    if data.ndim != 2:
        raise ValueError(f"Unsupported FITS image shape {data.shape} in {fits_path}")

    data = np.nan_to_num(data.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    if modality.startswith('euv_'):
        data = data / _get_exposure_time(fits_path)
        data = np.maximum(data, 0.0)
        data = np.flipud(data)
        aligned = _align_fits_image(data, fits_path, alignment_mode='full')
    elif modality == 'magnetogram':
        data = np.fliplr(data)
        aligned = _align_fits_image(data, fits_path, alignment_mode='disk_only')
    elif modality == 'halpha':
        exposure = _get_exposure_time(fits_path)
        data = data / exposure if exposure > 0 else data
        data = np.maximum(data, 0.0)
        data = np.flipud(data)
        aligned = _align_fits_image(data, fits_path, alignment_mode='disk_only')
        mask = _target_disk_mask(aligned.shape)
        aligned = aligned.copy()
        aligned[~mask] = 0.0
    else:
        aligned = _align_fits_image(data, fits_path, alignment_mode='full')

    resized = _resize_float_image(aligned, target_shape)
    return resized.astype(np.float32, copy=False)


def _cache_path_for_fits(cache_dir: Path, event_id: str, modality: str, fits_path: Path, target_shape: Tuple[int, int]) -> Path:
    target_h, target_w = target_shape
    name = f"{_fits_suffix_name(fits_path)}.r{target_h}x{target_w}.npy"
    return cache_dir / event_id / modality / name


def _ensure_numeric_cache(
    fits_path: Path,
    event_id: str,
    modality: str,
    cache_dir: Path,
    target_shape: Tuple[int, int],
) -> Path:
    cache_path = _cache_path_for_fits(cache_dir, event_id, modality, fits_path, target_shape)
    if cache_path.exists() and cache_path.stat().st_mtime >= fits_path.stat().st_mtime:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = _fits_to_physical_array(fits_path, modality, target_shape)
    np.save(cache_path, data.astype(np.float32, copy=False))
    return cache_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='创建HDF5数据集')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--events_csv', '--events', type=str, dest='events_file',
                        default=None, help='事件文件路径(xlsx/csv)，不指定则用配置文件')    # 使用data/raw/events_training.xlsx
    parser.add_argument('--output', type=str,
                        default=None,
                        help='输出HDF5文件路径，不指定则用配置文件中的hdf5_path')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    parser.add_argument('--append', action='store_true',
                        help='强制向已有 HDF5 追加新事件（配置需兼容）')
    parser.add_argument('--rebuild', action='store_true',
                        help='强制全量重建 HDF5，即使输出文件已存在')
    return parser.parse_args()


def _normalize_cme_column(df: pd.DataFrame) -> pd.DataFrame:
    """统一 CME 列名，支持 cme_associated / CME_asociated 等"""
    for alt in ['CME_asociated', 'CME_associated', 'cme_asociated']:
        if alt in df.columns and 'cme_associated' not in df.columns:
            df['cme_associated'] = df[alt]
            break
    return df


def _is_missing_event_id(value) -> bool:
    if pd.isna(value):
        return True
    s = str(value).strip()
    return (not s) or s.lower() in {'nan', 'none'}


def _generate_event_ids_like_downloader(df: pd.DataFrame) -> pd.Series:
    """按时间生成与其他流程一致的 event_id。"""
    event_ids = []
    time_count = {}

    for idx, row in df.iterrows():
        start_time_str = str(row.get('start_time', '') or '').strip()
        end_time_str = str(row.get('end_time', '') or '').strip()
        start_dt = pd.to_datetime(start_time_str, errors='coerce')
        end_dt = pd.to_datetime(end_time_str, errors='coerce')

        if pd.isna(start_dt) or pd.isna(end_dt):
            time_key = f'row_{idx}'
        else:
            time_key = f'{start_time_str}_{end_time_str}'

        time_count[time_key] = time_count.get(time_key, 0) + 1
        cnt = time_count[time_key]

        if pd.isna(start_dt):
            event_id = f'EVT_{idx + 1:04d}_{cnt}'
        else:
            event_id = f"EVT_{start_dt.strftime('%Y%m%d_%H%M%S')}_{cnt}"
        event_ids.append(event_id)

    return pd.Series(event_ids, index=df.index)


def _ensure_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    """确保事件表中的 event_id 可用于建目录和写 HDF5。"""
    df = df.copy()
    generated_event_ids = _generate_event_ids_like_downloader(df)

    if 'event_id' not in df.columns:
        df['event_id'] = generated_event_ids
        return df

    missing_mask = df['event_id'].apply(_is_missing_event_id)
    if missing_mask.any():
        df.loc[missing_mask, 'event_id'] = generated_event_ids[missing_mask]

    df['event_id'] = df['event_id'].astype(str).str.strip()
    return df


def load_events_metadata(file_path: str) -> pd.DataFrame:
    """加载事件元数据

    - 允许 peak_flux / duration / active_region 缺失，并自动填充：
      * peak_flux: 若缺失或不可解析，则置为 0.0
      * duration: 总是根据 start_time 和 end_time 计算得到（单位：分钟）
      * active_region: 若缺失，则置为 '0'（原始 Excel 中为 position of active region）
    """
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

    df = _normalize_cme_column(df)

    # 最小必需列：事件ID/时间/等级/CME/峰值时间
    required_columns = ['start_time', 'end_time', 'flare_class',
                        'cme_associated', 'peak_time']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"文件缺少必要列: {col}")

    # 统一转换所有时间列为 ISO 格式的字符串，便于后续排序和处理
    df = normalize_event_time_columns(df)

    # 若 active_region 缺失，尝试从 'position of active region' 映射；仍缺则置为 '0'
    if 'active_region' not in df.columns:
        if 'position of active region' in df.columns:
            df['active_region'] = df['position of active region']
        else:
            df['active_region'] = '0'
    # 保证 active_region 为字符串，缺失填 '0'
    df['active_region'] = df['active_region'].fillna('0').astype(str)

    # peak_flux: 若缺失，则整列置 0.0；若存在，则尝试转成 float，无法解析或 NaN 置 0.0
    if 'peak_flux' not in df.columns:
        df['peak_flux'] = 0.0
    else:
        df['peak_flux'] = pd.to_numeric(df['peak_flux'], errors='coerce').fillna(0.0)

    # duration: 始终根据 start_time 和 end_time 计算（单位：分钟），不依赖原始列
    def _compute_duration(row):
        try:
            st = pd.to_datetime(str(row['start_time']).replace('Z', '').replace('+00:00', '').strip())
            et = pd.to_datetime(str(row['end_time']).replace('Z', '').replace('+00:00', '').strip())
            return max(0.0, (et - st).total_seconds() / 60.0)
        except Exception:
            return 0.0

    df['duration'] = df.apply(_compute_duration, axis=1).astype(float)

    # 规范化 cme_associated 列: yes/True->1, no/False->0
    def _to_cme_bool(v):
        if pd.isna(v):
            return 0  # 缺失默认为0（不伴随CME）
        s = str(v).strip().lower()
        if s in ('yes', 'true', '1', 'y'):
            return 1
        elif s in ('no', 'false', '0', 'n'):
            return 0
        else:
            return 0  # 其他值默认为0
    df['cme_associated'] = df['cme_associated'].apply(_to_cme_bool)

    # 添加标签列: 基于 cme_associated 列，yes->1(爆发伴CME), no->2(爆发无CME), 其他->0
    if 'label' not in df.columns:
        def _to_label_from_cme(v):
            if pd.isna(v):
                return 0  # 缺失默认为0
            # 此时 v 已经被 _to_cme_bool 转换为 0 或 1
            try:
                v_int = int(v)
                if v_int == 1:
                    return 1  # yes -> label = 1
                elif v_int == 0:
                    return 2  # no -> label = 2
                else:
                    return 0
            except (ValueError, TypeError):
                return 0
        df['label'] = df['cme_associated'].apply(_to_label_from_cme)

    if 'data_available' not in df.columns:
        df['data_available'] = True

    # 按 start_time 排序，确保事件组按时间顺序创建
    # 此时 start_time 已经是 ISO 格式字符串，可以直接排序
    df = df.sort_values('start_time', na_position='last').reset_index(drop=True)
    df = ensure_event_ids_like_downloader(df)

    return df


def _select_image_for_timestamp(
    image_files: List[Path],
    target_dt: datetime,
    handle_missing: str,
    cadence_seconds: int,
    modality_cadence_seconds: int,
) -> Tuple[Optional[Path], bool]:
    """
    为给定时间戳选择最合适的图像文件
    返回 (图像路径, 是否缺失)
    """
    if not image_files:
        return None, True

    # 解析每个文件的时间
    file_times = []
    valid_files = []
    for p in image_files:
        dt = _parse_image_time_from_filename(p.stem)
        if dt is not None:
            file_times.append(dt)
            valid_files.append(p)

    if not valid_files:
        return None, True

    # 找时间最近的文件
    diffs = np.array([abs((ft - target_dt).total_seconds()) for ft in file_times])
    j = int(np.argmin(diffs))
    min_diff = float(diffs[j])
    best_path = valid_files[j]

    if handle_missing == "zero_fill":
        # 阈值：采样间隔的一半与模态 cadence 的较大者
        thr = max(cadence_seconds / 2, modality_cadence_seconds / 2) if modality_cadence_seconds > 0 else cadence_seconds / 2
        if min_diff > thr:
            return None, True  # 判定为缺失，填 0

    return best_path, False


def collect_event_data(event_row: pd.Series, config: Dict) -> Dict:
    """
    从预下载的图像文件中收集事件数据
    按 data_config 时间配置精确选择每个采样点对应的图像（时间最近），支持缺失处理
    """
    event_id = str(event_row['event_id']).strip()
    log = logging.getLogger(__name__)
    log.info(f"从图像文件收集事件数据: {event_id}")

    # 解析时间
    start_str = str(event_row['start_time']).replace('Z', '').replace('+00:00', '').strip()
    end_str = str(event_row['end_time']).replace('Z', '').replace('+00:00', '').strip()
    try:
        start_time = pd.to_datetime(start_str).to_pydatetime()
        end_time = pd.to_datetime(end_str).to_pydatetime()
    except Exception as e:
        log.warning(f"解析时间失败 {event_id}: {e}，使用默认")
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)

    pre_hours = config['time']['pre_event_hours']
    post_hours = config['time']['post_event_hours']
    cadence_minutes = config['time']['cadence_minutes']
    cadence_seconds = cadence_minutes * 60

    extended_start = start_time - timedelta(hours=pre_hours)
    extended_end = start_time + timedelta(hours=post_hours)

    timestamps = []
    current = extended_start
    while current <= extended_end:
        timestamps.append(current.isoformat())
        current += timedelta(minutes=cadence_minutes)

    num_frames = len(timestamps)
    time_series_data = {'timestamps': timestamps}

    preprocessing = config.get('preprocessing', {})
    handle_missing = preprocessing.get('handle_missing', 'nearest')
    if handle_missing not in ('zero_fill', 'nearest'):
        handle_missing = 'nearest'

    raw_dir = Path(config.get('raw_data_dir', 'data/raw'))
    processed_dir = Path(config.get('processed_data_dir', 'data/processed'))
    image_base_dir = raw_dir / 'images' / event_id
    if not image_base_dir.exists():
        image_base_dir = processed_dir / event_id

    if not image_base_dir.exists():
        log.warning(f"图像目录不存在: {image_base_dir}，将创建空数据")
        return create_empty_data(config, timestamps)

    modalities = config['modalities']
    for modality, modality_config in modalities.items():
        modality_dir = image_base_dir / modality
        if not modality_dir.exists():
            log.warning(f"模态 {modality} 的图像目录不存在: {modality_dir}")
            continue

        image_files = sorted(modality_dir.glob('*.png'))
        if not image_files:
            log.warning(f"模态 {modality} 没有找到图像文件")
            continue

        modality_cadence = _parse_cadence_to_seconds(modality_config.get('cadence', '0s'))
        resolution = modality_config.get('resolution', [256, 256])
        H, W = resolution[0], resolution[1]

        modality_array = np.zeros((num_frames, H, W), dtype=np.float32)
        normalization = modality_config.get('normalization')
        vmin, vmax = (normalization[0], normalization[1]) if normalization else (0.0, 255.0)

        for i, ts_str in enumerate(timestamps):
            try:
                target_dt = pd.to_datetime(ts_str).to_pydatetime()
            except Exception:
                continue

            img_path, is_missing = _select_image_for_timestamp(
                image_files, target_dt, handle_missing, cadence_seconds, modality_cadence
            )

            if is_missing or img_path is None:
                continue  # 保持该帧为 0

            try:
                image = Image.open(img_path)
                data = np.array(image, dtype=np.float32)
                if data.ndim == 3:
                    data = data[:, :, 0] if data.shape[2] >= 1 else data[:, :, 0]
                if data.shape[0] != H or data.shape[1] != W:
                    img_resized = Image.fromarray(data.astype(np.uint8)).resize((W, H), Image.Resampling.LANCZOS)
                    data = np.array(img_resized, dtype=np.float32)
                data = data / 255.0 * (vmax - vmin) + vmin
                modality_array[i] = data
            except Exception as e:
                log.error(f"读取图像 {img_path} 失败: {e}")

        time_series_data[modality] = modality_array
        log.info(f"模态 {modality} 数据形状: {time_series_data[modality].shape}")

    available_modalities = [m for m in modalities.keys() if m in time_series_data]
    if not available_modalities:
        log.warning(f"事件 {event_id} 没有可用的模态数据")
        return create_empty_data(config, timestamps)

    auxiliary_dir = image_base_dir / 'auxiliary'
    if auxiliary_dir.exists():
        time_series_data['auxiliary'] = load_auxiliary_data(auxiliary_dir, num_frames)

    return time_series_data


def collect_event_data_numeric_cache(event_row: pd.Series, config: Dict) -> Dict:
    """Collect event data from FITS-derived numeric .npy cache or processed PNGs."""
    image_source = str(config.get('hdf5_image_source', 'fits_cache')).strip().lower()
    if image_source == 'processed_png':
        return collect_event_data(event_row, config)
    if image_source not in {'fits_cache'}:
        raise ValueError(f"Unsupported hdf5_image_source={image_source!r}")

    event_id = str(event_row['event_id']).strip()
    log = logging.getLogger(__name__)
    log.info(f"Collecting FITS numeric-cache data for event {event_id}")

    start_str = str(event_row['start_time']).replace('Z', '').replace('+00:00', '').strip()
    try:
        start_time = pd.to_datetime(start_str).to_pydatetime()
    except Exception as e:
        log.warning(f"Failed to parse start_time for {event_id}: {e}; using current time")
        start_time = datetime.now()

    pre_hours = config['time']['pre_event_hours']
    post_hours = config['time']['post_event_hours']
    cadence_minutes = config['time']['cadence_minutes']
    cadence_seconds = cadence_minutes * 60

    extended_start = start_time - timedelta(hours=pre_hours)
    extended_end = start_time + timedelta(hours=post_hours)
    timestamps = []
    current = extended_start
    while current <= extended_end:
        timestamps.append(current.isoformat())
        current += timedelta(minutes=cadence_minutes)

    num_frames = len(timestamps)
    time_series_data = {'timestamps': timestamps}

    preprocessing = config.get('preprocessing', {})
    handle_missing = preprocessing.get('handle_missing', 'nearest')
    if handle_missing not in ('zero_fill', 'nearest'):
        handle_missing = 'nearest'

    raw_dir = Path(config.get('raw_data_dir', 'data/raw'))
    processed_dir = Path(config.get('processed_data_dir', 'data/processed'))
    numeric_cache_dir = Path(config.get('numeric_cache_dir', 'data/processed_arrays'))
    raw_event_dir = raw_dir / 'downloaded' / event_id
    processed_event_dir = processed_dir / event_id
    cache_event_dir = numeric_cache_dir / event_id

    if not raw_event_dir.exists() and not cache_event_dir.exists():
        log.warning(f"No FITS or numeric cache directory for {event_id}: raw={raw_event_dir}, cache={cache_event_dir}")
        return create_empty_data(config, timestamps)

    for modality, modality_config in config['modalities'].items():
        modality_cadence = _parse_cadence_to_seconds(modality_config.get('cadence', '0s'))
        resolution = modality_config.get('resolution', [256, 256])
        H, W = int(resolution[0]), int(resolution[1])
        target_shape = (H, W)
        modality_array = np.zeros((num_frames, H, W), dtype=np.float32)

        raw_modality_dir = raw_event_dir / modality
        cached_modality_dir = cache_event_dir / modality
        source_files: List[Path] = []
        if raw_modality_dir.exists():
            source_files = sorted(raw_modality_dir.glob('*.fits')) + sorted(raw_modality_dir.glob('*.fits.fz'))
        if not source_files and cached_modality_dir.exists():
            source_files = sorted(cached_modality_dir.glob(f'*.r{H}x{W}.npy'))

        if not source_files:
            log.warning(f"No FITS/cache files for {event_id}/{modality}")
            continue

        for i, ts_str in enumerate(timestamps):
            try:
                target_dt = pd.to_datetime(ts_str).to_pydatetime()
            except Exception:
                continue

            source_path, is_missing = _select_image_for_timestamp(
                source_files, target_dt, handle_missing, cadence_seconds, modality_cadence
            )
            if is_missing or source_path is None:
                continue

            try:
                if source_path.suffix.lower() == '.npy':
                    cache_path = source_path
                else:
                    cache_path = _ensure_numeric_cache(source_path, event_id, modality, numeric_cache_dir, target_shape)
                data = np.load(cache_path).astype(np.float32, copy=False)
                if data.shape != target_shape:
                    data = _resize_float_image(data, target_shape)
                modality_array[i] = data
            except Exception as e:
                log.error(f"Failed to load numeric data for {event_id}/{modality} from {source_path}: {e}")

        time_series_data[modality] = modality_array
        log.info(f"Modality {modality} data shape: {modality_array.shape}")

    available_modalities = [m for m in config['modalities'].keys() if m in time_series_data]
    if not available_modalities:
        log.warning(f"Event {event_id} has no available modalities")
        return create_empty_data(config, timestamps)

    auxiliary_dir = processed_event_dir / 'auxiliary'
    if auxiliary_dir.exists():
        time_series_data['auxiliary'] = load_auxiliary_data(auxiliary_dir, num_frames)

    return time_series_data


def create_empty_data(config: Dict, timestamps: List[str]) -> Dict:
    """创建空数据作为占位符"""
    num_frames = len(timestamps)
    time_series_data = {'timestamps': timestamps}

    modalities = config['modalities']
    for modality, modality_config in modalities.items():
        resolution = modality_config['resolution']
        time_series_data[modality] = np.zeros(
            (num_frames, resolution[0], resolution[1]),
            dtype=np.float32
        )

    return time_series_data


def load_auxiliary_data(auxiliary_dir: Path, num_frames: int) -> Dict:
    """加载辅助数据"""
    auxiliary = {}

    # 这里可以根据需要加载各种辅助数据
    # 例如：活动区掩码、太阳黑子数、磁通量等

    return auxiliary


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    logger = setup_logging(args.log_dir, 'create_hdf5_dataset', debug=args.debug)
    root_logger = logging.getLogger()
    root_logger.setLevel(logger.level)
    if not root_logger.handlers:
        for handler in logger.handlers:
            root_logger.addHandler(handler)

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 更新输出路径（如果指定了--output，则覆盖配置文件；否则使用配置文件中的hdf5_path）
    if args.output:
        config['data']['hdf5_path'] = args.output
    if hasattr(args, 'events_file') and args.events_file:
        config['data']['events_file'] = args.events_file

    # 从配置中获取事件文件路径
    events_file = config['data']['events_file']
    logger.info(f"使用事件文件: {events_file}")

    # 加载事件元数据
    logger.info(f"加载事件元数据: {events_file}")
    events_metadata = load_events_metadata(events_file)
    logger.info(f"加载到 {len(events_metadata)} 个事件")

    # 创建HDF5数据集
    creator = HDF5DatasetCreator(config['data'])
    output_path = Path(config['data']['hdf5_path'])
    append_mode = args.append or (output_path.exists() and not args.rebuild)
    if args.append and args.rebuild:
        raise ValueError('--append 与 --rebuild 不能同时使用')
    logger.info(f"HDF5 写入模式: {'append' if append_mode else 'rebuild'} | output={output_path}")

    creator.create_dataset(
        events_metadata,
        lambda row: collect_event_data_numeric_cache(row, config['data']),
        append=append_mode,
    )

    logger.info("HDF5数据集创建完成")


if __name__ == '__main__':
    main()
