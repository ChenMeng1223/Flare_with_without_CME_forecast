"""
Export a YOLO-style single-class activity-region detection dataset from HDF5.

Updated experiment D export:
- one image per event
- use the frame closest to event start_time by default
- one detection class: activity_region
- de-duplicate boxes within each image by unique region_id
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import yaml

SUPPORTED_MODALITIES = ["magnetogram", "euv_94", "euv_171", "euv_193", "halpha"]
DIFF_SUFFIX = "_diff_prev"
OFFSET_PATTERN = re.compile(r"^(?P<modality>.+)_(?P<direction>prev|next)(?P<steps>\d+)$")
FUSED_TEMPORAL_PATTERN = re.compile(
    r"^fused\((?P<spec_a>[^,]+),(?P<spec_b>[^)]+)\)_(?P<direction>prev|next)(?P<steps>\d+)$"
)
FUSED_CENTER_PATTERN = re.compile(r"^fused\((?P<spec_a>[^,]+),(?P<spec_b>[^)]+)\)$")
EVENT_ID_PATTERN = re.compile(r"^EVT_(?P<date>\d{8})_(?P<time>\d{6})")
FILENAME_DATETIME_PATTERNS = [
    re.compile(r"(?P<date>20\d{2}\d{2}\d{2})_(?P<time>\d{6})"),
    re.compile(r"(?P<date>20\d{2})_(?P<month>\d{2})_(?P<day>\d{2})T(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"),
]


USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    "data_config": "configs/data_config.yaml",
    "split_dir": "data/split_data",
    "output_dir": "outputs/yolo_dataset/activity_region_single_class_mag_euv94_euv171_processed_export_filtered20",
    "modality": "magnetogram",
    "channel_modalities": ["magnetogram", "euv_94", "euv_171"],
    "frame": "annotated",
    "skip_empty": True,
    "image_source": "processed",
    "label_source": "processed",
    "output_size": 512,
    "save_npy_sidecar": False,
    "max_time_delta_minutes": 20.0,
}


def _build_direct_run_args() -> List[str]:
    cfg = dict(DIRECT_RUN_CONFIG)
    args: List[str] = []
    for key in ("data_config", "split_dir", "output_dir", "modality", "frame", "image_source", "label_source", "output_size", "max_time_delta_minutes"):
        value = cfg.get(key)
        if value not in (None, ""):
            args.extend([f"--{key}", str(value)])
    channel_modalities = cfg.get("channel_modalities") or []
    if channel_modalities:
        args.append("--channel_modalities")
        args.extend(str(modality) for modality in channel_modalities)
    if cfg.get("skip_empty", False):
        args.append("--skip_empty")
    if cfg.get("save_npy_sidecar", False):
        args.append("--save_npy_sidecar")
    return args


def _load_hdf5_dataset_reader():
    """Load HDF5DatasetReader without importing the full data package."""
    repo_root = Path(__file__).resolve().parents[1]
    reader_path = repo_root / "data" / "hdf5_reader.py"
    spec = importlib.util.spec_from_file_location("hdf5_reader_only", reader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {reader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HDF5DatasetReader


HDF5DatasetReader = None


def get_hdf5_dataset_reader():
    global HDF5DatasetReader
    if HDF5DatasetReader is None:
        HDF5DatasetReader = _load_hdf5_dataset_reader()
    return HDF5DatasetReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO detection dataset from HDF5.")
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_config.yaml",
        help="Data config path.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="data/split_data",
        help="Directory containing <split>_events.txt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/yolo_dataset/activity_region_single_class",
        help="Output directory for YOLO dataset.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="magnetogram",
        choices=SUPPORTED_MODALITIES,
        help="Which modality to export as detector input.",
    )
    parser.add_argument(
        "--channel_modalities",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional RGB channel mapping. "
            "Examples: --channel_modalities magnetogram euv_171 euv_94, "
            "--channel_modalities magnetogram_prev1 magnetogram magnetogram_next1, "
            "or --channel_modalities magnetogram euv_171 euv_171_diff_prev. "
            "If omitted, the single selected --modality is replicated into RGB."
        ),
    )
    parser.add_argument(
        "--frame",
        type=str,
        default="event_start",
        choices=["first", "middle", "last", "event_start", "annotated"],
        help="Which frame to export for each event.",
    )
    parser.add_argument(
        "--skip_empty",
        action="store_true",
        help="Skip events without positive activity boxes.",
    )
    parser.add_argument(
        "--image_source",
        type=str,
        default="processed",
        choices=["hdf5", "processed"],
        help="Use images from HDF5 arrays or processed original-resolution images.",
    )
    parser.add_argument(
        "--label_source",
        type=str,
        default="processed",
        choices=["hdf5", "processed"],
        help="Use detection labels from HDF5 regions or processed/bboxes.json annotations.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=None,
        help="Optional square output image size, e.g. 512.",
    )
    parser.add_argument(
        "--save_npy_sidecar",
        action="store_true",
        help=(
            "Also save a same-stem .npy array beside each exported .png. "
            "This enables true multi-channel YOLO input while preserving a simple preview image."
        ),
    )
    parser.add_argument(
        "--max_time_delta_minutes",
        type=float,
        default=None,
        help=(
            "For processed exports, skip an event if any channel image is farther than this many minutes "
            "from the target timestamp. Set <= 0 to disable."
        ),
    )
    return parser.parse_args()


def load_data_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["data"]


def normalize_to_uint8(image: np.ndarray, modality: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite_mask = np.isfinite(image)
    if not finite_mask.any():
        return np.zeros_like(image, dtype=np.uint8)

    valid = image[finite_mask]
    if modality == "magnetogram":
        scale = max(np.percentile(np.abs(valid), 99), 1.0)
        normalized = np.clip((image + scale) / (2.0 * scale), 0.0, 1.0)
    else:
        lo = np.percentile(valid, 1)
        hi = np.percentile(valid, 99)
        if hi <= lo:
            normalized = np.zeros_like(image, dtype=np.float32)
        else:
            normalized = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def normalize_diff_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite_mask = np.isfinite(image)
    if not finite_mask.any():
        return np.zeros_like(image, dtype=np.uint8)

    valid = image[finite_mask]
    scale = max(np.percentile(np.abs(valid), 99), 1e-6)
    normalized = np.clip((image + scale) / (2.0 * scale), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "").replace("+00:00", ""))


def _parse_filename_datetime(filename: str) -> Optional[datetime]:
    for pattern in FILENAME_DATETIME_PATTERNS:
        match = pattern.search(filename)
        if not match:
            continue
        try:
            if "month" in match.groupdict():
                return datetime(
                    int(match.group("date")),
                    int(match.group("month")),
                    int(match.group("day")),
                    int(match.group("hour")),
                    int(match.group("minute")),
                    int(match.group("second")),
                )
            return datetime.strptime(
                f"{match.group('date')}_{match.group('time')}",
                "%Y%m%d_%H%M%S",
            )
        except ValueError:
            continue
    return None


def _event_id_to_datetime(event_id: str) -> Optional[datetime]:
    match = EVENT_ID_PATTERN.match(str(event_id).strip())
    if not match:
        return None
    try:
        return datetime.strptime(
            f"{match.group('date')}_{match.group('time')}",
            "%Y%m%d_%H%M%S",
        )
    except ValueError:
        return None


def _event_id_to_timestamp_string(event_id: str) -> Optional[str]:
    event_dt = _event_id_to_datetime(event_id)
    if event_dt is None:
        return None
    return event_dt.isoformat(timespec="seconds")


def _parse_channel_spec(channel_spec: str) -> Tuple[str, Optional[str], int]:
    if channel_spec.endswith(DIFF_SUFFIX):
        base_modality = channel_spec[: -len(DIFF_SUFFIX)]
        return base_modality, "diff_prev", 0
    match = OFFSET_PATTERN.match(channel_spec)
    if match:
        base_modality = match.group("modality")
        steps = int(match.group("steps"))
        direction = match.group("direction")
        offset = -steps if direction == "prev" else steps
        return base_modality, None, offset
    return channel_spec, None, 0


def _is_fused_channel_spec(channel_spec: str) -> bool:
    return FUSED_CENTER_PATTERN.match(channel_spec) is not None or FUSED_TEMPORAL_PATTERN.match(channel_spec) is not None


def _parse_fused_channel_spec(channel_spec: str) -> Tuple[str, str, int]:
    temporal_match = FUSED_TEMPORAL_PATTERN.match(channel_spec)
    if temporal_match:
        spec_a = temporal_match.group("spec_a").strip()
        spec_b = temporal_match.group("spec_b").strip()
        steps = int(temporal_match.group("steps"))
        direction = temporal_match.group("direction")
        offset = -steps if direction == "prev" else steps
        return spec_a, spec_b, offset

    center_match = FUSED_CENTER_PATTERN.match(channel_spec)
    if center_match:
        return center_match.group("spec_a").strip(), center_match.group("spec_b").strip(), 0

    raise ValueError(f"Unsupported fused channel spec: {channel_spec}")


def _validate_channel_specs(channel_specs: List[str]) -> None:
    invalid_specs = []
    for channel_spec in channel_specs:
        if _is_fused_channel_spec(channel_spec):
            try:
                spec_a, spec_b, _ = _parse_fused_channel_spec(channel_spec)
                _validate_channel_specs([spec_a, spec_b])
            except Exception:
                invalid_specs.append(channel_spec)
            continue
        base_modality, transform, _ = _parse_channel_spec(channel_spec)
        if base_modality not in SUPPORTED_MODALITIES:
            invalid_specs.append(channel_spec)
            continue
        if transform not in (None, "diff_prev"):
            invalid_specs.append(channel_spec)
    if invalid_specs:
        raise ValueError(f"Unsupported channel spec(s): {invalid_specs}")


def _load_bbox_info(processed_root: Path, event_id: str) -> Dict[str, Any]:
    bbox_path = processed_root / event_id / "bboxes.json"
    if not bbox_path.exists():
        return {}
    try:
        with open(bbox_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _find_processed_image_path(
    processed_root: Path,
    event_id: str,
    modality: str,
    target_timestamp: Optional[str],
    preferred_name: Optional[str],
    frame_mode: str,
) -> Optional[Path]:
    modality_dir = processed_root / event_id / modality
    if not modality_dir.exists():
        return None

    if preferred_name:
        preferred_path = modality_dir / Path(preferred_name).name
        if preferred_path.exists():
            return preferred_path

    image_paths = sorted(
        p for p in modality_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if not image_paths:
        return None

    if frame_mode == "last":
        return image_paths[-1]
    if frame_mode == "middle":
        return image_paths[len(image_paths) // 2]
    if frame_mode == "first":
        return image_paths[0]

    if not target_timestamp:
        return image_paths[0]

    target_dt = _parse_iso_datetime(target_timestamp)
    best_path: Optional[Path] = None
    best_delta: Optional[float] = None
    for image_path in image_paths:
        image_dt = _parse_filename_datetime(image_path.name)
        if image_dt is None:
            continue
        delta = abs((image_dt - target_dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_path = image_path

    return best_path or image_paths[0]


def _list_processed_image_paths(modality_dir: Path) -> List[Path]:
    if not modality_dir.exists():
        return []
    return sorted(
        p for p in modality_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )


def _find_previous_processed_image_path(
    processed_root: Path,
    event_id: str,
    modality: str,
    current_path: Path,
) -> Optional[Path]:
    modality_dir = processed_root / event_id / modality
    image_paths = _list_processed_image_paths(modality_dir)
    if not image_paths:
        return None
    try:
        current_index = image_paths.index(current_path)
    except ValueError:
        current_index = -1
    if current_index <= 0:
        return None
    return image_paths[current_index - 1]


def _find_adjacent_processed_image_path(
    processed_root: Path,
    event_id: str,
    modality: str,
    current_path: Path,
) -> Optional[Path]:
    modality_dir = processed_root / event_id / modality
    image_paths = _list_processed_image_paths(modality_dir)
    if not image_paths:
        return None
    try:
        current_index = image_paths.index(current_path)
    except ValueError:
        return None
    if current_index > 0:
        return image_paths[current_index - 1]
    if current_index + 1 < len(image_paths):
        return image_paths[current_index + 1]
    return None


def _find_offset_processed_image_path(
    processed_root: Path,
    event_id: str,
    modality: str,
    current_path: Path,
    offset: int,
) -> Optional[Path]:
    modality_dir = processed_root / event_id / modality
    image_paths = _list_processed_image_paths(modality_dir)
    if not image_paths:
        return None
    try:
        current_index = image_paths.index(current_path)
    except ValueError:
        return None
    target_index = min(max(current_index + offset, 0), len(image_paths) - 1)
    return image_paths[target_index]


def _resize_uint8_image(image: np.ndarray, output_size: Optional[int]) -> np.ndarray:
    if output_size is None:
        return image
    pil_image = Image.fromarray(image)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = pil_image.resize((output_size, output_size), resample=resampling)
    return np.asarray(resized, dtype=np.uint8)


def _load_processed_channel_image(
    processed_root: Path,
    event_id: str,
    modality: str,
    target_timestamp: Optional[str],
    preferred_name: Optional[str],
    frame_mode: str,
    output_size: Optional[int],
) -> Optional[np.ndarray]:
    processed_image_path = _find_processed_image_path(
        processed_root=processed_root,
        event_id=event_id,
        modality=modality,
        target_timestamp=target_timestamp,
        preferred_name=preferred_name,
        frame_mode=frame_mode,
    )
    if processed_image_path is None:
        return None
    with Image.open(processed_image_path) as img:
        image_2d = np.asarray(img.convert("L"), dtype=np.uint8)
    return _resize_uint8_image(image_2d, output_size)


def _select_processed_image_path_from_spec(
    processed_root: Path,
    event_id: str,
    channel_spec: str,
    target_timestamp: Optional[str],
    preferred_name: Optional[str],
    frame_mode: str,
) -> Optional[Path]:
    if _is_fused_channel_spec(channel_spec):
        spec_a, _, offset = _parse_fused_channel_spec(channel_spec)
        shifted_spec = _shift_channel_spec_offset(spec_a, offset)
        return _select_processed_image_path_from_spec(
            processed_root=processed_root,
            event_id=event_id,
            channel_spec=shifted_spec,
            target_timestamp=target_timestamp,
            preferred_name=preferred_name,
            frame_mode=frame_mode,
        )

    modality, transform, offset = _parse_channel_spec(channel_spec)
    processed_image_path = _find_processed_image_path(
        processed_root=processed_root,
        event_id=event_id,
        modality=modality,
        target_timestamp=target_timestamp,
        preferred_name=preferred_name,
        frame_mode=frame_mode,
    )
    if processed_image_path is None:
        return None

    selected_image_path = processed_image_path
    if offset != 0:
        offset_path = _find_offset_processed_image_path(
            processed_root=processed_root,
            event_id=event_id,
            modality=modality,
            current_path=processed_image_path,
            offset=offset,
        )
        if offset_path is not None:
            selected_image_path = offset_path

    if transform == "diff_prev":
        previous_path = _find_adjacent_processed_image_path(
            processed_root=processed_root,
            event_id=event_id,
            modality=modality,
            current_path=selected_image_path,
        )
        if previous_path is None:
            return None
    return selected_image_path


def _load_processed_channel_image_from_spec(
    processed_root: Path,
    event_id: str,
    channel_spec: str,
    target_timestamp: Optional[str],
    preferred_name: Optional[str],
    frame_mode: str,
    output_size: Optional[int],
) -> Optional[np.ndarray]:
    if _is_fused_channel_spec(channel_spec):
        spec_a, spec_b, offset = _parse_fused_channel_spec(channel_spec)
        shifted_a = _shift_channel_spec_offset(spec_a, offset)
        shifted_b = _shift_channel_spec_offset(spec_b, offset)
        image_a = _load_processed_channel_image_from_spec(
            processed_root=processed_root,
            event_id=event_id,
            channel_spec=shifted_a,
            target_timestamp=target_timestamp,
            preferred_name=preferred_name,
            frame_mode=frame_mode,
            output_size=output_size,
        )
        image_b = _load_processed_channel_image_from_spec(
            processed_root=processed_root,
            event_id=event_id,
            channel_spec=shifted_b,
            target_timestamp=target_timestamp,
            preferred_name=preferred_name,
            frame_mode=frame_mode,
            output_size=output_size,
        )
        return _fuse_uint8_images(image_a, image_b)

    modality, transform, _ = _parse_channel_spec(channel_spec)
    selected_image_path = _select_processed_image_path_from_spec(
        processed_root=processed_root,
        event_id=event_id,
        channel_spec=channel_spec,
        target_timestamp=target_timestamp,
        preferred_name=preferred_name,
        frame_mode=frame_mode,
    )
    if selected_image_path is None:
        return None

    with Image.open(selected_image_path) as img:
        current_image = np.asarray(img.convert("L"), dtype=np.float32)

    if transform == "diff_prev":
        previous_path = _find_adjacent_processed_image_path(
            processed_root=processed_root,
            event_id=event_id,
            modality=modality,
            current_path=selected_image_path,
        )
        if previous_path is None:
            return None
        with Image.open(previous_path) as img:
            previous_image = np.asarray(img.convert("L"), dtype=np.float32)
        image_2d = normalize_diff_to_uint8(current_image - previous_image)
    else:
        image_2d = current_image.astype(np.uint8)

    return _resize_uint8_image(image_2d, output_size)


def _load_hdf5_channel_image(
    event_data: Dict[str, Any],
    modality: str,
    frame_idx: int,
    output_size: Optional[int],
) -> Optional[np.ndarray]:
    images = event_data.get(modality)
    if images is None:
        return None
    image_2d = normalize_to_uint8(images[frame_idx], modality)
    return _resize_uint8_image(image_2d, output_size)


def _load_hdf5_channel_image_from_spec(
    event_data: Dict[str, Any],
    channel_spec: str,
    frame_idx: int,
    output_size: Optional[int],
) -> Optional[np.ndarray]:
    if _is_fused_channel_spec(channel_spec):
        spec_a, spec_b, offset = _parse_fused_channel_spec(channel_spec)
        shifted_a = _shift_channel_spec_offset(spec_a, offset)
        shifted_b = _shift_channel_spec_offset(spec_b, offset)
        image_a = _load_hdf5_channel_image_from_spec(
            event_data=event_data,
            channel_spec=shifted_a,
            frame_idx=frame_idx,
            output_size=output_size,
        )
        image_b = _load_hdf5_channel_image_from_spec(
            event_data=event_data,
            channel_spec=shifted_b,
            frame_idx=frame_idx,
            output_size=output_size,
        )
        return _fuse_uint8_images(image_a, image_b)

    modality, transform, offset = _parse_channel_spec(channel_spec)
    images = event_data.get(modality)
    if images is None:
        return None
    selected_frame_idx = min(max(frame_idx + offset, 0), images.shape[0] - 1)
    if transform == "diff_prev":
        adjacent_idx = selected_frame_idx - 1 if selected_frame_idx > 0 else selected_frame_idx + 1
        if adjacent_idx >= images.shape[0] or adjacent_idx < 0:
            return None
        image_2d = normalize_diff_to_uint8(images[selected_frame_idx] - images[adjacent_idx])
    else:
        image_2d = normalize_to_uint8(images[selected_frame_idx], modality)
    return _resize_uint8_image(image_2d, output_size)


def _build_channel_stack_from_modalities(
    event_id: str,
    event_data: Dict[str, Any],
    frame_idx: int,
    channel_modalities: List[str],
    image_source: str,
    processed_root: Path,
    target_timestamp: Optional[str],
    preferred_name: Optional[str],
    frame_mode: str,
    output_size: Optional[int],
) -> Optional[np.ndarray]:
    channel_images: List[np.ndarray] = []
    for channel_modality in channel_modalities:
        if image_source == "processed":
            image_2d = _load_processed_channel_image_from_spec(
                processed_root=processed_root,
                event_id=event_id,
                channel_spec=channel_modality,
                target_timestamp=target_timestamp,
                preferred_name=preferred_name,
                frame_mode=frame_mode,
                output_size=output_size,
            )
        else:
            image_2d = _load_hdf5_channel_image_from_spec(
                event_data=event_data,
                channel_spec=channel_modality,
                frame_idx=frame_idx,
                output_size=output_size,
            )
        if image_2d is None:
            return None
        channel_images.append(image_2d)

    return np.stack(channel_images, axis=-1)


def _build_rgb_image_from_modalities(
    event_id: str,
    event_data: Dict[str, Any],
    frame_idx: int,
    channel_modalities: List[str],
    image_source: str,
    processed_root: Path,
    target_timestamp: Optional[str],
    preferred_name: Optional[str],
    frame_mode: str,
    output_size: Optional[int],
) -> Optional[np.ndarray]:
    channel_stack = _build_channel_stack_from_modalities(
        event_id=event_id,
        event_data=event_data,
        frame_idx=frame_idx,
        channel_modalities=channel_modalities,
        image_source=image_source,
        processed_root=processed_root,
        target_timestamp=target_timestamp,
        preferred_name=preferred_name,
        frame_mode=frame_mode,
        output_size=output_size,
    )
    if channel_stack is None:
        return None

    if channel_stack.shape[-1] == 1:
        return np.repeat(channel_stack, 3, axis=-1)
    if channel_stack.shape[-1] == 2:
        return np.concatenate([channel_stack, channel_stack[..., -1:]], axis=-1)
    return channel_stack[..., :3]


def _shift_channel_spec_offset(channel_spec: str, offset_delta: int) -> str:
    if offset_delta == 0:
        return channel_spec
    if _is_fused_channel_spec(channel_spec):
        spec_a, spec_b, offset = _parse_fused_channel_spec(channel_spec)
        return _compose_fused_channel_spec(spec_a, spec_b, offset + offset_delta)

    base_modality, transform, offset = _parse_channel_spec(channel_spec)
    if transform == "diff_prev":
        if offset_delta != 0:
            raise ValueError(f"Offsetting diff spec is not supported: {channel_spec}")
        return channel_spec
    return _compose_offset_channel_spec(base_modality, offset + offset_delta)


def _compose_offset_channel_spec(modality: str, offset: int) -> str:
    if offset == 0:
        return modality
    direction = "prev" if offset < 0 else "next"
    return f"{modality}_{direction}{abs(offset)}"


def _compose_fused_channel_spec(spec_a: str, spec_b: str, offset: int) -> str:
    base = f"fused({spec_a},{spec_b})"
    if offset == 0:
        return base
    direction = "prev" if offset < 0 else "next"
    return f"{base}_{direction}{abs(offset)}"


def _fuse_uint8_images(image_a: Optional[np.ndarray], image_b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image_a is None or image_b is None:
        return None
    fused = 0.5 * image_a.astype(np.float32) + 0.5 * image_b.astype(np.float32)
    return np.clip(np.round(fused), 0, 255).astype(np.uint8)


def pick_frame_index(num_frames: int, frame_mode: str) -> int:
    if num_frames <= 0:
        return 0
    if frame_mode == "first":
        return 0
    if frame_mode == "middle":
        return num_frames // 2
    if frame_mode == "event_start":
        return max(0, min(num_frames - 1, 0))
    return num_frames - 1


def find_event_start_frame_index(timestamps: List[str], start_time: str) -> int:
    if not timestamps:
        return 0
    target = datetime.fromisoformat(start_time.replace("Z", "").replace("+00:00", ""))
    parsed = [
        datetime.fromisoformat(ts.replace("Z", "").replace("+00:00", ""))
        for ts in timestamps
    ]
    deltas = [abs((ts - target).total_seconds()) for ts in parsed]
    return int(np.argmin(deltas))


def clip_box_xyxy(box: List[float], width: float, height: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), width)
    x2 = min(max(x2, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    y2 = min(max(y2, 0.0), height)
    return [x1, y1, x2, y2]


def xyxy_to_yolo(box: List[float], width: float, height: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw * 0.5
    cy = y1 + bh * 0.5
    return cx / width, cy / height, bw / width, bh / height


def _compute_time_delta_minutes(target_timestamp: Optional[str], image_path: Optional[Path]) -> Optional[float]:
    if not target_timestamp or image_path is None:
        return None
    image_dt = _parse_filename_datetime(image_path.name)
    if image_dt is None:
        return None
    target_dt = _parse_iso_datetime(target_timestamp)
    return abs((image_dt - target_dt).total_seconds()) / 60.0


def _select_processed_channel_paths(
    processed_root: Path,
    event_id: str,
    channel_modalities: List[str],
    target_timestamp: Optional[str],
    preferred_name: Optional[str],
    frame_mode: str,
) -> Dict[str, Optional[Path]]:
    channel_paths: Dict[str, Optional[Path]] = {}
    for channel_spec in channel_modalities:
        channel_paths[channel_spec] = _select_processed_image_path_from_spec(
            processed_root=processed_root,
            event_id=event_id,
            channel_spec=channel_spec,
            target_timestamp=target_timestamp,
            preferred_name=preferred_name,
            frame_mode=frame_mode,
        )
    return channel_paths


def _summarize_time_delta_minutes(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "max": None,
            "gt60min": 0,
            "gt180min": 0,
        }
    values_sorted = sorted(float(v) for v in values)

    def percentile(p: float) -> float:
        idx = min(len(values_sorted) - 1, int(round((len(values_sorted) - 1) * p)))
        return values_sorted[idx]

    return {
        "count": len(values_sorted),
        "min": round(values_sorted[0], 4),
        "p50": round(percentile(0.5), 4),
        "p90": round(percentile(0.9), 4),
        "max": round(values_sorted[-1], 4),
        "gt60min": int(sum(v > 60.0 for v in values_sorted)),
        "gt180min": int(sum(v > 180.0 for v in values_sorted)),
    }


def get_split_event_ids(split_events_path: Path) -> List[str]:
    with open(split_events_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def ordered_unique(items: List[str]) -> List[str]:
    seen = set()
    unique_items: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def _build_processed_region_export(
    bbox_info: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], float, float, Optional[str], Optional[str], int, int]:
    bbox_resolution = bbox_info.get("bbox_resolution") or {}
    bbox_width = float(bbox_resolution.get("width", bbox_info.get("anno_width", 2048)) or 2048)
    bbox_height = float(bbox_resolution.get("height", bbox_info.get("anno_height", 2048)) or 2048)

    region_entries = bbox_info.get("regions", []) or []
    activities = bbox_info.get("activities", []) or []
    region_map: Dict[str, Dict[str, Any]] = {}
    ordered_region_ids: List[str] = []
    for region in region_entries:
        region_id = str(region.get("region_id", "")).strip()
        if not region_id:
            continue
        if region_id not in region_map:
            ordered_region_ids.append(region_id)
        region_map[region_id] = region

    activity_region_ids = ordered_unique(
        [
            str(activity.get("region_id", "")).strip()
            for activity in activities
            if str(activity.get("region_id", "")).strip()
        ]
    )
    export_region_ids = [region_id for region_id in activity_region_ids if region_id in region_map]
    if not export_region_ids:
        export_region_ids = ordered_region_ids

    export_regions = [region_map[region_id] for region_id in export_region_ids]
    preferred_region = next(
        (region for region in region_entries if bool(region.get("is_primary_region", False))),
        None,
    )
    if preferred_region is None and export_regions:
        preferred_region = export_regions[0]

    preferred_name = None
    preferred_timestamp = None
    if preferred_region is not None:
        preferred_name = str(preferred_region.get("annotation_frame_name", "") or "") or None
        preferred_timestamp = str(preferred_region.get("annotation_frame_timestamp", "") or "") or None

    duplicate_refs_removed = max(0, len(activity_region_ids) - len(export_regions))
    return (
        export_regions,
        bbox_width,
        bbox_height,
        preferred_name,
        preferred_timestamp,
        len(activity_region_ids),
        duplicate_refs_removed,
    )


def _required_modalities_for_channel_spec(channel_spec: str) -> List[str]:
    if _is_fused_channel_spec(channel_spec):
        spec_a, spec_b, _ = _parse_fused_channel_spec(channel_spec)
        return ordered_unique(
            _required_modalities_for_channel_spec(spec_a) +
            _required_modalities_for_channel_spec(spec_b)
        )
    base_modality, _, _ = _parse_channel_spec(channel_spec)
    return [base_modality]


def export_split(
    reader: Optional[Any],
    event_ids: List[str],
    split_name: str,
    modality: str,
    channel_modalities: List[str],
    frame_mode: str,
    output_root: Path,
    skip_empty: bool,
    image_source: str,
    label_source: str,
    output_size: Optional[int],
    processed_root: Path,
    save_npy_sidecar: bool,
    max_time_delta_minutes: Optional[float],
) -> Dict[str, Any]:
    split_image_dir = output_root / "images" / split_name
    split_label_dir = output_root / "labels" / split_name
    split_image_dir.mkdir(parents=True, exist_ok=True)
    split_label_dir.mkdir(parents=True, exist_ok=True)

    num_images = 0
    num_positive_images = 0
    num_boxes = 0
    skipped_missing_modality = 0
    skipped_no_boxes = 0
    total_activity_refs = 0
    total_unique_regions = 0
    duplicate_region_refs_removed = 0
    skipped_missing_processed_image = 0
    skipped_missing_processed_label = 0
    skipped_time_delta_exceeded = 0
    modality_time_delta_minutes: Dict[str, List[float]] = {spec: [] for spec in channel_modalities}

    for event_id in event_ids:
        requested_modalities = ordered_unique(
            [modality] + [
                required_modality
                for spec in channel_modalities
                for required_modality in _required_modalities_for_channel_spec(spec)
            ]
        )
        event_data: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}
        timestamps: List[str] = []
        primary_images: Optional[np.ndarray] = None
        if image_source == "hdf5" or label_source == "hdf5":
            if reader is None:
                raise ValueError("reader is required when image_source or label_source is hdf5")
            event_data = reader.get_event_data(event_id, modalities=requested_modalities)
            metadata = reader.get_event_metadata(event_id)
            timestamps = event_data.get("timestamps", [])
            if image_source == "hdf5":
                primary_images = event_data.get(modality)
                if primary_images is None:
                    skipped_missing_modality += 1
                    continue
                if any(event_data.get(required_modality) is None for required_modality in requested_modalities):
                    skipped_missing_modality += 1
                    continue

        bbox_info = _load_bbox_info(processed_root, event_id) if (image_source == "processed" or label_source == "processed") else {}
        yolo_lines: List[str] = []
        preferred_name = None
        preferred_timestamp = None
        if label_source == "processed":
            if not bbox_info or not bool(bbox_info.get("annotated", False)):
                skipped_missing_processed_label += 1
                continue
            (
                export_regions,
                bbox_width,
                bbox_height,
                preferred_name,
                preferred_timestamp,
                activity_ref_count,
                duplicate_refs_removed,
            ) = _build_processed_region_export(bbox_info)
            total_activity_refs += activity_ref_count
            total_unique_regions += len(export_regions)
            duplicate_region_refs_removed += duplicate_refs_removed
            for region in export_regions:
                norm_box = clip_box_xyxy(region.get("bbox", [0, 0, 0, 0]), bbox_width, bbox_height)
                cx, cy, bw, bh = xyxy_to_yolo(norm_box, bbox_width, bbox_height)
                if bw <= 0.0 or bh <= 0.0:
                    continue
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        else:
            assert primary_images is not None
            regions = event_data.get("regions")
            activity_region_ids = event_data.get("activity_region_ids", []) or []
            region_ids = event_data.get("region_ids", []) or []
            bbox_resolution = event_data.get("bbox_resolution", {}) or {}
            bbox_width = float(bbox_resolution.get("width", primary_images.shape[-1]) or primary_images.shape[-1])
            bbox_height = float(bbox_resolution.get("height", primary_images.shape[-2]) or primary_images.shape[-2])

            region_box_map: Dict[str, List[float]] = {}
            if regions is not None:
                for rid, box in zip(region_ids, regions):
                    region_box_map[str(rid)] = clip_box_xyxy(box.tolist(), bbox_width, bbox_height)

            total_activity_refs += len(activity_region_ids)
            unique_activity_region_ids = [
                rid for rid in ordered_unique([str(rid) for rid in activity_region_ids])
                if rid in region_box_map
            ]
            total_unique_regions += len(unique_activity_region_ids)
            duplicate_region_refs_removed += max(0, len(activity_region_ids) - len(unique_activity_region_ids))
            for rid in unique_activity_region_ids:
                norm_box = region_box_map[rid]
                cx, cy, bw, bh = xyxy_to_yolo(norm_box, bbox_width, bbox_height)
                if bw <= 0.0 or bh <= 0.0:
                    continue
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines and skip_empty:
            skipped_no_boxes += 1
            continue

        frame_idx = 0
        target_timestamp = None
        if image_source == "processed":
            if frame_mode == "annotated":
                target_timestamp = preferred_timestamp or _event_id_to_timestamp_string(event_id)
            elif frame_mode == "event_start":
                target_timestamp = metadata.get("start_time") or _event_id_to_timestamp_string(event_id)
        else:
            assert primary_images is not None
            if frame_mode == "event_start":
                frame_idx = find_event_start_frame_index(timestamps, metadata["start_time"])
            elif frame_mode == "annotated" and preferred_timestamp:
                frame_idx = find_event_start_frame_index(timestamps, preferred_timestamp)
            else:
                frame_idx = pick_frame_index(primary_images.shape[0], frame_mode)
            target_timestamp = timestamps[frame_idx] if timestamps and 0 <= frame_idx < len(timestamps) else None

        if image_source == "processed" and max_time_delta_minutes is not None and max_time_delta_minutes > 0:
            channel_paths = _select_processed_channel_paths(
                processed_root=processed_root,
                event_id=event_id,
                channel_modalities=channel_modalities,
                target_timestamp=target_timestamp,
                preferred_name=preferred_name,
                frame_mode=frame_mode,
            )
            channel_deltas = {
                channel_spec: _compute_time_delta_minutes(target_timestamp, channel_path)
                for channel_spec, channel_path in channel_paths.items()
            }
            if any(delta is None or delta > max_time_delta_minutes for delta in channel_deltas.values()):
                skipped_time_delta_exceeded += 1
                continue

        preview_channel_modalities = [modality] if save_npy_sidecar else channel_modalities
        image_rgb = _build_rgb_image_from_modalities(
            event_id=event_id,
            event_data=event_data,
            frame_idx=frame_idx,
            channel_modalities=preview_channel_modalities,
            image_source=image_source,
            processed_root=processed_root,
            target_timestamp=target_timestamp,
            preferred_name=preferred_name,
            frame_mode=frame_mode,
            output_size=output_size,
        )
        image_channel_stack = None
        if save_npy_sidecar:
            image_channel_stack = _build_channel_stack_from_modalities(
                event_id=event_id,
                event_data=event_data,
                frame_idx=frame_idx,
                channel_modalities=channel_modalities,
                image_source=image_source,
                processed_root=processed_root,
                target_timestamp=target_timestamp,
                preferred_name=preferred_name,
                frame_mode=frame_mode,
                output_size=output_size,
            )
        if image_rgb is None or (save_npy_sidecar and image_channel_stack is None):
            if image_source == "processed":
                skipped_missing_processed_image += 1
            else:
                skipped_missing_modality += 1
            continue

        if image_source == "processed":
            for channel_spec in channel_modalities:
                selected_image_path = _select_processed_image_path_from_spec(
                    processed_root=processed_root,
                    event_id=event_id,
                    channel_spec=channel_spec,
                    target_timestamp=target_timestamp,
                    preferred_name=preferred_name,
                    frame_mode=frame_mode,
                )
                delta_minutes = _compute_time_delta_minutes(target_timestamp, selected_image_path)
                if delta_minutes is not None:
                    modality_time_delta_minutes[channel_spec].append(delta_minutes)

        image_name = f"{event_id}.png"
        label_name = f"{event_id}.txt"
        Image.fromarray(image_rgb).save(split_image_dir / image_name)
        if image_channel_stack is not None:
            np.save(split_image_dir / f"{event_id}.npy", image_channel_stack.astype(np.uint8, copy=False))
        with open(split_label_dir / label_name, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        num_images += 1
        if yolo_lines:
            num_positive_images += 1
            num_boxes += len(yolo_lines)

    return {
        "split": split_name,
        "num_images": num_images,
        "num_positive_images": num_positive_images,
        "num_boxes": num_boxes,
        "skipped_missing_modality": skipped_missing_modality,
        "skipped_missing_processed_image": skipped_missing_processed_image,
        "skipped_missing_processed_label": skipped_missing_processed_label,
        "skipped_time_delta_exceeded": skipped_time_delta_exceeded,
        "skipped_no_boxes": skipped_no_boxes,
        "total_activity_refs": total_activity_refs,
        "total_unique_regions": total_unique_regions,
        "duplicate_region_refs_removed": duplicate_region_refs_removed,
        "processed_time_delta_minutes": {
            channel_spec: _summarize_time_delta_minutes(values)
            for channel_spec, values in modality_time_delta_minutes.items()
        },
    }


def main() -> int:
    args = parse_args()
    data_cfg = load_data_config(args.data_config)
    hdf5_path = str(data_cfg.get("hdf5_path", ""))
    reader: Optional[Any] = None
    if args.image_source == "hdf5" or args.label_source == "hdf5":
        if not hdf5_path:
            raise ValueError("data_config must provide hdf5_path when image_source or label_source is hdf5")
        reader = get_hdf5_dataset_reader()(hdf5_path)
    processed_root = Path(data_cfg["processed_data_dir"])
    channel_modalities = args.channel_modalities or [args.modality]
    _validate_channel_specs(channel_modalities)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)

    summaries: List[Dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        event_ids = get_split_event_ids(split_dir / f"{split_name}_events.txt")
        summary = export_split(
            reader=reader,
            event_ids=event_ids,
            split_name=split_name,
            modality=args.modality,
            channel_modalities=channel_modalities,
            frame_mode=args.frame,
            output_root=output_root,
            skip_empty=args.skip_empty,
            image_source=args.image_source,
            label_source=args.label_source,
            output_size=args.output_size,
            processed_root=processed_root,
            save_npy_sidecar=args.save_npy_sidecar,
            max_time_delta_minutes=args.max_time_delta_minutes,
        )
        summaries.append(summary)

    dataset_channels = len(channel_modalities) if args.save_npy_sidecar else 3
    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "channels": dataset_channels,
        "names": {
            0: "activity_region"
        },
    }
    with open(output_root / "dataset.yaml", "w", encoding="utf-8") as f:
        for key, value in dataset_yaml.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_val in value.items():
                    f.write(f"  {sub_key}: {sub_val}\n")
            else:
                f.write(f"{key}: {value}\n")

    export_summary = {
        "hdf5_path": hdf5_path,
        "modality": args.modality,
        "channel_modalities": channel_modalities,
        "frame": args.frame,
        "skip_empty": args.skip_empty,
        "image_source": args.image_source,
        "label_source": args.label_source,
        "output_size": args.output_size,
        "save_npy_sidecar": args.save_npy_sidecar,
        "max_time_delta_minutes": args.max_time_delta_minutes,
        "dataset_channels": dataset_channels,
        "export_mode": "one_image_per_event",
        "splits": summaries,
    }
    with open(output_root / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(export_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(export_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if USE_DIRECT_RUN_CONFIG and len(sys.argv) == 1:
        sys.argv.extend(_build_direct_run_args())
    raise SystemExit(main())
