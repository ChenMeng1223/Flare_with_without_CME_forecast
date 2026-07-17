"""
Generate a stage2 proposal cache from a trained YOLO11 detector.

This bridges the detector-first pipeline:
1. export YOLO dataset
2. train/evaluate YOLO11 detector
3. run YOLO11 on every stage2 window
4. save proposal cache keyed by window_id for stage2 training/inference
"""
import argparse
import json
import os
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
yolo_config_dir = project_root / "outputs" / "ultralytics_config"
yolo_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)


def _load_module_attr(module_name: str, relative_path: str, attr_name: str):
    module_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {attr_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def _load_module(module_name: str, relative_path: str):
    module_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HDF5DatasetReader = _load_module_attr("hdf5_reader_only", "data/hdf5_reader.py", "HDF5DatasetReader")
SolarFlareDataset = _load_module_attr("dataset_only", "data/dataset.py", "SolarFlareDataset")
load_config = _load_module_attr("config_utils_only", "utils/config_utils.py", "load_config")
yolo_export_utils = _load_module("yolo_export_utils", "scripts/j_export_yolo_dataset.py")


SUPPORTED_MODALITIES = ["magnetogram", "euv_94", "euv_171", "euv_193", "halpha"]


USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    # Change this when you want to use another YOLO detector checkpoint.
    "weights": "runs/detect/outputs/yolo_runs/yolo11n_activity_region_mag_euv94_euv171_processed_512_event_start_control/weights/best.pt",
    "data_config": "configs/data_config_stage2_yolo11_5modal_256.yaml",
    "output": "outputs/proposals/stage2_yolo11_mag_euv171_euv94_512_event_start_conf005_maxdet15.json",

    # These should match how the YOLO detector was trained/exported.
    "channel_modalities": ["magnetogram", "euv_94", "euv_171"],
    "frame_source": "event_start",
    "imgsz": 512,
    "device": "0",
    "conf": 0.005,
    "iou": 0.5,
    "max_det": 15,
    "model_channels": 3,
    "image_source": "processed",
}


def _build_direct_run_args() -> List[str]:
    cfg = dict(DIRECT_RUN_CONFIG)
    args: List[str] = []
    for key in [
        "weights",
        "data_config",
        "output",
        "frame_source",
        "imgsz",
        "device",
        "conf",
        "iou",
        "max_det",
        "model_channels",
        "image_source",
    ]:
        value = cfg.get(key)
        if value not in (None, ""):
            args.extend([f"--{key}", str(value)])

    channel_modalities = cfg.get("channel_modalities") or []
    if channel_modalities:
        args.append("--channel_modalities")
        args.extend(str(item) for item in channel_modalities)

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stage2 proposal cache from YOLO11.")
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        help="Path to trained YOLO11 weights, e.g. outputs/yolo_runs/.../weights/best.pt",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_config.yaml",
        help="Main data config path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/proposals/stage2_yolo11_proposals.json",
        help="Output proposal cache path.",
    )
    parser.add_argument(
        "--channel_modalities",
        type=str,
        nargs="+",
        default=["magnetogram"],
        help="Channel specs used to build detector input channels, aligned with scripts/j_export_yolo_dataset.py.",
    )
    parser.add_argument(
        "--frame_source",
        type=str,
        default="event_start",
        choices=["window_first", "window_middle", "window_last", "event_start"],
        help="Which frame inside each stage2 window is sent to YOLO11.",
    )
    parser.add_argument("--imgsz", type=int, default=512, help="YOLO inference image size.")
    parser.add_argument("--device", type=str, default="0", help="YOLO inference device, e.g. 0 or cpu.")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max_det", type=int, default=5, help="Maximum detections kept per window.")
    parser.add_argument(
        "--model_channels",
        type=int,
        default=3,
        help="Detector input channel count. For the current single-modality baseline this should stay 3.",
    )
    parser.add_argument(
        "--image_source",
        type=str,
        default="processed",
        choices=["hdf5", "processed"],
        help="Use images from HDF5 arrays or processed original-resolution images before resizing to imgsz.",
    )
    if USE_DIRECT_RUN_CONFIG and len(sys.argv) == 1:
        return parser.parse_args(_build_direct_run_args())
    args = parser.parse_args()
    if not args.weights:
        parser.error("--weights is required unless USE_DIRECT_RUN_CONFIG is enabled for direct file runs.")
    return args


def build_detector_input(
    event_id: str,
    event_data: Dict[str, Any],
    global_frame_idx: int,
    channel_modalities: List[str],
    model_channels: int,
    output_size: int,
    image_source: str,
    processed_root: Path,
    frame_mode: str,
) -> Optional[np.ndarray]:
    timestamps = event_data.get("timestamps", []) or []
    target_timestamp = (
        timestamps[global_frame_idx]
        if 0 <= global_frame_idx < len(timestamps)
        else None
    )
    stacked = yolo_export_utils._build_channel_stack_from_modalities(
        event_id=event_id,
        event_data=event_data,
        frame_idx=global_frame_idx,
        channel_modalities=channel_modalities,
        image_source=image_source,
        processed_root=processed_root,
        target_timestamp=target_timestamp,
        preferred_name=None,
        frame_mode=frame_mode,
        output_size=output_size,
    )
    if stacked is None:
        return None
    if model_channels == stacked.shape[-1]:
        return stacked
    if model_channels == 3 and stacked.shape[-1] == 1:
        return np.repeat(stacked, 3, axis=-1)
    if model_channels == 3 and stacked.shape[-1] == 2:
        return np.concatenate([stacked, stacked[..., -1:]], axis=-1)
    if model_channels < stacked.shape[-1]:
        return stacked[..., :model_channels]
    if model_channels > stacked.shape[-1]:
        last = stacked[..., -1:]
        repeat_count = model_channels - stacked.shape[-1]
        return np.concatenate([stacked, np.repeat(last, repeat_count, axis=-1)], axis=-1)
    return stacked


def pick_local_frame_index(sequence_length: int, frame_source: str) -> int:
    if sequence_length <= 0:
        return 0
    if frame_source == "window_first":
        return 0
    if frame_source == "window_middle":
        return sequence_length // 2
    return sequence_length - 1


def detector_frame_mode(frame_source: str) -> str:
    if frame_source == "window_first":
        return "first"
    if frame_source == "window_middle":
        return "middle"
    if frame_source == "window_last":
        return "last"
    return "event_start"


def _parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "").replace("+00:00", ""))


def pick_event_start_global_frame_index(
    timestamps: List[Any],
    event_start_time: Optional[str],
    start_idx: int,
    end_idx: int,
) -> int:
    if not timestamps:
        return max(0, int(start_idx))
    if not event_start_time:
        return max(0, min(len(timestamps) - 1, int(start_idx)))

    bounded_start = max(0, int(start_idx))
    bounded_end = min(len(timestamps), int(end_idx))
    if bounded_start >= bounded_end:
        return max(0, min(len(timestamps) - 1, bounded_start))

    target_dt = _parse_iso_datetime(event_start_time)
    best_idx = bounded_start
    best_delta: Optional[float] = None
    for global_idx in range(bounded_start, bounded_end):
        try:
            ts_dt = _parse_iso_datetime(timestamps[global_idx])
        except Exception:
            continue
        delta = abs((ts_dt - target_dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = global_idx
    return best_idx


def to_normalized_xyxy(box_xyxy: np.ndarray, image_size: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    x1, x2 = sorted((max(0.0, min(image_size, x1)), max(0.0, min(image_size, x2))))
    y1, y2 = sorted((max(0.0, min(image_size, y1)), max(0.0, min(image_size, y2))))
    return [
        x1 / image_size,
        y1 / image_size,
        x2 / image_size,
        y2 / image_size,
    ]


def main() -> int:
    args = parse_args()
    yolo_export_utils._validate_channel_specs(list(args.channel_modalities))

    data_cfg = load_config(args.data_config)["data"]
    hdf5_path = data_cfg.get("hdf5_path", "data/Solar_Flares_CME_dataset.h5")
    processed_root = Path(data_cfg.get("processed_data_dir", "data/processed"))
    if not processed_root.is_absolute():
        processed_root = (project_root / processed_root).resolve()
    required_modalities = yolo_export_utils.ordered_unique(
        required_modality
        for spec in args.channel_modalities
        for required_modality in yolo_export_utils._required_modalities_for_channel_spec(spec)
    )
    max_activities = int(data_cfg.get("max_activities", args.max_det))
    sequence_length = int(data_cfg.get("sequence_length", 9))
    stride = int(data_cfg.get("stride", 1))
    target_size = tuple(
        next(iter(data_cfg.get("modalities", {}).values())).get("resolution", [args.imgsz, args.imgsz])
    )

    reader = HDF5DatasetReader(hdf5_path)
    event_ids = reader.get_event_ids(available_only=True)

    dataset = SolarFlareDataset(
        reader=reader,
        event_ids=event_ids,
        modalities=required_modalities,
        sequence_length=sequence_length,
        stride=stride,
        target_size=target_size,
        max_activities=max_activities,
    )

    yolo_config_dir = project_root / "outputs" / "ultralytics_config"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    proposal_cache: Dict[str, Dict[str, Any]] = {}
    local_frame_idx = pick_local_frame_index(sequence_length, args.frame_source)
    frame_mode = detector_frame_mode(args.frame_source)
    event_cache: Dict[str, Dict[str, Any]] = {}
    skipped_missing_modality_windows = 0
    skipped_missing_modality_events = set()
    generated_windows = 0

    for window_info in dataset.windows:
        event_id = window_info["event_id"]
        start_idx = int(window_info["start_idx"])
        end_idx = int(window_info["end_idx"])
        window_id = str(window_info["window_id"])

        if event_id not in event_cache:
            event_cache[event_id] = reader.get_event_data(event_id, modalities=required_modalities)
        event_data = event_cache[event_id]

        effective_sequence_length = max(1, end_idx - start_idx)
        if args.frame_source == "event_start":
            event_meta = reader.get_event_metadata(event_id)
            global_frame_idx = pick_event_start_global_frame_index(
                timestamps=event_data.get("timestamps", []) or [],
                event_start_time=event_meta.get("start_time"),
                start_idx=start_idx,
                end_idx=end_idx,
            )
        else:
            chosen_local_idx = min(local_frame_idx, effective_sequence_length - 1)
            global_frame_idx = start_idx + chosen_local_idx

        detector_input = build_detector_input(
            event_id=event_id,
            event_data=event_data,
            global_frame_idx=global_frame_idx,
            channel_modalities=args.channel_modalities,
            model_channels=args.model_channels,
            output_size=args.imgsz,
            image_source=args.image_source,
            processed_root=processed_root,
            frame_mode=frame_mode,
        )

        if detector_input is None:
            skipped_missing_modality_windows += 1
            skipped_missing_modality_events.add(event_id)
            proposal_cache[window_id] = {
                "proposal_boxes": [],
                "proposal_scores": [],
                "event_id": event_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "frame_source": args.frame_source,
                "detector_modalities": list(args.channel_modalities),
                "image_source": args.image_source,
                "skipped_reason": "missing_detector_modality",
            }
            continue

        # Ultralytics treats numpy image inputs as already-loaded OpenCV images
        # (BGR). The exported YOLO PNGs are read that way during training and
        # native-path visualization, so mirror that channel order here.
        detector_input_bgr = detector_input[:, :, ::-1].copy()

        results = model.predict(
            source=[detector_input_bgr],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=min(args.max_det, max_activities),
            device=args.device,
            verbose=False,
            stream=False,
        )
        result = results[0]

        proposal_boxes: List[List[float]] = []
        proposal_scores: List[float] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            order = np.argsort(-confs)
            xyxy = xyxy[order]
            confs = confs[order]
            for box_xyxy, score in zip(xyxy[:max_activities], confs[:max_activities]):
                proposal_boxes.append(to_normalized_xyxy(box_xyxy.astype(np.float32), args.imgsz))
                proposal_scores.append(float(score))

        proposal_cache[window_id] = {
            "proposal_boxes": proposal_boxes,
            "proposal_scores": proposal_scores,
            "event_id": event_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "frame_source": args.frame_source,
            "detector_modalities": list(args.channel_modalities),
            "image_source": args.image_source,
        }
        generated_windows += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proposal_cache, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "weights": str(Path(args.weights).resolve()),
                "hdf5_path": hdf5_path,
                "frame_source": args.frame_source,
                "channel_modalities": list(args.channel_modalities),
                "model_channels": args.model_channels,
                "image_source": args.image_source,
                "processed_root": str(processed_root),
                "num_windows": len(dataset.windows),
                "generated_windows": generated_windows,
                "skipped_missing_modality_windows": skipped_missing_modality_windows,
                "skipped_missing_modality_events": len(skipped_missing_modality_events),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
