"""
End-to-end cascade evaluation:
YOLO11 proposals -> stage-2 multimodal model -> post-processing -> metrics.

This script intentionally does not use the proposal cache. It generates YOLO
proposals on the fly for each split window and feeds them directly into the
stage-2 model.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "outputs" / "ultralytics_config"))

from data.hdf5_reader import HDF5DatasetReader
from data.dataset import SolarFlareDataset
from models.multimodal_transformer import MultimodalTransformer
from scripts.p_generate_yolo11_stage2_proposals import (
    build_detector_input,
    detector_frame_mode,
    pick_event_start_global_frame_index,
    pick_local_frame_index,
    to_normalized_xyxy,
    yolo_export_utils,
)


USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    "yolo_weights": "runs/detect/outputs/yolo_runs/yolo11n_activity_region_mag_euv94_euv171_processed_512_event_start_control/weights/best.pt",
    "stage2_weights": "outputs/checkpoints_stage2_yolo11_5modal/solar_flare_cme_model_epoch_0114.pth",
    "data_config": "configs/data_config_stage2_yolo11_5modal_256.yaml",
    "model_config": "configs/model_config.yaml",
    "train_config": "configs/training_config_stage2_yolo11_5modal.yaml",
    "inference_config": "configs/inference_config_stage2_yolo11_5modal_256.yaml",
    "split": "test",
    "output_dir": "outputs/cascade_yolo11_stage2_5modal_test",
    "device": "0",
    "stage2_device": "",
    "imgsz": 512,
    "conf": 0.005,
    "iou": 0.5,
    "max_det": 15,
    "frame_source": "event_start",
    "image_source": "processed",
    "channel_modalities": "magnetogram,euv_94,euv_171",
    "model_channels": 3,
    "max_events": 0,
    "iou_threshold": 0.5,
    "save_visualizations": True,
    "viz_modality": "magnetogram",
    "viz_frame": "last",
}


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PostProcessor = _load_module("cascade_post_processing", PROJECT_ROOT / "inference" / "post_processing.py").PostProcessor


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end YOLO11 -> stage2 cascade inference/evaluation.")
    parser.add_argument("--yolo_weights", type=str, default=DIRECT_RUN_CONFIG["yolo_weights"])
    parser.add_argument("--stage2_weights", type=str, default=DIRECT_RUN_CONFIG["stage2_weights"])
    parser.add_argument("--data_config", type=str, default=DIRECT_RUN_CONFIG["data_config"])
    parser.add_argument("--model_config", type=str, default=DIRECT_RUN_CONFIG["model_config"])
    parser.add_argument("--train_config", type=str, default=DIRECT_RUN_CONFIG["train_config"])
    parser.add_argument("--inference_config", type=str, default=DIRECT_RUN_CONFIG["inference_config"])
    parser.add_argument("--split", type=str, default=DIRECT_RUN_CONFIG["split"], choices=["train", "val", "test"])
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=DIRECT_RUN_CONFIG["output_dir"])
    parser.add_argument("--device", type=str, default=DIRECT_RUN_CONFIG["device"], help="YOLO device, e.g. 0 or cpu.")
    parser.add_argument("--stage2_device", type=str, default=DIRECT_RUN_CONFIG["stage2_device"], help="Stage2 device. Empty means auto.")
    parser.add_argument("--imgsz", type=int, default=DIRECT_RUN_CONFIG["imgsz"])
    parser.add_argument("--conf", type=float, default=DIRECT_RUN_CONFIG["conf"])
    parser.add_argument("--iou", type=float, default=DIRECT_RUN_CONFIG["iou"])
    parser.add_argument("--max_det", type=int, default=DIRECT_RUN_CONFIG["max_det"])
    parser.add_argument("--frame_source", type=str, default=DIRECT_RUN_CONFIG["frame_source"])
    parser.add_argument("--image_source", type=str, default=DIRECT_RUN_CONFIG["image_source"], choices=["hdf5", "processed"])
    parser.add_argument("--channel_modalities", type=str, default=DIRECT_RUN_CONFIG["channel_modalities"])
    parser.add_argument("--model_channels", type=int, default=DIRECT_RUN_CONFIG["model_channels"])
    parser.add_argument("--max_events", type=int, default=DIRECT_RUN_CONFIG["max_events"])
    parser.add_argument("--iou_threshold", type=float, default=DIRECT_RUN_CONFIG["iou_threshold"])
    parser.add_argument("--viz_modality", type=str, default=DIRECT_RUN_CONFIG["viz_modality"])
    parser.add_argument("--viz_frame", type=str, default=DIRECT_RUN_CONFIG["viz_frame"], choices=["first", "middle", "last"])
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=DIRECT_RUN_CONFIG["save_visualizations"])
    if USE_DIRECT_RUN_CONFIG and len(sys.argv) == 1:
        args: List[str] = []
        for key, value in DIRECT_RUN_CONFIG.items():
            if isinstance(value, bool):
                args.append(f"--{key}" if value else f"--no-{key}")
                continue
            if value not in (None, ""):
                args.extend([f"--{key}", str(value)])
        return parser.parse_args(args)
    return parser.parse_args()


def enabled_modalities(data_cfg: Dict[str, Any]) -> List[str]:
    return [name for name, cfg in data_cfg.get("modalities", {}).items() if bool(cfg.get("enabled", True))]


def load_split_event_ids(split: str, split_file: str, reader: HDF5DatasetReader, max_events: int) -> List[str]:
    path = resolve_path(split_file) if split_file else PROJECT_ROOT / "data" / "split_data" / f"{split}_events.txt"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            event_ids = [line.strip() for line in f if line.strip()]
        available = set(reader.get_event_ids(available_only=True))
        event_ids = [event_id for event_id in event_ids if event_id in available]
    else:
        event_ids = reader.get_event_ids(available_only=True)
    return event_ids[:max_events] if max_events and max_events > 0 else event_ids


def resolve_stage2_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_config(data_cfg: Dict[str, Any], model_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = dict(model_cfg)
    mods = {name: cfg for name, cfg in data_cfg.get("modalities", {}).items() if bool(cfg.get("enabled", True))}
    first_mod = next(iter(mods.values()), {})
    model_cfg["modalities"] = mods
    model_cfg["max_activities"] = int(data_cfg.get("max_activities", model_cfg.get("max_activities", 5)))
    model_cfg["sequence_length"] = int(data_cfg.get("sequence_length", model_cfg.get("sequence_length", 10)))
    model_cfg["max_sequence_length"] = model_cfg["sequence_length"]
    model_cfg["input_size"] = list(first_mod.get("resolution", model_cfg.get("input_size", [256, 256])))
    model_cfg.setdefault("stage2", {})
    if train_cfg.get("two_stage_schedule", {}).get("enabled", False):
        model_cfg["stage2"]["roi_source_mix"] = train_cfg["two_stage_schedule"].get("roi_source_mix", {"predicted": 1.0})
    return model_cfg


def checkpoint_has_time_head(state_dict: Dict[str, Any]) -> bool:
    prefixes = (
        "stage_two_predictor.time_sequence_attn.",
        "stage_two_predictor.time_sequence_norm.",
        "stage_two_predictor.time_predictor.",
    )
    return any(str(key).startswith(prefixes) for key in state_dict.keys())


def load_stage2_model(weights: Path, model_cfg: Dict[str, Any], device: torch.device) -> MultimodalTransformer:
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    head_cfg = dict(model_cfg.get("prediction_heads", {}))
    time_cfg = dict(head_cfg.get("time", {}))
    time_cfg["enabled"] = checkpoint_has_time_head(state_dict)
    head_cfg["time"] = time_cfg
    model_cfg["prediction_heads"] = head_cfg
    model = MultimodalTransformer(model_cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def tensor_to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): tensor_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def make_stage2_inputs(sample: Dict[str, Any], proposal_boxes: torch.Tensor, proposal_scores: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    inputs: Dict[str, torch.Tensor] = {}
    for modality, value in (sample.get("data") or {}).items():
        if value is None:
            continue
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        inputs[modality] = value.unsqueeze(0).to(device)
    inputs["proposal_boxes"] = proposal_boxes.unsqueeze(0).to(device)
    inputs["proposal_scores"] = proposal_scores.unsqueeze(0).to(device)
    return inputs


def generate_yolo_proposals(
    yolo_model: Any,
    event_id: str,
    event_data: Dict[str, Any],
    event_meta: Dict[str, Any],
    start_idx: int,
    end_idx: int,
    args: argparse.Namespace,
    processed_root: Path,
    max_activities: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if args.frame_source == "event_start":
        global_frame_idx = pick_event_start_global_frame_index(
            timestamps=event_data.get("timestamps", []) or [],
            event_start_time=event_meta.get("start_time"),
            start_idx=start_idx,
            end_idx=end_idx,
        )
    else:
        local_idx = min(pick_local_frame_index(end_idx - start_idx, args.frame_source), max(0, end_idx - start_idx - 1))
        global_frame_idx = start_idx + local_idx

    detector_input = build_detector_input(
        event_id=event_id,
        event_data=event_data,
        global_frame_idx=global_frame_idx,
        channel_modalities=[item.strip() for item in args.channel_modalities.split(",") if item.strip()],
        model_channels=args.model_channels,
        output_size=args.imgsz,
        image_source=args.image_source,
        processed_root=processed_root,
        frame_mode=detector_frame_mode(args.frame_source),
    )

    boxes = torch.zeros((max_activities, 4), dtype=torch.float32)
    scores = torch.zeros((max_activities, 1), dtype=torch.float32)
    debug: Dict[str, Any] = {
        "detector_frame_idx": int(global_frame_idx),
        "num_yolo_proposals": 0,
        "proposal_scores": [],
    }
    if detector_input is None:
        debug["skipped_reason"] = "missing_detector_input"
        return boxes, scores, debug

    detector_input_bgr = detector_input[:, :, ::-1].copy()
    results = yolo_model.predict(
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
    if result.boxes is None or len(result.boxes) == 0:
        return boxes, scores, debug

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    order = np.argsort(-confs)
    xyxy = xyxy[order]
    confs = confs[order]
    kept = min(max_activities, len(confs))
    for idx, (box_xyxy, score) in enumerate(zip(xyxy[:kept], confs[:kept])):
        boxes[idx] = torch.tensor(to_normalized_xyxy(box_xyxy.astype(np.float32), args.imgsz), dtype=torch.float32)
        scores[idx, 0] = float(score)
    debug["num_yolo_proposals"] = int(kept)
    debug["proposal_scores"] = [float(v) for v in confs[:kept].tolist()]
    return boxes, scores, debug


def clip_valid_boxes(boxes: Any) -> np.ndarray:
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    if torch.is_tensor(boxes):
        arr = boxes.detach().cpu().numpy()
    else:
        arr = np.asarray(boxes)
    arr = arr.reshape(-1, 4).astype(np.float32) if arr.size else np.zeros((0, 4), dtype=np.float32)
    out = []
    for box in arr:
        if not np.isfinite(box).all() or float(np.abs(box).sum()) <= 0.0:
            continue
        x1, y1, x2, y2 = box.tolist()
        box = np.asarray([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)
        box = np.clip(box, 0.0, 1.0)
        if (box[2] - box[0]) > 1e-6 and (box[3] - box[1]) > 1e-6:
            out.append(box)
    return np.asarray(out, dtype=np.float32).reshape(-1, 4) if out else np.zeros((0, 4), dtype=np.float32)


def unique_gt_regions(sample: Dict[str, Any]) -> Tuple[np.ndarray, List[int]]:
    boxes = sample.get("bbox")
    labels = sample.get("label")
    mask = sample.get("activity_mask")
    if torch.is_tensor(boxes):
        boxes_arr = boxes.detach().cpu().numpy()
    else:
        boxes_arr = np.asarray(boxes)
    if torch.is_tensor(labels):
        labels_arr = labels.detach().cpu().numpy()
    else:
        labels_arr = np.asarray(labels)
    if torch.is_tensor(mask):
        mask_arr = mask.detach().cpu().numpy().astype(bool)
    else:
        mask_arr = np.asarray(mask).astype(bool)

    region_ids = list((sample.get("metadata") or {}).get("activity_region_ids", []))
    seen_ids = set()
    seen_boxes = set()
    out_boxes: List[np.ndarray] = []
    out_labels: List[int] = []
    for idx, box in enumerate(boxes_arr.reshape(-1, 4)):
        if idx >= len(mask_arr) or not mask_arr[idx]:
            continue
        label = int(labels_arr[idx]) if idx < len(labels_arr) else 0
        if label <= 0:
            continue
        region_id = str(region_ids[idx]).strip() if idx < len(region_ids) and region_ids[idx] is not None else ""
        if region_id:
            if region_id in seen_ids:
                continue
            seen_ids.add(region_id)
        else:
            key = tuple(np.round(np.asarray(box, dtype=np.float32), 6).tolist())
            if key in seen_boxes:
                continue
            seen_boxes.add(key)
        out_boxes.append(np.asarray(box, dtype=np.float32))
        out_labels.append(label)
    return clip_valid_boxes(out_boxes), out_labels


def safe_filename(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "window"


def select_background_frame(sample: Dict[str, Any], modality: str, frame: str) -> Optional[np.ndarray]:
    data = (sample.get("data") or {}).get(modality)
    if data is None:
        sample_data = sample.get("data") or {}
        if not sample_data:
            return None
        data = next(iter(sample_data.values()))
    if torch.is_tensor(data):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    if arr.ndim == 4:
        # T, C, H, W
        t = arr.shape[0]
        idx = 0 if frame == "first" else (t // 2 if frame == "middle" else t - 1)
        arr = arr[idx, 0]
    elif arr.ndim == 3:
        # T, H, W or C, H, W
        idx = 0 if frame == "first" else (arr.shape[0] // 2 if frame == "middle" else arr.shape[0] - 1)
        arr = arr[idx]
    elif arr.ndim != 2:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def draw_label(draw: Any, xy: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 2
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text(xy, text, fill=color, font=font)


def draw_normalized_box(
    draw: Any,
    box: np.ndarray,
    image_size: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int,
    label: str = "",
) -> None:
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    coords = (
        int(round(x1 * w)),
        int(round(y1 * h)),
        int(round(x2 * w)),
        int(round(y2 * h)),
    )
    draw.rectangle(coords, outline=color, width=width)
    if label:
        draw_label(draw, (coords[0] + 2, max(2, coords[1] - 13)), label, color)


def save_cascade_visualization(
    sample: Dict[str, Any],
    output_path: Path,
    gt_boxes: np.ndarray,
    yolo_boxes: torch.Tensor,
    yolo_scores: torch.Tensor,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    window_id: str,
    modality: str,
    frame: str,
) -> Optional[str]:
    if not PIL_AVAILABLE:
        return None
    bg = select_background_frame(sample, modality, frame)
    if bg is None:
        return None
    image = Image.fromarray(bg, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)

    yolo_arr = clip_valid_boxes(yolo_boxes)
    yolo_score_arr = yolo_scores.detach().cpu().numpy().reshape(-1) if torch.is_tensor(yolo_scores) else np.asarray(yolo_scores).reshape(-1)

    for idx, box in enumerate(yolo_arr, start=1):
        score = float(yolo_score_arr[idx - 1]) if idx - 1 < len(yolo_score_arr) else float("nan")
        label = f"Y{idx}:{score:.3f}" if idx <= 15 and np.isfinite(score) else (f"Y{idx}" if idx <= 15 else "")
        draw_normalized_box(draw, box, image.size, color=(255, 140, 0), width=1, label=label)

    for idx, box in enumerate(gt_boxes, start=1):
        draw_normalized_box(draw, box, image.size, color=(0, 255, 0), width=2, label=f"GT{idx}")

    for idx, box in enumerate(pred_boxes, start=1):
        cls = int(pred_classes[idx - 1]) if idx - 1 < len(pred_classes) else -1
        draw_normalized_box(draw, box, image.size, color=(0, 220, 255), width=2, label=f"S{idx}:c{cls}")

    header_lines = [
        str(window_id),
        f"GT={len(gt_boxes)} YOLO={len(yolo_arr)} Stage2={len(pred_boxes)}",
        "green=GT orange=YOLO cyan=stage2",
    ]
    y = 4
    for line in header_lines:
        draw_label(draw, (4, y), line, (255, 255, 255))
        y += 14

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def box_iou_matrix(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(target) == 0:
        return np.zeros((len(pred), len(target)), dtype=np.float32)
    px1, py1, px2, py2 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    ix1 = np.maximum(px1, tx1)
    iy1 = np.maximum(py1, ty1)
    ix2 = np.minimum(px2, tx2)
    iy2 = np.minimum(py2, ty2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    p_area = np.maximum(0.0, px2 - px1) * np.maximum(0.0, py2 - py1)
    t_area = np.maximum(0.0, tx2 - tx1) * np.maximum(0.0, ty2 - ty1)
    return (inter / np.maximum(p_area + t_area - inter, 1e-8)).astype(np.float32)


def greedy_match(pred: np.ndarray, target: np.ndarray) -> List[Tuple[int, int, float]]:
    ious = box_iou_matrix(pred, target)
    pairs: List[Tuple[int, int, float]] = []
    used_p, used_t = set(), set()
    flat = [(i, j, float(ious[i, j])) for i in range(ious.shape[0]) for j in range(ious.shape[1])]
    for i, j, score in sorted(flat, key=lambda item: item[2], reverse=True):
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        pairs.append((i, j, score))
    return pairs


def summarize_detection(rows: List[Dict[str, Any]], iou_threshold: float) -> Dict[str, Any]:
    tp = sum(int(row["tp_iou50"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    num_pred = sum(int(row["num_pred"]) for row in rows)
    num_gt = sum(int(row["num_gt"]) for row in rows)
    avg_iou = float(np.mean([float(row["avg_target_iou"]) for row in rows])) if rows else 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        "iou_threshold": float(iou_threshold),
        "num_windows": len(rows),
        "num_predictions": int(num_pred),
        "num_targets": int(num_gt),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_target_iou": avg_iou,
    }


def summarize_classification(matched_pred_classes: List[int], matched_gt_classes: List[int]) -> Dict[str, Any]:
    if not matched_pred_classes:
        return {"num_matched": 0, "accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
    pred = np.asarray(matched_pred_classes, dtype=np.int64)
    gt = np.asarray(matched_gt_classes, dtype=np.int64)
    labels = [1, 2]
    accuracy = float((pred == gt).mean())
    precisions, recalls, f1s = [], [], []
    for cls in labels:
        tp = int(np.sum((pred == cls) & (gt == cls)))
        fp = int(np.sum((pred == cls) & (gt != cls)))
        fn = int(np.sum((pred != cls) & (gt == cls)))
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2.0 * p * r / (p + r + 1e-8)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    return {
        "num_matched": int(len(pred)),
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "pred_class_counts": {str(cls): int(np.sum(pred == cls)) for cls in labels},
        "gt_class_counts": {str(cls): int(np.sum(gt == cls)) for cls in labels},
    }


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = load_yaml(args.data_config)["data"]
    model_cfg = load_yaml(args.model_config)["model"]
    train_cfg = load_yaml(args.train_config)["training"]
    inference_cfg = load_yaml(args.inference_config)

    hdf5_path = resolve_path(data_cfg.get("hdf5_path", "data/Solar_Flares_CME_dataset.h5"))
    processed_root = resolve_path(data_cfg.get("processed_data_dir", "data/processed"))
    max_activities = int(data_cfg.get("max_activities", args.max_det))
    sequence_length = int(data_cfg.get("sequence_length", 10))
    stride = int(data_cfg.get("stride", 10))
    target_size = tuple(next(iter(data_cfg.get("modalities", {}).values())).get("resolution", [256, 256]))

    reader = HDF5DatasetReader(str(hdf5_path))
    event_ids = load_split_event_ids(args.split, args.split_file, reader, args.max_events)
    dataset = SolarFlareDataset(
        reader=reader,
        event_ids=event_ids,
        modalities=enabled_modalities(data_cfg),
        sequence_length=sequence_length,
        stride=stride,
        target_size=target_size,
        max_activities=max_activities,
        config={"max_activities": max_activities},
    )

    channel_modalities = [item.strip() for item in args.channel_modalities.split(",") if item.strip()]
    yolo_export_utils._validate_channel_specs(channel_modalities)
    required_modalities = yolo_export_utils.ordered_unique(
        required for spec in channel_modalities for required in yolo_export_utils._required_modalities_for_channel_spec(spec)
    )

    from ultralytics import YOLO

    yolo_model = YOLO(str(resolve_path(args.yolo_weights)))
    stage2_device = resolve_stage2_device(args.stage2_device)
    full_model_cfg = build_model_config(data_cfg, model_cfg, train_cfg)
    stage2_model = load_stage2_model(resolve_path(args.stage2_weights), full_model_cfg, stage2_device)
    postprocessor = PostProcessor(inference_cfg.get("inference", inference_cfg))

    event_cache: Dict[str, Dict[str, Any]] = {}
    meta_cache: Dict[str, Dict[str, Any]] = {}
    predictions: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []
    matched_pred_classes: List[int] = []
    matched_gt_classes: List[int] = []

    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        metadata = sample.get("metadata", {}) or {}
        event_id = str(metadata.get("event_id", ""))
        start_idx = int(metadata.get("start_idx", 0))
        end_idx = int(metadata.get("end_idx", start_idx + sequence_length))
        window_id = str(metadata.get("window_id", f"{event_id}_{start_idx}_{end_idx}"))

        if event_id not in event_cache:
            event_cache[event_id] = reader.get_event_data(event_id, modalities=required_modalities)
            meta_cache[event_id] = reader.get_event_metadata(event_id)

        proposal_boxes, proposal_scores, proposal_debug = generate_yolo_proposals(
            yolo_model=yolo_model,
            event_id=event_id,
            event_data=event_cache[event_id],
            event_meta=meta_cache[event_id],
            start_idx=start_idx,
            end_idx=end_idx,
            args=args,
            processed_root=processed_root,
            max_activities=max_activities,
        )

        inputs = make_stage2_inputs(sample, proposal_boxes, proposal_scores, stage2_device)
        with torch.no_grad():
            outputs = stage2_model(inputs)
            pred = {
                "class_probs_mean": outputs["class_probs"],
                "bbox_pred_mean": outputs.get("bbox_pred"),
                "event_prob_mean": outputs["event_prob"],
                "event_gate_mean": outputs.get("event_gate"),
                "proposal_boxes": outputs.get("proposal_boxes"),
                "context_boxes": outputs.get("context_boxes"),
                "proposal_scores": outputs.get("proposal_scores"),
                "bbox_size_gated": outputs.get("bbox_size_gated"),
                "refine_delta": outputs.get("refine_delta"),
                "predicted_class": outputs["class_probs"].argmax(dim=-1),
            }
            if "time_pred" in outputs:
                pred["time_pred_mean"] = outputs.get("time_pred")
            pred = postprocessor.process(pred)

        pred_boxes = clip_valid_boxes(pred.get("processed_bboxes"))
        pred_classes_raw = pred.get("final_classes")
        pred_classes = np.asarray(tensor_to_jsonable(pred_classes_raw), dtype=np.int64).reshape(-1) if pred_classes_raw is not None else np.zeros((0,), dtype=np.int64)
        gt_boxes, gt_classes = unique_gt_regions(sample)
        pairs = greedy_match(pred_boxes, gt_boxes)
        tp_pairs = [(pi, gi, iou) for pi, gi, iou in pairs if iou >= args.iou_threshold]
        target_ious = np.zeros((len(gt_boxes),), dtype=np.float32)
        for _, gi, iou_value in pairs:
            target_ious[gi] = max(target_ious[gi], float(iou_value))
        for pi, gi, _ in tp_pairs:
            if pi < len(pred_classes) and gi < len(gt_classes):
                matched_pred_classes.append(int(pred_classes[pi]))
                matched_gt_classes.append(int(gt_classes[gi]))

        visualization_path = ""
        if args.save_visualizations:
            vis_path = output_dir / f"cascade_{sample_idx:04d}_{safe_filename(window_id)}.png"
            saved_path = save_cascade_visualization(
                sample=sample,
                output_path=vis_path,
                gt_boxes=gt_boxes,
                yolo_boxes=proposal_boxes,
                yolo_scores=proposal_scores,
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                window_id=window_id,
                modality=args.viz_modality,
                frame=args.viz_frame,
            )
            visualization_path = saved_path or ""

        row = {
            "window_index": sample_idx,
            "window_id": window_id,
            "event_id": event_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_yolo_proposals": int(proposal_debug.get("num_yolo_proposals", 0)),
            "num_pred": int(len(pred_boxes)),
            "num_gt": int(len(gt_boxes)),
            "tp_iou50": int(len(tp_pairs)),
            "false_positives": int(len(pred_boxes) - len(tp_pairs)),
            "false_negatives": int(len(gt_boxes) - len(tp_pairs)),
            "avg_target_iou": float(target_ious.mean()) if len(target_ious) else 0.0,
            "matched_ious": json.dumps([float(iou) for _, _, iou in pairs], ensure_ascii=False),
            "pred_classes": json.dumps([int(v) for v in pred_classes.tolist()], ensure_ascii=False),
            "gt_classes": json.dumps([int(v) for v in gt_classes], ensure_ascii=False),
            "proposal_scores": json.dumps(proposal_debug.get("proposal_scores", []), ensure_ascii=False),
            "visualization_path": visualization_path,
        }
        window_rows.append(row)
        predictions.append(
            {
                "metadata": {
                    **metadata,
                    "window_id": window_id,
                    "event_id": event_id,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                },
                "yolo": proposal_debug,
                "proposal_boxes": tensor_to_jsonable(proposal_boxes),
                "proposal_scores": tensor_to_jsonable(proposal_scores),
                "gt_boxes": gt_boxes.tolist(),
                "gt_classes": [int(v) for v in gt_classes],
                "pred_boxes": pred_boxes.tolist(),
                "pred_classes": [int(v) for v in pred_classes.tolist()],
                "visualization_path": visualization_path,
                "processed": tensor_to_jsonable(
                    {
                        "processed_bboxes": pred.get("processed_bboxes"),
                        "final_classes": pred.get("final_classes"),
                        "classification_confidences": pred.get("classification_confidences"),
                        "processed_event_probs": pred.get("processed_event_probs"),
                        "bbox_confidences": pred.get("bbox_confidences"),
                        "kept_bbox_slot_indices": pred.get("kept_bbox_slot_indices"),
                    }
                ),
            }
        )

        if (sample_idx + 1) % 10 == 0 or sample_idx + 1 == len(dataset):
            print(f"[cascade] processed {sample_idx + 1}/{len(dataset)} windows")

    detection_metrics = summarize_detection(window_rows, args.iou_threshold)
    classification_metrics = summarize_classification(matched_pred_classes, matched_gt_classes)
    metrics = {
        "split": args.split,
        "yolo_weights": str(resolve_path(args.yolo_weights)),
        "stage2_weights": str(resolve_path(args.stage2_weights)),
        "data_config": str(resolve_path(args.data_config)),
        "model_config": str(resolve_path(args.model_config)),
        "train_config": str(resolve_path(args.train_config)),
        "inference_config": str(resolve_path(args.inference_config)),
        "yolo_params": {
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "frame_source": args.frame_source,
            "image_source": args.image_source,
            "channel_modalities": channel_modalities,
        },
        "detection": detection_metrics,
        "classification_on_iou_matched_detections": classification_metrics,
    }

    with open(output_dir / "cascade_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with open(output_dir / "cascade_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(output_dir / "cascade_window_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()) if window_rows else ["window_id"])
        writer.writeheader()
        writer.writerows(window_rows)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[cascade] outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
