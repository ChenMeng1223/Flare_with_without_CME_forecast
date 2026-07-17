"""
Visualize YOLO predictions on original-resolution magnetogram images across thresholds.

Outputs:
- all-test contact sheets at selected thresholds
- representative per-event threshold comparison sheet
- summary JSON/Markdown with score-distribution diagnostics
"""
import argparse
import json
import math
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    "data": "outputs/yolo_dataset/activity_region_single_class_mag_euv94_euv171_processed_export/dataset.yaml",
    "weights": "runs/detect/outputs/yolo_runs/yolo11n_activity_region_mag_euv94_euv171_processed_512/weights/best.pt",
    "split": "test",
    "imgsz": 512,
    "device": "0",
    "confs": "0.001,0.005,0.01,0.05,0.1,0.2",
    "predict_iou": 0.5,
    "max_det": 300,
    "post_nms_iou": 0.5,
    "post_nms_ios": 0.9,
    "modality": "magnetogram",
    "frame_mode": "annotated",
    "output_dir": "outputs/yolo_visualizations",
    "render_source": "original",
    "prediction_source_mode": "native_path",
}


def _build_direct_run_args() -> list[str]:
    cfg = dict(DIRECT_RUN_CONFIG)
    args: list[str] = []
    for key in (
        "data",
        "weights",
        "split",
        "imgsz",
        "device",
        "confs",
        "predict_iou",
        "max_det",
        "post_nms_iou",
        "post_nms_ios",
        "modality",
        "frame_mode",
        "output_dir",
        "render_source",
        "prediction_source_mode",
    ):
        value = cfg.get(key)
        if value not in (None, ""):
            args.extend([f"--{key}", str(value)])
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO predictions across thresholds.")
    parser.add_argument(
        "--data",
        type=str,
        default=DIRECT_RUN_CONFIG["data"],
        help="YOLO dataset yaml path.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DIRECT_RUN_CONFIG["weights"],
        help="YOLO weights path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DIRECT_RUN_CONFIG["split"],
        choices=["train", "val", "test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument("--imgsz", type=int, default=DIRECT_RUN_CONFIG["imgsz"], help="Inference image size.")
    parser.add_argument("--device", type=str, default=DIRECT_RUN_CONFIG["device"], help="Inference device.")
    parser.add_argument(
        "--confs",
        type=str,
        default=DIRECT_RUN_CONFIG["confs"],
        help="Comma-separated confidence thresholds.",
    )
    parser.add_argument("--predict_iou", type=float, default=DIRECT_RUN_CONFIG["predict_iou"], help="YOLO prediction NMS IoU threshold.")
    parser.add_argument("--max_det", type=int, default=DIRECT_RUN_CONFIG["max_det"], help="Maximum detections after NMS.")
    parser.add_argument(
        "--post_nms_iou",
        type=float,
        default=DIRECT_RUN_CONFIG["post_nms_iou"],
        help="Secondary greedy NMS IoU threshold applied after YOLO output. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--post_nms_ios",
        type=float,
        default=DIRECT_RUN_CONFIG["post_nms_ios"],
        help=(
            "Secondary suppression threshold on intersection over the smaller box area. "
            "Useful for removing near-contained duplicate boxes. Set <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=DIRECT_RUN_CONFIG["modality"],
        help="Modality used to recover original-resolution images.",
    )
    parser.add_argument(
        "--frame_mode",
        type=str,
        default=DIRECT_RUN_CONFIG["frame_mode"],
        choices=["event_start", "annotated"],
        help="Which processed image to align with the exported dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DIRECT_RUN_CONFIG["output_dir"],
        help="Directory for visualization outputs.",
    )
    parser.add_argument(
        "--render_source",
        type=str,
        default=DIRECT_RUN_CONFIG["render_source"],
        choices=["dataset", "original"],
        help="Render overlays on exported dataset images or original processed images.",
    )
    parser.add_argument(
        "--prediction_source_mode",
        type=str,
        default=DIRECT_RUN_CONFIG["prediction_source_mode"],
        choices=["native_path", "array"],
        help="Use Ultralytics native image-path loading or preloaded numpy arrays for inference.",
    )
    return parser.parse_args(argv)


def load_dataset_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_key = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith("  "):
                key, value = line.split(":", 1)
                value = value.strip()
                if value:
                    data[key.strip()] = value
                    current_key = None
                else:
                    current_key = key.strip()
                    data[current_key] = {}
            elif current_key is not None:
                sub_key, sub_val = line.strip().split(":", 1)
                data[current_key][sub_key.strip()] = sub_val.strip()
    return data


def yolo_txt_to_xyxy(line: str, image_size: Tuple[int, int]) -> np.ndarray:
    parts = line.strip().split()
    _, cx, cy, bw, bh = [float(x) for x in parts]
    width, height = image_size
    cx *= width
    cy *= height
    bw *= width
    bh *= height
    x1 = cx - bw * 0.5
    y1 = cy - bh * 0.5
    x2 = cx + bw * 0.5
    y2 = cy + bh * 0.5
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "").replace("+00:00", ""))


def parse_filename_datetime(filename: str) -> Optional[datetime]:
    match = re.search(r"(20\d{6}_\d{6})", filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def find_event_start_frame_index(timestamps: List[str], start_time: str) -> int:
    if not timestamps:
        return 0
    target = parse_iso_datetime(start_time)
    parsed = [parse_iso_datetime(ts) for ts in timestamps]
    deltas = [abs((ts - target).total_seconds()) for ts in parsed]
    return int(np.argmin(deltas))


def load_bbox_info(processed_root: Path, event_id: str) -> Dict[str, Any]:
    bbox_path = processed_root / event_id / "bboxes.json"
    if not bbox_path.exists():
        return {}
    try:
        with open(bbox_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_processed_image_path(
    processed_root: Path,
    event_id: str,
    modality: str,
    target_timestamp: Optional[str],
    preferred_name: Optional[str],
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
    if not target_timestamp:
        return image_paths[0]

    target_dt = parse_iso_datetime(target_timestamp)
    best_path: Optional[Path] = None
    best_delta: Optional[float] = None
    for image_path in image_paths:
        image_dt = parse_filename_datetime(image_path.name)
        if image_dt is None:
            continue
        delta = abs((image_dt - target_dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_path = image_path
    return best_path or image_paths[0]


def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


def compute_intersection_over_smaller(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    smaller = min(box_area(box_a), box_area(box_b))
    return inter / (smaller + 1e-8) if smaller > 0 else 0.0


def apply_secondary_nms(
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    iou_threshold: float,
    ios_threshold: float,
    max_det: int,
) -> Tuple[List[np.ndarray], List[float]]:
    if not pred_boxes:
        return [], []

    order = np.argsort(-np.asarray(pred_scores, dtype=np.float32))
    kept_boxes: List[np.ndarray] = []
    kept_scores: List[float] = []
    for idx in order.tolist():
        candidate_box = pred_boxes[idx]
        candidate_score = float(pred_scores[idx])
        is_duplicate = False
        for kept_box in kept_boxes:
            if iou_threshold > 0 and compute_iou(candidate_box, kept_box) >= iou_threshold:
                is_duplicate = True
                break
            if ios_threshold > 0 and compute_intersection_over_smaller(candidate_box, kept_box) >= ios_threshold:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        kept_boxes.append(candidate_box)
        kept_scores.append(candidate_score)
        if max_det > 0 and len(kept_boxes) >= max_det:
            break
    return kept_boxes, kept_scores


def xyxy_from_array(arr: Sequence[float]) -> np.ndarray:
    return np.asarray([float(v) for v in arr[:4]], dtype=np.float32)


def color_for_score(score: float) -> Tuple[int, int, int]:
    score = max(0.0, min(1.0, float(score)))
    red = 255
    green = int(64 + 160 * score)
    blue = int(64 * (1.0 - score))
    return red, green, blue


def draw_text_box(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text(xy, text, fill=fill, font=font)


def render_overlay(
    image_path: Path,
    gt_boxes: List[np.ndarray],
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    eval_image_size: Tuple[int, int],
    header_lines: List[str],
    render_image_size: Optional[Tuple[int, int]] = None,
    gt_coord_size: Optional[Tuple[int, int]] = None,
    show_pred_labels_limit: int = 12,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    orig_w, orig_h = image.size
    if render_image_size is None:
        render_w, render_h = orig_w, orig_h
    else:
        render_w, render_h = render_image_size
    eval_w, eval_h = eval_image_size
    scale_x = render_w / float(eval_w)
    scale_y = render_h / float(eval_h)

    if gt_coord_size is None:
        gt_coord_w, gt_coord_h = orig_w, orig_h
    else:
        gt_coord_w, gt_coord_h = gt_coord_size
    gt_scale_x = render_w / float(gt_coord_w)
    gt_scale_y = render_h / float(gt_coord_h)

    for gt_box in gt_boxes:
        scaled_gt = np.array([
            gt_box[0] * gt_scale_x,
            gt_box[1] * gt_scale_y,
            gt_box[2] * gt_scale_x,
            gt_box[3] * gt_scale_y,
        ], dtype=np.float32)
        x1, y1, x2, y2 = [int(round(v)) for v in scaled_gt.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=5)

    pred_order = np.argsort([-score for score in pred_scores]) if pred_scores else np.array([], dtype=int)
    for rank, idx in enumerate(pred_order.tolist()):
        pred_box = pred_boxes[idx]
        score = pred_scores[idx]
        scaled = np.array([
            pred_box[0] * scale_x,
            pred_box[1] * scale_y,
            pred_box[2] * scale_x,
            pred_box[3] * scale_y,
        ], dtype=np.float32)
        x1, y1, x2, y2 = [int(round(v)) for v in scaled.tolist()]
        color = color_for_score(score)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if rank < show_pred_labels_limit:
            draw_text_box(draw, (x1 + 2, max(2, y1 - 12)), f"{score:.3f}", fill=color)

    y = 4
    for line in header_lines:
        draw_text_box(draw, (4, y), line, fill=(255, 255, 255))
        y += 12

    return image


def thumbnail_with_border(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (16, 16, 16))
    paste_x = (size[0] - thumb.size[0]) // 2
    paste_y = (size[1] - thumb.size[1]) // 2
    canvas.paste(thumb, (paste_x, paste_y))
    return canvas


def create_contact_sheet(
    images: Sequence[Image.Image],
    captions: Sequence[str],
    output_path: Path,
    columns: int = 4,
    tile_size: Tuple[int, int] = (420, 420),
) -> None:
    rows = int(math.ceil(len(images) / columns))
    caption_h = 28
    sheet = Image.new(
        "RGB",
        (columns * tile_size[0], rows * (tile_size[1] + caption_h)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (image, caption) in enumerate(zip(images, captions)):
        row = idx // columns
        col = idx % columns
        x = col * tile_size[0]
        y = row * (tile_size[1] + caption_h)
        thumb = thumbnail_with_border(image, tile_size)
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + tile_size[1] + 6), caption, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def select_representative_events(event_stats: Dict[str, Dict[str, Any]]) -> List[str]:
    non_empty = [item for item in event_stats.items()]
    if not non_empty:
        return []

    avg_iou_values = np.array([float(stats["avg_best_iou"]) for _, stats in non_empty], dtype=np.float32)
    median_iou = float(np.median(avg_iou_values))

    best_event = max(non_empty, key=lambda kv: kv[1]["avg_best_iou"])[0]
    median_event = min(non_empty, key=lambda kv: abs(kv[1]["avg_best_iou"] - median_iou))[0]
    failure_event = min(non_empty, key=lambda kv: kv[1]["avg_best_iou"])[0]

    low_conf_good_candidates = [
        (event_id, stats)
        for event_id, stats in non_empty
        if float(stats["avg_best_iou"]) >= 0.5
    ]
    if low_conf_good_candidates:
        low_conf_good_event = min(low_conf_good_candidates, key=lambda kv: kv[1]["avg_best_score"])[0]
    else:
        low_conf_good_event = best_event

    ordered = OrderedDict()
    for event_id in [best_event, median_event, low_conf_good_event, failure_event]:
        ordered[event_id] = None
    return list(ordered.keys())


def load_gt_boxes_from_dataset_label(label_path: Path, image_size: Tuple[int, int]) -> List[np.ndarray]:
    gt_boxes: List[np.ndarray] = []
    if not label_path.exists():
        return gt_boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gt_boxes.append(yolo_txt_to_xyxy(line, image_size))
    return gt_boxes


def build_event_records(
    image_paths: Sequence[Path],
    label_dir: Path,
    processed_root: Path,
    modality: str,
    frame_mode: str,
) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for image_path in image_paths:
        event_id = image_path.stem
        bbox_info = load_bbox_info(processed_root, event_id)
        target_timestamp = bbox_info.get("annotation_frame_timestamp")
        preferred_name = None
        if frame_mode == "annotated":
            regions_meta = bbox_info.get("regions", [])
            if regions_meta:
                preferred_name = regions_meta[0].get("annotation_frame_name")

        original_image_path = find_processed_image_path(
            processed_root=processed_root,
            event_id=event_id,
            modality=modality,
            target_timestamp=target_timestamp,
            preferred_name=preferred_name,
        )
        if original_image_path is None:
            continue
        original_image = Image.open(original_image_path)
        original_image_size = original_image.size
        original_image.close()
        with Image.open(image_path) as dataset_img:
            dataset_image_size = dataset_img.size
        gt_boxes = load_gt_boxes_from_dataset_label(label_dir / f"{event_id}.txt", dataset_image_size)

        records[event_id] = {
            "dataset_image_path": image_path,
            "dataset_image_size": dataset_image_size,
            "original_image_path": original_image_path,
            "original_image_size": original_image_size,
            "gt_boxes": gt_boxes,
        }
    return records


def compute_event_stats(
    event_records: Dict[str, Dict[str, Any]],
    predictions: Dict[str, Dict[str, List[Any]]],
) -> Dict[str, Dict[str, Any]]:
    event_stats: Dict[str, Dict[str, Any]] = {}
    for event_id, record in event_records.items():
        gt_boxes = record["gt_boxes"]
        pred_boxes = [xyxy_from_array(box) for box in predictions[event_id]["boxes"]]
        pred_scores = [float(score) for score in predictions[event_id]["scores"]]

        gt_best_ious: List[float] = []
        gt_best_scores: List[float] = []
        for gt_box in gt_boxes:
            if pred_boxes:
                ious = [compute_iou(pred_box, gt_box) for pred_box in pred_boxes]
                best_idx = int(np.argmax(ious))
                gt_best_ious.append(float(ious[best_idx]))
                gt_best_scores.append(float(pred_scores[best_idx]))
            else:
                gt_best_ious.append(0.0)
                gt_best_scores.append(0.0)

        top_pred_max_iou = 0.0
        top_pred_score = 0.0
        if pred_boxes:
            top_idx = int(np.argmax(pred_scores))
            top_pred_score = float(pred_scores[top_idx])
            if gt_boxes:
                top_pred_max_iou = max(compute_iou(pred_boxes[top_idx], gt_box) for gt_box in gt_boxes)

        event_stats[event_id] = {
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_boxes),
            "avg_best_iou": float(np.mean(gt_best_ious)) if gt_best_ious else 0.0,
            "min_best_iou": float(np.min(gt_best_ious)) if gt_best_ious else 0.0,
            "avg_best_score": float(np.mean(gt_best_scores)) if gt_best_scores else 0.0,
            "gt_best_ious": gt_best_ious,
            "gt_best_scores": gt_best_scores,
            "top_pred_score": top_pred_score,
            "top_pred_max_iou": float(top_pred_max_iou),
        }
    return event_stats


def summarize_score_distribution(event_stats: Dict[str, Dict[str, Any]], thresholds: Sequence[float]) -> Dict[str, Any]:
    gt_best_scores = np.array(
        [score for stats in event_stats.values() for score in stats["gt_best_scores"]],
        dtype=np.float32,
    )
    gt_best_ious = np.array(
        [iou for stats in event_stats.values() for iou in stats["gt_best_ious"]],
        dtype=np.float32,
    )
    top_pred_scores = np.array([stats["top_pred_score"] for stats in event_stats.values()], dtype=np.float32)
    top_pred_ious = np.array([stats["top_pred_max_iou"] for stats in event_stats.values()], dtype=np.float32)

    summary = {
        "num_events": int(len(event_stats)),
        "num_gt_boxes": int(len(gt_best_scores)),
        "gt_best_score_quantiles": {
            "min": float(gt_best_scores.min()) if len(gt_best_scores) else 0.0,
            "p25": float(np.percentile(gt_best_scores, 25)) if len(gt_best_scores) else 0.0,
            "median": float(np.percentile(gt_best_scores, 50)) if len(gt_best_scores) else 0.0,
            "p75": float(np.percentile(gt_best_scores, 75)) if len(gt_best_scores) else 0.0,
            "max": float(gt_best_scores.max()) if len(gt_best_scores) else 0.0,
        },
        "top_pred_score_quantiles": {
            "min": float(top_pred_scores.min()) if len(top_pred_scores) else 0.0,
            "p25": float(np.percentile(top_pred_scores, 25)) if len(top_pred_scores) else 0.0,
            "median": float(np.percentile(top_pred_scores, 50)) if len(top_pred_scores) else 0.0,
            "p75": float(np.percentile(top_pred_scores, 75)) if len(top_pred_scores) else 0.0,
            "max": float(top_pred_scores.max()) if len(top_pred_scores) else 0.0,
        },
        "top_pred_iou_quantiles": {
            "min": float(top_pred_ious.min()) if len(top_pred_ious) else 0.0,
            "median": float(np.percentile(top_pred_ious, 50)) if len(top_pred_ious) else 0.0,
            "max": float(top_pred_ious.max()) if len(top_pred_ious) else 0.0,
        },
        "threshold_retention": [],
    }

    for threshold in thresholds:
        keep_mask = gt_best_scores >= threshold
        summary["threshold_retention"].append({
            "threshold": float(threshold),
            "fraction_gt_best_scores_kept": float(keep_mask.mean()) if len(keep_mask) else 0.0,
            "fraction_gt_with_iou_ge_0p3_and_score_kept": float(((gt_best_ious >= 0.3) & keep_mask).mean()) if len(keep_mask) else 0.0,
            "fraction_gt_with_iou_ge_0p5_and_score_kept": float(((gt_best_ious >= 0.5) & keep_mask).mean()) if len(keep_mask) else 0.0,
            "fraction_images_top_pred_score_kept": float((top_pred_scores >= threshold).mean()) if len(top_pred_scores) else 0.0,
        })
    return summary


def write_summary_markdown(path: Path, summary: Dict[str, Any], representative_events: Sequence[str], event_stats: Dict[str, Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# YOLO Threshold Visualization Summary\n\n")
        f.write("## Score Distribution\n\n")
        f.write(f"- num_events: `{summary['num_events']}`\n")
        f.write(f"- num_gt_boxes: `{summary['num_gt_boxes']}`\n")
        f.write(f"- gt_best_score_quantiles: `{summary['gt_best_score_quantiles']}`\n")
        f.write(f"- top_pred_score_quantiles: `{summary['top_pred_score_quantiles']}`\n")
        f.write(f"- top_pred_iou_quantiles: `{summary['top_pred_iou_quantiles']}`\n\n")
        f.write("## Threshold Retention\n\n")
        for row in summary["threshold_retention"]:
            f.write(f"- {row}\n")
        f.write("\n## Representative Events\n\n")
        for event_id in representative_events:
            f.write(f"- {event_id}: `{event_stats[event_id]}`\n")


def sanitize_conf_label(conf: float) -> str:
    return str(conf).replace(".", "p")


def load_prediction_source(image_path: Path) -> np.ndarray:
    sidecar_path = image_path.with_suffix(".npy")
    if sidecar_path.exists():
        return np.load(sidecar_path)
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"))


def build_prediction_sources(image_paths: Sequence[Path], mode: str) -> List[object]:
    if mode == "native_path":
        return [str(path) for path in image_paths]
    return [load_prediction_source(path) for path in image_paths]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["YOLO_CONFIG_DIR"] = str((repo_root / "outputs" / "ultralytics_config").resolve())

    from ultralytics import YOLO

    confs = [float(item.strip()) for item in args.confs.split(",") if item.strip()]
    data_yaml = load_dataset_yaml(Path(args.data))
    dataset_root = Path(data_yaml["path"])
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    processed_root = repo_root / "data" / "processed"
    event_records = build_event_records(image_paths, label_dir, processed_root, args.modality, args.frame_mode)
    kept_image_paths = [event_records[event_id]["dataset_image_path"] for event_id in event_records]
    prediction_sources = build_prediction_sources(kept_image_paths, args.prediction_source_mode)

    model = YOLO(args.weights)
    predictions_by_conf: Dict[float, Dict[str, Dict[str, List[Any]]]] = {}
    for conf in confs:
        results = model.predict(
            source=prediction_sources,
            imgsz=args.imgsz,
            conf=conf,
            iou=args.predict_iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            stream=False,
        )
        per_event: Dict[str, Dict[str, List[Any]]] = {}
        for dataset_image_path, result in zip(kept_image_paths, results):
            event_id = dataset_image_path.stem
            boxes: List[Any] = []
            scores: List[float] = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                conf_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
                dedup_boxes, dedup_scores = apply_secondary_nms(
                    pred_boxes=[box.astype(np.float32) for box in xyxy],
                    pred_scores=[float(score) for score in conf_scores],
                    iou_threshold=args.post_nms_iou,
                    ios_threshold=args.post_nms_ios,
                    max_det=args.max_det,
                )
                boxes = [box.tolist() for box in dedup_boxes]
                scores = [float(score) for score in dedup_scores]
            per_event[event_id] = {"boxes": boxes, "scores": scores}
        predictions_by_conf[conf] = per_event

    base_conf = confs[0]
    event_stats = compute_event_stats(event_records, predictions_by_conf[base_conf])
    representative_events = select_representative_events(event_stats)

    output_root = Path(args.output_dir) / Path(args.weights).stem / args.split
    output_root.mkdir(parents=True, exist_ok=True)

    for conf in confs:
        conf_label = sanitize_conf_label(conf)
        conf_dir = output_root / f"conf_{conf_label}"
        conf_dir.mkdir(parents=True, exist_ok=True)
        for event_id in sorted(event_records.keys()):
            record = event_records[event_id]
            preds = predictions_by_conf[conf][event_id]
            if args.render_source == "dataset":
                render_image_path = record["dataset_image_path"]
                render_image_size = (args.imgsz, args.imgsz)
            else:
                render_image_path = record["original_image_path"]
                render_image_size = None

            headers = [
                event_id,
                f"conf={conf:.3f} preds={len(preds['boxes'])}",
                f"avgBestIoU={event_stats[event_id]['avg_best_iou']:.3f}",
                f"avgBestScore={event_stats[event_id]['avg_best_score']:.3f}",
            ]
            overlay = render_overlay(
                image_path=render_image_path,
                gt_boxes=record["gt_boxes"],
                pred_boxes=[xyxy_from_array(box) for box in preds["boxes"]],
                pred_scores=preds["scores"],
                eval_image_size=(args.imgsz, args.imgsz),
                render_image_size=render_image_size,
                gt_coord_size=record["dataset_image_size"],
                header_lines=headers,
            )
            overlay.save(conf_dir / f"{event_id}.png")

    summary = summarize_score_distribution(event_stats, confs)
    summary["representative_events"] = representative_events
    summary["event_stats"] = event_stats
    summary["render_source"] = args.render_source
    summary["predict_iou"] = float(args.predict_iou)
    summary["post_nms_iou"] = float(args.post_nms_iou)
    summary["post_nms_ios"] = float(args.post_nms_ios)
    summary["max_det"] = int(args.max_det)
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_markdown(output_root / "summary.md", summary, representative_events, event_stats)

    print(json.dumps({
        "output_dir": str(output_root.resolve()),
        "representative_events": representative_events,
        "confs": confs,
        "render_source": args.render_source,
        "predict_iou": args.predict_iou,
        "post_nms_iou": args.post_nms_iou,
        "post_nms_ios": args.post_nms_ios,
        "prediction_source_mode": args.prediction_source_mode,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if USE_DIRECT_RUN_CONFIG and len(sys.argv) == 1:
        sys.argv.extend(_build_direct_run_args())
    raise SystemExit(main())
