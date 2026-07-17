"""
Evaluate a YOLO single-class localizer with experiment-C-style localization metrics.

Metrics are GT-centric:
- mean best IoU per ground-truth box
- Recall@IoU 0.1 / 0.3 / 0.5
- false positives per image at a chosen IoU threshold
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    "data": "outputs/yolo_dataset/activity_region_single_class_mag_euv94_euv171_processed_export/dataset.yaml",
    "weights": "runs/detect/outputs/yolo_runs/yolo11n_activity_region_mag_euv94_euv171_processed_512/weights/best.pt",
    "split": "test",
    "imgsz": 512,
    "device": "0",
    "conf": 0.001,
    "predict_iou": 0.5,
    "max_det": 300,
    "post_nms_iou": 0.5,
    "post_nms_ios": 0.9,
    "iou_match": 0.3,
    "output_dir": "outputs/yolo_eval",
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
        "conf",
        "predict_iou",
        "max_det",
        "post_nms_iou",
        "post_nms_ios",
        "iou_match",
        "output_dir",
        "prediction_source_mode",
    ):
        value = cfg.get(key)
        if value not in (None, ""):
            args.extend([f"--{key}", str(value)])
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO localization baseline.")
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
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--imgsz", type=int, default=DIRECT_RUN_CONFIG["imgsz"], help="Inference image size.")
    parser.add_argument("--device", type=str, default=DIRECT_RUN_CONFIG["device"], help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--conf", type=float, default=DIRECT_RUN_CONFIG["conf"], help="Confidence threshold for prediction export.")
    parser.add_argument("--predict_iou", type=float, default=DIRECT_RUN_CONFIG["predict_iou"], help="YOLO prediction NMS IoU threshold.")
    parser.add_argument("--max_det", type=int, default=DIRECT_RUN_CONFIG["max_det"], help="Maximum detections per image after NMS.")
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
    parser.add_argument("--iou_match", type=float, default=DIRECT_RUN_CONFIG["iou_match"], help="IoU threshold used for FP accounting.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DIRECT_RUN_CONFIG["output_dir"],
        help="Output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--prediction_source_mode",
        type=str,
        default=DIRECT_RUN_CONFIG["prediction_source_mode"],
        choices=["native_path", "array"],
        help="Use Ultralytics native image-path loading or preloaded numpy arrays for inference.",
    )
    return parser.parse_args(argv)


def load_dataset_yaml(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
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


def greedy_match(pred_boxes: List[np.ndarray], gt_boxes: List[np.ndarray], iou_threshold: float) -> Tuple[int, int]:
    matched_pred = set()
    matched_gt = set()
    triples: List[Tuple[float, int, int]] = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                triples.append((iou, pred_idx, gt_idx))
    triples.sort(reverse=True, key=lambda x: x[0])
    for _, pred_idx, gt_idx in triples:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
    return len(matched_pred), len(pred_boxes) - len(matched_pred)


def build_eval_output_dir(output_root: Path, weights_path: Path, split: str) -> Path:
    """Keep evaluation outputs unique per training run, not just per weight filename."""
    run_dir = weights_path.resolve().parent.parent
    run_name = run_dir.name if run_dir.name else weights_path.stem
    weight_name = weights_path.stem
    return output_root / f"{run_name}_{weight_name}" / split


def load_prediction_source(image_path: Path) -> np.ndarray:
    sidecar_path = image_path.with_suffix(".npy")
    if sidecar_path.exists():
        return np.load(sidecar_path)

    from PIL import Image

    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"))


def build_prediction_sources(image_paths: List[Path], mode: str) -> List[object]:
    if mode == "native_path":
        return [str(path) for path in image_paths]
    return [load_prediction_source(path) for path in image_paths]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    yolo_config_dir = repo_root / "outputs" / "ultralytics_config"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)

    from PIL import Image
    from ultralytics import YOLO

    print(f"[YOLO Eval] data={args.data}")
    print(f"[YOLO Eval] weights={args.weights}")
    print(
        f"[YOLO Eval] split={args.split}, imgsz={args.imgsz}, conf={args.conf}, "
        f"predict_iou={args.predict_iou}, post_nms_iou={args.post_nms_iou}, "
        f"post_nms_ios={args.post_nms_ios}, max_det={args.max_det}, device={args.device}, "
        f"prediction_source_mode={args.prediction_source_mode}"
    )

    data_yaml = load_dataset_yaml(Path(args.data))
    dataset_root = Path(data_yaml["path"])
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    model = YOLO(args.weights)
    prediction_sources = build_prediction_sources(image_paths, args.prediction_source_mode)
    results = model.predict(
        source=prediction_sources,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.predict_iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
        stream=False,
    )

    per_gt_rows: List[Dict[str, float]] = []
    per_image_rows: List[Dict[str, float]] = []

    for image_path, result in zip(image_paths, results):
        label_path = label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as img:
            width, height = img.size

        gt_boxes: List[np.ndarray] = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        gt_boxes.append(yolo_txt_to_xyxy(line, (width, height)))

        pred_boxes: List[np.ndarray] = []
        pred_scores: List[float] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            for box, score in zip(xyxy, confs):
                pred_boxes.append(box.astype(np.float32))
                pred_scores.append(float(score))
        pred_boxes, pred_scores = apply_secondary_nms(
            pred_boxes,
            pred_scores,
            iou_threshold=args.post_nms_iou,
            ios_threshold=args.post_nms_ios,
            max_det=args.max_det,
        )

        for gt_idx, gt_box in enumerate(gt_boxes):
            if pred_boxes:
                ious = [compute_iou(pred_box, gt_box) for pred_box in pred_boxes]
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])
                best_score = pred_scores[best_idx]
                best_box = pred_boxes[best_idx].tolist()
            else:
                best_iou = 0.0
                best_score = 0.0
                best_box = []

            per_gt_rows.append(
                {
                    "image_id": image_path.stem,
                    "gt_idx": gt_idx,
                    "best_iou": best_iou,
                    "best_score": best_score,
                    "gt_box": gt_box.tolist(),
                    "best_pred_box": best_box,
                    "num_preds_in_image": len(pred_boxes),
                }
            )

        matched_count, fp_count = greedy_match(pred_boxes, gt_boxes, args.iou_match)
        per_image_rows.append(
            {
                "image_id": image_path.stem,
                "num_gt": len(gt_boxes),
                "num_pred": len(pred_boxes),
                "matched_pred_at_iou_match": matched_count,
                "false_positives_at_iou_match": fp_count,
            }
        )

    best_ious = np.array([row["best_iou"] for row in per_gt_rows], dtype=np.float32)
    fp_values = np.array([row["false_positives_at_iou_match"] for row in per_image_rows], dtype=np.float32)
    pred_values = np.array([row["num_pred"] for row in per_image_rows], dtype=np.float32)

    summary = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": args.split,
        "num_images": int(len(per_image_rows)),
        "num_gt_boxes": int(len(per_gt_rows)),
        "mean_iou": float(best_ious.mean()) if len(best_ious) else 0.0,
        "median_iou": float(np.median(best_ious)) if len(best_ious) else 0.0,
        "recall_at_0.1": float((best_ious >= 0.1).mean()) if len(best_ious) else 0.0,
        "recall_at_0.3": float((best_ious >= 0.3).mean()) if len(best_ious) else 0.0,
        "recall_at_0.5": float((best_ious >= 0.5).mean()) if len(best_ious) else 0.0,
        "mean_predictions_per_image": float(pred_values.mean()) if len(pred_values) else 0.0,
        "mean_false_positives_per_image_at_iou_match": float(fp_values.mean()) if len(fp_values) else 0.0,
        "iou_match_for_fp": float(args.iou_match),
        "predict_iou": float(args.predict_iou),
        "post_nms_iou": float(args.post_nms_iou),
        "post_nms_ios": float(args.post_nms_ios),
        "max_det": int(args.max_det),
    }

    weights_path = Path(args.weights)
    output_dir = build_eval_output_dir(Path(args.output_dir), weights_path, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(output_dir / "per_gt_details.json", "w", encoding="utf-8") as f:
        json.dump(per_gt_rows, f, ensure_ascii=False, indent=2)
    with open(output_dir / "per_image_details.json", "w", encoding="utf-8") as f:
        json.dump(per_image_rows, f, ensure_ascii=False, indent=2)

    summary_md = output_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# YOLO Localizer Evaluation\n\n")
        for key, value in summary.items():
            f.write(f"- {key}: `{value}`\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if USE_DIRECT_RUN_CONFIG and len(sys.argv) == 1:
        sys.argv.extend(_build_direct_run_args())
    raise SystemExit(main())
