"""
Visualize test-set YOLO localization performance across multiple trained models.

Outputs:
- overall metric comparison chart
- per-image mean-IoU heatmap
- representative side-by-side prediction grid on selected test events
- summary markdown/json manifest
"""
import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    "experiments": [
        "Magnetogram-256::outputs/yolo_eval/yolo11n_activity_region_magnetogram_event_start_unique_region_best/test/summary.json",
        "Magnetogram-512::outputs/yolo_eval/yolo11n_activity_region_magnetogram_event_start_unique_region_512_best/test/summary.json",
        "Mag+EUV94+EUV171-512::outputs/yolo_eval/yolo11n_activity_region_mag_euv94_euv171_processed_512_best/test/summary.json",
    ],
    "conf": 0.05,
    "device": "0",
    "imgsz": 512,
    "max_examples": 6,
    "top_pred_labels": 8,
    "output_dir": "outputs/yolo_model_comparison_testset",
    "prediction_source_mode": "native_path",
}


def _build_direct_run_args() -> list[str]:
    cfg = dict(DIRECT_RUN_CONFIG)
    args: list[str] = []
    for spec in cfg.get("experiments", []):
        if spec:
            args.extend(["--experiment", str(spec)])
    for key in ("conf", "device", "imgsz", "max_examples", "top_pred_labels", "output_dir", "prediction_source_mode"):
        value = cfg.get(key)
        if value not in (None, ""):
            args.extend([f"--{key}", str(value)])
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple YOLO localizers on the test set.")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help=(
            "Experiment spec in the form "
            "'label::path/to/summary.json'. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DIRECT_RUN_CONFIG["conf"],
        help="Confidence threshold used for visualization overlays.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DIRECT_RUN_CONFIG["device"],
        help="Inference device used when re-rendering selected examples.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DIRECT_RUN_CONFIG["imgsz"],
        help="Inference image size for selected examples.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=DIRECT_RUN_CONFIG["max_examples"],
        help="Maximum number of representative test events to render.",
    )
    parser.add_argument(
        "--top_pred_labels",
        type=int,
        default=DIRECT_RUN_CONFIG["top_pred_labels"],
        help="Maximum number of prediction score labels drawn per tile.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DIRECT_RUN_CONFIG["output_dir"],
        help="Directory for generated comparison figures.",
    )
    parser.add_argument(
        "--prediction_source_mode",
        type=str,
        default=DIRECT_RUN_CONFIG["prediction_source_mode"],
        choices=["native_path", "array"],
        help="Use Ultralytics native image-path loading or preloaded numpy arrays for inference.",
    )
    return parser.parse_args(argv)


def default_experiment_specs(repo_root: Path) -> List[str]:
    return list(DIRECT_RUN_CONFIG["experiments"])


def parse_experiment_spec(spec: str, repo_root: Path) -> Tuple[str, Path]:
    if "::" not in spec:
        raise ValueError(f"Invalid experiment spec: {spec}")
    label, raw_path = spec.split("::", 1)
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return label.strip(), path.resolve()


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


def draw_text_box(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text(xy, text, fill=fill, font=font)


def color_for_score(score: float) -> Tuple[int, int, int]:
    score = max(0.0, min(1.0, float(score)))
    red = 255
    green = int(80 + 160 * score)
    blue = int(80 * (1.0 - score))
    return red, green, blue


def render_overlay(
    image_path: Path,
    gt_boxes: List[np.ndarray],
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    header_lines: List[str],
    top_pred_labels: int,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for gt_box in gt_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in gt_box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)

    order = np.argsort([-score for score in pred_scores]) if pred_scores else np.array([], dtype=int)
    for rank, idx in enumerate(order.tolist()):
        pred_box = pred_boxes[idx]
        score = pred_scores[idx]
        x1, y1, x2, y2 = [int(round(v)) for v in pred_box.tolist()]
        color = color_for_score(score)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if rank < top_pred_labels:
            draw_text_box(draw, (x1 + 2, max(2, y1 - 12)), f"{score:.3f}", fill=color)

    y = 4
    for line in header_lines:
        draw_text_box(draw, (4, y), line, fill=(255, 255, 255))
        y += 12
    return image


def thumbnail(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (16, 16, 16))
    x = (size[0] - thumb.size[0]) // 2
    y = (size[1] - thumb.size[1]) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def create_grid(
    tiles: Sequence[Image.Image],
    output_path: Path,
    columns: int,
    tile_size: Tuple[int, int] = (360, 360),
    padding: int = 16,
    bg: Tuple[int, int, int] = (248, 248, 248),
) -> None:
    if not tiles:
        return
    rows = int(math.ceil(len(tiles) / columns))
    width = columns * tile_size[0] + (columns + 1) * padding
    height = rows * tile_size[1] + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), bg)
    for idx, image in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        x = padding + col * (tile_size[0] + padding)
        y = padding + row * (tile_size[1] + padding)
        canvas.paste(thumbnail(image, tile_size), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def group_mean_iou(per_gt_rows: List[Dict[str, Any]]) -> OrderedDict[str, float]:
    grouped: "OrderedDict[str, List[float]]" = OrderedDict()
    for row in per_gt_rows:
        grouped.setdefault(row["image_id"], []).append(float(row["best_iou"]))
    return OrderedDict((image_id, float(np.mean(values))) for image_id, values in grouped.items())


def select_representative_images(
    experiments: List[Dict[str, Any]],
    baseline_label: str,
    max_examples: int,
) -> List[str]:
    baseline = next(exp for exp in experiments if exp["label"] == baseline_label)
    common_ids = set(baseline["image_mean_iou"].keys())
    for exp in experiments:
        common_ids &= set(exp["image_mean_iou"].keys())
    if not common_ids:
        return []

    base_scores = OrderedDict(
        (image_id, score)
        for image_id, score in baseline["image_mean_iou"].items()
        if image_id in common_ids
    )
    selected: List[str] = []

    for exp in experiments:
        if exp["label"] == baseline_label:
            continue
        gains = []
        for image_id, base_value in base_scores.items():
            if image_id not in exp["image_mean_iou"]:
                continue
            delta = exp["image_mean_iou"][image_id] - base_value
            gains.append((delta, image_id))
        gains.sort()
        for _, image_id in reversed(gains):
            if image_id not in selected:
                selected.append(image_id)
                break
        for _, image_id in gains:
            if image_id not in selected:
                selected.append(image_id)
                break

    if len(selected) < max_examples:
        variability = []
        candidate_ids = list(base_scores.keys())
        for image_id in candidate_ids:
            values = [exp["image_mean_iou"].get(image_id, np.nan) for exp in experiments]
            finite = [v for v in values if not np.isnan(v)]
            if finite:
                variability.append((float(np.std(finite)), image_id))
        variability.sort(reverse=True)
        for _, image_id in variability:
            if image_id not in selected:
                selected.append(image_id)
            if len(selected) >= max_examples:
                break

    return selected[:max_examples]


def plot_metric_comparison(experiments: List[Dict[str, Any]], output_path: Path) -> None:
    labels = [exp["label"] for exp in experiments]
    metrics = [
        ("mean_iou", "Mean IoU", False),
        ("recall_at_0.3", "Recall@0.3", False),
        ("recall_at_0.5", "Recall@0.5", False),
        ("mean_false_positives_per_image_at_iou_match", "FP / Image", True),
    ]
    colors = ["#34495e", "#d35400", "#2c7a7b", "#8e5ea2"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    y = np.arange(len(labels))
    for ax, (metric_key, title, lower_is_better) in zip(axes, metrics):
        values = [float(exp["summary"][metric_key]) for exp in experiments]
        ax.barh(y, values, color=colors[: len(labels)])
        ax.set_yticks(y, labels)
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        for idx, value in enumerate(values):
            ax.text(value, idx, f" {value:.3f}", va="center", ha="left", fontsize=10)
        if not lower_is_better:
            ax.set_xlim(0, max(values) * 1.18 if values else 1.0)
        else:
            ax.set_xlim(0, max(values) * 1.08 if values else 1.0)
    fig.suptitle("YOLO Test-Set Comparison Across Input Modalities", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_image_heatmap(experiments: List[Dict[str, Any]], output_path: Path) -> None:
    all_image_ids = list(experiments[0]["image_mean_iou"].keys())
    labels = [exp["label"] for exp in experiments]
    matrix = np.array(
        [
            [exp["image_mean_iou"].get(image_id, np.nan) for exp in experiments]
            for image_id in all_image_ids
        ],
        dtype=np.float32,
    )

    fig_h = max(8, 0.36 * len(all_image_ids))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(0.9, float(np.nanmax(matrix))))
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(all_image_ids)), labels=all_image_ids)
    ax.set_title("Per-Image Mean Best-IoU on Test Set")
    fig.colorbar(im, ax=ax, label="Mean best IoU")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_gt_boxes(label_path: Path, image_size: Tuple[int, int]) -> List[np.ndarray]:
    gt_boxes: List[np.ndarray] = []
    if not label_path.exists():
        return gt_boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gt_boxes.append(yolo_txt_to_xyxy(line, image_size))
    return gt_boxes


def load_prediction_source(image_path: Path) -> np.ndarray:
    sidecar_path = image_path.with_suffix(".npy")
    if sidecar_path.exists():
        return np.load(sidecar_path)
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"))


def rerender_selected_examples(
    experiments: List[Dict[str, Any]],
    selected_ids: List[str],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    if not selected_ids:
        return
    from ultralytics import YOLO

    tiles: List[Image.Image] = []
    for exp in experiments:
        dataset = load_dataset_yaml(Path(exp["summary"]["data"]))
        dataset_root = Path(dataset["path"])
        image_dir = dataset_root / "images" / "test"
        label_dir = dataset_root / "labels" / "test"
        valid_pairs = [
            (image_id, image_dir / f"{image_id}.png")
            for image_id in selected_ids
            if (image_dir / f"{image_id}.png").exists()
        ]
        if not valid_pairs:
            continue
        valid_ids = [image_id for image_id, _ in valid_pairs]
        image_paths = [image_path for _, image_path in valid_pairs]
        if args.prediction_source_mode == "native_path":
            prediction_sources = [str(image_path) for image_path in image_paths]
        else:
            prediction_sources = [load_prediction_source(image_path) for image_path in image_paths]

        model = YOLO(exp["summary"]["weights"])
        results = model.predict(
            source=prediction_sources,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=float(exp["summary"].get("predict_iou", 0.7)),
            max_det=int(exp["summary"].get("max_det", 300)),
            device=args.device,
            verbose=False,
            stream=False,
        )

        for image_id, image_path, result in zip(valid_ids, image_paths, results):
            with Image.open(image_path) as img:
                image_size = img.size
            gt_boxes = load_gt_boxes(label_dir / f"{image_id}.txt", image_size)
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
                iou_threshold=float(exp["summary"].get("post_nms_iou", 0.0)),
                ios_threshold=float(exp["summary"].get("post_nms_ios", 0.0)),
                max_det=int(exp["summary"].get("max_det", 300)),
            )

            image_mean_iou = exp["image_mean_iou"].get(image_id, float("nan"))
            matched_row = exp["per_image_map"].get(image_id, {})
            fp_count = matched_row.get("false_positives_at_iou_match", "n/a")
            header_lines = [
                exp["label"],
                image_id,
                f"meanIoU={image_mean_iou:.3f}" if not np.isnan(image_mean_iou) else "meanIoU=n/a",
                f"pred={len(pred_boxes)} fp={fp_count}",
                f"conf>={args.conf:.3f}",
            ]
            tiles.append(
                render_overlay(
                    image_path=image_path,
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    header_lines=header_lines,
                    top_pred_labels=args.top_pred_labels,
                )
            )
    create_grid(tiles, output_path=output_path, columns=len(experiments))


def write_summary(
    experiments: List[Dict[str, Any]],
    selected_ids: List[str],
    output_dir: Path,
    conf: float,
) -> None:
    baseline = experiments[0]
    common_ids = set(baseline["image_mean_iou"].keys())
    for exp in experiments:
        common_ids &= set(exp["image_mean_iou"].keys())
    rows = []
    for exp in experiments:
        summary = exp["summary"]
        rows.append(
            {
                "label": exp["label"],
                "mean_iou": float(summary["mean_iou"]),
                "recall_at_0.3": float(summary["recall_at_0.3"]),
                "recall_at_0.5": float(summary["recall_at_0.5"]),
                "mean_false_positives_per_image_at_iou_match": float(
                    summary["mean_false_positives_per_image_at_iou_match"]
                ),
            }
        )

    manifest = {
        "visualization_confidence_threshold": conf,
        "baseline_label": baseline["label"],
        "num_common_test_images": len(common_ids),
        "experiments": rows,
        "selected_examples": selected_ids,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    metric_rank = sorted(rows, key=lambda row: row["mean_iou"], reverse=True)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# YOLO Test-Set Model Comparison\n\n")
        f.write(f"- Visualization confidence threshold: `{conf}`\n")
        f.write(f"- Baseline: `{baseline['label']}`\n")
        f.write(f"- Common test images across experiments: `{len(common_ids)}`\n")
        f.write("- Mean-IoU ranking:\n")
        for row in metric_rank:
            f.write(
                f"  - `{row['label']}`: "
                f"mean_iou={row['mean_iou']:.3f}, "
                f"recall@0.3={row['recall_at_0.3']:.3f}, "
                f"recall@0.5={row['recall_at_0.5']:.3f}, "
                f"fp/image={row['mean_false_positives_per_image_at_iou_match']:.2f}\n"
            )
        if selected_ids:
            f.write("- Selected representative test events:\n")
            for image_id in selected_ids:
                deltas = []
                base_value = baseline["image_mean_iou"].get(image_id, float("nan"))
                for exp in experiments[1:]:
                    exp_value = exp["image_mean_iou"].get(image_id, float("nan"))
                    if np.isnan(base_value) or np.isnan(exp_value):
                        continue
                    deltas.append(f"{exp['label']} - {baseline['label']} = {exp_value - base_value:+.3f}")
                f.write(f"  - `{image_id}`: {', '.join(deltas)}\n")
        else:
            f.write("- Selected representative test events: `none (no shared test images across all experiments)`\n")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["YOLO_CONFIG_DIR"] = str(repo_root / "outputs" / "ultralytics_config")

    specs = args.experiment or default_experiment_specs(repo_root)
    experiments: List[Dict[str, Any]] = []
    for spec in specs:
        label, summary_path = parse_experiment_spec(spec, repo_root)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(summary_path.parent / "per_gt_details.json", "r", encoding="utf-8") as f:
            per_gt_rows = json.load(f)
        with open(summary_path.parent / "per_image_details.json", "r", encoding="utf-8") as f:
            per_image_rows = json.load(f)
        experiments.append(
            {
                "label": label,
                "summary_path": str(summary_path),
                "summary": summary,
                "per_gt_rows": per_gt_rows,
                "per_image_rows": per_image_rows,
                "per_image_map": {row["image_id"]: row for row in per_image_rows},
                "image_mean_iou": group_mean_iou(per_gt_rows),
            }
        )

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_comparison(experiments, output_dir / "metrics_overview.png")
    plot_per_image_heatmap(experiments, output_dir / "per_image_mean_iou_heatmap.png")
    selected_ids = select_representative_images(
        experiments=experiments,
        baseline_label=experiments[0]["label"],
        max_examples=args.max_examples,
    )
    rerender_selected_examples(
        experiments=experiments,
        selected_ids=selected_ids,
        args=args,
        output_path=output_dir / "selected_examples_grid.png",
    )
    write_summary(experiments, selected_ids, output_dir, args.conf)

    print(f"[Done] outputs written to {output_dir}")
    print(f"[Done] selected examples: {selected_ids}")
    return 0


if __name__ == "__main__":
    if USE_DIRECT_RUN_CONFIG and len(sys.argv) == 1:
        sys.argv.extend(_build_direct_run_args())
    raise SystemExit(main())
