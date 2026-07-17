"""
Analyze proposal/refined localization errors on a fixed split.

This script focuses on experiment C:
- proposal vs refined bbox metrics
- recall@IoU thresholds
- center/size errors
- slot-level error type breakdown
- representative success/failure visualizations
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import SolarFlareDataset
from data.hdf5_reader import HDF5DatasetReader
from models.multimodal_transformer import MultimodalTransformer
from scripts.f_train_model import custom_collate
from utils.config_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze localizer errors for proposal/refined boxes.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/checkpoints/solar_flare_cme_model_epoch_0093.pth",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_config.yaml",
        help="Data config path.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="Model config path.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/training_config.yaml",
        help="Training config path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to analyze.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cpu/cuda. Empty means auto.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/localizer_error_analysis",
        help="Output directory.",
    )
    parser.add_argument(
        "--max_visualizations",
        type=int,
        default=10,
        help="How many success/failure examples to save for refined boxes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible visualization sampling.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_enabled_modalities(modalities_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: cfg
        for name, cfg in modalities_cfg.items()
        if bool(cfg.get("enabled", True))
    }


def build_model_config(
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
) -> Dict[str, Any]:
    model_cfg = dict(model_config)
    model_cfg["modalities"] = get_enabled_modalities(data_config["modalities"])
    model_cfg["max_activities"] = data_config.get("max_activities", model_cfg.get("max_activities", 5))
    model_cfg["sequence_length"] = data_config.get("sequence_length", model_cfg.get("sequence_length", 9))
    model_cfg.setdefault("stage2", {})
    if train_config.get("two_stage_schedule", {}).get("enabled", False):
        model_cfg["stage2"]["roi_source_mix"] = train_config["two_stage_schedule"].get(
            "roi_source_mix", {"predicted": 1.0}
        )
    return model_cfg


def load_split_events(reader: HDF5DatasetReader, data_config: Dict[str, Any], split_name: str) -> List[str]:
    split_dir = Path(data_config.get("split_data_dir", "data/split_data"))
    split_path = split_dir / f"{split_name}_events.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        event_ids = [line.strip() for line in f if line.strip()]
    available = set(reader.get_event_ids(available_only=True))
    return [event_id for event_id in event_ids if event_id in available]


def create_loader(
    hdf5_path: str,
    data_config: Dict[str, Any],
    split_name: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[SolarFlareDataset, DataLoader]:
    reader = HDF5DatasetReader(hdf5_path)
    event_ids = load_split_events(reader, data_config, split_name)
    enabled_modalities = get_enabled_modalities(data_config["modalities"])
    modalities_list = list(enabled_modalities.keys())
    first_mod = next(iter(enabled_modalities.values()), {})
    target_size = tuple(first_mod.get("resolution", [256, 256]))

    dataset = SolarFlareDataset(
        reader=reader,
        event_ids=event_ids,
        modalities=modalities_list,
        sequence_length=data_config.get("sequence_length", 9),
        stride=data_config.get("stride", 1),
        target_size=target_size,
        max_activities=data_config.get("max_activities", 5),
        config={
            "max_activities": data_config.get("max_activities", 5),
            "proposal_cache_path": data_config.get("proposal_cache_path"),
        },
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=custom_collate,
    )
    return dataset, loader


def load_model(model_path: str, model_cfg: Dict[str, Any], device: torch.device) -> MultimodalTransformer:
    model = MultimodalTransformer(model_cfg)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def box_area(box: np.ndarray) -> float:
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    return width * height


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


def center_and_size_errors(pred_box: np.ndarray, gt_box: np.ndarray) -> Tuple[float, float, float]:
    pred_center = np.array([(pred_box[0] + pred_box[2]) * 0.5, (pred_box[1] + pred_box[3]) * 0.5], dtype=np.float32)
    gt_center = np.array([(gt_box[0] + gt_box[2]) * 0.5, (gt_box[1] + gt_box[3]) * 0.5], dtype=np.float32)
    center_error = float(np.linalg.norm(pred_center - gt_center))

    pred_w = max(1e-6, float(pred_box[2] - pred_box[0]))
    pred_h = max(1e-6, float(pred_box[3] - pred_box[1]))
    gt_w = max(1e-6, float(gt_box[2] - gt_box[0]))
    gt_h = max(1e-6, float(gt_box[3] - gt_box[1]))
    width_rel = abs(pred_w - gt_w) / gt_w
    height_rel = abs(pred_h - gt_h) / gt_h
    size_error = float((width_rel + height_rel) * 0.5)
    return center_error, size_error, float(abs(math.log((pred_w * pred_h) / (gt_w * gt_h))))


def classify_error_type(
    own_iou: float,
    best_any_iou: float,
    best_any_slot: int,
    slot_id: int,
    center_error: float,
    size_error: float,
) -> str:
    if best_any_iou < 0.1:
        return "missed"
    if best_any_slot != slot_id and best_any_iou >= max(own_iou + 0.05, 0.1):
        return "wrong_activity"
    if own_iou >= 0.5:
        return "good"
    offset_bad = center_error > 0.15
    size_bad = size_error > 0.5
    if offset_bad and size_bad:
        return "offset_and_size"
    if offset_bad:
        return "offset"
    if size_bad:
        return "size"
    return "low_overlap"


def prepare_image(sample_data: Dict[str, torch.Tensor], modality_name: str) -> np.ndarray:
    tensor = sample_data[modality_name]
    if tensor.ndim == 4:
        image = tensor[-1, 0].detach().cpu().numpy()
    elif tensor.ndim == 3:
        image = tensor[-1].detach().cpu().numpy()
    else:
        image = tensor.detach().cpu().numpy()
    image = image.astype(np.float32)
    finite_mask = np.isfinite(image)
    if finite_mask.any():
        lo = np.percentile(image[finite_mask], 1)
        hi = np.percentile(image[finite_mask], 99)
        if hi > lo:
            image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
        else:
            image = np.zeros_like(image)
    else:
        image = np.zeros_like(image)
    return image


def draw_box(ax: Any, box: np.ndarray, color: str, label: str, linewidth: float = 2.0) -> None:
    x1, y1, x2, y2 = box.tolist()
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)
    ax.text(x1, max(0.0, y1 - 4), label, color=color, fontsize=8, bbox={"facecolor": "black", "alpha": 0.4, "pad": 1})


def save_visualization(
    sample_record: Dict[str, Any],
    output_path: Path,
    modality_name: str,
) -> None:
    image = sample_record["image"]
    height, width = image.shape[-2], image.shape[-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray", origin="upper")
    gt_box = np.asarray(sample_record["gt_box"], dtype=np.float32)
    proposal_box = np.asarray(sample_record["proposal_box"], dtype=np.float32)
    refined_box = np.asarray(sample_record["refined_box"], dtype=np.float32)

    draw_box(ax, gt_box * np.array([width, height, width, height], dtype=np.float32), "#00ff7f", "GT")
    draw_box(ax, proposal_box * np.array([width, height, width, height], dtype=np.float32), "#ffb000", "Proposal")
    draw_box(ax, refined_box * np.array([width, height, width, height], dtype=np.float32), "#00b7ff", "Refined")

    title = (
        f"{sample_record['event_id']} | slot {sample_record['slot_id']} | "
        f"proposal IoU={sample_record['proposal_iou']:.3f}, refined IoU={sample_record['refined_iou']:.3f}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def aggregate_stage_metrics(stage_df: pd.DataFrame, stage_name: str) -> Dict[str, Any]:
    if stage_df.empty:
        return {
            "stage": stage_name,
            "num_slots": 0,
        }

    summary = {
        "stage": stage_name,
        "num_slots": int(len(stage_df)),
        "mean_iou": float(stage_df["iou"].mean()),
        "median_iou": float(stage_df["iou"].median()),
        "recall_at_0.1": float((stage_df["iou"] >= 0.1).mean()),
        "recall_at_0.3": float((stage_df["iou"] >= 0.3).mean()),
        "recall_at_0.5": float((stage_df["iou"] >= 0.5).mean()),
        "mean_center_error": float(stage_df["center_error"].mean()),
        "median_center_error": float(stage_df["center_error"].median()),
        "mean_size_error": float(stage_df["size_error"].mean()),
        "median_size_error": float(stage_df["size_error"].median()),
        "mean_log_area_error": float(stage_df["log_area_error"].mean()),
        "error_type_counts": stage_df["error_type"].value_counts().to_dict(),
    }
    return summary


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = load_config(args.data_config)["data"]
    model_root = load_config(args.model_config)["model"]
    train_root = load_config(args.train_config)["training"]
    hdf5_path = data_root["hdf5_path"]

    dataset, loader = create_loader(
        hdf5_path=hdf5_path,
        data_config=data_root,
        split_name=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_cfg = build_model_config(data_root, model_root, train_root)
    device = resolve_device(args.device)
    model = load_model(args.model_path, model_cfg, device)

    modality_name = dataset.modalities[0]
    rows: List[Dict[str, Any]] = []
    viz_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = {k: v.to(device) for k, v in batch["data"].items()}
            targets = {
                "label": batch["label"].to(device),
                "bbox": batch["bbox"].to(device),
                "time_features": batch["time_features"].to(device),
                "activity_mask": batch["activity_mask"].to(device),
            }
            outputs = model(inputs, targets=targets)

            labels = targets["label"].detach().cpu().numpy()
            activity_mask = targets["activity_mask"].detach().cpu().numpy().astype(bool)
            gt_bboxes = targets["bbox"].detach().cpu().numpy()
            proposal_boxes = outputs["proposal_boxes"].detach().cpu().numpy()
            refined_boxes = outputs["bbox_pred"].detach().cpu().numpy()
            metadata_list = batch["metadata"]

            for sample_offset in range(labels.shape[0]):
                positive_slots = np.where((labels[sample_offset] > 0) & activity_mask[sample_offset])[0].tolist()
                all_gt_boxes = gt_bboxes[sample_offset, positive_slots] if positive_slots else np.empty((0, 4), dtype=np.float32)
                if not positive_slots:
                    continue

                image = prepare_image({modality_name: batch["data"][modality_name][sample_offset]}, modality_name)
                metadata = metadata_list[sample_offset]
                event_id = metadata.get("event_id", f"sample_{batch_idx}_{sample_offset}")
                window_id = metadata.get("window_id", f"{event_id}_{batch_idx}_{sample_offset}")

                for slot_id in positive_slots:
                    gt_box = gt_bboxes[sample_offset, slot_id]
                    proposal_box = proposal_boxes[sample_offset, slot_id]
                    refined_box = refined_boxes[sample_offset, slot_id]

                    proposal_ious_any = [compute_iou(proposal_box, other_gt) for other_gt in all_gt_boxes]
                    refined_ious_any = [compute_iou(refined_box, other_gt) for other_gt in all_gt_boxes]
                    proposal_best_idx = int(np.argmax(proposal_ious_any)) if proposal_ious_any else -1
                    refined_best_idx = int(np.argmax(refined_ious_any)) if refined_ious_any else -1
                    proposal_best_slot = positive_slots[proposal_best_idx] if proposal_best_idx >= 0 else -1
                    refined_best_slot = positive_slots[refined_best_idx] if refined_best_idx >= 0 else -1

                    proposal_iou = compute_iou(proposal_box, gt_box)
                    refined_iou = compute_iou(refined_box, gt_box)

                    proposal_center_error, proposal_size_error, proposal_log_area_error = center_and_size_errors(proposal_box, gt_box)
                    refined_center_error, refined_size_error, refined_log_area_error = center_and_size_errors(refined_box, gt_box)

                    proposal_error_type = classify_error_type(
                        own_iou=proposal_iou,
                        best_any_iou=max(proposal_ious_any) if proposal_ious_any else 0.0,
                        best_any_slot=proposal_best_slot,
                        slot_id=slot_id,
                        center_error=proposal_center_error,
                        size_error=proposal_size_error,
                    )
                    refined_error_type = classify_error_type(
                        own_iou=refined_iou,
                        best_any_iou=max(refined_ious_any) if refined_ious_any else 0.0,
                        best_any_slot=refined_best_slot,
                        slot_id=slot_id,
                        center_error=refined_center_error,
                        size_error=refined_size_error,
                    )

                    base_info = {
                        "event_id": event_id,
                        "window_id": window_id,
                        "slot_id": int(slot_id),
                        "label": int(labels[sample_offset, slot_id]),
                        "num_positive_slots": int(len(positive_slots)),
                        "gt_box": gt_box.tolist(),
                        "proposal_box": proposal_box.tolist(),
                        "refined_box": refined_box.tolist(),
                    }

                    rows.append(
                        {
                            **base_info,
                            "stage": "proposal",
                            "iou": proposal_iou,
                            "best_iou_any": float(max(proposal_ious_any) if proposal_ious_any else 0.0),
                            "best_gt_slot": int(proposal_best_slot),
                            "center_error": proposal_center_error,
                            "size_error": proposal_size_error,
                            "log_area_error": proposal_log_area_error,
                            "error_type": proposal_error_type,
                        }
                    )
                    rows.append(
                        {
                            **base_info,
                            "stage": "refined",
                            "iou": refined_iou,
                            "best_iou_any": float(max(refined_ious_any) if refined_ious_any else 0.0),
                            "best_gt_slot": int(refined_best_slot),
                            "center_error": refined_center_error,
                            "size_error": refined_size_error,
                            "log_area_error": refined_log_area_error,
                            "error_type": refined_error_type,
                        }
                    )

                    viz_records.append(
                        {
                            **base_info,
                            "proposal_iou": proposal_iou,
                            "refined_iou": refined_iou,
                            "proposal_error_type": proposal_error_type,
                            "refined_error_type": refined_error_type,
                            "image": image,
                        }
                    )

    slot_df = pd.DataFrame(rows)
    proposal_df = slot_df[slot_df["stage"] == "proposal"].reset_index(drop=True)
    refined_df = slot_df[slot_df["stage"] == "refined"].reset_index(drop=True)

    summary = {
        "split": args.split,
        "model_path": args.model_path,
        "hdf5_path": hdf5_path,
        "num_windows": int(len(dataset)),
        "num_positive_slots": int(len(proposal_df)),
        "proposal": aggregate_stage_metrics(proposal_df, "proposal"),
        "refined": aggregate_stage_metrics(refined_df, "refined"),
    }

    summary["delta"] = {
        "mean_iou_gain": float(summary["refined"]["mean_iou"] - summary["proposal"]["mean_iou"]) if proposal_df.size else 0.0,
        "recall_at_0.1_gain": float(summary["refined"]["recall_at_0.1"] - summary["proposal"]["recall_at_0.1"]) if proposal_df.size else 0.0,
        "recall_at_0.3_gain": float(summary["refined"]["recall_at_0.3"] - summary["proposal"]["recall_at_0.3"]) if proposal_df.size else 0.0,
        "recall_at_0.5_gain": float(summary["refined"]["recall_at_0.5"] - summary["proposal"]["recall_at_0.5"]) if proposal_df.size else 0.0,
        "center_error_change": float(summary["refined"]["mean_center_error"] - summary["proposal"]["mean_center_error"]) if proposal_df.size else 0.0,
        "size_error_change": float(summary["refined"]["mean_size_error"] - summary["proposal"]["mean_size_error"]) if proposal_df.size else 0.0,
    }

    output_root = Path(args.output_dir) / f"{args.split}_epoch_analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    slot_df.to_csv(output_root / "slot_localization_errors.csv", index=False, encoding="utf-8-sig")

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_root / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Localizer Error Analysis\n\n")
        f.write(f"- split: `{summary['split']}`\n")
        f.write(f"- model: `{summary['model_path']}`\n")
        f.write(f"- num_windows: `{summary['num_windows']}`\n")
        f.write(f"- num_positive_slots: `{summary['num_positive_slots']}`\n\n")
        for stage_name in ("proposal", "refined"):
            stage = summary[stage_name]
            f.write(f"## {stage_name.title()}\n\n")
            f.write(f"- mean IoU: `{stage['mean_iou']:.6f}`\n")
            f.write(f"- median IoU: `{stage['median_iou']:.6f}`\n")
            f.write(f"- Recall@0.1: `{stage['recall_at_0.1']:.6f}`\n")
            f.write(f"- Recall@0.3: `{stage['recall_at_0.3']:.6f}`\n")
            f.write(f"- Recall@0.5: `{stage['recall_at_0.5']:.6f}`\n")
            f.write(f"- mean center error: `{stage['mean_center_error']:.6f}`\n")
            f.write(f"- mean size error: `{stage['mean_size_error']:.6f}`\n")
            f.write(f"- error types: `{stage['error_type_counts']}`\n\n")
        f.write("## Delta (Refined - Proposal)\n\n")
        for key, value in summary["delta"].items():
            f.write(f"- {key}: `{value:.6f}`\n")

    refined_viz_df = refined_df.sort_values("iou", ascending=False)
    success_candidates = refined_viz_df.head(max(args.max_visualizations * 3, args.max_visualizations))
    failure_candidates = refined_df.sort_values("iou", ascending=True).head(max(args.max_visualizations * 3, args.max_visualizations))

    viz_by_key = {(record["event_id"], record["window_id"], record["slot_id"]): record for record in viz_records}

    def pick_records(source_df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
        picked: List[Dict[str, Any]] = []
        seen: set = set()
        for _, row in source_df.iterrows():
            key = (row["event_id"], row["window_id"], int(row["slot_id"]))
            if key in seen or key not in viz_by_key:
                continue
            picked.append(viz_by_key[key])
            seen.add(key)
            if len(picked) >= limit:
                break
        return picked

    success_records = pick_records(success_candidates, args.max_visualizations)
    failure_records = pick_records(failure_candidates, args.max_visualizations)

    for idx, record in enumerate(success_records):
        save_visualization(record, output_root / "visualizations" / "success" / f"{idx:02d}_{record['event_id']}_slot{record['slot_id']}.png", modality_name)
    for idx, record in enumerate(failure_records):
        save_visualization(record, output_root / "visualizations" / "failure" / f"{idx:02d}_{record['event_id']}_slot{record['slot_id']}.png", modality_name)

    sample_payload = {
        "success_examples": [
            {
                "event_id": r["event_id"],
                "window_id": r["window_id"],
                "slot_id": r["slot_id"],
                "proposal_iou": r["proposal_iou"],
                "refined_iou": r["refined_iou"],
                "refined_error_type": r["refined_error_type"],
            }
            for r in success_records
        ],
        "failure_examples": [
            {
                "event_id": r["event_id"],
                "window_id": r["window_id"],
                "slot_id": r["slot_id"],
                "proposal_iou": r["proposal_iou"],
                "refined_iou": r["refined_iou"],
                "refined_error_type": r["refined_error_type"],
            }
            for r in failure_records
        ],
    }
    with open(output_root / "sample_examples.json", "w", encoding="utf-8") as f:
        json.dump(sample_payload, f, ensure_ascii=False, indent=2)

    return summary


def main() -> int:
    args = parse_args()
    summary = analyze(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
