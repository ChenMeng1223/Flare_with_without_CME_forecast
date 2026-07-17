"""
Visualize stage-2 proposal cache overlays for one split.

The script uses the same SolarFlareDataset path as stage-2 training, so the
rendered proposal slots and GT boxes match what the model consumes.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data.dataset import SolarFlareDataset
from data.hdf5_reader import HDF5DatasetReader


def _load_module_attr(module_name: str, relative_path: str, attr_name: str):
    module_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {attr_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


load_config = _load_module_attr("config_utils_only", "utils/config_utils.py", "load_config")


USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    "data_config": "configs/data_config_stage2_yolo11_5modal_256.yaml",
    "split": "test",
    "output_dir": "outputs/proposal_visualizations/stage2_yolo11_5modal_256_test_maxdet15",
    "background_modality": "magnetogram",
    "frame": "last",
    "max_images": 0,
    "columns": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize retained stage-2 proposals with GT boxes.")
    parser.add_argument("--data_config", type=str, default=DIRECT_RUN_CONFIG["data_config"])
    parser.add_argument("--split", type=str, default=DIRECT_RUN_CONFIG["split"], choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default=DIRECT_RUN_CONFIG["output_dir"])
    parser.add_argument("--background_modality", type=str, default=DIRECT_RUN_CONFIG["background_modality"])
    parser.add_argument("--frame", type=str, default=DIRECT_RUN_CONFIG["frame"], choices=["first", "middle", "last"])
    parser.add_argument("--max_images", type=int, default=DIRECT_RUN_CONFIG["max_images"], help="0 means all windows.")
    parser.add_argument("--columns", type=int, default=DIRECT_RUN_CONFIG["columns"])
    if USE_DIRECT_RUN_CONFIG and len(sys.argv) == 1:
        args: List[str] = []
        for key, value in DIRECT_RUN_CONFIG.items():
            if value not in (None, ""):
                args.extend([f"--{key}", str(value)])
        return parser.parse_args(args)
    return parser.parse_args()


def load_event_ids(split: str) -> List[str]:
    path = project_root / "data" / "split_data" / f"{split}_events.txt"
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def enabled_modalities(data_cfg: Dict[str, Any]) -> List[str]:
    modalities = data_cfg.get("modalities", {})
    return [name for name, cfg in modalities.items() if bool(cfg.get("enabled", True))]


def first_target_size(data_cfg: Dict[str, Any]) -> Tuple[int, int]:
    for cfg in data_cfg.get("modalities", {}).values():
        if bool(cfg.get("enabled", True)):
            resolution = cfg.get("resolution", [256, 256])
            return int(resolution[0]), int(resolution[1])
    return 256, 256


def pick_frame_index(num_frames: int, frame: str) -> int:
    if num_frames <= 1:
        return 0
    if frame == "first":
        return 0
    if frame == "middle":
        return num_frames // 2
    return num_frames - 1


def tensor_to_background(sample: Dict[str, Any], modality: str, frame: str) -> np.ndarray:
    data = sample.get("data", {})
    if modality not in data:
        modality = next(iter(data.keys()))
    arr = data[modality]
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    # Expected: T,C,H,W or T,H,W. Collapse channels for display.
    idx = pick_frame_index(arr.shape[0], frame)
    image = arr[idx]
    if image.ndim == 3:
        image = image.mean(axis=0)
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape[-2:], dtype=np.uint8)
    lo, hi = np.percentile(image[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(image[finite].min()), float(image[finite].max())
    norm = (image - lo) / (hi - lo + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def valid_boxes(boxes: Any, mask: Optional[Any] = None, scores: Optional[Any] = None) -> List[Tuple[np.ndarray, Optional[float]]]:
    if isinstance(boxes, torch.Tensor):
        boxes_arr = boxes.detach().cpu().numpy()
    else:
        boxes_arr = np.asarray(boxes)
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_arr = mask.detach().cpu().numpy().astype(bool)
        else:
            mask_arr = np.asarray(mask).astype(bool)
    else:
        mask_arr = np.ones((len(boxes_arr),), dtype=bool)
    if scores is not None:
        if isinstance(scores, torch.Tensor):
            score_arr = scores.detach().cpu().numpy().reshape(-1)
        else:
            score_arr = np.asarray(scores).reshape(-1)
    else:
        score_arr = np.full((len(boxes_arr),), np.nan, dtype=np.float32)

    out: List[Tuple[np.ndarray, Optional[float]]] = []
    for idx, box in enumerate(boxes_arr):
        if idx >= len(mask_arr) or not bool(mask_arr[idx]):
            continue
        arr = np.asarray(box, dtype=np.float32).reshape(-1)
        if arr.size < 4 or not np.isfinite(arr[:4]).all():
            continue
        if float(np.abs(arr[:4]).sum()) <= 0.0:
            continue
        score = float(score_arr[idx]) if idx < len(score_arr) and np.isfinite(score_arr[idx]) else None
        out.append((np.clip(arr[:4], 0.0, 1.0), score))
    return out


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 2
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text(xy, text, fill=fill, font=font)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: np.ndarray,
    image_size: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int,
    label: Optional[str] = None,
) -> None:
    w, h = image_size
    x1, y1, x2, y2 = box.tolist()
    coords = (int(round(x1 * w)), int(round(y1 * h)), int(round(x2 * w)), int(round(y2 * h)))
    draw.rectangle(coords, outline=color, width=width)
    if label:
        draw_label(draw, (coords[0] + 2, max(2, coords[1] - 12)), label, color)


def render_window(sample: Dict[str, Any], modality: str, frame: str) -> Tuple[Image.Image, Dict[str, Any]]:
    bg = tensor_to_background(sample, modality, frame)
    image = Image.fromarray(bg, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    metadata = sample.get("metadata", {})

    gt_items = valid_boxes(sample.get("region_bbox"), sample.get("region_mask"))
    if not gt_items:
        gt_items = valid_boxes(sample.get("bbox"), sample.get("activity_mask"))
    proposal_items = valid_boxes(sample.get("proposal_boxes"), scores=sample.get("proposal_scores"))
    proposal_items = sorted(proposal_items, key=lambda item: -1.0 if item[1] is None else -item[1])

    for gt_idx, (box, _) in enumerate(gt_items, start=1):
        draw_box(draw, box, image.size, color=(0, 255, 0), width=4, label=f"GT{gt_idx}")
    for pred_idx, (box, score) in enumerate(proposal_items, start=1):
        label = f"P{pred_idx}:{score:.3f}" if score is not None else f"P{pred_idx}"
        draw_box(draw, box, image.size, color=(255, 80, 40), width=2, label=label if pred_idx <= 15 else None)

    header = [
        f"{metadata.get('event_id')} | {metadata.get('window_id')}",
        f"start={metadata.get('start_idx')} end={metadata.get('end_idx')} | proposals={len(proposal_items)} gt={len(gt_items)}",
        "green=GT, orange=retained proposals",
    ]
    y = 4
    for line in header:
        draw_label(draw, (4, y), line, (255, 255, 255))
        y += 13

    stats = {
        "window_id": metadata.get("window_id"),
        "event_id": metadata.get("event_id"),
        "start_idx": metadata.get("start_idx"),
        "end_idx": metadata.get("end_idx"),
        "num_gt": len(gt_items),
        "num_proposals": len(proposal_items),
        "proposal_scores": [score for _, score in proposal_items if score is not None],
    }
    return image, stats


def make_contact_sheet(images: List[Image.Image], captions: List[str], output_path: Path, columns: int) -> None:
    if not images:
        return
    columns = max(1, columns)
    thumb_size = (320, 320)
    caption_h = 34
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + caption_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (image, caption) in enumerate(zip(images, captions)):
        row = idx // columns
        col = idx % columns
        thumb = image.copy()
        thumb.thumbnail(thumb_size)
        x = col * thumb_size[0] + (thumb_size[0] - thumb.width) // 2
        y = row * (thumb_size[1] + caption_h)
        sheet.paste(thumb, (x, y))
        draw.text((col * thumb_size[0] + 4, y + thumb_size[1] + 4), caption[:56], fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    args = parse_args()
    data_config_path = Path(args.data_config)
    if not data_config_path.is_absolute():
        data_config_path = project_root / data_config_path
    data_cfg = load_config(str(data_config_path))["data"]

    hdf5_path = Path(data_cfg.get("hdf5_path", "data/Solar_Flares_CME_dataset.h5"))
    if not hdf5_path.is_absolute():
        hdf5_path = project_root / hdf5_path
    reader = HDF5DatasetReader(str(hdf5_path))

    dataset = SolarFlareDataset(
        reader=reader,
        event_ids=load_event_ids(args.split),
        modalities=enabled_modalities(data_cfg),
        sequence_length=int(data_cfg.get("sequence_length", 10)),
        stride=int(data_cfg.get("stride", 10)),
        target_size=first_target_size(data_cfg),
        max_activities=int(data_cfg.get("max_activities", 15)),
        config={
            "max_activities": int(data_cfg.get("max_activities", 15)),
            "proposal_cache_path": data_cfg.get("proposal_cache_path"),
        },
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    image_dir = output_dir / "windows"
    image_dir.mkdir(parents=True, exist_ok=True)

    images: List[Image.Image] = []
    captions: List[str] = []
    stats_rows: List[Dict[str, Any]] = []
    total = len(dataset) if args.max_images <= 0 else min(len(dataset), args.max_images)
    for idx in range(total):
        sample = dataset[idx]
        image, stats = render_window(sample, args.background_modality, args.frame)
        event_id = str(stats["event_id"])
        window_id = str(stats["window_id"])
        path = image_dir / f"{idx:04d}_{event_id}_{window_id}.png"
        image.save(path)
        images.append(image)
        captions.append(f"{idx:04d} {event_id} p={stats['num_proposals']} gt={stats['num_gt']}")
        stats["image_path"] = str(path)
        stats_rows.append(stats)

    make_contact_sheet(images, captions, output_dir / "contact_sheet.png", args.columns)
    summary = {
        "data_config": str(data_config_path),
        "proposal_cache_path": data_cfg.get("proposal_cache_path"),
        "split": args.split,
        "num_rendered": total,
        "output_dir": str(output_dir),
        "proposal_count_distribution": {
            str(k): int(sum(1 for row in stats_rows if row["num_proposals"] == k))
            for k in sorted({row["num_proposals"] for row in stats_rows})
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "windows": stats_rows}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
