"""Export same-frame HDF5 input images for modality alignment inspection."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODALITIES = ["magnetogram", "halpha", "euv_94", "euv_171", "euv_304"]


def _decode_timestamp(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _stretch_for_display(image: np.ndarray, modality: str) -> tuple[np.ndarray, tuple[float, float]]:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32), (float("nan"), float("nan"))

    values = arr[finite]
    if modality == "magnetogram":
        limit = float(np.nanpercentile(np.abs(values), 99.0))
        limit = max(limit, 1e-6)
        low, high = -limit, limit
    else:
        low, high = np.nanpercentile(values, [1.0, 99.5])
        low, high = float(low), float(high)
        if high <= low:
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if high <= low:
            high = low + 1e-6

    display = np.clip((arr - low) / (high - low), 0.0, 1.0)
    display = np.nan_to_num(display, nan=0.0, posinf=1.0, neginf=0.0)
    return display.astype(np.float32), (low, high)


def _save_single_image(
    display: np.ndarray,
    output_path: Path,
    event_id: str,
    modality: str,
    frame_index: int,
    timestamp: str,
) -> None:
    image = _display_to_rgb_with_grid(display, title=f"{modality} | frame {frame_index}\n{timestamp}")
    image.save(output_path)


def _display_to_rgb_with_grid(display: np.ndarray, title: str = "") -> Image.Image:
    gray = np.clip(display * 255.0, 0, 255).astype(np.uint8)
    base = Image.fromarray(gray, mode="L").convert("RGB")
    scale = 2
    base = base.resize((base.width * scale, base.height * scale), Image.Resampling.NEAREST)
    title_h = 44 if title else 0
    canvas = Image.new("RGB", (base.width, base.height + title_h), "white")
    canvas.paste(base, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if title:
        y = 4
        for line in title.split("\n"):
            draw.text((6, y), line, fill=(0, 0, 0), font=font)
            y += 14
    grid_color = (0, 220, 220)
    for pos in [0, 64, 128, 192, 255]:
        p = pos * scale
        draw.line((p, title_h, p, title_h + base.height - 1), fill=grid_color, width=1)
        draw.line((0, title_h + p, base.width - 1, title_h + p), fill=grid_color, width=1)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default="data/Solar_Flares_CME_dataset_fitscache.h5")
    parser.add_argument("--event-id", default="EVT_20250331_101600_1")
    parser.add_argument("--frame", default="last")
    parser.add_argument("--out-dir", default="outputs/diagnostics/input_alignment")
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    root_out = Path(args.out_dir)
    output_dir = root_out / f"{args.event_id}_frame_{args.frame}"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_lines: list[str] = []
    display_images: dict[str, np.ndarray] = {}

    with h5py.File(hdf5_path, "r") as h5:
        event = h5[f"events/{args.event_id}"]
        timestamps = [_decode_timestamp(v) for v in event["timestamps"][...]]
        if args.frame == "last":
            frame_index = len(timestamps) - 1
        elif args.frame == "first":
            frame_index = 0
        elif args.frame == "middle":
            frame_index = len(timestamps) // 2
        else:
            frame_index = int(args.frame)
            if frame_index < 0:
                frame_index = len(timestamps) + frame_index
        frame_index = max(0, min(frame_index, len(timestamps) - 1))
        timestamp = timestamps[frame_index]

        for modality in args.modalities:
            data_path = f"data/{modality}/images"
            if data_path not in event:
                stats_lines.append(f"{modality}: missing")
                continue
            image = np.asarray(event[data_path][frame_index, :, :], dtype=np.float32)
            display, limits = _stretch_for_display(image, modality)
            display_images[modality] = display

            values = image[np.isfinite(image)]
            stats_lines.append(
                f"{modality}: shape={image.shape}, raw_min={float(np.nanmin(values)):.6g}, "
                f"raw_p1={float(np.nanpercentile(values, 1)):.6g}, "
                f"raw_p50={float(np.nanpercentile(values, 50)):.6g}, "
                f"raw_p99={float(np.nanpercentile(values, 99)):.6g}, "
                f"raw_max={float(np.nanmax(values)):.6g}, "
                f"display_limits=({limits[0]:.6g},{limits[1]:.6g})"
            )

            _save_single_image(
                display,
                output_dir / f"{modality}_frame_{frame_index}.png",
                args.event_id,
                modality,
                frame_index,
                timestamp,
            )
            print(f"saved {modality}", flush=True)

    available = [m for m in args.modalities if m in display_images]
    tiles = [_display_to_rgb_with_grid(display_images[m], title=m) for m in available]
    gap = 12
    header_h = 34
    width = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
    height = max(tile.height for tile in tiles) + header_h
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    draw.text(
        (6, 8),
        f"{args.event_id} | frame {frame_index} | {timestamp} | same HDF5 input frame",
        fill=(0, 0, 0),
        font=ImageFont.load_default(),
    )
    x = 0
    for tile in tiles:
        grid.paste(tile, (x, header_h))
        x += tile.width + gap
    grid.save(output_dir / "all_modalities_same_frame_alignment_grid.png")

    with open(output_dir / "stats.txt", "w", encoding="utf-8") as handle:
        handle.write(f"hdf5={hdf5_path}\n")
        handle.write(f"event_id={args.event_id}\n")
        handle.write(f"frame_index={frame_index}\n")
        handle.write(f"timestamp={timestamp}\n")
        handle.write("\n".join(stats_lines) + "\n")

    print(output_dir.resolve())
    print(f"timestamp={timestamp}")
    print("\n".join(stats_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
