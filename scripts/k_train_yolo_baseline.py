"""
Train a YOLO11 single-class activity-region localization baseline.

Experiment D baseline:
- detector only
- one class: activity_region
- report outputs into a project-local runs directory
"""
import argparse
import os
import sys
from pathlib import Path


USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    "data": "outputs/yolo_dataset/activity_region_single_class_mag_euv94_euv171_processed_export_filtered20/dataset.yaml",
    "model": "yolo11n.pt",
    "epochs": 100,
    "imgsz": 512,
    "batch": 16,
    "device": "0",
    "workers": 4,
    "project": "outputs/yolo_runs", # 输出目录
    "name": "yolo11n_activity_region_mag_euv94_euv171_processed_512_filtered20", # 输出名称
    "channels": 3,
    "seed": 42, # 随机种子
    "patience": 30,
}


def _build_direct_run_args() -> list[str]:
    cfg = dict(DIRECT_RUN_CONFIG)
    args: list[str] = []
    for key in ("data", "model", "epochs", "imgsz", "batch", "device", "workers", "project", "name", "channels", "seed", "patience"):
        value = cfg.get(key)
        if value not in (None, ""):
            args.extend([f"--{key}", str(value)])
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 localization baseline.")
    parser.add_argument(
        "--data",
        type=str,
        default=DIRECT_RUN_CONFIG["data"],
        help="YOLO dataset yaml path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model checkpoint or model name.",
    )
    parser.add_argument("--epochs", type=int, default=DIRECT_RUN_CONFIG["epochs"], help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=DIRECT_RUN_CONFIG["imgsz"], help="Training image size.")
    parser.add_argument("--batch", type=int, default=DIRECT_RUN_CONFIG["batch"], help="Batch size.")
    parser.add_argument("--device", type=str, default=DIRECT_RUN_CONFIG["device"], help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=DIRECT_RUN_CONFIG["workers"], help="Dataloader workers.")
    parser.add_argument("--project", type=str, default=DIRECT_RUN_CONFIG["project"], help="Output project directory.")
    parser.add_argument(
        "--name",
        type=str,
        default=DIRECT_RUN_CONFIG["name"],
        help="Run name.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Expected input channels. Use >3 together with same-stem .npy sidecar arrays.",
    )
    parser.add_argument("--seed", type=int, default=DIRECT_RUN_CONFIG["seed"], help="Random seed.")
    parser.add_argument("--patience", type=int, default=DIRECT_RUN_CONFIG["patience"], help="Early stopping patience.")
    return parser.parse_args(argv)


def adapt_first_conv_to_channels(model, channels: int) -> None:
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")

    detect_model = model.model
    first_block = detect_model.model[0]
    first_conv = getattr(first_block, "conv", None)
    if first_conv is None:
        raise AttributeError("Unable to locate the first convolution layer on the YOLO backbone.")
    if first_conv.in_channels == channels:
        detect_model.channels = channels
        detect_model.yaml["channels"] = channels
        return

    import torch
    from torch import nn

    new_conv = nn.Conv2d(
        in_channels=channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
        device=first_conv.weight.device,
        dtype=first_conv.weight.dtype,
    )

    with torch.no_grad():
        new_conv.weight.zero_()
        shared_channels = min(first_conv.in_channels, channels)
        new_conv.weight[:, :shared_channels].copy_(first_conv.weight[:, :shared_channels])
        if channels > first_conv.in_channels:
            mean_weight = first_conv.weight.mean(dim=1, keepdim=True)
            repeat_count = channels - first_conv.in_channels
            new_conv.weight[:, first_conv.in_channels:].copy_(mean_weight.repeat(1, repeat_count, 1, 1))
        if first_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    first_block.conv = new_conv
    detect_model.channels = channels
    detect_model.yaml["channels"] = channels


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    yolo_config_dir = repo_root / "outputs" / "ultralytics_config"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)

    from ultralytics import YOLO

    print(f"[YOLO Train] data={args.data}")
    print(
        f"[YOLO Train] model={args.model}, channels={args.channels}, "
        f"epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={args.device}"
    )
    print(f"[YOLO Train] output={args.project}/{args.name}")

    model = YOLO(args.model)
    adapt_first_conv_to_channels(model, args.channels)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    if USE_DIRECT_RUN_CONFIG and len(sys.argv) == 1:
        sys.argv.extend(_build_direct_run_args())
    raise SystemExit(main())
