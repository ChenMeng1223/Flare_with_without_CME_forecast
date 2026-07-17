#!/usr/bin/env python3
"""
批量绘制视向磁场图像脚本

遍历data/raw/downloaded/下所有事件的magnetogram文件夹，
读取FITS文件，绘制磁场图像，并保存到data/processed/下。
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import yaml

import os
import sys
from pathlib import Path
from typing import List, Optional
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
import yaml

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_magnetogram(fits_path: Path, output_path: Path) -> bool:
    """绘制单个磁场FITS文件"""
    try:
        from astropy.io import fits

        # 读取FITS文件
        with fits.open(fits_path) as hdul:
            # HMI磁场数据通常在第二个HDU中
            if len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
            elif hdul[0].data is not None:
                data = hdul[0].data
            else:
                logger.warning(f"FITS文件 {fits_path} 所有HDU数据都为空")
                return False

        if data is None:
            logger.warning(f"FITS文件 {fits_path} 数据为空")
            return False

        # 处理NaN和无穷大值
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # 归一化到[-2000, 2000]范围
        vmin, vmax = -2000, 2000
        data = np.clip(data, vmin, vmax)

        # 创建圆形mask（太阳圆盘）
        height, width = data.shape
        center_y, center_x = height // 2, width // 2
        radius = min(center_x, center_y) * 0.95  # 稍微小一点，避免边缘

        # 创建圆形mask
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

        # 创建背景全黑的图像
        output_image = np.zeros((height, width), dtype=np.uint8)

        # 归一化数据到0-255范围（黑白图像）
        data_normalized = ((data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        data_normalized = np.clip(data_normalized, 0, 255)

        # 只在圆形区域内应用数据
        output_image[mask] = data_normalized[mask]

        # 创建图像，只显示圆形区域
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')

        # 显示图像
        im = ax.imshow(output_image, cmap='gray', origin='lower')

        # 去掉坐标轴和边框
        ax.axis('off')
        ax.set_facecolor('black')

        # 设置显示范围为圆形区域
        ax.set_xlim(center_x - radius, center_x + radius)
        ax.set_ylim(center_y - radius, center_y + radius)

        # 保存图像
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black', transparent=False)
        plt.close()

        logger.info(f"成功绘制: {output_path}")
        return True

    except Exception as e:
        logger.error(f"绘制失败 {fits_path}: {e}")
        return False

def process_event_magnetograms(event_dir: Path, processed_base_dir: Path) -> int:
    """处理单个事件的磁场数据"""
    event_id = event_dir.name
    magnetogram_dir = event_dir / 'magnetogram'

    if not magnetogram_dir.exists():
        logger.warning(f"事件 {event_id} 没有magnetogram文件夹")
        return 0

    # 创建输出目录
    output_dir = processed_base_dir / event_id / 'magnetogram'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有FITS文件
    fits_files = list(magnetogram_dir.glob('*.fits'))
    if not fits_files:
        logger.warning(f"事件 {event_id} 的magnetogram文件夹为空")
        return 0

    logger.info(f"处理事件 {event_id}: {len(fits_files)} 个FITS文件")

    success_count = 0
    for fits_file in fits_files:
        output_file = output_dir / f"{fits_file.stem}.png"
        if plot_magnetogram(fits_file, output_file):
            success_count += 1

    logger.info(f"事件 {event_id} 处理完成: {success_count}/{len(fits_files)}")
    return success_count

def batch_plot_magnetograms(downloaded_dir: Path, processed_dir: Path) -> None:
    """批量绘制所有事件的磁场图像"""
    if not downloaded_dir.exists():
        logger.error(f"下载目录不存在: {downloaded_dir}")
        return

    # 查找所有事件文件夹
    event_dirs = [d for d in downloaded_dir.iterdir() if d.is_dir() and d.name.startswith('EVT_')]
    event_dirs.sort()

    if not event_dirs:
        logger.warning("没有找到事件文件夹")
        return

    logger.info(f"找到 {len(event_dirs)} 个事件文件夹")

    total_success = 0
    total_files = 0

    for event_dir in event_dirs:
        success = process_event_magnetograms(event_dir, processed_dir)
        total_success += success
        # 粗略估计文件数（每个事件可能不同）
        magnetogram_dir = event_dir / 'magnetogram'
        if magnetogram_dir.exists():
            fits_count = len(list(magnetogram_dir.glob('*.fits')))
            total_files += fits_count

    logger.info(f"批量处理完成: {total_success}/{total_files} 个文件成功绘制")

def main():
    parser = argparse.ArgumentParser(description='批量绘制视向磁场图像')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml',
                       help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置路径
    raw_data_dir = Path(config['data']['raw_data_dir'])
    downloaded_dir = raw_data_dir / 'downloaded'
    processed_dir = Path(config['data']['processed_data_dir'])

    logger.info("开始批量绘制磁场图像...")
    logger.info(f"下载目录: {downloaded_dir}")
    logger.info(f"处理目录: {processed_dir}")

    batch_plot_magnetograms(downloaded_dir, processed_dir)

if __name__ == "__main__":
    main()