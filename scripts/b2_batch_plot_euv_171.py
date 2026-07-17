#!/usr/bin/env python3
"""
批量绘制 EUV 171Å 波段完整图像脚本

调用统一脚本 batch_plot_euv，仅处理 171Å 波段。
基于 IDL 代码 plot_aia193_fullmap 的 Python 实现。
"""

import argparse
import logging
from pathlib import Path

import yaml

from b2_batch_plot_euv import batch_plot_euv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='批量绘制 EUV 171Å 波段图像')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml', help='配置文件路径')
    parser.add_argument('--save-npy', action='store_true', help='同时保存标准化数据为 .npy 文件')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已存在的输出文件')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_data_dir = Path(config['data']['raw_data_dir'])
    downloaded_dir = raw_data_dir / 'downloaded'
    processed_dir = Path(config['data']['processed_data_dir'])

    logger.info("开始批量绘制 EUV 171Å 图像...")
    batch_plot_euv(
        downloaded_dir, processed_dir, config,
        bands=['euv_171'],
        save_npy=args.save_npy,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()