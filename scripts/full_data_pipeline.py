#!/usr/bin/env python3
"""
SDO数据处理完整流程脚本

执行完整的SDO数据处理流程：
1. 根据events.xlsx下载各波段数据
2. 批量将FITS文件转换为图像
3. 从图像创建HDF5数据集

使用方法：
python scripts/full_data_pipeline.py --config configs/data_config.yaml
"""
import sys
import os
from pathlib import Path
import argparse
import logging
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_logging


def run_download_step(config_path: str, events_path: str, max_workers: int, log_dir: str):
    """执行下载步骤"""
    logger = logging.getLogger(__name__)
    logger.info("开始执行下载步骤...")

    from scripts.a_download_sdo_data import main as download_main

    # 构造命令行参数
    import sys
    original_argv = sys.argv.copy()

    sys.argv = [
        'a_download_sdo_data.py',
        '--config', config_path,
        '--events', events_path,
        '--max_workers', str(max_workers),
        '--log_dir', log_dir
    ]

    try:
        download_main()
        logger.info("下载步骤完成")
        return True
    except Exception as e:
        logger.error(f"下载步骤失败: {e}")
        return False
    finally:
        sys.argv = original_argv


def run_image_processing_step(config_path: str, events_path: str, max_workers: int, log_dir: str):
    """执行图像处理步骤"""
    logger = logging.getLogger(__name__)
    logger.info("开始执行图像处理步骤...")

    from scripts.batch_process_images import main as process_main

    # 构造命令行参数
    import sys
    original_argv = sys.argv.copy()

    sys.argv = [
        'batch_process_images.py',
        '--config', config_path,
        '--events', events_path,
        '--max_workers', str(max_workers),
        '--log_dir', log_dir
    ]

    try:
        process_main()
        logger.info("图像处理步骤完成")
        return True
    except Exception as e:
        logger.error(f"图像处理步骤失败: {e}")
        return False
    finally:
        sys.argv = original_argv


def run_hdf5_creation_step(config_path: str, events_path: str, output_path: str, log_dir: str):
    """执行HDF5创建步骤"""
    logger = logging.getLogger(__name__)
    logger.info("开始执行HDF5创建步骤...")

    from scripts.d_create_hdf5_dataset import main as hdf5_main

    # 构造命令行参数
    import sys
    original_argv = sys.argv.copy()

    sys.argv = [
        'd_create_hdf5_dataset.py',
        '--config', config_path,
        '--events_csv', events_path,
        '--output', output_path,
        '--log_dir', log_dir
    ]

    try:
        hdf5_main()
        logger.info("HDF5创建步骤完成")
        return True
    except Exception as e:
        logger.error(f"HDF5创建步骤失败: {e}")
        return False
    finally:
        sys.argv = original_argv


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='完整的SDO数据处理流程')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--output', type=str, default='data/solar_flares_dataset.h5',
                        help='HDF5输出文件路径')
    parser.add_argument('--max_workers', type=int, default=2,
                        help='最大并发数')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--skip_download', action='store_true',
                        help='跳过下载步骤（如果已经下载过）')
    parser.add_argument('--skip_images', action='store_true',
                        help='跳过图像处理步骤（如果已经处理过）')
    parser.add_argument('--skip_hdf5', action='store_true',
                        help='跳过HDF5创建步骤')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(log_dir=args.log_dir, log_level='INFO')
    logger = logging.getLogger(__name__)

    # 加载配置以获取事件文件路径
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    events_file = config['data']['events_file']

    logger.info("开始完整的SDO数据处理流程")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"事件文件: {events_file}")
    logger.info(f"输出文件: {args.output}")

    success = True

    # 步骤1: 下载数据
    if not args.skip_download:
        if not run_download_step(args.config, events_file, args.max_workers, args.log_dir):
            success = False
    else:
        logger.info("跳过下载步骤")

    # 步骤2: 处理图像
    if success and not args.skip_images:
        if not run_image_processing_step(args.config, events_file, args.max_workers, args.log_dir):
            success = False
    else:
        logger.info("跳过图像处理步骤")

    # 步骤3: 创建HDF5数据集
    if success and not args.skip_hdf5:
        if not run_hdf5_creation_step(args.config, events_file, args.output, args.log_dir):
            success = False
    else:
        logger.info("跳过HDF5创建步骤")

    if success:
        logger.info("完整的SDO数据处理流程成功完成！")
        logger.info(f"HDF5数据集已保存到: {args.output}")
    else:
        logger.error("数据处理流程失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()