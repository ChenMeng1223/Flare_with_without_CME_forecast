"""
评估模型脚本
"""
import argparse
import sys
from pathlib import Path
import logging
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from data import create_data_loaders
from models.multimodal_transformer import MultimodalTransformer
from training.trainer import SolarFlareTrainer
from utils.config_utils import load_config

logger = logging.getLogger(__name__)


def main(args=None):
    """主函数"""
    parser = argparse.ArgumentParser(description='评估太阳耀斑预测模型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help='HDF5数据集路径')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='数据划分')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                        help='输出目录')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # 设置日志
    log_file = Path(args.output_dir) / 'evaluation.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    logger.info("开始评估模型")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"数据集: {args.hdf5_path}")
    logger.info(f"数据划分: {args.split}")
    logger.info(f"输出目录: {args.output_dir}")

    try:
        # 加载模型
        logger.info("加载模型...")
        checkpoint = torch.load(args.model_path, map_location='cpu')

        # 从检查点获取配置
        config = checkpoint.get('config', {})
        if not config:
            logger.warning("检查点中没有配置，使用默认配置")
            config = load_config('configs/training_config.yaml')

        # 创建模型
        model_config = config.get('model', {})
        if not model_config:
            # 如果没有模型配置，使用默认配置
            model_config = {
                'image_encoder': {
                    'in_channels': 3,
                    'hidden_dim': 128,
                    'num_heads': 8,
                    'num_layers': 4,
                    'dropout': 0.1
                },
                'physics_encoder': {
                    'input_dim': 10,
                    'hidden_dim': 64,
                    'num_layers': 2,
                    'dropout': 0.1
                },
                'temporal_encoder': {
                    'input_dim': 192,
                    'hidden_dim': 256,
                    'num_heads': 8,
                    'num_layers': 4,
                    'dropout': 0.1
                },
                'num_classes': 3,
                'bbox_output': True,
                'time_prediction': True
            }

        model = MultimodalTransformer(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 创建训练器
        trainer_config = config.get('training', {})
        trainer = SolarFlareTrainer(
            model=model,
            config=trainer_config,
            device=device
        )

        # 根据划分创建数据加载器
        logger.info("创建数据加载器...")

        if args.split == 'all':
            # 使用所有数据
            train_loader, val_loader, test_loader = create_data_loaders(
                hdf5_path=args.hdf5_path,
                batch_size=16,
                num_workers=4,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0
            )
            loaders = [('train', train_loader)]
        else:
            # 标准划分
            train_loader, val_loader, test_loader = create_data_loaders(
                hdf5_path=args.hdf5_path,
                batch_size=16,
                num_workers=4,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )

            if args.split == 'train':
                loaders = [('train', train_loader)]
            elif args.split == 'val':
                loaders = [('val', val_loader)]
            elif args.split == 'test':
                if test_loader:
                    loaders = [('test', test_loader)]
                else:
                    logger.warning("测试集为空，使用验证集")
                    loaders = [('val', val_loader)]

        # 评估每个数据集
        evaluation_results = {}

        for split_name, loader in loaders:
            logger.info(f"评估 {split_name} 集...")

            metrics = trainer.validate(loader)

            evaluation_results[split_name] = {
                'num_samples': len(loader.dataset),
                'metrics': metrics
            }

            logger.info(f"{split_name} 集结果:")
            logger.info(f"  样本数: {len(loader.dataset)}")
            logger.info(f"  损失: {metrics['loss']:.4f}")
            logger.info(f"  准确率: {metrics['accuracy']:.4f}")
            logger.info(f"  F1分数: {metrics['f1']:.4f}")
            logger.info(f"  精确率: {metrics['precision']:.4f}")
            logger.info(f"  召回率: {metrics['recall']:.4f}")

        # 保存评估结果
        output_path = Path(args.output_dir)

        # 保存为JSON
        json_path = output_path / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            # 转换张量为可序列化的格式
            def convert_tensors(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_tensors(item) for item in obj]
                else:
                    return obj

            json.dump(convert_tensors(evaluation_results), f, indent=2)

        logger.info(f"评估结果保存到: {json_path}")

        # 保存为CSV
        try:
            import pandas as pd

            rows = []
            for split_name, result in evaluation_results.items():
                row = {
                    'split': split_name,
                    'num_samples': result['num_samples'],
                    'loss': result['metrics']['loss'],
                    'accuracy': result['metrics']['accuracy'],
                    'f1': result['metrics']['f1'],
                    'precision': result['metrics']['precision'],
                    'recall': result['metrics']['recall']
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            csv_path = output_path / 'evaluation_results.csv'
            df.to_csv(csv_path, index=False)

            logger.info(f"CSV结果保存到: {csv_path}")

        except ImportError:
            logger.warning("pandas未安装，跳过CSV导出")

        # 生成评估报告
        report_path = output_path / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("太阳耀斑预测模型评估报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"模型: {args.model_path}\n")
            f.write(f"数据集: {args.hdf5_path}\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for split_name, result in evaluation_results.items():
                f.write(f"{split_name.upper()} 集:\n")
                f.write(f"  样本数: {result['num_samples']}\n")
                f.write(f"  损失: {result['metrics']['loss']:.4f}\n")
                f.write(f"  准确率: {result['metrics']['accuracy']:.4f}\n")
                f.write(f"  F1分数: {result['metrics']['f1']:.4f}\n")
                f.write(f"  精确率: {result['metrics']['precision']:.4f}\n")
                f.write(f"  召回率: {result['metrics']['recall']:.4f}\n\n")

            f.write("=" * 60 + "\n")

        logger.info(f"评估报告保存到: {report_path}")

        logger.info("评估完成!")

    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    from datetime import datetime

    sys.exit(main())