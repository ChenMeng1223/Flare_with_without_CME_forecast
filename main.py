"""
主程序入口
"""
import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='太阳耀斑预测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s train --config configs/training_config.yaml
  %(prog)s predict --model outputs/best_model.pth --data data/test.h5
  %(prog)s evaluate --model outputs/best_model.pth --dataset data/solar_flares.h5
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, required=True,
                              help='训练配置文件')
    train_parser.add_argument('--data', type=str,
                              default='data/solar_flares_dataset.h5',
                              help='HDF5数据集路径')
    train_parser.add_argument('--output', type=str, default='outputs',
                              help='输出目录')

    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='运行预测')
    predict_parser.add_argument('--model', type=str, required=True,
                                help='模型文件路径')
    predict_parser.add_argument('--data', type=str, required=True,
                                help='输入数据路径（HDF5文件或目录）')
    predict_parser.add_argument('--event', type=str,
                                help='事件ID（如果输入是HDF5文件）')
    predict_parser.add_argument('--output', type=str, default='predictions',
                                help='输出目录')
    predict_parser.add_argument('--config', type=str,
                                default='configs/inference_config.yaml',
                                help='推理配置文件')

    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--model', type=str, required=True,
                             help='模型文件路径')
    eval_parser.add_argument('--dataset', type=str, required=True,
                             help='HDF5数据集路径')
    eval_parser.add_argument('--split', type=str, default='test',
                             help='数据划分（train/val/test）')
    eval_parser.add_argument('--output', type=str, default='evaluation',
                             help='输出目录')

    # 数据准备命令
    data_parser = subparsers.add_parser('prepare-data', help='准备数据')
    data_parser.add_argument('--config', type=str, required=True,
                             help='数据配置文件')
    data_parser.add_argument('--events', type=str, required=True,
                             help='事件元数据CSV文件')
    data_parser.add_argument('--output', type=str,
                             default='data/solar_flares_dataset.h5',
                             help='输出HDF5文件路径')

    args = parser.parse_args()

    if args.command == 'train':
        from scripts.f_train_model import main as train_main
        train_args = argparse.Namespace(
            hdf5_path=args.data,
            output_dir=args.output,
            train_config=args.config
        )
        train_main()

    elif args.command == 'predict':
        from scripts.g_inference_pipeline import main as predict_main
        predict_args = argparse.Namespace(
            model_path=args.model,
            data_path=args.data,
            event_id=args.event,
            output_dir=args.output,
            config_path=args.config
        )
        predict_main()

    elif args.command == 'evaluate':
        from scripts.evaluate_model import main as eval_main
        eval_args = argparse.Namespace(
            model_path=args.model,
            hdf5_path=args.dataset,
            split=args.split,
            output_dir=args.output
        )
        eval_main()

    elif args.command == 'prepare-data':
        from scripts.d_create_hdf5_dataset import main as data_main
        data_args = argparse.Namespace(
            config=args.config,
            events_csv=args.events,
            output=args.output
        )
        data_main()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()