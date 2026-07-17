"""
训练模型脚本
"""
import sys
import os
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate
import logging
from data.hdf5_reader import HDF5DatasetReader
from data.dataset import SolarFlareDataset
from models.multimodal_transformer import MultimodalTransformer
from training.trainer import SolarFlareTrainer
from utils.logging_utils import setup_logging
from utils.config_utils import load_config


USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    'data_config': 'configs/data_config_stage2_yolo11_5modal_256.yaml',
    'model_config': 'configs/model_config.yaml',
    'train_config': 'configs/training_config_stage2_yolo11_5modal.yaml',
    'hdf5_path': 'data/Solar_Flares_CME_dataset.h5',
    'output_dir': 'outputs/stage2_yolo11_5modal_256',
    'log_dir': 'logs',
    'debug': False,
}


def _get_enabled_modalities(modalities_cfg: Dict) -> Dict:
    return {
        name: cfg
        for name, cfg in modalities_cfg.items()
        if bool(cfg.get('enabled', True))
    }


def _is_cuda_device_supported() -> bool:
    """检查当前 PyTorch 是否支持已检测到的 CUDA 设备架构。"""
    if not torch.cuda.is_available():
        return False

    try:
        major, minor = torch.cuda.get_device_capability()
        target_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        return target_arch in supported_arches
    except Exception:
        return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练太阳耀斑预测模型')
    parser.add_argument('--data_config', type=str,
                        default='configs/data_config.yaml',
                        help='数据配置文件路径')
    parser.add_argument('--model_config', type=str,
                        default='configs/model_config.yaml',
                        help='模型配置文件路径')
    parser.add_argument('--train_config', type=str,
                        default='configs/training_config.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--hdf5_path', type=str,
                        default='data/solar_flares_dataset.h5',
                        help='HDF5数据集路径')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--resume', type=str, default='',
                        help='从指定 checkpoint 恢复训练')
    parser.add_argument('--resume_latest', action='store_true',
                        help='从 checkpoint.save_dir 下最新 checkpoint 恢复训练')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    return parser.parse_args()


def _build_direct_run_args() -> List[str]:
    cfg = dict(DIRECT_RUN_CONFIG)
    args: List[str] = []
    for key in ['data_config', 'model_config', 'train_config', 'hdf5_path', 'output_dir', 'log_dir']:
        value = cfg.get(key)
        if value not in (None, ''):
            args.extend([f'--{key}', str(value)])
    if cfg.get('debug', False):
        args.append('--debug')
    return args


def custom_collate(batch):
    """自定义 collate：metadata 保持为列表，避免变长 missing_modalities 导致报错"""
    meta = [b.pop('metadata') for b in batch]
    out = default_collate(batch)
    out['metadata'] = meta
    return out


def _get_window_slot_counts(dataset: SolarFlareDataset, window_info: Dict) -> Counter:
    """统计窗口内真实类别 1/2 的槽位数量。"""
    labels = dataset.get_window_labels(window_info)
    return Counter(int(label) for label in labels if int(label) in (1, 2))


def _resolve_hdf5_open_kwargs(data_config: Dict) -> Dict:
    runtime_cfg = data_config.get('runtime', {})
    open_kwargs: Dict[str, object] = {}
    locking = runtime_cfg.get('hdf5_locking')
    if locking is not None:
        open_kwargs['locking'] = bool(locking)
    return open_kwargs


def _build_model_config(model_config: Dict, data_config: Dict, train_config: Dict) -> Dict:
    enabled_modalities = _get_enabled_modalities(data_config['modalities'])
    first_mod = next(iter(enabled_modalities.values()), {})
    target_size = tuple(first_mod.get('resolution', model_config.get('input_size', [256, 256])))

    model_config_merged = dict(model_config)
    model_config_merged['modalities'] = enabled_modalities
    model_config_merged['max_activities'] = data_config.get('max_activities', model_config.get('max_activities', 5))
    model_config_merged['input_size'] = list(target_size)
    model_config_merged['sequence_length'] = data_config.get('sequence_length', model_config.get('sequence_length', 9))
    model_config_merged['max_sequence_length'] = model_config_merged['sequence_length']

    spatial_cfg = dict(model_config.get('spatial', {}))
    model_config_merged['spatial'] = spatial_cfg
    if 'encoder_downsample_factor' in model_config:
        spatial_cfg.setdefault('encoder_downsample_factor', model_config['encoder_downsample_factor'])

    model_config_merged.setdefault('stage2', {})
    if train_config.get('two_stage_schedule', {}).get('enabled', False):
        model_config_merged['stage2']['roi_source_mix'] = train_config['two_stage_schedule'].get('roi_source_mix', {'predicted': 1.0})

    return model_config_merged


def _default_training_history() -> Dict[str, List[float]]:
    return {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1': [], 'val_f1': [],
        'train_f1_all': [], 'val_f1_all': [],
        'train_tss': [], 'val_tss': [],
        'train_tss_all': [], 'val_tss_all': [],
        'train_rss': [], 'val_rss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_iou': [], 'val_iou': [],
        'train_composite_score': [], 'val_composite_score': []
    }


def _load_history_if_exists(history_path: Path) -> Dict[str, List[float]]:
    if not history_path.exists():
        return _default_training_history()
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        base = _default_training_history()
        for key in base:
            if isinstance(data.get(key), list):
                base[key] = data[key]
        return base
    except Exception:
        return _default_training_history()


def _resolve_resume_checkpoint(args, checkpoint_manager) -> Optional[str]:
    if args.resume:
        return args.resume
    if not args.resume_latest:
        return None
    return checkpoint_manager.get_latest_checkpoint()


def _build_weighted_sampler_and_class_weights(
        train_dataset: SolarFlareDataset,
        balance_config: Dict,
        logger: logging.Logger) -> Tuple[WeightedRandomSampler, List[float]]:
    """基于训练集窗口槽位数量构建加权采样器和自动类别权重。"""
    window_slot_counts = [_get_window_slot_counts(train_dataset, w) for w in train_dataset.windows]
    class_counts = Counter()
    for slot_count in window_slot_counts:
        class_counts.update(slot_count)

    if not class_counts:
        raise ValueError("训练集窗口中未找到类别 1/2 槽位，无法构建平衡采样器")

    sampler_cfg = balance_config.get('sampler', {})
    sampling_power = float(sampler_cfg.get('power', 1.0))
    min_weight = float(sampler_cfg.get('min_weight', 1e-6))

    class_weight_map = {
        label: (1.0 / class_counts[label]) ** sampling_power
        for label in (1, 2) if class_counts.get(label, 0) > 0
    }

    sample_weights = []
    for slot_count in window_slot_counts:
        if slot_count:
            weight = sum(
                slot_count.get(label, 0) * class_weight_map.get(label, 0.0)
                for label in (1, 2)
            )
            sample_weights.append(max(weight, min_weight))
        else:
            sample_weights.append(min_weight)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    total_count = sum(class_counts.values())
    num_classes = len(class_counts)
    auto_class_weights = []
    for label in (1, 2):
        count = class_counts.get(label, 0)
        if count > 0:
            auto_class_weights.append(total_count / (num_classes * count))
        else:
            auto_class_weights.append(1.0)

    logger.info(
        "训练集槽位类别统计(1/2): %s, 自动类别权重[1,2]=%s, 采样幂次=%.3f",
        dict(class_counts),
        [round(x, 6) for x in auto_class_weights],
        sampling_power,
    )

    return sampler, auto_class_weights


def create_data_loaders(hdf5_path: str, data_config: Dict,
                        batch_size: int, num_workers: int = 4,
                        balance_config: Dict = None):
    """创建数据加载器"""
    logger = logging.getLogger(__name__)
    reader = HDF5DatasetReader(hdf5_path, open_kwargs=_resolve_hdf5_open_kwargs(data_config))
    balance_config = balance_config or {}

    # 从预处理的划分文件中读取事件ID（与 data_config.split_data_dir 对齐）
    splits_dir = Path(data_config.get('split_data_dir', 'data/split_data'))

    def load_split_events(split_name: str) -> List[str]:
        """从划分文件中加载事件ID"""
        split_file = splits_dir / f'{split_name}_events.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            # 如果划分文件不存在，回退到动态划分
            logger.warning(f"未找到划分文件 {split_file}，使用动态划分")
            from sklearn.model_selection import train_test_split
            all_event_ids = reader.get_event_ids(available_only=True)
            split_config = data_config['splits']
            train_ratio = split_config['train_ratio']
            val_ratio = split_config['val_ratio']
            test_ratio = split_config['test_ratio']
            random_seed = split_config['random_seed']

            events_df = reader.index_df[reader.index_df['event_id'].isin(all_event_ids)]

            if split_name == 'train':
                train_events, _ = train_test_split(
                    all_event_ids,
                    test_size=1 - train_ratio,
                    random_state=random_seed,
                    stratify=events_df.set_index('event_id').loc[all_event_ids]['label']
                )
                return sorted(train_events)
            elif split_name == 'val':
                _, temp_events = train_test_split(
                    all_event_ids,
                    test_size=1 - train_ratio,
                    random_state=random_seed,
                    stratify=events_df.set_index('event_id').loc[all_event_ids]['label']
                )
                val_events, _ = train_test_split(
                    temp_events,
                    test_size=test_ratio / (val_ratio + test_ratio),
                    random_state=random_seed,
                    stratify=events_df.set_index('event_id').loc[temp_events]['label']
                )
                return sorted(val_events)
            else:  # test
                _, temp_events = train_test_split(
                    all_event_ids,
                    test_size=1 - train_ratio,
                    random_state=random_seed,
                    stratify=events_df.set_index('event_id').loc[all_event_ids]['label']
                )
                _, test_events = train_test_split(
                    temp_events,
                    test_size=test_ratio / (val_ratio + test_ratio),
                    random_state=random_seed,
                    stratify=events_df.set_index('event_id').loc[temp_events]['label']
                )
                return sorted(test_events)

    train_events = load_split_events('train')
    val_events = load_split_events('val')
    test_events = load_split_events('test')

    logger.info(f"数据划分: 训练集 {len(train_events)}, "
                f"验证集 {len(val_events)}, 测试集 {len(test_events)}")

    # 创建数据集：与 data_config 和 HDF5 一致
    enabled_modalities = _get_enabled_modalities(data_config['modalities'])
    modalities_list = list(enabled_modalities.keys())
    seq_len = data_config.get('sequence_length', 9)
    # target_size 从首模态 resolution 获取，与 HDF5 图像一致
    first_mod = next(iter(enabled_modalities.values()), {})
    target_size = tuple(first_mod.get('resolution', [256, 256]))
    stride = data_config.get('stride', 1)  # 滑动步长，sequence_length=9 时用 1 增加窗口数

    dataset_config = {
        'modalities': modalities_list,
        'sequence_length': seq_len,
        'target_size': target_size,
        'stride': stride,
        'max_activities': data_config.get('max_activities', 5),
        'config': {
            'max_activities': data_config.get('max_activities', 5),
            'proposal_cache_path': data_config.get('proposal_cache_path'),
        },
    }

    train_dataset = SolarFlareDataset(
        reader, train_events, **dataset_config
    )
    val_dataset = SolarFlareDataset(
        reader, val_events, **dataset_config
    )
    test_dataset = SolarFlareDataset(
        reader, test_events, **dataset_config
    )

    sampler, auto_class_weights = _build_weighted_sampler_and_class_weights(
        train_dataset, balance_config, logger
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    return train_loader, val_loader, test_loader, auto_class_weights


# 计算模型参数量
def count_parameters(model: nn.Module) -> None:
    # 总参数
    total_params = sum(p.numel() for p in model.parameters())
    # 可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 格式化输出（转成 M，更直观：1M = 100万）
    print(f"Total params: {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f} M)")


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(args.log_dir, 'train_model', debug=args.debug)
    logger = logging.getLogger(__name__)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    data_config = load_config(args.data_config)['data']  # 在configs/data_config.yaml中配置
    enabled_modalities = _get_enabled_modalities(data_config['modalities'])
    if not enabled_modalities:
        raise ValueError("data_config.modalities 中没有任何 enabled=true 的模态，无法训练")
    data_config = dict(data_config)
    data_config['modalities'] = enabled_modalities
    model_config = load_config(args.model_config)['model']  # 在configs/model_config.yaml中配置
    train_config = load_config(args.train_config)['training']  # 在configs/training_config.yaml中配置

    first_mod = next(iter(data_config['modalities'].values()), {})
    target_size = tuple(first_mod.get('resolution', model_config.get('input_size', [256, 256])))
    logger.info("当前目标分辨率: %s, 序列长度: %s", target_size, data_config.get('sequence_length', model_config.get('sequence_length', 9)))

    # 设置设备
    cuda_supported = _is_cuda_device_supported()
    device = torch.device('cuda' if cuda_supported else 'cpu')
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available() and not cuda_supported:
        try:
            major, minor = torch.cuda.get_device_capability()
            logger.warning(
                "检测到当前 GPU 架构 sm_%s%s 不受当前 PyTorch 支持，已自动回退到 CPU 训练，避免出现 NaN。",
                major,
                minor,
            )
        except Exception:
            logger.warning("检测到当前 GPU 不受当前 PyTorch 支持，已自动回退到 CPU 训练，避免出现 NaN。")

    # 创建模型：优先同步 data_config 中的分辨率、时序长度、模态和空间下采样配置
    model_config_merged = _build_model_config(model_config, data_config, train_config)
    logger.info("创建模型...")
    model = MultimodalTransformer(model_config_merged)    # 创建模型，model_config_merged中包含了data_config和model_config中的配置
    print("模型加载完成！")
    # 计算模型参数量
    count_parameters(model)

    # 创建数据加载器（优先使用 data_config 中的 hdf5_path）
    hdf5_path = data_config.get('hdf5_path', args.hdf5_path)
    logger.info("创建数据加载器...")
    balance_config = train_config.get('balance', {})
    train_loader, val_loader, test_loader, auto_class_weights = create_data_loaders(
        hdf5_path=hdf5_path,
        data_config=data_config,
        batch_size=train_config['batch_size'],
        num_workers=data_config['preprocessing'].get('num_workers', 4),
        balance_config=balance_config,
    )
    if balance_config.get('auto_class_weights', True):
        train_config['class_weights'] = [1.0] + auto_class_weights
        logger.info("自动更新训练类别权重为: %s", train_config['class_weights'])
    print("数据加载器创建完成！")

    # 创建训练器
    logger.info("创建训练器...")
    trainer_config = dict(train_config)
    trainer_config['model'] = model_config_merged
    trainer = SolarFlareTrainer(model, trainer_config, device)
    print("训练器创建完成！")

    resume_checkpoint = _resolve_resume_checkpoint(args, trainer.checkpoint_manager)
    history = _load_history_if_exists(output_dir / 'training_history.json')
    start_epoch = 0
    if resume_checkpoint:
        logger.info("准备从 checkpoint 恢复训练: %s", resume_checkpoint)
        checkpoint = trainer.checkpoint_manager.load_checkpoint(
            resume_checkpoint,
            trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            device=device,
        )
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        additional_info = checkpoint.get('additional_info', {}) or {}
        best_metrics = additional_info.get('best_metrics')
        if isinstance(best_metrics, dict) and best_metrics:
            trainer.best_metrics = best_metrics
        else:
            val_metrics = additional_info.get('val_metrics', {}) or {}
            if val_metrics:
                trainer.best_metrics = {
                    'epoch': int(checkpoint.get('epoch', start_epoch - 1)),
                    'val_loss': float(val_metrics.get('loss', 0.0)),
                    'val_accuracy': float(val_metrics.get('accuracy', 0.0)),
                    'val_f1': float(val_metrics.get('f1', 0.0)),
                    'val_f1_all': float(val_metrics.get('f1_all', 0.0)),
                    'val_composite_score': float(val_metrics.get('composite_score', checkpoint.get('metric_value', 0.0) or 0.0)),
                }
        if trainer.best_metrics:
            trainer.early_stopping.best_score = float(trainer.best_metrics.get('val_composite_score', checkpoint.get('metric_value', 0.0) or 0.0))
            trainer.early_stopping.best_epoch = int(trainer.best_metrics.get('epoch', start_epoch - 1))
            trainer.early_stopping.counter = max(
                0,
                start_epoch - trainer.early_stopping.best_epoch - 1,
            )
        logger.info("已恢复到 epoch=%d，下一个训练 epoch=%d", start_epoch, start_epoch + 1)

    # 训练模型
    logger.info("开始训练...")
    history = trainer.fit(train_loader, val_loader, history=history, start_epoch=start_epoch)

    # 保存训练历史
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    logger.info(f"训练历史已保存到: {history_path}")
    logger.info("训练完成！")


if __name__ == '__main__':
    if USE_DIRECT_RUN_CONFIG and len(sys.argv) == 1:
        sys.argv.extend(_build_direct_run_args())
    main()
