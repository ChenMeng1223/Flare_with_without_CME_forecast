"""
数据加载器模块 - 创建PyTorch DataLoader
"""
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import h5py

from data.hdf5_reader import HDF5DatasetReader
from data.dataset import SolarFlareDataset
from data.transforms import create_transforms  # 如果需要的话

logger = logging.getLogger(__name__)


def create_data_loaders(
        hdf5_path: str,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        modalities: Optional[List[str]] = None,
        sequence_length: int = 48,
        stride: int = 6,
        target_size: Tuple[int, int] = (512, 512),
        handle_missing: str = 'interpolate',
        fill_value: float = 0.0,
        use_augmentation: bool = True,
        train_transform: Optional[Any] = None,
        val_transform: Optional[Any] = None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
        drop_last: bool = False,
        max_activities: int = 5
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器

    Args:
        hdf5_path: HDF5文件路径
        batch_size: 批次大小
        num_workers: 数据加载工作线程数
        pin_memory: 是否固定内存（用于CUDA）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        modalities: 使用的模态列表
        sequence_length: 序列长度
        stride: 滑动步长
        target_size: 目标图像尺寸
        handle_missing: 缺失数据处理方式
        fill_value: 填充值
        use_augmentation: 是否使用数据增强
        train_transform: 训练数据变换
        val_transform: 验证数据变换
        shuffle_train: 是否打乱训练集
        shuffle_val: 是否打乱验证集
        shuffle_test: 是否打乱测试集
        drop_last: 是否丢弃最后一个不完整的批次
        max_activities: 每条样本中允许的最大活动数量，用于标签/ bbox 填充

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 验证比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-3:  # 放宽容差
        raise ValueError(f"数据集比例总和应为1.0，但得到: {total_ratio}")

    logger.info(f"创建数据加载器: {hdf5_path}")
    logger.info(f"批次大小: {batch_size}, 工作线程: {num_workers}")
    logger.info(f"数据集比例: 训练({train_ratio}), 验证({val_ratio}), 测试({test_ratio})")

    # 创建HDF5读取器
    try:
        reader = HDF5DatasetReader(hdf5_path)
    except Exception as e:
        logger.error(f"无法加载HDF5文件: {hdf5_path}, 错误: {e}")
        raise

    # 获取所有可用事件ID
    all_event_ids = reader.get_event_ids(available_only=True)

    if not all_event_ids:
        raise ValueError(f"HDF5文件中没有可用的事件: {hdf5_path}")

    logger.info(f"总事件数: {len(all_event_ids)}")

    # 设置随机种子以确保可重复性
    rng = np.random.RandomState(random_seed)

    # 打乱事件ID
    shuffled_event_ids = all_event_ids.copy()
    rng.shuffle(shuffled_event_ids)

    # 计算分割点
    n_total = len(shuffled_event_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    # 分配事件ID到不同集合
    train_event_ids = shuffled_event_ids[:n_train]
    val_event_ids = shuffled_event_ids[n_train:n_train + n_val]
    test_event_ids = shuffled_event_ids[n_train + n_val:]

    logger.info(f"训练集事件: {len(train_event_ids)}")
    logger.info(f"验证集事件: {len(val_event_ids)}")
    logger.info(f"测试集事件: {len(test_event_ids)}")

    # 验证 sequence_length 是否合理
    # 检查实际事件的帧数，确保 sequence_length 不会导致空数据集
    sample_event_ids = all_event_ids[:min(10, len(all_event_ids))]  # 采样前10个事件
    min_frames = float('inf')
    max_frames = 0
    for event_id in sample_event_ids:
        try:
            with h5py.File(hdf5_path, 'r') as f:
                event_path = f'events/{event_id}'
                if event_path in f:
                    event_group = f[event_path]
                    num_frames = event_group.attrs.get('num_frames', 0)
                    
                    # 如果属性中没有num_frames（可能是旧版本文件），从时间戳数组长度获取
                    if num_frames == 0 and 'timestamps' in event_group:
                        num_frames = len(event_group['timestamps'])
                    
                    min_frames = min(min_frames, num_frames)
                    max_frames = max(max_frames, num_frames)
        except Exception:
            continue
    
    if min_frames != float('inf'):
        logger.info(f"采样事件帧数范围: {min_frames} - {max_frames} 帧")
        if sequence_length > min_frames:
            logger.warning(
                f"警告: sequence_length ({sequence_length}) 大于某些事件的最小帧数 ({min_frames})。"
                f"这可能导致某些事件无法创建滑动窗口，从而产生空数据集。"
                f"建议将 sequence_length 设置为小于等于最小帧数的值。"
            )
        else:
            logger.info(f"sequence_length ({sequence_length}) 设置合理，小于最小帧数 ({min_frames})")

    # 创建数据集
    train_dataset = SolarFlareDataset(
        reader=reader,
        event_ids=train_event_ids,
        modalities=modalities,
        sequence_length=sequence_length,
        stride=stride,
        target_size=target_size,
        handle_missing=handle_missing,
        fill_value=fill_value,
        transform=train_transform,
        max_activities=max_activities
    )

    val_dataset = SolarFlareDataset(
        reader=reader,
        event_ids=val_event_ids,
        modalities=modalities,
        sequence_length=sequence_length,
        stride=stride,
        target_size=target_size,
        handle_missing=handle_missing,
        fill_value=fill_value,
        transform=val_transform,
        max_activities=max_activities
    )

    # 创建测试集（如果有测试数据）
    if n_test > 0:
        test_dataset = SolarFlareDataset(
            reader=reader,
            event_ids=test_event_ids,
            modalities=modalities,
            sequence_length=sequence_length,
            stride=stride,
            target_size=target_size,
            handle_missing=handle_missing,
            fill_value=fill_value,
            transform=val_transform,  # 测试集使用与验证集相同的变换
            max_activities=max_activities
        )
    else:
        test_dataset = None
        logger.warning("测试集大小为0，不创建测试集数据加载器")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # 训练集不丢弃最后一个不完整批次，确保小数据集也能训练
        collate_fn=collate_fn if hasattr(train_dataset, 'collate_fn') else default_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # 验证集不需要drop_last
        collate_fn=collate_fn if hasattr(val_dataset, 'collate_fn') else default_collate_fn
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,  # 测试集不需要drop_last
            collate_fn=collate_fn if hasattr(test_dataset, 'collate_fn') else default_collate_fn
        )
    else:
        test_loader = None

    # 打印统计信息
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset) if test_dataset else 0
    
    logger.info(f"训练集窗口数: {train_size}")
    logger.info(f"验证集窗口数: {val_size}")
    if test_dataset:
        logger.info(f"测试集窗口数: {test_size}")

    # 检查数据集是否为空
    if train_size == 0:
        raise ValueError(f"训练集为空！可能原因：所有事件的帧数都小于sequence_length({sequence_length})")
    if val_size == 0:
        raise ValueError(f"验证集为空！可能原因：所有事件的帧数都小于sequence_length({sequence_length})")
    if test_size == 0 and test_dataset is not None:
        logger.warning(f"测试集为空！可能原因：所有事件的帧数都小于sequence_length({sequence_length})")

    return train_loader, val_loader, test_loader


def default_collate_fn(batch: List[Dict]) -> Dict:
    """
    默认的批处理函数

    Args:
        batch: 批次样本列表

    Returns:
        批处理后的字典
    """
    if not batch:
        return {}

    # 获取第一个样本作为参考
    first_sample = batch[0]

    collated_batch = {}

    # 处理每个键
    for key in first_sample.keys():
        if key == 'metadata':
            # 元数据保持为列表
            collated_batch[key] = [sample[key] for sample in batch]

        elif key == 'auxiliary' and first_sample[key] is None:
            # 辅助数据可能为None
            collated_batch[key] = None

        elif isinstance(first_sample[key], torch.Tensor):
            # 张量数据堆叠
            collated_batch[key] = torch.stack([sample[key] for sample in batch])

        elif isinstance(first_sample[key], dict):
            # 字典数据（如多模态数据）
            collated_dict = {}
            for subkey in first_sample[key].keys():
                if isinstance(first_sample[key][subkey], torch.Tensor):
                    collated_dict[subkey] = torch.stack(
                        [sample[key][subkey] for sample in batch]
                    )
                else:
                    collated_dict[subkey] = [sample[key][subkey] for sample in batch]
            collated_batch[key] = collated_dict

        else:
            # 其他类型保持为列表
            collated_batch[key] = [sample[key] for sample in batch]

    return collated_batch


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义批处理函数，处理多模态数据

    Args:
        batch: 批次样本列表

    Returns:
        批处理后的字典
    """
    return default_collate_fn(batch)


def get_dataset_statistics(
        hdf5_path: str,
        modalities: Optional[List[str]] = None,
        sample_fraction: float = 0.1
) -> Dict[str, Any]:
    """
    获取数据集统计信息（均值、标准差等）

    Args:
        hdf5_path: HDF5文件路径
        modalities: 模态列表
        sample_fraction: 采样比例（用于加速计算）

    Returns:
        统计信息字典
    """
    logger.info(f"计算数据集统计信息: {hdf5_path}")

    reader = HDF5DatasetReader(hdf5_path)

    # 获取所有事件ID
    all_event_ids = reader.get_event_ids(available_only=True)

    if not modalities:
        modalities = reader.modalities

    # 随机采样部分事件
    n_sample = max(1, int(len(all_event_ids) * sample_fraction))
    sample_event_ids = np.random.choice(all_event_ids, size=n_sample, replace=False)

    statistics = {}

    for modality in modalities:
        logger.info(f"计算模态 {modality} 的统计信息...")

        all_values = []

        for event_id in sample_event_ids:
            try:
                # 获取事件数据
                event_data = reader.get_event_data(event_id, [modality])

                if event_data[modality] is not None:
                    # 展平数据并添加到列表
                    modality_data = event_data[modality]
                    valid_values = modality_data[~np.isnan(modality_data)]

                    if len(valid_values) > 0:
                        all_values.append(valid_values)
            except Exception as e:
                logger.warning(f"处理事件 {event_id} 时出错: {e}")

        if all_values:
            # 合并所有值
            combined_values = np.concatenate(all_values)

            statistics[modality] = {
                'mean': float(np.mean(combined_values)),
                'std': float(np.std(combined_values)),
                'min': float(np.min(combined_values)),
                'max': float(np.max(combined_values)),
                'median': float(np.median(combined_values)),
                'num_samples': len(combined_values)
            }

            logger.info(
                f"{modality}: 均值={statistics[modality]['mean']:.4f}, "
                f"标准差={statistics[modality]['std']:.4f}"
            )
        else:
            statistics[modality] = {
                'mean': 0.0,
                'std': 1.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'num_samples': 0
            }
            logger.warning(f"模态 {modality} 没有有效数据")

    return statistics


def create_subset_data_loaders(
        hdf5_path: str,
        event_ids: List[str],
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        modalities: Optional[List[str]] = None,
        sequence_length: int = 48,
        stride: int = 6,
        target_size: Tuple[int, int] = (512, 512),
        shuffle: bool = False,
        drop_last: bool = False
) -> DataLoader:
    """
    为指定事件ID创建数据加载器

    Args:
        hdf5_path: HDF5文件路径
        event_ids: 事件ID列表
        batch_size: 批次大小
        num_workers: 数据加载工作线程数
        pin_memory: 是否固定内存
        modalities: 模态列表
        sequence_length: 序列长度
        stride: 滑动步长
        target_size: 目标图像尺寸
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后一个不完整的批次

    Returns:
        数据加载器
    """
    logger.info(f"为 {len(event_ids)} 个事件创建数据加载器")

    reader = HDF5DatasetReader(hdf5_path)

    dataset = SolarFlareDataset(
        reader=reader,
        event_ids=event_ids,
        modalities=modalities,
        sequence_length=sequence_length,
        stride=stride,
        target_size=target_size
    )

    # 检查数据集是否为空
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError(
            f"数据集为空！无法创建数据加载器。"
            f"可能原因：所有事件的帧数都小于sequence_length({sequence_length})"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

    logger.info(f"数据集大小: {dataset_size} 个窗口")

    return loader

