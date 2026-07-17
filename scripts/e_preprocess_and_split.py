"""
数据预处理与划分脚本
功能：对HDF5数据集进行预处理、划分训练/验证/测试集，并创建数据集索引
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any

from utils.logging_utils import setup_logging


def _decode_hdf5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8').replace('\x00', '').strip()
    return str(value).strip()


def _collect_valid_event_rows(h5_file: h5py.File, logger: logging.Logger) -> List[Dict[str, Any]]:
    valid_rows: List[Dict[str, Any]] = []
    skipped_missing_group = 0
    skipped_incomplete_group = 0

    index_table = h5_file['index_table']
    events_group = h5_file['events']

    for row in index_table:
        if not bool(row['data_available']):
            continue

        event_id = _decode_hdf5_string(row['event_id'])
        if not event_id:
            continue

        if event_id not in events_group:
            skipped_missing_group += 1
            logger.warning(f"跳过索引脏记录：events/{event_id} 不存在")
            continue

        event_group = events_group[event_id]
        has_timestamps = 'timestamps' in event_group
        has_data_group = 'data' in event_group and len(event_group['data'].keys()) > 0
        num_frames = int(event_group.attrs.get('num_frames', 0) or 0)
        if (not has_timestamps) or (not has_data_group) or num_frames <= 0:
            skipped_incomplete_group += 1
            logger.warning(f"跳过不完整事件：{event_id} 缺少有效时间序列数据")
            continue

        valid_rows.append({
            'row': row,
            'event_id': event_id,
            'event_group': event_group,
            'num_frames': num_frames,
        })

    if skipped_missing_group or skipped_incomplete_group:
        logger.warning(
            "HDF5 清洗完成：跳过 %d 条缺失事件组索引，跳过 %d 条不完整事件记录",
            skipped_missing_group,
            skipped_incomplete_group,
        )

    return valid_rows


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='预处理和划分HDF5数据集')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--hdf5_path', type=str, default='data/Solar_Flares_CME_dataset.h5',
                        help='HDF5数据集路径')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--force', action='store_true',
                        help='强制重新处理')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    return parser.parse_args()


def validate_hdf5_structure(hdf5_path: str) -> Dict[str, Any]:
    """
    验证HDF5文件结构并返回统计信息

    Args:
        hdf5_path: HDF5文件路径

    Returns:
        数据集统计信息字典
    """
    logger = logging.getLogger(__name__)
    logger.info(f"验证HDF5文件结构: {hdf5_path}")

    stats = {
        'total_events': 0,
        'available_events': 0,
        'modality_availability': {},
        'label_distribution': {},
        'quality_stats': {}
    }

    with h5py.File(hdf5_path, 'r') as f:
        # ????
        if 'index_table' not in f:
            raise ValueError("HDF5???????")

        if 'events' not in f:
            raise ValueError("HDF5???????")

        valid_entries = _collect_valid_event_rows(f, logger)
        index_table = f['index_table'][:]
        stats['total_events'] = len(index_table)
        stats['available_events'] = len(valid_entries)

        # ??????
        labels = np.array([entry['row']['label'] for entry in valid_entries], dtype=np.uint8)
        unique_labels, counts = np.unique(labels, return_counts=True)
        stats['label_distribution'] = dict(zip(unique_labels.astype(str), counts))

        # ???????
        modalities = json.loads(f.attrs.get('modalities', '[]'))
        modality_counts = {modality: 0 for modality in modalities}

        # ????????
        sample_size = min(10, stats['available_events'])
        if len(valid_entries) > 0:
            sample_indices = np.random.choice(
                np.arange(len(valid_entries)),
                size=min(sample_size, len(valid_entries)),
                replace=False
            )

            for idx in sample_indices:
                event_group = valid_entries[int(idx)]['event_group']

                for modality in modalities:
                    if event_group.attrs.get(f'{modality}_available', False):
                        modality_counts[modality] += 1

                        # ??????
                        if f'data/{modality}/images' in event_group:
                            data = event_group[f'data/{modality}/images'][:]
                            nan_ratio = np.isnan(data).sum() / data.size

                            if modality not in stats['quality_stats']:
                                stats['quality_stats'][modality] = {
                                    'nan_ratios': [],
                                    'mean_values': [],
                                    'std_values': []
                                }

                            stats['quality_stats'][modality]['nan_ratios'].append(nan_ratio)
                            stats['quality_stats'][modality]['mean_values'].append(np.nanmean(data))
                            stats['quality_stats'][modality]['std_values'].append(np.nanstd(data))

        stats['modality_availability'] = modality_counts

        # ??????
        logger.info(f"HDF5??????:")
        logger.info(f"  ????: {stats['total_events']}")
        logger.info(f"  ?????: {stats['available_events']}")
        logger.info(f"  ????: {stats['label_distribution']}")

        for modality, count in modality_counts.items():
            percentage = count / sample_size * 100 if sample_size > 0 else 0
            logger.info(f"  ?? '{modality}' ???: {count}/{sample_size} ({percentage:.1f}%)")

    return stats


def create_event_statistics(hdf5_path: str, output_dir: Path) -> pd.DataFrame:
    """
    创建事件统计信息并保存

    Args:
        hdf5_path: HDF5文件路径
        output_dir: 输出目录

    Returns:
        事件统计DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("创建事件统计信息...")

    event_stats = []

    with h5py.File(hdf5_path, 'r') as f:
        valid_entries = _collect_valid_event_rows(f, logger)
        modalities = json.loads(f.attrs.get('modalities', '[]'))
        dtype_names = f['index_table'].dtype.names

        for entry in valid_entries:
            row = entry['row']
            event_id = entry['event_id']
            event_group = entry['event_group']

            stats = {
                'event_id': event_id,
                'label': int(row['label']),
                'flare_class': _decode_hdf5_string(row['flare_class']),
                'active_region': _decode_hdf5_string(row['active_region']),
                'start_time': _decode_hdf5_string(row['start_time']),
                'duration_minutes': float(row['duration']),
                'peak_flux': float(row['peak_flux']),
                'num_frames': int(entry['num_frames']),
            }
            if 'cme_associated' in dtype_names:
                stats['cme_associated'] = bool(row['cme_associated'])
            else:
                # ???????? cme_associated??? label ???1=??CME?2=???CME
                stats['cme_associated'] = int(row['label']) == 1
            # position_x/y/r ?? index_table ???????????????????
            for col in ('position_x', 'position_y', 'position_r'):
                if col in dtype_names:
                    stats[col] = float(row[col])
                else:
                    stats[col] = 0.0

            # ?????
            for modality in modalities:
                stats[f'{modality}_available'] = event_group.attrs.get(
                    f'{modality}_available', False
                )

                if stats[f'{modality}_available']:
                    # ????????
                    try:
                        data = event_group[f'data/{modality}/images'][:]
                        stats[f'{modality}_nan_ratio'] = np.isnan(data).sum() / data.size
                        stats[f'{modality}_mean'] = np.nanmean(data)
                        stats[f'{modality}_std'] = np.nanstd(data)
                    except Exception:
                        stats[f'{modality}_nan_ratio'] = np.nan
                        stats[f'{modality}_mean'] = np.nan
                        stats[f'{modality}_std'] = np.nan

            event_stats.append(stats)

    # 创建DataFrame
    df_stats = pd.DataFrame(event_stats)

    # 保存统计信息
    stats_path = output_dir / 'event_statistics.csv'
    df_stats.to_csv(stats_path, index=False)

    # 保存汇总统计
    summary = {
        'total_events': len(df_stats),
        'label_distribution': df_stats['label'].value_counts().to_dict(),
        'flare_class_distribution': df_stats['flare_class'].value_counts().to_dict(),
        'cme_distribution': df_stats['cme_associated'].value_counts().to_dict(),
        'modality_availability': {
            modality: df_stats[f'{modality}_available'].sum()
            for modality in modalities
        }
    }

    summary_path = output_dir / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else float(x))

    logger.info(f"事件统计信息已保存到: {stats_path}")
    logger.info(f"数据集摘要已保存到: {summary_path}")

    return df_stats


def _split_random(df_stats: pd.DataFrame, split_config: Dict) -> Tuple[List[str], List[str], List[str]]:
    """随机/分层抽样划分，保证各类别比例相似"""
    logger = logging.getLogger(__name__)
    event_ids = df_stats['event_id'].tolist()
    train_ratio = split_config['train_ratio']
    val_ratio = split_config['val_ratio']
    test_ratio = split_config['test_ratio']
    random_seed = split_config['random_seed']

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    stratify_col = df_stats['active_region'].astype(str).str.strip() + '_' + df_stats['label'].astype(str)

    def _do_split(arr, train_sz, strat=None):
        try:
            return train_test_split(arr, train_size=train_sz, random_state=random_seed, stratify=strat)
        except ValueError:
            logger.warning("分层抽样失败（可能某层样本过少），改用随机划分")
            return train_test_split(arr, train_size=train_sz, random_state=random_seed)

    train_ids, remaining_ids = _do_split(event_ids, train_ratio, stratify_col)
    remaining_df = df_stats[df_stats['event_id'].isin(remaining_ids)]
    remaining_stratify = remaining_df['active_region'].astype(str).str.strip() + '_' + remaining_df['label'].astype(str)
    val_ratio_adj = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = _do_split(remaining_ids, val_ratio_adj, remaining_stratify)
    return train_ids, val_ids, test_ids


def _split_temporal(df_stats: pd.DataFrame, split_config: Dict) -> Tuple[List[str], List[str], List[str]]:
    """按时间先后划分：训练集=最早、验证集=中间、测试集=最新，降低数据泄露风险"""
    logger = logging.getLogger(__name__)
    df = df_stats.sort_values('start_time').reset_index(drop=True)
    n = len(df)
    train_ratio = split_config['train_ratio']
    val_ratio = split_config['val_ratio']
    test_ratio = split_config['test_ratio']
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    train_ids = df.iloc[:n_train]['event_id'].tolist()
    val_ids = df.iloc[n_train:n_train + n_val]['event_id'].tolist()
    test_ids = df.iloc[n_train + n_val:]['event_id'].tolist()

    logger.info(f"时间划分: 训练 {n_train}(最早), 验证 {n_val}(中间), 测试 {n_test}(最新)")
    return train_ids, val_ids, test_ids


def create_data_splits(df_stats: pd.DataFrame, split_config: Dict,
                       output_dir: Path, split_data_dir: Path = None) -> Dict[str, List[str]]:
    """
    创建数据划分

    Args:
        df_stats: 事件统计DataFrame（含 start_time 用于时间划分）
        split_config: 划分配置，含 split_method: "random" | "temporal"
        output_dir: 输出目录（兼容参数，实际写入 split_data_dir）
        split_data_dir: 划分结果输出目录，默认 data/split_data

    Returns:
        划分字典 {split_name: [event_ids]}
    """
    logger = logging.getLogger(__name__)
    split_method = split_config.get('split_method', 'temporal')
    logger.info(f"创建数据划分（方法: {split_method}）...")

    splits_dir = Path(split_data_dir) if split_data_dir else Path('data/split_data')
    splits_dir.mkdir(parents=True, exist_ok=True)

    if split_method == 'temporal':
        train_ids, val_ids, test_ids = _split_temporal(df_stats, split_config)
    else:
        train_ids, val_ids, test_ids = _split_random(df_stats, split_config)

    splits = {
        'train': sorted(train_ids),
        'val': sorted(val_ids),
        'test': sorted(test_ids)
    }

    # 保存划分文件
    for split_name, split_event_ids in splits.items():
        split_path = splits_dir / f'{split_name}_events.txt'
        with open(split_path, 'w') as f:
            for event_id in split_event_ids:
                f.write(f"{event_id}\n")

        # 计算划分统计
        split_df = df_stats[df_stats['event_id'].isin(split_event_ids)]
        split_stats = {
            'num_events': len(split_event_ids),
            'label_distribution': split_df['label'].value_counts().to_dict(),
            'flare_class_distribution': split_df['flare_class'].value_counts().to_dict(),
            'cme_distribution': split_df['cme_associated'].value_counts().to_dict(),
            'avg_duration': split_df['duration_minutes'].mean(),
            'avg_peak_flux': split_df['peak_flux'].mean()
        }

        stats_path = splits_dir / f'{split_name}_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(split_stats, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else float(x))

        logger.info(f"{split_name}集: {len(split_event_ids)}个事件")
        logger.info(f"  标签分布: {split_stats['label_distribution']}")

    return splits


def create_dataset_config(hdf5_path: str, data_config: Dict,
                          splits: Dict[str, List[str]],
                          output_dir: Path) -> Dict:
    """
    创建数据集配置文件

    Args:
        hdf5_path: HDF5文件路径
        data_config: 数据配置
        splits: 数据划分
        output_dir: 输出目录

    Returns:
        数据集配置字典
    """
    logger = logging.getLogger(__name__)
    logger.info("创建数据集配置文件...")

    # 从HDF5文件读取全局属性
    with h5py.File(hdf5_path, 'r') as f:
        dataset_config = {
            'name': f.attrs.get('dataset_name', 'Solar Flare Multimodal Dataset'),
            'version': f.attrs.get('version', '1.0'),
            'creation_date': datetime.now().isoformat(),
            'preprocessing_date': datetime.now().isoformat(),
            'hdf5_path': str(Path(hdf5_path).absolute()),
            'modalities': json.loads(f.attrs.get('modalities', '[]')),
            'time_config': json.loads(f.attrs.get('time_config', '{}')),
            'data_config': data_config,
            'splits': {
                'train': {
                    'num_events': len(splits['train']),
                    'event_ids': splits['train']
                },
                'val': {
                    'num_events': len(splits['val']),
                    'event_ids': splits['val']
                },
                'test': {
                    'num_events': len(splits['test']),
                    'event_ids': splits['test']
                }
            },
            'statistics': {
                'total_events': sum(len(ids) for ids in splits.values()),
                'split_ratios': {
                    'train': len(splits['train']) / sum(len(ids) for ids in splits.values()),
                    'val': len(splits['val']) / sum(len(ids) for ids in splits.values()),
                    'test': len(splits['test']) / sum(len(ids) for ids in splits.values())
                }
            }
        }

    # 保存数据集配置
    config_path = output_dir / 'dataset_config.json'
    with open(config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)

    # 同时保存为YAML格式
    config_yaml_path = output_dir / 'dataset_config.yaml'
    with open(config_yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    logger.info(f"数据集配置已保存到: {config_path}")

    return dataset_config


def create_sliding_window_index(hdf5_path: str, splits: Dict[str, List[str]],
                                split_data_dir: Path, config: Dict) -> None:
    """
    创建滑动窗口索引

    Args:
        hdf5_path: HDF5文件路径
        splits: 数据划分
        split_data_dir: 划分结果输出目录（与 create_data_splits 一致）
        config: 配置字典
    """
    logger = logging.getLogger(__name__)
    logger.info("创建滑动窗口索引...")

    # 获取配置参数：sequence_length 在 data 级，time 下无此字段
    sequence_length = config.get('sequence_length') or config.get('time', {}).get('sequence_length', 9)
    stride = config.get('stride') or config.get('preprocessing', {}).get('stride', 1)
    modalities = list(config.get('modalities', {}).keys())

    window_indices = {
        'train': [],
        'val': [],
        'test': []
    }

    with h5py.File(hdf5_path, 'r') as f:
        for split_name, event_ids in splits.items():
            logger.info(f"处理{split_name}集的滑动窗口...")

            for event_id in event_ids:
                try:
                    event_path = f'events/{event_id}'
                    if event_path not in f:
                        continue

                    event_group = f[event_path]
                    num_frames = event_group.attrs.get('num_frames', 0)

                    if num_frames < sequence_length:
                        logger.warning(f"事件 {event_id} 帧数不足: {num_frames} < {sequence_length}")
                        continue

                    # 检查模态可用性
                    available_modalities = []
                    for modality in modalities:
                        if event_group.attrs.get(f'{modality}_available', False):
                            available_modalities.append(modality)

                    if len(available_modalities) == 0:
                        logger.warning(f"事件 {event_id} 没有可用模态")
                        continue

                    # 生成滑动窗口索引
                    for start_idx in range(0, num_frames - sequence_length + 1, stride):
                        end_idx = start_idx + sequence_length

                        window_info = {
                            'event_id': event_id,
                            'start_idx': int(start_idx),
                            'end_idx': int(end_idx),
                            'window_id': f"{event_id}_{start_idx}_{end_idx}",
                            'available_modalities': available_modalities,
                            'num_modalities': len(available_modalities),
                            'split': split_name
                        }

                        # 获取窗口时间戳
                        timestamps = event_group['timestamps'][start_idx:end_idx]
                        window_info['start_time'] = timestamps[0].decode('utf-8')
                        window_info['end_time'] = timestamps[-1].decode('utf-8')

                        window_indices[split_name].append(window_info)

                except Exception as e:
                    logger.error(f"处理事件 {event_id} 时出错: {e}")
                    continue

            logger.info(f"{split_name}集: {len(window_indices[split_name])}个窗口")

    # 保存窗口索引
    for split_name, windows in window_indices.items():
        if windows:
            # 保存为CSV
            df_windows = pd.DataFrame(windows)
            csv_path = split_data_dir / f'window_index_{split_name}.csv'
            df_windows.to_csv(csv_path, index=False)

            # 保存为Parquet（更高效）
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(df_windows)
                pq_path = split_data_dir / f'window_index_{split_name}.parquet'
                pq.write_table(table, pq_path)
                logger.info(f"{split_name}窗口索引已保存为Parquet: {pq_path}")
            except ImportError:
                logger.warning("未安装pyarrow，跳过Parquet保存")

            # 保存统计信息
            stats = {
                'num_windows': len(windows),
                'events_covered': len(set(w['event_id'] for w in windows)),
                'avg_modalities_per_window': np.mean([w['num_modalities'] for w in windows]),
                'window_size': sequence_length,
                'stride': stride
            }

            stats_path = split_data_dir / f'window_stats_{split_name}.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

    # 保存总窗口索引
    all_windows = []
    for split_name, windows in window_indices.items():
        for window in windows:
            all_windows.append(window)

    if all_windows:
        df_all = pd.DataFrame(all_windows)
        all_path = split_data_dir / 'window_index_all.csv'
        df_all.to_csv(all_path, index=False)

        logger.info(f"总窗口数: {len(all_windows)}")
        logger.info(f"窗口索引已保存到: {all_path}")


def create_preprocessing_report(output_dir: Path, stats: Dict,
                                df_stats: pd.DataFrame,
                                dataset_config: Dict) -> None:
    """
    创建预处理报告

    Args:
        output_dir: 输出目录
        stats: HDF5验证统计
        df_stats: 事件统计DataFrame
        dataset_config: 数据集配置
    """
    def convert_numpy_types(obj):
        """递归转换NumPy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        elif hasattr(obj, 'item'):  # NumPy标量
            return obj.item()
        else:
            return obj

    logger = logging.getLogger(__name__)
    logger.info("创建预处理报告...")

    report = {
        'preprocessing_info': {
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(output_dir.absolute()),
            'python_version': sys.version,
            'platform': sys.platform
        },
        'dataset_overview': {
            'total_events': stats['total_events'],
            'available_events': stats['available_events'],
            'availability_rate': stats['available_events'] / stats['total_events'] if stats['total_events'] > 0 else 0,
            'label_distribution': stats['label_distribution'],
            'modality_availability': stats['modality_availability']
        },
        'quality_assessment': {
            'avg_nan_ratios': {},
            'avg_values': {},
            'value_ranges': {}
        },
        'splits_summary': dataset_config.get('statistics', {}),
        'recommendations': []
    }

    # 添加质量评估
    for modality, quality_stats in stats['quality_stats'].items():
        report['quality_assessment']['avg_nan_ratios'][modality] = {
            'mean': np.mean(quality_stats['nan_ratios']),
            'std': np.std(quality_stats['nan_ratios']),
            'max': np.max(quality_stats['nan_ratios'])
        }

        report['quality_assessment']['avg_values'][modality] = {
            'mean': np.mean(quality_stats['mean_values']),
            'std': np.std(quality_stats['mean_values'])
        }

        report['quality_assessment']['value_ranges'][modality] = {
            'min': np.min(quality_stats['mean_values']),
            'max': np.max(quality_stats['mean_values'])
        }

    # 添加建议
    recommendations = []

    # 检查数据平衡性
    label_dist = report['dataset_overview']['label_distribution']
    if len(label_dist) > 0:
        min_count = min(label_dist.values())
        max_count = max(label_dist.values())
        if max_count / min_count > 3:  # 类别不平衡超过3倍
            recommendations.append({
                'type': 'class_imbalance',
                'severity': 'warning',
                'message': f'检测到类别不平衡，建议使用加权损失或过采样',
                'details': label_dist
            })

    # 检查缺失数据
    for modality, nan_stats in report['quality_assessment']['avg_nan_ratios'].items():
        if nan_stats['mean'] > 0.1:  # 缺失率超过10%
            recommendations.append({
                'type': 'missing_data',
                'severity': 'warning',
                'message': f'模态 {modality} 缺失率较高: {nan_stats["mean"]:.1%}',
                'details': nan_stats
            })

    report['recommendations'] = recommendations

    # 保存报告
    report_path = output_dir / 'preprocessing_report.json'
    with open(report_path, 'w') as f:
        json.dump(convert_numpy_types(report), f, indent=2, ensure_ascii=False)

    # 生成Markdown格式报告
    md_report = f"""# 太阳耀斑数据集预处理报告

## 预处理信息
- **时间戳**: {report['preprocessing_info']['timestamp']}
- **输出目录**: {report['preprocessing_info']['output_directory']}
- **Python版本**: {report['preprocessing_info']['python_version']}
- **平台**: {report['preprocessing_info']['platform']}

## 数据集概览
- **总事件数**: {report['dataset_overview']['total_events']}
- **可用事件数**: {report['dataset_overview']['available_events']}
- **可用率**: {report['dataset_overview']['availability_rate']:.1%}

### 标签分布
"""

    for label, count in report['dataset_overview']['label_distribution'].items():
        percentage = count / report['dataset_overview']['total_events'] * 100
        md_report += f"- 类别 {label}: {count} ({percentage:.1f}%)\n"

    md_report += f"""
### 模态可用性
"""

    for modality, count in report['dataset_overview']['modality_availability'].items():
        sample_size = 10  # 假设采样大小为10
        percentage = count / sample_size * 100 if sample_size > 0 else 0
        md_report += f"- {modality}: {count}/{sample_size} ({percentage:.1f}%)\n"

    md_report += f"""
## 数据划分
- **训练集**: {report['splits_summary']['split_ratios']['train']:.1%} ({dataset_config['splits']['train']['num_events']}个事件)
- **验证集**: {report['splits_summary']['split_ratios']['val']:.1%} ({dataset_config['splits']['val']['num_events']}个事件)
- **测试集**: {report['splits_summary']['split_ratios']['test']:.1%} ({dataset_config['splits']['test']['num_events']}个事件)

## 质量评估
### 缺失数据统计
| 模态 | 平均缺失率 | 标准差 | 最大缺失率 |
|------|------------|--------|------------|
"""

    for modality, nan_stats in report['quality_assessment']['avg_nan_ratios'].items():
        md_report += f"| {modality} | {nan_stats['mean']:.2%} | {nan_stats['std']:.2%} | {nan_stats['max']:.2%} |\n"

    md_report += f"""
## 建议
"""

    if not report['recommendations']:
        md_report += "- 无重大问题，数据集质量良好\n"
    else:
        for rec in report['recommendations']:
            md_report += f"### {rec['type']} ({rec['severity']})\n"
            md_report += f"- {rec['message']}\n"
            if 'details' in rec:
                md_report += f"- 详情: {rec['details']}\n"

    # 保存Markdown报告
    md_path = output_dir / 'preprocessing_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)

    logger.info(f"预处理报告已保存到: {report_path}")
    logger.info(f"Markdown报告已保存到: {md_path}")


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(args.log_dir, 'preprocess_data', debug=args.debug)
    logger = logging.getLogger(__name__)

    # 检查HDF5文件是否存在
    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        logger.error(f"HDF5文件不存在: {hdf5_path}")
        return

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已经处理过
    done_file = output_dir / '.preprocessing_done'
    if done_file.exists() and not args.force:
        logger.info(f"检测到已完成的预处理，使用 --force 参数强制重新处理")
        return

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)['data']

    logger.info(f"开始预处理数据集: {hdf5_path}")

    try:
        # 步骤1: 验证HDF5文件结构
        stats = validate_hdf5_structure(str(hdf5_path))

        # 步骤2: 创建事件统计
        df_stats = create_event_statistics(str(hdf5_path), output_dir)

        # 步骤3: 创建数据划分（保存到 data/split_data）
        split_data_dir = Path(config.get('split_data_dir', 'data/split_data'))
        splits = create_data_splits(df_stats, config['splits'], output_dir, split_data_dir)

        # 步骤4: 创建数据集配置
        dataset_config = create_dataset_config(
            str(hdf5_path), config, splits, output_dir
        )

        # 步骤5: 创建滑动窗口索引（保存到 data/split_data）
        create_sliding_window_index(str(hdf5_path), splits, split_data_dir, config)

        # 步骤6: 创建预处理报告
        create_preprocessing_report(output_dir, stats, df_stats, dataset_config)

        # 标记处理完成
        with open(done_file, 'w') as f:
            f.write(datetime.now().isoformat())

        logger.info(f"数据预处理完成！输出目录: {output_dir}")

    except Exception as e:
        logger.error(f"预处理过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
