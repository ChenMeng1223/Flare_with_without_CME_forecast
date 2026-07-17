"""
指标跟踪器 - 记录和计算训练过程中的指标
"""
import numpy as np
from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import torch
import matplotlib.pyplot as plt

# 与脚本 `f_train_model.py` 中 `setup_logging(..., log_name="train_model", ...)` 对齐，
# 让指标跟踪相关日志也写入同一个日志文件。
logger = logging.getLogger("train_model")

class MetricsTracker:
    """
    跟踪和记录训练指标
    """

    def __init__(self, config: Dict[str, Any], save_dir: Optional[str] = None):
        """
        从配置字典初始化指标跟踪器

        Args:
            config: 训练配置字典
            save_dir: 保存目录，如果为None则从config中读取
        """
        # 获取日志配置
        logging_config = config.get('logging', {})

        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            log_dir = logging_config.get('log_dir', 'logs')
            self.save_dir = Path(log_dir) / "metrics"

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 配置参数
        self.track_averages = logging_config.get('track_averages', True)
        self.window_size = logging_config.get('window_size', 50)
        self.log_frequency = logging_config.get('log_frequency', 10)

        # 存储指标
        self.metrics = defaultdict(list)
        self.moving_averages = defaultdict(list)

        # 当前epoch的临时存储
        self.current_epoch_metrics = defaultdict(list)

        # 当前batch的临时存储
        self.current_batch_metrics = defaultdict(list)

        # 最佳指标跟踪
        self.best_metrics = {}

        # 配置文件
        self.config = config

        logger.info(f"MetricsTracker initialized. Save directory: {self.save_dir}")
        logger.info(f"Log frequency: {self.log_frequency} batches, Window size: {self.window_size}")

    def reset_epoch(self):
        """
        重置当前epoch的指标
        """
        self.current_epoch_metrics.clear()
        self.current_batch_metrics.clear()

    def update(self, metrics_dict: Dict[str, Union[float, torch.Tensor]],
               batch_size: int = 1, is_batch_metrics: bool = False):
        """
        更新指标

        Args:
            metrics_dict: 指标字典
            batch_size: 批次大小，用于加权平均
            is_batch_metrics: 是否为批次指标（用于频繁记录）
        """
        for name, value in metrics_dict.items():
            # 转换tensor为python数值
            if isinstance(value, torch.Tensor):
                value = value.item()

            if is_batch_metrics:
                self.current_batch_metrics[name].append((value, batch_size))
            else:
                self.current_epoch_metrics[name].append((value, batch_size))

    def get_batch_metrics(self) -> Dict[str, float]:
        """
        获取当前批次指标的平均值

        Returns:
            dict: 批次指标
        """
        return self._compute_weighted_averages(self.current_batch_metrics)

    def compute_epoch_metrics(self) -> Dict[str, float]:
        """
        计算当前epoch的平均指标

        Returns:
            dict: 平均指标
        """
        epoch_metrics = self._compute_weighted_averages(self.current_epoch_metrics)

        # 保存到历史记录
        for name, value in epoch_metrics.items():
            self.metrics[name].append(value)

            # 计算移动平均
            if self.track_averages:
                recent_values = self.metrics[name][-self.window_size:]
                moving_avg = sum(recent_values) / len(recent_values)
                self.moving_averages[name].append(moving_avg)

        # 记录最佳指标
        self._update_best_metrics(epoch_metrics)

        return epoch_metrics

    def _compute_weighted_averages(self, metrics_dict: Dict[str, List[tuple]]) -> Dict[str, float]:
        """
        计算加权平均值

        Args:
            metrics_dict: 指标字典

        Returns:
            dict: 加权平均指标
        """
        result = {}

        for name, values in metrics_dict.items():
            if not values:
                continue

            # 计算加权平均
            total_value = 0.0
            total_weight = 0

            for value, weight in values:
                total_value += value * weight
                total_weight += weight

            if total_weight > 0:
                result[name] = total_value / total_weight

        return result

    def _update_best_metrics(self, epoch_metrics: Dict[str, float]):
        """
        更新最佳指标记录

        Args:
            epoch_metrics: 当前epoch的指标
        """
        for name, value in epoch_metrics.items():
            if name not in self.best_metrics:
                self.best_metrics[name] = {
                    'value': value,
                    'epoch': len(self.metrics[name]) - 1
                }
            else:
                # 对于损失类指标，越小越好；对于准确率类指标，越大越好
                # 这里假设指标名包含'loss'的是损失类指标
                if 'loss' in name.lower():
                    if value < self.best_metrics[name]['value']:
                        self.best_metrics[name] = {
                            'value': value,
                            'epoch': len(self.metrics[name]) - 1
                        }
                else:
                    if value > self.best_metrics[name]['value']:
                        self.best_metrics[name] = {
                            'value': value,
                            'epoch': len(self.metrics[name]) - 1
                        }

    def get_metrics(self, metric_name: Optional[str] = None) -> Union[List[float], Dict[str, List[float]]]:
        """
        获取指标历史

        Args:
            metric_name: 指标名称，为None时返回所有指标

        Returns:
            指标历史
        """
        if metric_name:
            return self.metrics.get(metric_name, [])
        return dict(self.metrics)

    def get_moving_averages(self, metric_name: Optional[str] = None) -> Union[List[float], Dict[str, List[float]]]:
        """
        获取移动平均

        Args:
            metric_name: 指标名称，为None时返回所有指标

        Returns:
            移动平均历史
        """
        if metric_name:
            return self.moving_averages.get(metric_name, [])
        return dict(self.moving_averages)

    def get_best_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取最佳指标

        Returns:
            最佳指标信息
        """
        return self.best_metrics.copy()

    def get_latest_metrics(self) -> Dict[str, float]:
        """
        获取最新的指标值

        Returns:
            最新指标值
        """
        latest_metrics = {}
        for name, values in self.metrics.items():
            if values:
                latest_metrics[name] = values[-1]
        return latest_metrics

    def save_metrics(self, filename: str = "metrics.json"):
        """
        保存指标到文件

        Args:
            filename: 文件名
        """
        filepath = self.save_dir / filename

        save_data = {
            'config': self.config,
            'metrics': dict(self.metrics),
            'moving_averages': dict(self.moving_averages),
            'best_metrics': self.best_metrics
        }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        logger.info(f"Metrics saved to {filepath}")

        # 同时保存为CSV格式以便分析
        self.save_metrics_csv()

    def save_metrics_csv(self, filename: str = "metrics.csv"):
        """
        保存指标为CSV格式

        Args:
            filename: 文件名
        """
        try:
            import pandas as pd

            # 收集所有指标数据
            data = {'epoch': list(range(len(self.metrics.get('train_loss', []))))}

            for metric_name, values in self.metrics.items():
                if values:
                    data[metric_name] = values

            df = pd.DataFrame(data)
            csv_path = self.save_dir / filename
            df.to_csv(csv_path, index=False)

            logger.info(f"Metrics CSV saved to {csv_path}")
        except ImportError:
            logger.warning("Pandas not installed. Skipping CSV export.")

    def load_metrics(self, filename: str = "metrics.json") -> bool:
        """
        从文件加载指标

        Args:
            filename: 文件名

        Returns:
            bool: 是否加载成功
        """
        filepath = self.save_dir / filename

        if not filepath.exists():
            logger.warning(f"Metrics file not found: {filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)

            self.metrics = defaultdict(list, save_data.get('metrics', {}))
            self.moving_averages = defaultdict(list, save_data.get('moving_averages', {}))
            self.best_metrics = save_data.get('best_metrics', {})

            logger.info(f"Metrics loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return False

    def plot_metrics(self, metric_names: Optional[List[str]] = None,
                     save_path: Optional[str] = None, show_plot: bool = True):
        """
        绘制指标曲线

        Args:
            metric_names: 要绘制的指标名称列表
            save_path: 保存路径
            show_plot: 是否显示图表
        """
        try:
            if not metric_names:
                metric_names = [name for name in self.metrics.keys()
                              if not name.endswith('_loss')]
                # 添加损失指标
                loss_names = [name for name in self.metrics.keys()
                            if name.endswith('_loss')]
                metric_names = loss_names + metric_names

            fig, axes = plt.subplots(len(metric_names), 1,
                                    figsize=(12, 4 * len(metric_names)))
            if len(metric_names) == 1:
                axes = [axes]

            for idx, name in enumerate(metric_names):
                if name not in self.metrics:
                    logger.warning(f"Metric '{name}' not found")
                    continue

                values = self.metrics[name]
                if not values:
                    continue

                epochs = range(len(values))
                axes[idx].plot(epochs, values, label=name, linewidth=2,
                              color='blue', alpha=0.8)

                if self.track_averages and name in self.moving_averages:
                    ma_values = self.moving_averages[name]
                    ma_epochs = range(len(ma_values))
                    axes[idx].plot(ma_epochs, ma_values, label=f'{name} (MA)',
                                  linestyle='--', linewidth=1.5, color='red')

                # 标记最佳值
                if name in self.best_metrics:
                    best_epoch = self.best_metrics[name]['epoch']
                    best_value = self.best_metrics[name]['value']
                    axes[idx].scatter(best_epoch, best_value,
                                     color='green', s=100, zorder=5,
                                     label=f'Best: {best_value:.4f}')

                axes[idx].set_xlabel('Epoch', fontsize=12)
                axes[idx].set_ylabel(name.replace('_', ' ').title(), fontsize=12)
                axes[idx].set_title(f'{name.replace("_", " ").title()} over Epochs',
                                   fontsize=14, fontweight='bold')
                axes[idx].legend(fontsize=10)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_axisbelow(True)

            plt.suptitle('Training Metrics', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Metrics plot saved to {save_path}")

            if show_plot:
                plt.show()

            plt.close(fig)

        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot metrics.")
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")

    def summary(self, print_to_console: bool = True) -> str:
        """
        生成指标摘要

        Args:
            print_to_console: 是否打印到控制台

        Returns:
            str: 摘要字符串
        """
        lines = []
        lines.append("=" * 80)
        lines.append("METRICS SUMMARY")
        lines.append("=" * 80)

        for name, values in self.metrics.items():
            if not values:
                continue

            latest = values[-1]
            if name in self.best_metrics:
                best = self.best_metrics[name]['value']
                best_epoch = self.best_metrics[name]['epoch']
                lines.append(f"{name:<25} Latest: {latest:>8.6f} | Best: {best:>8.6f} (epoch {best_epoch+1})")
            else:
                lines.append(f"{name:<25} Latest: {latest:>8.6f}")

        lines.append("=" * 80)

        summary = "\n".join(lines)

        if print_to_console:
            logger.info("\n" + summary)

        return summary