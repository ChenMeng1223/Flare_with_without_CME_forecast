"""
早停机制 - 防止模型过拟合
"""
import numpy as np
import logging
from typing import Optional, Dict, Any

# 与脚本 `f_train_model.py` 中 `setup_logging(..., log_name="train_model", ...)` 对齐，
# 让早停相关日志写入同一个日志文件。
logger = logging.getLogger("train_model")

class EarlyStopping:
    """
    早停机制实现
    """

    def __init__(self, config: Dict[str, Any]):
        """
        从配置字典初始化早停机制

        Args:
            config: 早停配置字典
        """
        self.patience = config.get('patience', 20)
        self.min_delta = config.get('min_delta', 0.001)
        self.restore_best_weights = config.get('restore_best_weights', True)
        self.mode = config.get('mode', 'min')  # 默认为最小化指标
        self.verbose = config.get('verbose', True)

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.best_state_dict = None

        if self.mode not in ['min', 'max']:
            raise ValueError(f"mode should be 'min' or 'max', got {self.mode}")

        logger.info(f"EarlyStopping initialized with patience={self.patience}, "
                   f"min_delta={self.min_delta}, mode={self.mode}")

    def __call__(self, current_score: float, epoch: int, model_state_dict: Optional[Dict] = None) -> bool:
        """
        检查是否需要早停

        Args:
            current_score: 当前指标值
            epoch: 当前epoch
            model_state_dict: 当前模型状态字典，用于恢复最佳权重

        Returns:
            bool: 是否需要早停
        """
        if self.best_score is None:
            # 第一次调用，保存为最佳
            self.best_score = current_score
            self.best_epoch = epoch
            if self.restore_best_weights and model_state_dict is not None:
                self.best_state_dict = model_state_dict.copy()
            return False

        if self.mode == 'min':
            # 指标越小越好
            improvement = self.best_score - current_score
            is_better = improvement > self.min_delta
        else:  # mode == 'max'
            # 指标越大越好
            improvement = current_score - self.best_score
            is_better = improvement > self.min_delta

        if is_better:
            # 有改善，重置计数器
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights and model_state_dict is not None:
                self.best_state_dict = model_state_dict.copy()
            if self.verbose:
                improvement_str = f"{improvement:.6f}"
                logger.info(f"Improvement detected: {improvement_str}. "
                           f"Best score updated: {current_score:.6f} (epoch {epoch+1})")
        else:
            # 没有改善，计数器加1
            self.counter += 1
            if self.verbose:
                logger.info(f"No improvement for {self.counter} epoch(s). "
                           f"Best: {self.best_score:.6f} (epoch {self.best_epoch+1}), "
                           f"Current: {current_score:.6f}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")

        return self.early_stop

    def should_stop(self) -> bool:
        """
        返回是否应该停止训练

        Returns:
            bool: 是否应该停止
        """
        return self.early_stop

    def reset(self):
        """
        重置早停状态
        """
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.best_state_dict = None
        logger.info("EarlyStopping reset")

    def restore_best_model(self, model):
        """
        恢复最佳模型权重

        Args:
            model: 模型实例
        """
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info(f"Restored best model weights from epoch {self.best_epoch+1}")

    def get_best_info(self) -> tuple:
        """
        获取最佳信息

        Returns:
            tuple: (最佳分数, 最佳epoch)
        """
        return self.best_score, self.best_epoch

    def get_progress(self) -> dict:
        """
        获取当前进度信息

        Returns:
            dict: 进度信息
        """
        return {
            'counter': self.counter,
            'patience': self.patience,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop,
            'remaining_patience': self.patience - self.counter
        }