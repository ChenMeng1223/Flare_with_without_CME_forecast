"""
检查点管理器 - 用于保存和加载模型训练状态
"""
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# 与脚本 `f_train_model.py` 中 `setup_logging(..., log_name="train_model", ...)` 对齐，
# 让 checkpoint 的保存/最佳指标信息写入同一个日志文件。
logger = logging.getLogger("train_model")

class CheckpointManager:
    """
    管理模型检查点的保存和加载
    """

    def __init__(self,
                 config: Dict[str, Any],
                 model_name: str = "solar_flare_cme_model",
                 save_dir: Optional[str] = None):
        """
        初始化检查点管理器

        Args:
            config: 训练配置字典
            model_name: 模型名称，用于生成文件名
            save_dir: 保存目录，如果为None则从config中读取
        """
        self.config = config
        self.model_name = model_name

        # 从config中读取检查点配置（支持传入完整训练config或只传checkpoint子配置）
        checkpoint_config = config.get('checkpoint', config if isinstance(config, dict) else {})
        self.save_best_only = checkpoint_config.get('save_best_only', True)
        self.max_checkpoints = checkpoint_config.get('max_checkpoints', 10)
        self.save_frequency = checkpoint_config.get('save_frequency', 5)
        self.save_every_epoch_start = checkpoint_config.get('save_every_epoch_start', None)

        # 设置保存目录（优先级：函数参数 > checkpoint.save_dir > checkpoint.checkpoint_dir > logging.log_dir/checkpoints > outputs/checkpoints）
        if save_dir:
            self.save_dir = Path(save_dir)
        elif checkpoint_config.get('save_dir'):
            self.save_dir = Path(checkpoint_config.get('save_dir'))
        elif checkpoint_config.get('checkpoint_dir'):
            self.save_dir = Path(checkpoint_config.get('checkpoint_dir'))
        else:
            log_dir = None
            if isinstance(config, dict):
                log_dir = config.get('logging', {}).get('log_dir')
            if log_dir:
                self.save_dir = Path(log_dir) / 'checkpoints'
            else:
                self.save_dir = Path('outputs/checkpoints')

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 跟踪最佳指标
        self.best_metric = None
        self.best_checkpoint_path = None

        # 保存的检查点列表
        self.checkpoints = []

        logger.info(f"CheckpointManager initialized. Save directory: {self.save_dir}")
        logger.info(
            f"Save frequency: {self.save_frequency}, "
            f"Save every epoch start: {self.save_every_epoch_start}, "
            f"Max checkpoints: {self.max_checkpoints}"
        )

    def should_save_checkpoint(self, epoch: int, is_best: bool = False) -> bool:
        """
        判断是否应该保存检查点

        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型

        Returns:
            bool: 是否应该保存
        """
        if is_best:
            return True
        elif self.save_best_only:
            return False
        elif self.save_every_epoch_start is not None and (epoch + 1) >= self.save_every_epoch_start:
            return True
        else:
            return (epoch + 1) % self.save_frequency == 0

    def save_checkpoint(self,
                        epoch: int,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                        metric_value: Optional[float] = None,
                        is_best: bool = False,
                        best_tag: Optional[str] = None,
                        additional_info: Optional[Dict[str, Any]] = None,
                        filename_suffix: Optional[str] = None) -> Optional[str]:
        """
        保存检查点

        Args:
            epoch: 当前epoch
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            metric_value: 指标值
            is_best: 是否为最佳模型
            additional_info: 额外信息
            filename_suffix: 文件名后缀

        Returns:
            保存的文件路径或None（如果不保存）
        """
        # 检查是否应该保存
        if not self.should_save_checkpoint(epoch, is_best):
            return None

        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_value': metric_value,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'additional_info': additional_info or {}
        }
        if best_tag is not None:
            checkpoint['best_tag'] = best_tag

        # 添加调度器状态
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # 生成文件名
        if is_best:
            # best_tag 用于区分不同“最优准则”（例如 F1 / loss）
            if best_tag:
                filename = f"{self.model_name}_{best_tag}_best.pth"
            else:
                filename = f"{self.model_name}_best.pth"
        elif filename_suffix:
            filename = f"{self.model_name}_{filename_suffix}.pth"
        else:
            filename = f"{self.model_name}_epoch_{epoch+1:04d}.pth"

        filepath = self.save_dir / filename

        # 保存检查点
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

        # 如果是最佳检查点，更新跟踪信息
        # 记录“新最优”的日志；仅当 best_tag=None 时才更新默认 best 的跟踪信息
        if is_best and metric_value is not None:
            tag_display = best_tag if best_tag is not None else "composite_score"
            logger.info(f"New best metric ({tag_display}): {metric_value:.6f}")

            # 只跟踪默认 best（即 best_tag=None，对应当前 trainer 的 val_f1 选择逻辑）
            if best_tag is None:
                self.best_metric = metric_value
                self.best_checkpoint_path = filepath

        # 管理检查点数量（只对普通检查点）
        if not is_best and not self.save_best_only:
            self.checkpoints.append(filepath)
            if len(self.checkpoints) > self.max_checkpoints:
                oldest_checkpoint = self.checkpoints.pop(0)
                if oldest_checkpoint.exists():
                    oldest_checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {oldest_checkpoint}")

        return str(filepath)

    def save_full_checkpoint(self,
                            epoch: int,
                            model: torch.nn.Module,
                            optimizer: torch.optim.Optimizer,
                            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                            metrics: Optional[Dict[str, float]] = None,
                            is_best: bool = False) -> Optional[str]:
        """
        保存完整检查点（简化接口）

        Args:
            epoch: 当前epoch
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            metrics: 指标字典
            is_best: 是否为最佳模型

        Returns:
            保存的文件路径或None
        """
        metric_value = metrics.get('val_f1', None) if metrics else None

        additional_info = {
            'metrics': metrics or {},
            'model_name': type(model).__name__
        }

        return self.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metric_value=metric_value,
            is_best=is_best,
            additional_info=additional_info
        )

    def load_checkpoint(self,
                        checkpoint_path: str,
                        model: torch.nn.Module,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                        device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备

        Returns:
            检查点信息
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        raw_epoch = checkpoint.get('epoch', 'N/A')
        if isinstance(raw_epoch, int):
            logger.info(f"Epoch: {raw_epoch + 1} (stored_index={raw_epoch})")
        else:
            logger.info(f"Epoch: {raw_epoch}")
        logger.info(f"Metric value: {checkpoint.get('metric_value', 'N/A')}")

        return checkpoint

    def load_best_checkpoint(self,
                            model: torch.nn.Module,
                            optimizer: Optional[torch.optim.Optimizer] = None,
                            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                            device: torch.device = torch.device('cpu'),
                            best_tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        加载最佳检查点

        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备

        Returns:
            检查点信息或None
        """
        # best_tag=None：沿用历史跟踪 best_checkpoint_path（对应 val_f1 best）
        if best_tag is None:
            if self.best_checkpoint_path is None:
                # 尝试在目录中查找最佳检查点
                best_checkpoint = self.save_dir / f"{self.model_name}_best.pth"
                if best_checkpoint.exists():
                    self.best_checkpoint_path = best_checkpoint

            if self.best_checkpoint_path is None:
                logger.warning("No best checkpoint found")
                return None

            best_path = self.best_checkpoint_path
        else:
            best_path = self.save_dir / f"{self.model_name}_{best_tag}_best.pth"
            if not best_path.exists():
                logger.warning(f"No best checkpoint found for tag={best_tag}: {best_path}")
                return None

        return self.load_checkpoint(best_path, model, optimizer, scheduler, device)

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        获取最新的检查点路径

        Returns:
            最新检查点路径或None
        """
        checkpoints = list(self.save_dir.glob(f"{self.model_name}_epoch_*.pth"))
        if not checkpoints:
            return None

        # 按文件名中的epoch排序
        def extract_epoch_number(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return 0

        checkpoints.sort(key=extract_epoch_number)
        return str(checkpoints[-1])
