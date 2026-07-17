"""
模型训练器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import logging
from tqdm import tqdm
import wandb

# 混合精度训练
try:
    from torch.amp import GradScaler, autocast
    HAS_AMP = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        HAS_AMP = True
    except ImportError:
        HAS_AMP = False
        print("警告: PyTorch版本不支持自动混合精度，将使用FP32训练")

from training.metrics_tracker import MetricsTracker
from utils.metrics_calculation import calculate_metrics, BoundingBoxMetrics
from training.checkpoint_manager import CheckpointManager
from training.early_stopping import EarlyStopping

# 与脚本 `f_train_model.py` 中 `setup_logging(..., log_name="train_model", ...)` 对齐，
# 让训练过程的 checkpoint 相关信息写入同一个日志文件。
logger = logging.getLogger("train_model")


class SolarFlareTrainer:
    """太阳耀斑训练器"""

    def __init__(self, model: nn.Module, config: Dict,
                 device: Optional[torch.device] = None):
        """
        初始化训练器

        Args:
            model: 模型
            config: 训练配置
            device: 设备
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # 将模型移到设备
        self.model = self.model.to(self.device)

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()

        # 初始化损失函数
        self.criterion = self._create_criterion()

        # 将损失函数中的权重移动到正确设备
        for key, loss_fn in self.criterion.items():
            if hasattr(loss_fn, 'weight') and loss_fn.weight is not None:
                loss_fn.weight = loss_fn.weight.to(self.device)

        # 梯度累积
        self.accumulation_steps = config.get('accumulation_steps', 1)

        # 混合精度训练
        self.use_mixed_precision = config.get('use_mixed_precision', False) and HAS_AMP
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("启用混合精度训练 (FP16)")
        else:
            self.scaler = None
            logger.info("使用FP32训练")

        # 初始化组件
        self.checkpoint_manager = CheckpointManager(config['checkpoint'])
        self.early_stopping = EarlyStopping(config['early_stopping'])
        self.metrics_tracker = MetricsTracker(config)

        # 训练状态
        self.current_epoch = 0
        self.best_metrics = {}
        self.two_stage_schedule = config.get('two_stage_schedule', {})
        self.enable_time_prediction = bool(
            config.get('model', {})
            .get('prediction_heads', {})
            .get('time', {})
            .get('enabled', True)
        )

        # 日志配置
        self.use_wandb = config['logging'].get('wandb', False)
        if self.use_wandb:
            wandb.init(entity="502025760001-nanjing-university",project="Solar_Flare_CME_prediction", config=config)

        logger.info(f"训练器初始化完成，设备: {self.device}, 梯度累积: {self.accumulation_steps}步")

    @staticmethod
    def _get_region_targets(targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        region_bbox = targets.get('region_bbox', targets['bbox'])
        region_mask = targets.get('region_mask')
        if region_mask is None:
            region_mask = targets.get('activity_mask', torch.ones_like(targets['label'], dtype=torch.bool))
        return region_bbox, region_mask

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.config['optimizer']
        optimizer_type = optimizer_config['type']

        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                eps=optimizer_config.get('eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 0)
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                eps=optimizer_config.get('eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"未知的优化器类型: {optimizer_type}")

        return optimizer

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', None)

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('gamma', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            scheduler = None

        return scheduler

    def _create_criterion(self) -> Dict[str, nn.Module]:
        """创建损失函数"""
        # 分类损失类别权重（仅对真实 bbox 槽位做 1/2 二分类）
        class_weights_cfg = self.config.get('class_weights', None)
        class_weight_tensor = None
        if class_weights_cfg is not None:
            try:
                class_weights = torch.tensor(class_weights_cfg, dtype=torch.float32)
                if class_weights.numel() >= 3:
                    class_weight_tensor = class_weights[:3].clone()
                elif class_weights.numel() == 2:
                    class_weight_tensor = torch.tensor([1.0, class_weights[0].item(), class_weights[1].item()], dtype=torch.float32)
                else:
                    class_weight_tensor = None
            except Exception as e:
                logger.warning("无法解析 class_weights 配置，将不使用类别权重: %s", e)
                class_weight_tensor = None

        criterion = {
            # 仅对类别 1/2 做二分类；类别 0 留给空槽位，不参与分类损失
            'classification': nn.CrossEntropyLoss(weight=class_weight_tensor),
            'time': nn.SmoothL1Loss(beta=0.05),
            'time_duration': nn.SmoothL1Loss(beta=0.05),
            'event_prob': nn.BCEWithLogitsLoss(),  # 改为BCEWithLogitsLoss，兼容混合精度训练
            'proposal_score': nn.BCEWithLogitsLoss(),
            'bbox_l1': nn.L1Loss(),
            'suppression': nn.L1Loss(),
        }

        return criterion

    def _get_slot_level_targets(self, targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构造槽位级监督目标。

        Returns:
            slot_labels: (B, A) 槽位标签，padding 槽位被置为 0
            slot_mask: (B, A) 有效槽位掩码
            has_any_event: (B,) 每个样本是否至少包含一个真实事件(label>0)
        """
        labels = targets['label']
        slot_mask = targets.get('activity_mask', torch.ones_like(labels, dtype=torch.bool))
        slot_labels = labels.clone()
        slot_labels[~slot_mask] = 0
        has_any_event = ((slot_labels > 0) & slot_mask).any(dim=1)
        return slot_labels, slot_mask, has_any_event

    @staticmethod
    def _safe_binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        tss = recall - (fp / (fp + tn + 1e-8))
        return {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'tss': float(tss),
        }

    def _compute_activity_binary_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        target_positive = targets > 0
        pred_positive = predictions > 0
        tp = int(np.sum(target_positive & pred_positive))
        fp = int(np.sum(~target_positive & pred_positive))
        fn = int(np.sum(target_positive & ~pred_positive))
        tn = int(np.sum(~target_positive & ~pred_positive))
        return self._safe_binary_metrics(tp, fp, fn, tn)

    def _compute_positive_class_ovr_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        per_class: Dict[str, Any] = {}
        for cls in (1, 2):
            target_positive = targets == cls
            pred_positive = predictions == cls
            tp = int(np.sum(target_positive & pred_positive))
            fp = int(np.sum(~target_positive & pred_positive))
            fn = int(np.sum(target_positive & ~pred_positive))
            tn = int(np.sum(~target_positive & ~pred_positive))
            per_class[str(cls)] = self._safe_binary_metrics(tp, fp, fn, tn)

        return {
            'per_class': per_class,
            'macro_precision': float(np.mean([v['precision'] for v in per_class.values()])),
            'macro_recall': float(np.mean([v['recall'] for v in per_class.values()])),
            'macro_f1': float(np.mean([v['f1'] for v in per_class.values()])),
            'macro_tss': float(np.mean([v['tss'] for v in per_class.values()])),
        }

    def _compute_slot_classification_metrics(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            num_classes: int
    ) -> Dict[str, Any]:
        """计算槽位级分类指标与每类统计。"""
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()

        metrics = calculate_metrics(pred_np, target_np, num_classes=num_classes)
        metrics['activity_binary'] = self._compute_activity_binary_metrics(target_np, pred_np)
        metrics['positive_class_ovr'] = self._compute_positive_class_ovr_metrics(target_np, pred_np)

        pred_counts = np.bincount(pred_np, minlength=num_classes)
        target_counts = np.bincount(target_np, minlength=num_classes)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for pred, target in zip(pred_np, target_np):
            confusion[target, pred] += 1

        positive_mask = target_np > 0
        if np.any(positive_mask):
            pos_pred = pred_np[positive_mask]
            pos_target = target_np[positive_mask]
            confusion_12 = np.zeros((2, 2), dtype=np.int64)
            confusion_12_with_background = np.zeros((2, 3), dtype=np.int64)
            for pred, target in zip(pos_pred, pos_target):
                target_idx = int(target - 1)
                if pred in (1, 2):
                    confusion_12[target_idx, int(pred - 1)] += 1
                if pred in (0, 1, 2):
                    confusion_12_with_background[target_idx, int(pred)] += 1

            accuracy_12 = float(np.mean(pos_pred == pos_target))
            class_precision_12 = {}
            class_recall_12 = {}
            class_f1_12 = {}
            class_tss_12 = {}
            precision_values_12 = []
            recall_values_12 = []
            f1_values_12 = []
            tss_values_12 = []
            support_values_12 = []

            for cls_idx, original_label in enumerate((1, 2)):
                tp = float(np.sum((pos_target == original_label) & (pos_pred == original_label)))
                fp = float(np.sum((pos_target != original_label) & (pos_pred == original_label)))
                fn = float(np.sum((pos_target == original_label) & (pos_pred != original_label)))
                tn = float(np.sum((pos_target != original_label) & (pos_pred != original_label)))
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                tss = recall - (fp / (fp + tn + 1e-8))
                label_key = str(original_label)
                class_precision_12[label_key] = float(precision)
                class_recall_12[label_key] = float(recall)
                class_f1_12[label_key] = float(f1)
                class_tss_12[label_key] = float(tss)
                precision_values_12.append(float(precision))
                recall_values_12.append(float(recall))
                f1_values_12.append(float(f1))
                tss_values_12.append(float(tss))
                support_values_12.append(float(np.sum(pos_target == original_label)))

            total_support_12 = float(sum(support_values_12))
            if total_support_12 > 0:
                weighted_precision_12 = float(sum(p * s for p, s in zip(precision_values_12, support_values_12)) / total_support_12)
                weighted_recall_12 = float(sum(r * s for r, s in zip(recall_values_12, support_values_12)) / total_support_12)
                weighted_f1_12 = float(sum(f * s for f, s in zip(f1_values_12, support_values_12)) / total_support_12)
                weighted_tss_12 = float(sum(t * s for t, s in zip(tss_values_12, support_values_12)) / total_support_12)
            else:
                weighted_precision_12 = 0.0
                weighted_recall_12 = 0.0
                weighted_f1_12 = 0.0
                weighted_tss_12 = 0.0
        else:
            confusion_12 = np.zeros((2, 2), dtype=np.int64)
            confusion_12_with_background = np.zeros((2, 3), dtype=np.int64)
            accuracy_12 = 0.0
            class_precision_12 = {}
            class_recall_12 = {}
            class_f1_12 = {}
            class_tss_12 = {}
            weighted_precision_12 = 0.0
            weighted_recall_12 = 0.0
            weighted_f1_12 = 0.0
            weighted_tss_12 = 0.0

        metrics['accuracy_12'] = float(accuracy_12)
        metrics['macro_precision_12'] = float(np.mean(list(class_precision_12.values()))) if class_precision_12 else 0.0
        metrics['macro_recall_12'] = float(np.mean(list(class_recall_12.values()))) if class_recall_12 else 0.0
        metrics['macro_f1_12'] = float(np.mean(list(class_f1_12.values()))) if class_f1_12 else 0.0
        metrics['macro_tss_12'] = float(np.mean(list(class_tss_12.values()))) if class_tss_12 else 0.0
        metrics['weighted_precision_12'] = float(weighted_precision_12)
        metrics['weighted_recall_12'] = float(weighted_recall_12)
        metrics['weighted_f1_12'] = float(weighted_f1_12)
        metrics['weighted_tss_12'] = float(weighted_tss_12)
        metrics['pred_class_counts_12'] = {
            '0': int(np.sum(pred_np[positive_mask] == 0)) if np.any(positive_mask) else 0,
            '1': int(np.sum(pred_np[positive_mask] == 1)) if np.any(positive_mask) else 0,
            '2': int(np.sum(pred_np[positive_mask] == 2)) if np.any(positive_mask) else 0,
        }
        metrics['target_class_counts_12'] = {
            '1': int(np.sum(target_np == 1)),
            '2': int(np.sum(target_np == 2)),
        }
        metrics['confusion_matrix_12'] = confusion_12.tolist()
        metrics['confusion_matrix_12_with_background'] = confusion_12_with_background.tolist()
        metrics['pred_class_counts'] = {str(i): int(pred_counts[i]) for i in range(num_classes)}
        metrics['target_class_counts'] = {str(i): int(target_counts[i]) for i in range(num_classes)}
        metrics['class_precision'] = {}
        metrics['class_recall'] = {}
        metrics['class_f1'] = {}
        metrics['class_precision_12'] = class_precision_12
        metrics['class_recall_12'] = class_recall_12
        metrics['class_f1_12'] = class_f1_12
        metrics['class_tss_12'] = class_tss_12
        for cls_idx in range(num_classes):
            tp = float(confusion[cls_idx, cls_idx])
            fp = float(confusion[:, cls_idx].sum() - tp)
            fn = float(confusion[cls_idx, :].sum() - tp)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            metrics['class_precision'][str(cls_idx)] = float(precision)
            metrics['class_recall'][str(cls_idx)] = float(recall)
            metrics['class_f1'][str(cls_idx)] = float(f1)

        return metrics

    def _box_area(self, boxes: torch.Tensor) -> torch.Tensor:
        """计算 xyxy 边界框面积。"""
        widths = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0)
        heights = (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)
        return widths * heights

    def _compute_composite_validation_score(self, val_metrics: Dict[str, float]) -> float:
        """综合验证指标：兼顾分类、定位与损失稳定性，越大越好。"""
        val_f1 = float(val_metrics.get('f1', 0.0))
        val_iou = float(val_metrics.get('iou', 0.0))
        val_tss = float(val_metrics.get('tss', 0.0))
        val_loss = float(val_metrics.get('loss', 0.0))
        classification_loss = float(val_metrics.get('classification_loss', 0.0))
        time_rmse = float(val_metrics.get('rmse', 0.0)) if self.enable_time_prediction else 0.0

        loss_term = 1.0 / (1.0 + max(val_loss, 0.0))
        cls_loss_term = 1.0 / (1.0 + max(classification_loss, 0.0))
        rmse_term = 1.0 / (1.0 + max(time_rmse, 0.0))

        composite = (
            0.35 * val_f1 +
            0.25 * val_iou +
            0.20 * max(val_tss, 0.0) +
            0.10 * loss_term +
            0.05 * cls_loss_term +
            0.05 * rmse_term
        )
        return float(composite)

    def _generalized_iou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """计算归一化 xyxy 边界框的 GIoU loss。"""
        inter_x1 = torch.maximum(pred_boxes[..., 0], target_boxes[..., 0])
        inter_y1 = torch.maximum(pred_boxes[..., 1], target_boxes[..., 1])
        inter_x2 = torch.minimum(pred_boxes[..., 2], target_boxes[..., 2])
        inter_y2 = torch.minimum(pred_boxes[..., 3], target_boxes[..., 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter_area = inter_w * inter_h

        pred_area = self._box_area(pred_boxes)
        target_area = self._box_area(target_boxes)
        union_area = pred_area + target_area - inter_area
        iou = inter_area / union_area.clamp(min=1e-8)

        enc_x1 = torch.minimum(pred_boxes[..., 0], target_boxes[..., 0])
        enc_y1 = torch.minimum(pred_boxes[..., 1], target_boxes[..., 1])
        enc_x2 = torch.maximum(pred_boxes[..., 2], target_boxes[..., 2])
        enc_y2 = torch.maximum(pred_boxes[..., 3], target_boxes[..., 3])
        enc_area = ((enc_x2 - enc_x1).clamp(min=0.0) * (enc_y2 - enc_y1).clamp(min=0.0)).clamp(min=1e-8)

        giou = iou - (enc_area - union_area) / enc_area
        return (1.0 - giou).mean()

    def _pairwise_box_iou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """计算预测框与真实框之间的两两 IoU 矩阵。"""
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return pred_boxes.new_zeros((pred_boxes.size(0), target_boxes.size(0)))

        inter_x1 = torch.maximum(pred_boxes[:, None, 0], target_boxes[None, :, 0])
        inter_y1 = torch.maximum(pred_boxes[:, None, 1], target_boxes[None, :, 1])
        inter_x2 = torch.minimum(pred_boxes[:, None, 2], target_boxes[None, :, 2])
        inter_y2 = torch.minimum(pred_boxes[:, None, 3], target_boxes[None, :, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter_area = inter_w * inter_h

        pred_area = self._box_area(pred_boxes)[:, None]
        target_area = self._box_area(target_boxes)[None, :]
        union_area = pred_area + target_area - inter_area
        return inter_area / union_area.clamp(min=1e-8)

    def _solve_assignment_min_cost(self, cost_matrix: torch.Tensor) -> List[Tuple[int, int]]:
        """对小规模 cost matrix 做精确一对一最优匹配。"""
        if cost_matrix.numel() == 0:
            return []

        cost_matrix = cost_matrix.detach().cpu()
        num_predictions, num_targets = cost_matrix.shape
        if num_predictions == 0 or num_targets == 0:
            return []

        if num_targets <= num_predictions:
            working_matrix = cost_matrix.transpose(0, 1)
            row_is_target = True
        else:
            working_matrix = cost_matrix
            row_is_target = False

        num_rows, num_cols = working_matrix.shape
        dp: Dict[int, Tuple[float, List[Tuple[int, int]]]] = {0: (0.0, [])}

        for row_idx in range(num_rows):
            next_dp: Dict[int, Tuple[float, List[Tuple[int, int]]]] = {}
            for used_mask, (score, pairs) in dp.items():
                for col_idx in range(num_cols):
                    if used_mask & (1 << col_idx):
                        continue
                    next_mask = used_mask | (1 << col_idx)
                    next_score = score + float(working_matrix[row_idx, col_idx].item())
                    best = next_dp.get(next_mask)
                    if best is None or next_score < best[0]:
                        next_dp[next_mask] = (next_score, pairs + [(row_idx, col_idx)])
            dp = next_dp

        _, best_pairs = min(dp.values(), key=lambda item: item[0])
        assignments: List[Tuple[int, int]] = []
        for row_idx, col_idx in best_pairs:
            if row_is_target:
                target_idx = row_idx
                pred_idx = col_idx
            else:
                pred_idx = row_idx
                target_idx = col_idx
            assignments.append((pred_idx, target_idx))
        return assignments

    def _build_hungarian_supervision_targets(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """基于 proposal boxes 做样本内 Hungarian matching，生成槽位监督目标。"""
        proposal_boxes = outputs.get('proposal_boxes', outputs['bbox_pred']).detach()
        raw_labels = targets['label']
        raw_slot_mask = targets.get('activity_mask', torch.ones_like(raw_labels, dtype=torch.bool))
        raw_bboxes = targets['bbox']
        raw_time_features = targets['time_features']

        assigned_labels = torch.zeros_like(raw_labels)
        assigned_bboxes = torch.zeros_like(raw_bboxes)
        assigned_time_features = torch.zeros_like(raw_time_features)
        positive_slot_mask = torch.zeros_like(raw_slot_mask)

        batch_size = proposal_boxes.size(0)
        for batch_idx in range(batch_size):
            gt_mask = (raw_labels[batch_idx] > 0) & raw_slot_mask[batch_idx]
            gt_indices = torch.nonzero(gt_mask, as_tuple=False).flatten()
            if gt_indices.numel() == 0:
                continue

            pred_boxes = proposal_boxes[batch_idx]
            gt_boxes = raw_bboxes[batch_idx, gt_indices]
            iou_matrix = self._pairwise_box_iou(pred_boxes, gt_boxes)
            l1_matrix = torch.abs(pred_boxes[:, None, :] - gt_boxes[None, :, :]).mean(dim=-1)
            cost_matrix = (1.0 - iou_matrix) + l1_matrix
            assignments = self._solve_assignment_min_cost(cost_matrix)

            for pred_idx, local_gt_idx in assignments:
                gt_idx = int(gt_indices[local_gt_idx].item())
                positive_slot_mask[batch_idx, pred_idx] = True
                assigned_labels[batch_idx, pred_idx] = raw_labels[batch_idx, gt_idx]
                assigned_bboxes[batch_idx, pred_idx] = raw_bboxes[batch_idx, gt_idx]
                assigned_time_features[batch_idx, pred_idx] = raw_time_features[batch_idx, gt_idx]

        return assigned_labels, positive_slot_mask, assigned_bboxes, assigned_time_features

    def _build_region_hungarian_supervision_targets(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于唯一活动区 GT 做样本内 Hungarian matching，生成定位监督目标。"""
        proposal_boxes = outputs.get('proposal_boxes', outputs['bbox_pred']).detach()
        raw_region_bboxes, raw_region_mask = self._get_region_targets(targets)

        assigned_bboxes = torch.zeros_like(raw_region_bboxes)
        positive_slot_mask = torch.zeros_like(raw_region_mask)

        batch_size = proposal_boxes.size(0)
        for batch_idx in range(batch_size):
            gt_indices = torch.nonzero(raw_region_mask[batch_idx], as_tuple=False).flatten()
            if gt_indices.numel() == 0:
                continue

            pred_boxes = proposal_boxes[batch_idx]
            gt_boxes = raw_region_bboxes[batch_idx, gt_indices]
            iou_matrix = self._pairwise_box_iou(pred_boxes, gt_boxes)
            l1_matrix = torch.abs(pred_boxes[:, None, :] - gt_boxes[None, :, :]).mean(dim=-1)
            cost_matrix = (1.0 - iou_matrix) + l1_matrix
            assignments = self._solve_assignment_min_cost(cost_matrix)

            for pred_idx, local_gt_idx in assignments:
                gt_idx = int(gt_indices[local_gt_idx].item())
                positive_slot_mask[batch_idx, pred_idx] = True
                assigned_bboxes[batch_idx, pred_idx] = raw_region_bboxes[batch_idx, gt_idx]

        return positive_slot_mask, assigned_bboxes

    def _get_roi_source_mix(self) -> Dict[str, float]:
        """读取第二阶段 ROI source mixing 配置。"""
        default_mix = {'gt': 1.0, 'jittered_gt': 0.0, 'predicted': 0.0}
        if not self.two_stage_schedule.get('enabled', False):
            return default_mix
        roi_mix = self.two_stage_schedule.get('roi_source_mix', {})
        total = float(sum(float(v) for v in roi_mix.values()))
        if total <= 0:
            return default_mix
        return {key: float(value) / total for key, value in roi_mix.items()}

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失

        Args:
            outputs: 模型输出
            targets: 目标值

        Returns:
            (总损失, 各损失分量字典)
        """
        loss_components = {}

        def _assert_finite(name: str, tensor: torch.Tensor, extra: Optional[Dict[str, Any]] = None) -> None:
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                details = extra or {}
                raise ValueError(f"检测到非有限{name}: {details}")

        # 1. 基于样本内 Hungarian matching 构造槽位监督，避免固定槽位顺序带来的错配
        slot_labels, positive_event_slot_mask, matched_bboxes, matched_time_features = \
            self._build_hungarian_supervision_targets(outputs, targets)
        positive_region_slot_mask, matched_region_bboxes = self._build_region_hungarian_supervision_targets(outputs, targets)
        slot_mask = torch.ones_like(positive_region_slot_mask, dtype=torch.bool, device=positive_region_slot_mask.device)
        external_proposals = bool(outputs.get('external_proposals', False))

        # 2. 分类损失：仅对匹配到真实 activity 的槽位做 1/2 二分类，类别 0 保留给未匹配槽位
        if positive_event_slot_mask.any():
            class_logits = outputs['class_logits'].reshape(-1, outputs['class_logits'].size(-1))
            class_targets = slot_labels.view(-1)
            class_mask = torch.ones_like(class_targets, dtype=torch.bool)
            selected_class_logits = class_logits[class_mask]
            selected_class_targets = class_targets[class_mask]
            _assert_finite(
                '分类logits',
                selected_class_logits,
                {
                    'class_logits_min': float(selected_class_logits.detach().min().item()),
                    'class_logits_max': float(selected_class_logits.detach().max().item()),
                    'num_slots': int(class_mask.sum().item()),
                }
            )
            class_loss = self.criterion['classification'](
                selected_class_logits,
                selected_class_targets
            )
        else:
            class_loss = torch.tensor(0.0, device=self.device)
        _assert_finite('分类损失', class_loss)
        loss_components['classification'] = class_loss

        # 3. 第一阶段 proposal/objectness 损失
        if external_proposals:
            proposal_score_logits = None
            proposal_targets = None
            proposal_score_loss = torch.tensor(0.0, device=self.device)
        else:
            proposal_score_logits = outputs.get('proposal_scores', outputs['event_prob']).squeeze(-1)
            proposal_targets = positive_region_slot_mask.float()
            _assert_finite('proposal logits', proposal_score_logits)
            proposal_score_loss = self.criterion['proposal_score'](proposal_score_logits, proposal_targets)
        _assert_finite('proposal损失', proposal_score_loss)
        loss_components['proposal_score'] = proposal_score_loss

        # 4. 第一阶段 proposal bbox 损失：对匹配后的正槽位计算 proposal L1 + GIoU
        bbox_mask = positive_region_slot_mask
        if external_proposals:
            proposal_bbox_l1_loss = torch.tensor(0.0, device=self.device)
            proposal_bbox_giou_loss = torch.tensor(0.0, device=self.device)
            proposal_bbox_loss = torch.tensor(0.0, device=self.device)
        elif bbox_mask.any():
            proposal_bbox_flat = outputs.get('proposal_boxes', outputs['bbox_pred']).view(-1, 4)
            bbox_target_flat = matched_region_bboxes.view(-1, 4)
            bbox_mask_flat = bbox_mask.view(-1)
            selected_proposal_bbox = proposal_bbox_flat[bbox_mask_flat]
            selected_bbox_target = bbox_target_flat[bbox_mask_flat]
            _assert_finite(
                'proposal bbox目标',
                selected_bbox_target,
                {
                    'bbox_target_min': float(selected_bbox_target.detach().min().item()),
                    'bbox_target_max': float(selected_bbox_target.detach().max().item()),
                    'num_bbox_slots': int(bbox_mask_flat.sum().item()),
                }
            )
            _assert_finite(
                'proposal bbox预测',
                selected_proposal_bbox,
                {
                    'bbox_pred_min': float(selected_proposal_bbox.detach().min().item()),
                    'bbox_pred_max': float(selected_proposal_bbox.detach().max().item()),
                }
            )
            proposal_bbox_l1_loss = self.criterion['bbox_l1'](selected_proposal_bbox, selected_bbox_target)
            proposal_bbox_giou_loss = self._generalized_iou_loss(selected_proposal_bbox, selected_bbox_target)
            proposal_bbox_loss = proposal_bbox_l1_loss + proposal_bbox_giou_loss
        else:
            proposal_bbox_l1_loss = torch.tensor(0.0, device=self.device)
            proposal_bbox_giou_loss = torch.tensor(0.0, device=self.device)
            proposal_bbox_loss = torch.tensor(0.0, device=self.device)
        _assert_finite('proposal bbox损失', proposal_bbox_loss)
        loss_components['proposal_bbox_l1'] = proposal_bbox_l1_loss
        loss_components['proposal_bbox_giou'] = proposal_bbox_giou_loss
        loss_components['proposal_bbox'] = proposal_bbox_loss

        # 5. 第二阶段 refined bbox 损失：对匹配后的正槽位计算 refined L1 + GIoU
        if bbox_mask.any():
            refined_bbox_flat = outputs['bbox_pred'].view(-1, 4)
            bbox_target_flat = matched_region_bboxes.view(-1, 4)
            bbox_mask_flat = bbox_mask.view(-1)
            selected_refined_bbox = refined_bbox_flat[bbox_mask_flat]
            selected_bbox_target = bbox_target_flat[bbox_mask_flat]
            _assert_finite(
                'refined bbox预测',
                selected_refined_bbox,
                {
                    'bbox_pred_min': float(selected_refined_bbox.detach().min().item()),
                    'bbox_pred_max': float(selected_refined_bbox.detach().max().item()),
                }
            )
            bbox_l1_loss = self.criterion['bbox_l1'](selected_refined_bbox, selected_bbox_target)
            bbox_giou_loss = self._generalized_iou_loss(selected_refined_bbox, selected_bbox_target)
            bbox_loss = bbox_l1_loss + bbox_giou_loss
        else:
            bbox_l1_loss = torch.tensor(0.0, device=self.device)
            bbox_giou_loss = torch.tensor(0.0, device=self.device)
            bbox_loss = torch.tensor(0.0, device=self.device)
        _assert_finite('bbox损失', bbox_loss)
        loss_components['bbox_l1'] = bbox_l1_loss
        loss_components['bbox_giou'] = bbox_giou_loss
        loss_components['bbox'] = bbox_loss

        # 6. 时间预测损失：仅对匹配后的正槽位计算（显式监督 start / peak / end）
        time_mask = positive_event_slot_mask
        if self.enable_time_prediction and time_mask.any():
            time_target_flat = matched_time_features.reshape(-1, 3)
            time_mask_flat = time_mask.reshape(-1)
            selected_time_target = time_target_flat[time_mask_flat]
            selected_time_pred = outputs['time_pred'].reshape(-1, 3)[time_mask_flat]

            target_start = selected_time_target[..., 0:1]
            target_peak = selected_time_target[..., 1:2]
            target_end = selected_time_target[..., 2:3]
            target_center = 0.5 * (target_start + target_end)
            target_duration = (target_end - target_start).clamp(min=1e-3)
            selected_time_center = outputs.get('time_center', 0.5 * (selected_time_pred[..., 0:1] + selected_time_pred[..., 2:3])).reshape(-1, 1)[time_mask_flat]
            selected_time_duration = outputs.get('time_duration', (selected_time_pred[..., 2:3] - selected_time_pred[..., 0:1]).clamp(min=1e-3)).reshape(-1, 1)[time_mask_flat]
            target_log_duration = torch.log(target_duration)
            pred_log_duration = torch.log(selected_time_duration.clamp(min=1e-3))

            _assert_finite(
                '时间目标',
                selected_time_target,
                {
                    'time_target_min': float(selected_time_target.detach().min().item()),
                    'time_target_max': float(selected_time_target.detach().max().item()),
                    'num_time_slots': int(time_mask_flat.sum().item()),
                }
            )
            _assert_finite(
                '时间预测',
                selected_time_pred,
                {
                    'time_pred_min': float(selected_time_pred.detach().min().item()),
                    'time_pred_max': float(selected_time_pred.detach().max().item()),
                }
            )
            center_loss = self.criterion['time'](selected_time_center, target_center)
            duration_loss = self.criterion['time_duration'](pred_log_duration, target_log_duration)
            boundary_loss = self.criterion['time'](selected_time_pred, selected_time_target)
            time_loss = 0.25 * center_loss + 0.25 * duration_loss + 0.50 * boundary_loss
        else:
            time_loss = torch.tensor(0.0, device=self.device)
        _assert_finite('时间损失', time_loss)
        loss_components['time'] = time_loss

        # 7. 事件概率损失：按匹配后的槽位监督（匹配到真实事件的槽位为 1）
        event_prob_targets = positive_region_slot_mask.float()
        _assert_finite('事件概率logits', outputs['event_prob'])
        _assert_finite('事件概率目标', event_prob_targets)
        event_prob_loss = self.criterion['event_prob'](
            outputs['event_prob'].squeeze(-1),
            event_prob_targets
        )
        _assert_finite('事件概率损失', event_prob_loss)
        loss_components['event_prob'] = event_prob_loss

        # 8. 负槽位抑制损失：未匹配槽位应收缩为空框/极短时间
        negative_slot_mask = slot_mask & (~positive_region_slot_mask)
        if negative_slot_mask.any():
            negative_mask_flat = negative_slot_mask.view(-1)
            bbox_size_gated = outputs.get('bbox_size_gated')
            time_duration_raw = outputs.get('time_pred') if self.enable_time_prediction else None
            if bbox_size_gated is not None:
                neg_bbox_size = bbox_size_gated.reshape(-1, bbox_size_gated.size(-1))[negative_mask_flat]
                bbox_suppression_loss = self.criterion['suppression'](
                    neg_bbox_size,
                    torch.zeros_like(neg_bbox_size)
                )
            else:
                bbox_suppression_loss = torch.tensor(0.0, device=self.device)
            if time_duration_raw is not None:
                neg_time_pred = time_duration_raw.reshape(-1, 3)[negative_mask_flat]
                time_suppression_target = torch.zeros_like(neg_time_pred)
                time_suppression_loss = self.criterion['time'](
                    neg_time_pred,
                    time_suppression_target
                )
            else:
                time_suppression_loss = torch.tensor(0.0, device=self.device)
        else:
            bbox_suppression_loss = torch.tensor(0.0, device=self.device)
            time_suppression_loss = torch.tensor(0.0, device=self.device)
        _assert_finite('负槽位bbox抑制损失', bbox_suppression_loss)
        _assert_finite('负槽位时间抑制损失', time_suppression_loss)
        loss_components['bbox_suppression'] = bbox_suppression_loss
        loss_components['time_suppression'] = time_suppression_loss

        # 9. 物理约束损失
        physics_loss = outputs.get('physics_loss', torch.tensor(0.0, device=self.device))
        if not isinstance(physics_loss, torch.Tensor):
            physics_loss = torch.tensor(physics_loss, device=self.device, dtype=torch.float32)
        _assert_finite('物理约束损失', physics_loss)
        loss_components['physics'] = physics_loss

        # 10. 计算总损失
        suppression_weight = float(self.config['loss_weights'].get('negative_suppression', 0.2))
        total_loss = (
                self.config['loss_weights']['classification'] * class_loss +
                self.config['loss_weights']['bbox_regression'] * proposal_bbox_loss +
                self.config['loss_weights'].get('bbox_refine', self.config['loss_weights']['bbox_regression']) * bbox_loss +
                self.config['loss_weights']['time_prediction'] * time_loss +
                self.config['loss_weights']['event_probability'] * event_prob_loss +
                self.config['loss_weights']['event_probability'] * proposal_score_loss +
                suppression_weight * (bbox_suppression_loss + time_suppression_loss) +
                self.config['loss_weights']['physics_constraint'] * physics_loss
        )
        _assert_finite('总损失', total_loss)

        return total_loss, loss_components

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            训练指标字典
        """
        self.model.train()

        epoch_metrics = {
            'loss': 0.0,
            'classification_loss': 0.0,
            'proposal_score_loss': 0.0,
            'proposal_bbox_loss': 0.0,
            'bbox_loss': 0.0,
            'time_loss': 0.0,
            'event_prob_loss': 0.0,
            'physics_loss': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tss': 0.0,
            'rss': 0.0,
            'rmse': 0.0,
            'iou': 0.0,
        }

        all_predictions = []
        all_targets = []
        all_slot_predictions = []
        all_slot_targets = []
        all_time_predictions = []
        all_time_targets = []
        all_bbox_predictions = []
        all_bbox_targets = []

        num_batches = len(train_loader)

        with tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} - Train") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 准备数据
                inputs = {k: v.to(self.device) for k, v in batch['data'].items()}
                if 'proposal_boxes' in batch:
                    inputs['proposal_boxes'] = batch['proposal_boxes'].to(self.device)
                if 'proposal_scores' in batch:
                    inputs['proposal_scores'] = batch['proposal_scores'].to(self.device)
                targets = {
                    'label': batch['label'].to(self.device),
                    'bbox': batch['bbox'].to(self.device),
                    'region_bbox': batch.get('region_bbox', batch['bbox']).to(self.device),
                    'time_features': batch['time_features'].to(self.device),
                    'activity_mask': batch.get('activity_mask', torch.ones_like(batch['label'], dtype=torch.bool)).to(self.device),
                    'region_mask': batch.get('region_mask', batch.get('activity_mask', torch.ones_like(batch['label'], dtype=torch.bool))).to(self.device),
                }

                # 物理输入（如果有）
                physics_inputs = batch.get('physics_inputs')
                if physics_inputs is not None:
                    physics_inputs = {k: v.to(self.device) for k, v in physics_inputs.items()}

                if not torch.isfinite(targets['bbox']).all():
                    bad_idx = torch.nonzero(~torch.isfinite(targets['bbox']), as_tuple=False)
                    sample_indices = batch.get('sample_idx')
                    event_ids = batch.get('event_id')
                    details = []
                    for idx_triplet in bad_idx[:8]:
                        b_idx, slot_idx, coord_idx = [int(v.item()) for v in idx_triplet]
                        sample_desc = f"batch={b_idx},slot={slot_idx},coord={coord_idx}"
                        if sample_indices is not None:
                            try:
                                sample_desc += f",sample_idx={sample_indices[b_idx]}"
                            except Exception:
                                pass
                        if event_ids is not None:
                            try:
                                sample_desc += f",event_id={event_ids[b_idx]}"
                            except Exception:
                                pass
                        details.append(sample_desc)
                    raise ValueError("训练批次 targets['bbox'] 存在非有限值: " + "; ".join(details))

                # 混合精度前向传播
                if self.use_mixed_precision:
                    with autocast(device_type="cuda"):
                        outputs = self.model(inputs, physics_inputs, targets=targets)
                        total_loss, loss_components = self.compute_loss(outputs, targets)
                    
                    # 缩放损失进行反向传播
                    self.scaler.scale(total_loss).backward()
                else:
                    # 前向传播
                    outputs = self.model(inputs, physics_inputs, targets=targets)
                    # 计算损失
                    total_loss, loss_components = self.compute_loss(outputs, targets)
                    # 反向传播
                    total_loss.backward()

                # 梯度累积：只有在积累足够步数后才更新参数
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # 梯度裁剪
                    if self.config.get('gradient_clip', 0) > 0:
                        if self.use_mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['gradient_clip']
                        )

                    # 优化器步骤
                    if self.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # 清零梯度
                    self.optimizer.zero_grad()

                # 更新指标（使用未缩放的损失）
                batch_size = targets['label'].size(0)
                epoch_metrics['loss'] += total_loss.item() * batch_size
                epoch_metrics['classification_loss'] += loss_components['classification'].item() * batch_size
                epoch_metrics['proposal_score_loss'] += loss_components['proposal_score'].item() * batch_size
                epoch_metrics['proposal_bbox_loss'] += loss_components['proposal_bbox'].item() * batch_size
                epoch_metrics['bbox_loss'] += loss_components['bbox'].item() * batch_size
                epoch_metrics['time_loss'] += loss_components['time'].item() * batch_size
                epoch_metrics['event_prob_loss'] += loss_components['event_prob'].item() * batch_size
                epoch_metrics['physics_loss'] += loss_components['physics'].item() * batch_size

                # 计算分类指标：仅对真实 activity 槽位统计，并把预测限制在 1/2
                pred_slots = outputs['class_logits'].argmax(dim=-1)  # (B, A)
                slot_labels, eval_mask, matched_bboxes, matched_time_features = \
                    self._build_hungarian_supervision_targets(outputs, targets)
                region_eval_mask, matched_region_bboxes = self._build_region_hungarian_supervision_targets(outputs, targets)
                pred_slots_flat = pred_slots[eval_mask]
                target_slots_flat = slot_labels[eval_mask]
                all_slot_predictions.extend(pred_slots.reshape(-1).detach().cpu().numpy())
                all_slot_targets.extend(slot_labels.reshape(-1).detach().cpu().numpy())

                bbox_mask = region_eval_mask
                if bbox_mask.any():
                    for sample_idx in range(bbox_mask.size(0)):
                        sample_mask = bbox_mask[sample_idx]
                        if sample_mask.any():
                            all_bbox_predictions.append(
                                outputs['bbox_pred'][sample_idx][sample_mask].detach().cpu().numpy()
                            )
                            all_bbox_targets.append(
                                matched_region_bboxes[sample_idx][sample_mask].detach().cpu().numpy()
                            )
                    if self.enable_time_prediction:
                        all_time_predictions.append(outputs['time_pred'][bbox_mask].detach().cpu().numpy())
                        all_time_targets.append(matched_time_features[bbox_mask].detach().cpu().numpy())
                    epoch_metrics.setdefault('proposal_score_mean', 0.0)
                    epoch_metrics['proposal_score_mean'] += float(outputs['proposal_scores'].sigmoid().mean().item()) * batch_size

                if pred_slots_flat.numel() > 0:
                    all_predictions.extend(pred_slots_flat.detach().cpu().numpy())
                    all_targets.extend(target_slots_flat.detach().cpu().numpy())
                    metrics = self._compute_slot_classification_metrics(
                        pred_slots_flat,
                        target_slots_flat,
                        num_classes=self.config.get('num_classes', 3)
                    )
                else:
                    metrics = {
                        'accuracy': 0.0,
                        'macro_precision': 0.0,
                        'macro_recall': 0.0,
                        'macro_f1': 0.0,
                        'macro_tss': 0.0,
                        'accuracy_12': 0.0,
                        'macro_precision_12': 0.0,
                        'macro_recall_12': 0.0,
                        'macro_f1_12': 0.0,
                        'macro_tss_12': 0.0,
                        'pred_class_counts': {},
                        'target_class_counts': {},
                        'pred_class_counts_12': {},
                        'target_class_counts_12': {},
                        'class_precision': {},
                        'class_recall': {},
                        'class_f1': {},
                        'class_precision_12': {},
                        'class_recall_12': {},
                        'class_f1_12': {},
                        'class_tss_12': {},
                    }

                # 防止 nan 传播
                if np.isnan(metrics.get('accuracy', 0.0)):
                    metrics['accuracy'] = 0.0

                # calculate_metrics 返回的是 macro_precision/macro_recall/macro_f1 等字段；
                # 训练器历史记录使用 precision/recall/f1，这里做一次映射，避免这些值长期为 0。
                epoch_metrics['accuracy'] += float(metrics.get('accuracy_12', 0.0)) * batch_size
                epoch_metrics['precision'] += float(metrics.get('macro_precision_12', 0.0)) * batch_size
                epoch_metrics['recall'] += float(metrics.get('macro_recall_12', 0.0)) * batch_size
                epoch_metrics['f1'] += float(metrics.get('macro_f1_12', 0.0)) * batch_size

                # 更新进度条
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'acc': metrics.get('accuracy_12', metrics['accuracy'])
                })

        # 计算平均指标
        total_samples = len(train_loader.dataset)
        for key in ['loss', 'classification_loss', 'proposal_score_loss', 'proposal_bbox_loss', 'bbox_loss', 'time_loss', 'event_prob_loss', 'physics_loss', 'accuracy', 'precision', 'recall', 'f1', 'proposal_score_mean']:
            epoch_metrics[key] /= total_samples

        if len(all_predictions) > 0:
            epoch_cls_metrics = self._compute_slot_classification_metrics(
                torch.as_tensor(np.asarray(all_predictions, dtype=np.int64)),
                torch.as_tensor(np.asarray(all_targets, dtype=np.int64)),
                num_classes=self.config.get('num_classes', 3)
            )
            epoch_metrics['train_pred_class_counts'] = epoch_cls_metrics.get('pred_class_counts', {})
            epoch_metrics['train_target_class_counts'] = epoch_cls_metrics.get('target_class_counts', {})
            epoch_metrics['train_pred_class_counts_12'] = epoch_cls_metrics.get('pred_class_counts_12', {})
            epoch_metrics['train_target_class_counts_12'] = epoch_cls_metrics.get('target_class_counts_12', {})
            epoch_metrics['train_class_precision'] = epoch_cls_metrics.get('class_precision', {})
            epoch_metrics['train_class_recall'] = epoch_cls_metrics.get('class_recall', {})
            epoch_metrics['train_class_f1'] = epoch_cls_metrics.get('class_f1', {})
            epoch_metrics['train_class_precision_12'] = epoch_cls_metrics.get('class_precision_12', {})
            epoch_metrics['train_class_recall_12'] = epoch_cls_metrics.get('class_recall_12', {})
            epoch_metrics['train_class_f1_12'] = epoch_cls_metrics.get('class_f1_12', {})
            epoch_metrics['train_class_tss_12'] = epoch_cls_metrics.get('class_tss_12', {})
            epoch_metrics['train_confusion_matrix_12'] = epoch_cls_metrics.get('confusion_matrix_12', [[0, 0], [0, 0]])
            epoch_metrics['train_confusion_matrix_12_with_background'] = epoch_cls_metrics.get('confusion_matrix_12_with_background', [[0, 0, 0], [0, 0, 0]])

            epoch_metrics['accuracy_all'] = float(epoch_cls_metrics.get('accuracy', 0.0))
            epoch_metrics['precision_all'] = float(epoch_cls_metrics.get('macro_precision', 0.0))
            epoch_metrics['recall_all'] = float(epoch_cls_metrics.get('macro_recall', 0.0))
            epoch_metrics['f1_all'] = float(epoch_cls_metrics.get('macro_f1', 0.0))
            epoch_metrics['tss_all'] = float(epoch_cls_metrics.get('macro_tss', 0.0))

            epoch_metrics['accuracy'] = float(epoch_cls_metrics.get('accuracy_12', 0.0))
            epoch_metrics['precision'] = float(epoch_cls_metrics.get('macro_precision_12', 0.0))
            epoch_metrics['recall'] = float(epoch_cls_metrics.get('macro_recall_12', 0.0))
            epoch_metrics['f1'] = float(epoch_cls_metrics.get('macro_f1_12', 0.0))
            epoch_metrics['tss'] = float(epoch_cls_metrics.get('macro_tss_12', 0.0))
            epoch_metrics['macro_precision_12'] = float(epoch_cls_metrics.get('macro_precision_12', 0.0))
            epoch_metrics['macro_recall_12'] = float(epoch_cls_metrics.get('macro_recall_12', 0.0))
            epoch_metrics['macro_f1_12'] = float(epoch_cls_metrics.get('macro_f1_12', 0.0))
            epoch_metrics['macro_tss_12'] = float(epoch_cls_metrics.get('macro_tss_12', 0.0))
        else:
            epoch_metrics['train_pred_class_counts'] = {}
            epoch_metrics['train_target_class_counts'] = {}
            epoch_metrics['train_pred_class_counts_12'] = {}
            epoch_metrics['train_target_class_counts_12'] = {}
            epoch_metrics['train_class_precision'] = {}
            epoch_metrics['train_class_recall'] = {}
            epoch_metrics['train_class_f1'] = {}
            epoch_metrics['train_class_precision_12'] = {}
            epoch_metrics['train_class_recall_12'] = {}
            epoch_metrics['train_class_f1_12'] = {}
            epoch_metrics['train_class_tss_12'] = {}
            epoch_metrics['train_confusion_matrix_12'] = [[0, 0], [0, 0]]
            epoch_metrics['train_confusion_matrix_12_with_background'] = [[0, 0, 0], [0, 0, 0]]
            epoch_metrics['accuracy_all'] = 0.0
            epoch_metrics['precision_all'] = 0.0
            epoch_metrics['recall_all'] = 0.0
            epoch_metrics['f1_all'] = 0.0
            epoch_metrics['tss_all'] = 0.0
            epoch_metrics['accuracy'] = 0.0
            epoch_metrics['precision'] = 0.0
            epoch_metrics['recall'] = 0.0
            epoch_metrics['f1'] = 0.0
            epoch_metrics['tss'] = 0.0
            epoch_metrics['macro_precision_12'] = 0.0
            epoch_metrics['macro_recall_12'] = 0.0
            epoch_metrics['macro_f1_12'] = 0.0
            epoch_metrics['macro_tss_12'] = 0.0

        if len(all_slot_predictions) > 0:
            train_all_slot_metrics = self._compute_slot_classification_metrics(
                torch.as_tensor(np.asarray(all_slot_predictions, dtype=np.int64)),
                torch.as_tensor(np.asarray(all_slot_targets, dtype=np.int64)),
                num_classes=self.config.get('num_classes', 3)
            )
        else:
            train_all_slot_metrics = {}
        epoch_metrics['classification_activity_only'] = epoch_cls_metrics if len(all_predictions) > 0 else {}
        epoch_metrics['classification_all_slots'] = train_all_slot_metrics
        epoch_metrics['activity_binary_activity_only'] = epoch_metrics['classification_activity_only'].get('activity_binary', {})
        epoch_metrics['activity_binary_all_slots'] = train_all_slot_metrics.get('activity_binary', {}) if train_all_slot_metrics else {}
        epoch_metrics['positive_class_ovr_activity_only'] = epoch_metrics['classification_activity_only'].get('positive_class_ovr', {})
        epoch_metrics['positive_class_ovr_all_slots'] = train_all_slot_metrics.get('positive_class_ovr', {}) if train_all_slot_metrics else {}
        epoch_metrics['num_eval_activity_slots'] = int(len(all_predictions))
        epoch_metrics['num_eval_all_slots'] = int(len(all_slot_predictions))

        if all_time_predictions:
            train_time_pred = np.concatenate(all_time_predictions, axis=0)
            train_time_tgt = np.concatenate(all_time_targets, axis=0)
            rss = np.sum((train_time_tgt - train_time_pred) ** 2)
            rmse = np.sqrt(np.mean((train_time_tgt - train_time_pred) ** 2))
            epoch_metrics['rss'] = float(rss)
            epoch_metrics['rmse'] = float(rmse)
        else:
            epoch_metrics['rss'] = 0.0
            epoch_metrics['rmse'] = 0.0

        if all_bbox_predictions:
            bbox_metrics = BoundingBoxMetrics.compute_metrics_per_sample(
                all_bbox_predictions,
                all_bbox_targets
            )
            epoch_metrics['iou'] = float(bbox_metrics.get('average_iou', 0.0))
        else:
            epoch_metrics['iou'] = 0.0

        # 记录到wandb
        if self.use_wandb:
            wandb.log({f'train/{k}': v for k, v in epoch_metrics.items() if np.isscalar(v)}, step=self.current_epoch)

        return epoch_metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证

        Args:
            val_loader: 验证数据加载器

        Returns:
            验证指标字典
        """
        self.model.eval()

        epoch_metrics = {
            'loss': 0.0,
            'classification_loss': 0.0,
            'proposal_score_loss': 0.0,
            'proposal_bbox_loss': 0.0,
            'bbox_loss': 0.0,
            'time_loss': 0.0,
            'event_prob_loss': 0.0,
            'physics_loss': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tss': 0.0,
            'rss': 0.0,
            'rmse': 0.0,
            'iou': 0.0,
            'confusion_matrix': None
        }

        all_predictions = []
        all_targets = []
        all_slot_predictions = []
        all_slot_targets = []
        all_time_predictions = []
        all_time_targets = []
        all_bbox_predictions = []
        all_bbox_targets = []

        with tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} - Val") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 准备数据
                inputs = {k: v.to(self.device) for k, v in batch['data'].items()}
                if 'proposal_boxes' in batch:
                    inputs['proposal_boxes'] = batch['proposal_boxes'].to(self.device)
                if 'proposal_scores' in batch:
                    inputs['proposal_scores'] = batch['proposal_scores'].to(self.device)
                targets = {
                    'label': batch['label'].to(self.device),
                    'bbox': batch['bbox'].to(self.device),
                    'region_bbox': batch.get('region_bbox', batch['bbox']).to(self.device),
                    'time_features': batch['time_features'].to(self.device),
                    'activity_mask': batch.get('activity_mask', torch.ones_like(batch['label'], dtype=torch.bool)).to(self.device),
                    'region_mask': batch.get('region_mask', batch.get('activity_mask', torch.ones_like(batch['label'], dtype=torch.bool))).to(self.device),
                }

                # 物理输入（如果有）
                physics_inputs = batch.get('physics_inputs')
                if physics_inputs is not None:
                    physics_inputs = {k: v.to(self.device) for k, v in physics_inputs.items()}

                if not torch.isfinite(targets['bbox']).all():
                    bad_idx = torch.nonzero(~torch.isfinite(targets['bbox']), as_tuple=False)
                    sample_indices = batch.get('sample_idx')
                    event_ids = batch.get('event_id')
                    details = []
                    for idx_triplet in bad_idx[:8]:
                        b_idx, slot_idx, coord_idx = [int(v.item()) for v in idx_triplet]
                        sample_desc = f"batch={b_idx},slot={slot_idx},coord={coord_idx}"
                        if sample_indices is not None:
                            try:
                                sample_desc += f",sample_idx={sample_indices[b_idx]}"
                            except Exception:
                                pass
                        if event_ids is not None:
                            try:
                                sample_desc += f",event_id={event_ids[b_idx]}"
                            except Exception:
                                pass
                        details.append(sample_desc)
                    raise ValueError("验证批次 targets['bbox'] 存在非有限值: " + "; ".join(details))

                # 前向传播
                outputs = self.model(inputs, physics_inputs, targets=targets)

                # 计算损失
                total_loss, loss_components = self.compute_loss(outputs, targets)

                # 更新指标
                batch_size = targets['label'].size(0)
                epoch_metrics['loss'] += total_loss.item() * batch_size
                epoch_metrics['classification_loss'] += loss_components['classification'].item() * batch_size
                epoch_metrics['proposal_score_loss'] += loss_components['proposal_score'].item() * batch_size
                epoch_metrics['proposal_bbox_loss'] += loss_components['proposal_bbox'].item() * batch_size
                epoch_metrics['bbox_loss'] += loss_components['bbox'].item() * batch_size
                epoch_metrics['time_loss'] += loss_components['time'].item() * batch_size
                epoch_metrics['event_prob_loss'] += loss_components['event_prob'].item() * batch_size
                epoch_metrics['physics_loss'] += loss_components['physics'].item() * batch_size

                # 收集预测和槽位级主标签：仅对真实 activity 槽位统计，并把预测限制在 1/2
                pred_slots = outputs['class_logits'].argmax(dim=-1)  # (B, A)
                slot_labels, eval_mask, matched_bboxes, matched_time_features = \
                    self._build_hungarian_supervision_targets(outputs, targets)
                region_eval_mask, matched_region_bboxes = self._build_region_hungarian_supervision_targets(outputs, targets)
                pred_slots_flat = pred_slots[eval_mask]
                target_slots_flat = slot_labels[eval_mask]
                all_slot_predictions.extend(pred_slots.reshape(-1).detach().cpu().numpy())
                all_slot_targets.extend(slot_labels.reshape(-1).detach().cpu().numpy())

                bbox_mask = region_eval_mask
                if bbox_mask.any():
                    for sample_idx in range(bbox_mask.size(0)):
                        sample_mask = bbox_mask[sample_idx]
                        if sample_mask.any():
                            all_bbox_predictions.append(
                                outputs['bbox_pred'][sample_idx][sample_mask].detach().cpu().numpy()
                            )
                            all_bbox_targets.append(
                                matched_region_bboxes[sample_idx][sample_mask].detach().cpu().numpy()
                            )
                    if self.enable_time_prediction:
                        all_time_predictions.append(outputs['time_pred'][bbox_mask].detach().cpu().numpy())
                        all_time_targets.append(matched_time_features[bbox_mask].detach().cpu().numpy())
                    epoch_metrics.setdefault('proposal_score_mean', 0.0)
                    epoch_metrics['proposal_score_mean'] += float(outputs['proposal_scores'].sigmoid().mean().item()) * batch_size

                if pred_slots_flat.numel() > 0:
                    all_predictions.extend(pred_slots_flat.detach().cpu().numpy())
                    all_targets.extend(target_slots_flat.detach().cpu().numpy())

                    # 更新进度条
                    batch_metrics = self._compute_slot_classification_metrics(
                        pred_slots_flat,
                        target_slots_flat,
                        num_classes=self.config.get('num_classes', 3)
                    )
                else:
                    batch_metrics = {
                        'accuracy': 0.0,
                        'macro_precision': 0.0,
                        'macro_recall': 0.0,
                        'macro_f1': 0.0,
                        'macro_tss': 0.0,
                        'accuracy_12': 0.0,
                        'macro_precision_12': 0.0,
                        'macro_recall_12': 0.0,
                        'macro_f1_12': 0.0,
                        'macro_tss_12': 0.0,
                        'pred_class_counts': {},
                        'target_class_counts': {},
                        'pred_class_counts_12': {},
                        'target_class_counts_12': {},
                        'class_precision': {},
                        'class_recall': {},
                        'class_f1': {},
                        'class_precision_12': {},
                        'class_recall_12': {},
                        'class_f1_12': {},
                        'class_tss_12': {},
                    }
                pbar.set_postfix({'loss': total_loss.item(), 'acc': batch_metrics.get('accuracy_12', batch_metrics['accuracy'])})

        # 计算平均指标
        total_samples = len(val_loader.dataset)
        for key in ['loss', 'classification_loss', 'proposal_score_loss', 'proposal_bbox_loss', 'bbox_loss', 'time_loss',
                    'event_prob_loss', 'physics_loss']:
            if key in epoch_metrics:
                epoch_metrics[key] /= total_samples

        # 计算分类指标
        if len(all_predictions) > 0:
            metrics = self._compute_slot_classification_metrics(
                torch.as_tensor(np.array(all_predictions)),
                torch.as_tensor(np.array(all_targets)),
                num_classes=self.config.get('num_classes', 3)
            )
        else:
            metrics = {
                'accuracy': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'macro_f1': 0.0,
                'macro_tss': 0.0,
                'accuracy_12': 0.0,
                'macro_precision_12': 0.0,
                'macro_recall_12': 0.0,
                'macro_f1_12': 0.0,
                'macro_tss_12': 0.0,
                'pred_class_counts': {},
                'target_class_counts': {},
                'pred_class_counts_12': {},
                'target_class_counts_12': {},
                'class_precision': {},
                'class_recall': {},
                'class_f1': {},
                'class_precision_12': {},
                'class_recall_12': {},
                'class_f1_12': {},
                'class_tss_12': {},
                'confusion_matrix_12': [[0, 0], [0, 0]],
            }
        if len(all_slot_predictions) > 0:
            all_slot_metrics = self._compute_slot_classification_metrics(
                torch.as_tensor(np.array(all_slot_predictions)),
                torch.as_tensor(np.array(all_slot_targets)),
                num_classes=self.config.get('num_classes', 3)
            )
        else:
            all_slot_metrics = {}
        if np.isnan(metrics.get('accuracy', 0.0)):
            metrics['accuracy'] = 0.0
        if all_slot_metrics and np.isnan(all_slot_metrics.get('accuracy', 0.0)):
            all_slot_metrics['accuracy'] = 0.0
        epoch_metrics.update(metrics)
        epoch_metrics['classification_activity_only'] = metrics
        epoch_metrics['classification_all_slots'] = all_slot_metrics
        epoch_metrics['activity_binary_activity_only'] = metrics.get('activity_binary', {})
        epoch_metrics['activity_binary_all_slots'] = all_slot_metrics.get('activity_binary', {}) if all_slot_metrics else {}
        epoch_metrics['positive_class_ovr_activity_only'] = metrics.get('positive_class_ovr', {})
        epoch_metrics['positive_class_ovr_all_slots'] = all_slot_metrics.get('positive_class_ovr', {}) if all_slot_metrics else {}
        epoch_metrics['num_eval_activity_slots'] = int(len(all_predictions))
        epoch_metrics['num_eval_all_slots'] = int(len(all_slot_predictions))
        # 同步写入训练器期望字段名
        epoch_metrics['accuracy_all'] = float(metrics.get('accuracy', 0.0))
        epoch_metrics['precision_all'] = float(metrics.get('macro_precision', 0.0))
        epoch_metrics['recall_all'] = float(metrics.get('macro_recall', 0.0))
        epoch_metrics['f1_all'] = float(metrics.get('macro_f1', 0.0))
        epoch_metrics['tss_all'] = float(metrics.get('macro_tss', 0.0))
        epoch_metrics['accuracy'] = float(metrics.get('accuracy_12', 0.0))
        epoch_metrics['precision'] = float(metrics.get('macro_precision_12', 0.0))
        epoch_metrics['recall'] = float(metrics.get('macro_recall_12', 0.0))
        epoch_metrics['f1'] = float(metrics.get('macro_f1_12', 0.0))
        epoch_metrics['tss'] = float(metrics.get('macro_tss_12', 0.0))

        if all_time_predictions:
            val_time_pred = np.concatenate(all_time_predictions, axis=0)
            val_time_tgt = np.concatenate(all_time_targets, axis=0)
            rss = np.sum((val_time_tgt - val_time_pred) ** 2)
            rmse = np.sqrt(np.mean((val_time_tgt - val_time_pred) ** 2))
            epoch_metrics['rss'] = float(rss)
            epoch_metrics['rmse'] = float(rmse)
        else:
            epoch_metrics['rss'] = 0.0
            epoch_metrics['rmse'] = 0.0

        if all_bbox_predictions:
            bbox_metrics = BoundingBoxMetrics.compute_metrics_per_sample(
                all_bbox_predictions,
                all_bbox_targets
            )
            epoch_metrics['iou'] = float(bbox_metrics.get('average_iou', 0.0))
        else:
            epoch_metrics['iou'] = 0.0

        epoch_metrics['val_pred_class_counts'] = metrics.get('pred_class_counts', {})
        epoch_metrics['val_target_class_counts'] = metrics.get('target_class_counts', {})
        epoch_metrics['val_pred_class_counts_12'] = metrics.get('pred_class_counts_12', {})
        epoch_metrics['val_target_class_counts_12'] = metrics.get('target_class_counts_12', {})
        epoch_metrics['val_class_precision'] = metrics.get('class_precision', {})
        epoch_metrics['val_class_recall'] = metrics.get('class_recall', {})
        epoch_metrics['val_class_f1'] = metrics.get('class_f1', {})
        epoch_metrics['val_class_precision_12'] = metrics.get('class_precision_12', {})
        epoch_metrics['val_class_recall_12'] = metrics.get('class_recall_12', {})
        epoch_metrics['val_class_f1_12'] = metrics.get('class_f1_12', {})
        epoch_metrics['val_class_tss_12'] = metrics.get('class_tss_12', {})
        epoch_metrics['val_confusion_matrix_12'] = metrics.get('confusion_matrix_12', [[0, 0], [0, 0]])
        epoch_metrics['val_confusion_matrix_12_with_background'] = metrics.get('confusion_matrix_12_with_background', [[0, 0, 0], [0, 0, 0]])

        # 诊断：记录验证集预测/真实类别分布，便于判断指标“恒定”是否来自预测塌缩或类别极不平衡
        try:
            num_classes = int(self.config.get('num_classes', 3))
            pred_counts = np.bincount(np.array(all_predictions, dtype=np.int64), minlength=num_classes)
            tgt_counts = np.bincount(np.array(all_targets, dtype=np.int64), minlength=num_classes)
            epoch_metrics['val_pred_class_counts'] = {str(i): int(pred_counts[i]) for i in range(num_classes)}
            epoch_metrics['val_target_class_counts'] = {str(i): int(tgt_counts[i]) for i in range(num_classes)}
        except Exception:
            epoch_metrics['val_pred_class_counts'] = epoch_metrics.get('val_pred_class_counts', {})
            epoch_metrics['val_target_class_counts'] = epoch_metrics.get('val_target_class_counts', {})

        # 记录到wandb
        if self.use_wandb:
            wandb.log({f'val/{k}': v for k, v in epoch_metrics.items() if np.isscalar(v)}, step=self.current_epoch)

        return epoch_metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            num_epochs: Optional[int] = None,
            history: Optional[Dict[str, List[float]]] = None,
            start_epoch: int = 0) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数

        Returns:
            训练历史
        """
        if num_epochs is None:
            num_epochs = self.config['epochs']

        history = history or {
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

        # 同时保存：F1 best（默认 best.pth）与 val_loss best（额外 best_tag=loss）
        best_val_loss = None

        logger.info(f"开始训练，共{num_epochs}个epoch，起始epoch={start_epoch + 1}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.validate(val_loader)

            # 更新学习率（在epoch结束后调用，避免PyTorch警告）
            if self.scheduler is not None and epoch > 0:  # 第一个epoch后开始调度
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # 记录历史
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1'])
            history['val_f1'].append(val_metrics['f1'])
            history['train_f1_all'].append(train_metrics.get('f1_all', 0.0))
            history['val_f1_all'].append(val_metrics.get('f1_all', 0.0))
            history['train_tss'].append(train_metrics.get('tss', 0.0))
            history['val_tss'].append(val_metrics.get('tss', 0.0))
            history['train_tss_all'].append(train_metrics.get('tss_all', 0.0))
            history['val_tss_all'].append(val_metrics.get('tss_all', 0.0))
            history['train_rss'].append(train_metrics.get('rss', 0.0))
            history['val_rss'].append(val_metrics.get('rss', 0.0))
            history['train_rmse'].append(train_metrics.get('rmse', 0.0))
            history['val_rmse'].append(val_metrics.get('rmse', 0.0))
            history['train_iou'].append(train_metrics.get('iou', 0.0))
            history['val_iou'].append(val_metrics.get('iou', 0.0))

            # 记录综合验证指标（用于 early stopping / best model）
            train_metrics['composite_score'] = self._compute_composite_validation_score(train_metrics)
            val_metrics['composite_score'] = self._compute_composite_validation_score(val_metrics)
            history['train_composite_score'].append(train_metrics.get('composite_score', 0.0))
            history['val_composite_score'].append(val_metrics.get('composite_score', 0.0))

            # 打印epoch结果
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc12: {train_metrics['accuracy']:.4f}, "
                f"F1_12: {train_metrics['f1']:.4f}, TSS_12: {train_metrics.get('tss', 0.0):.4f}, "
                f"F1_all: {train_metrics.get('f1_all', 0.0):.4f}, TSS_all: {train_metrics.get('tss_all', 0.0):.4f}, "
                f"RSS: {train_metrics.get('rss', 0.0):.4f}, RMSE: {train_metrics.get('rmse', 0.0):.4f}, IoU: {train_metrics.get('iou', 0.0):.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc12: {val_metrics['accuracy']:.4f}, "
                f"F1_12: {val_metrics['f1']:.4f}, TSS_12: {val_metrics.get('tss', 0.0):.4f}, "
                f"F1_all: {val_metrics.get('f1_all', 0.0):.4f}, TSS_all: {val_metrics.get('tss_all', 0.0):.4f}, "
                f"RSS: {val_metrics.get('rss', 0.0):.4f}, RMSE: {val_metrics.get('rmse', 0.0):.4f}, IoU: {val_metrics.get('iou', 0.0):.4f}, Composite: {val_metrics['composite_score']:.4f}"
            )

            # 保存最佳模型
            is_best = False
            if not self.best_metrics or val_metrics['composite_score'] > self.best_metrics.get('val_composite_score', -1e9):
                self.best_metrics = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_f1': val_metrics['f1'],
                    'train_f1_all': train_metrics.get('f1_all', 0.0),
                    'val_f1_all': val_metrics.get('f1_all', 0.0),
                    'val_composite_score': val_metrics['composite_score']
                }
                is_best = True

            # 计算 val_loss 的 best（越小越好），仅做辅助保存
            is_best_loss = False
            if best_val_loss is None:
                best_val_loss = val_metrics['loss']
                is_best_loss = True
            else:
                mode = getattr(self.early_stopping, "mode", "min")
                min_delta = getattr(self.early_stopping, "min_delta", 0.0)
                if mode == 'min':
                    if val_metrics['loss'] < best_val_loss - min_delta:
                        best_val_loss = val_metrics['loss']
                        is_best_loss = True
                else:  # mode == 'max'
                    if val_metrics['loss'] > best_val_loss + min_delta:
                        best_val_loss = val_metrics['loss']
                        is_best_loss = True

            # 保存检查点
            self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metric_value=val_metrics.get('composite_score', None),
                is_best=is_best,
                additional_info={
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_metrics': self.best_metrics,
                    'config': self.config
                }
            )

            # 保存 val_loss best（文件名：{model_name}_loss_best.pth）
            if is_best_loss:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metric_value=val_metrics.get('loss', None),
                    is_best=True,
                    best_tag="loss",
                    additional_info={
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'best_metrics': self.best_metrics,
                        'config': self.config
                    }
                )

            # 早停检查：改为监控综合验证指标（越大越好）
            if self.early_stopping(val_metrics['composite_score'], epoch, self.model.state_dict()):
                logger.info(f"早停在epoch {epoch + 1}")
                break

        # 记录最佳指标
        logger.info("训练完成！")
        logger.info(f"最佳epoch: {self.best_metrics['epoch'] + 1}")
        logger.info(f"最佳验证F1_12: {self.best_metrics['val_f1']:.4f}")
        logger.info(f"最佳验证F1_all: {self.best_metrics.get('val_f1_all', 0.0):.4f}")
        logger.info(f"最佳验证综合指标: {self.best_metrics.get('val_composite_score', 0.0):.4f}")
        logger.info(f"最佳验证准确率12: {self.best_metrics['val_accuracy']:.4f}")

        # 加载最佳模型
        self.checkpoint_manager.load_best_checkpoint(self.model, self.optimizer)

        # 关闭wandb
        if self.use_wandb:
            wandb.finish()

        return history
