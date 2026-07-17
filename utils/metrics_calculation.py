"""
指标计算模块
用于计算和跟踪模型性能指标
定位损失用iou
分类损失用precision、recall、f1等
回归损失用了R²分数、对称平均绝对百分比误差 (SMAPE)
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
import sklearn.metrics as sk_metrics
from sklearn.preprocessing import label_binarize
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def _safe_tss(tp: float, fp: float, fn: float, tn: float) -> float:
    """计算单类 one-vs-rest TSS。"""
    hit_rate_den = tp + fn
    false_alarm_den = fp + tn
    if hit_rate_den <= 0 or false_alarm_den <= 0:
        return 0.0
    hit_rate = tp / hit_rate_den
    false_alarm_rate = fp / false_alarm_den
    return float(hit_rate - false_alarm_rate)


class ConfusionMatrix:
    """混淆矩阵类"""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        初始化混淆矩阵

        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        """重置混淆矩阵"""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        更新混淆矩阵

        Args:
            predictions: 预测标签 (N,)
            targets: 真实标签 (N,)
        """
        for pred, target in zip(predictions, targets):
            self.matrix[target][pred] += 1
        self.total_samples += len(predictions)

    def compute_metrics(self) -> Dict[str, float]:
        """
        计算基于混淆矩阵的指标

        Returns:
            指标字典
        """
        metrics = {}

        # 计算每个类别的指标
        tp = np.diag(self.matrix)  # 真正例
        fp = np.sum(self.matrix, axis=0) - tp  # 假正例
        fn = np.sum(self.matrix, axis=1) - tp  # 假负例
        tn = np.sum(self.matrix) - (fp + fn + tp)  # 真负例

        # 准确率
        total = np.sum(self.matrix)
        if total == 0:
            # 没有样本时避免除零，返回全 0 指标
            metrics['overall_accuracy'] = 0.0
            metrics['macro_precision'] = 0.0
            metrics['macro_recall'] = 0.0
            metrics['macro_f1'] = 0.0
            metrics['weighted_precision'] = 0.0
            metrics['weighted_recall'] = 0.0
            metrics['weighted_f1'] = 0.0
            metrics['macro_tss'] = 0.0
            metrics['weighted_tss'] = 0.0
            for i, class_name in enumerate(self.class_names):
                metrics[f'{class_name}_precision'] = 0.0
                metrics[f'{class_name}_recall'] = 0.0
                metrics[f'{class_name}_f1'] = 0.0
                metrics[f'{class_name}_tss'] = 0.0
                metrics[f'{class_name}_support'] = 0
            return metrics

        metrics['overall_accuracy'] = np.sum(tp) / total

        # 每个类别的精确率、召回率、F1分数、TSS
        precision_per_class = np.zeros(self.num_classes)
        recall_per_class = np.zeros(self.num_classes)
        f1_per_class = np.zeros(self.num_classes)
        tss_per_class = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            # 精确率
            if tp[i] + fp[i] > 0:
                precision_per_class[i] = tp[i] / (tp[i] + fp[i])
            else:
                precision_per_class[i] = 0.0

            # 召回率
            if tp[i] + fn[i] > 0:
                recall_per_class[i] = tp[i] / (tp[i] + fn[i])
            else:
                recall_per_class[i] = 0.0

            # F1分数
            if precision_per_class[i] + recall_per_class[i] > 0:
                f1_per_class[i] = 2 * (precision_per_class[i] * recall_per_class[i]) / (
                            precision_per_class[i] + recall_per_class[i])
            else:
                f1_per_class[i] = 0.0

            # TSS 分数
            tss_per_class[i] = _safe_tss(tp[i], fp[i], fn[i], tn[i])

        # 宏平均
        metrics['macro_precision'] = np.mean(precision_per_class)
        metrics['macro_recall'] = np.mean(recall_per_class)
        metrics['macro_f1'] = np.mean(f1_per_class)
        metrics['macro_tss'] = np.mean(tss_per_class)

        # 微平均（加权平均）
        class_weights = np.sum(self.matrix, axis=1) / np.sum(self.matrix)
        metrics['weighted_precision'] = np.sum(precision_per_class * class_weights)
        metrics['weighted_recall'] = np.sum(recall_per_class * class_weights)
        metrics['weighted_f1'] = np.sum(f1_per_class * class_weights)
        metrics['weighted_tss'] = np.sum(tss_per_class * class_weights)

        # 添加每个类别的指标
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision_per_class[i]
            metrics[f'{class_name}_recall'] = recall_per_class[i]
            metrics[f'{class_name}_f1'] = f1_per_class[i]
            metrics[f'{class_name}_tss'] = tss_per_class[i]
            metrics[f'{class_name}_support'] = int(tp[i] + fn[i])

        return metrics

    def get_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return self.matrix

    def get_normalized_matrix(self) -> np.ndarray:
        """获取归一化的混淆矩阵（按行归一化）"""
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        return self.matrix / row_sums


class ROC_AUC_Calculator:
    """ROC曲线和AUC计算器"""

    def __init__(self, num_classes: int):
        """
        初始化ROC AUC计算器

        Args:
            num_classes: 类别数量
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """重置计算器"""
        self.all_targets = []
        self.all_probs = []

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        更新数据

        Args:
            predictions: 预测概率 (N, num_classes)
            targets: 真实标签 (N,)
        """
        self.all_targets.append(targets)
        self.all_probs.append(predictions)

    def compute_metrics(self) -> Dict[str, Any]:
        """
        计算ROC曲线和AUC指标

        Returns:
            包含ROC曲线数据和AUC值的字典
        """
        if not self.all_targets:
            return {}

        # 合并所有批次的数据
        all_targets = np.concatenate(self.all_targets)
        all_probs = np.concatenate(self.all_probs)

        # 二值化标签（用于多类别ROC）
        if self.num_classes > 2:
            y_true_bin = label_binarize(all_targets, classes=range(self.num_classes))
        else:
            y_true_bin = all_targets.reshape(-1, 1)

        results = {
            'auc_scores': {},
            'roc_curves': {}
        }

        # 计算每个类别的ROC曲线和AUC
        for i in range(self.num_classes):
            if self.num_classes > 2:
                y_true_class = y_true_bin[:, i]
                y_score_class = all_probs[:, i]
            else:
                y_true_class = y_true_bin
                y_score_class = all_probs[:, 1]  # 正类的概率

            # 计算ROC曲线
            fpr, tpr, thresholds = sk_metrics.roc_curve(y_true_class, y_score_class)
            auc_score = sk_metrics.auc(fpr, tpr)

            results['auc_scores'][f'class_{i}'] = auc_score
            results['roc_curves'][f'class_{i}'] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }

        # 计算宏平均和微平均AUC
        if self.num_classes > 2:
            # 宏平均AUC
            auc_scores = list(results['auc_scores'].values())
            results['macro_auc'] = np.mean(auc_scores)

            # 微平均AUC
            y_true_flat = y_true_bin.ravel()
            y_score_flat = all_probs.ravel()
            if len(np.unique(y_true_flat)) > 1:  # 确保有正负样本
                results['micro_auc'] = sk_metrics.roc_auc_score(y_true_flat, y_score_flat)
            else:
                results['micro_auc'] = 0.0

        return results


class PrecisionRecallCalculator:
    """精确率-召回率计算器"""

    def __init__(self, num_classes: int):
        """
        初始化精确率-召回率计算器

        Args:
            num_classes: 类别数量
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """重置计算器"""
        self.all_targets = []
        self.all_probs = []

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        更新数据

        Args:
            predictions: 预测概率 (N, num_classes)
            targets: 真实标签 (N,)
        """
        self.all_targets.append(targets)
        self.all_probs.append(predictions)

    def compute_metrics(self) -> Dict[str, Any]:
        """
        计算精确率-召回率曲线和AP

        Returns:
            包含PR曲线数据和AP值的字典
        """
        if not self.all_targets:
            return {}

        # 合并所有批次的数据
        all_targets = np.concatenate(self.all_targets)
        all_probs = np.concatenate(self.all_probs)

        results = {
            'ap_scores': {},
            'pr_curves': {},
            'average_precision': {}
        }

        # 计算每个类别的精确率-召回率曲线
        for i in range(self.num_classes):
            y_true_class = (all_targets == i).astype(int)
            y_score_class = all_probs[:, i]

            # 计算精确率-召回率曲线
            precision, recall, thresholds = sk_metrics.precision_recall_curve(
                y_true_class, y_score_class
            )

            # 计算平均精确率 (AP)
            ap_score = sk_metrics.average_precision_score(y_true_class, y_score_class)

            results['ap_scores'][f'class_{i}'] = ap_score
            results['pr_curves'][f'class_{i}'] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }

        # 计算宏平均AP
        ap_scores = list(results['ap_scores'].values())
        results['macro_ap'] = np.mean(ap_scores)

        return results


class RegressionMetrics:
    """回归指标计算器"""

    @staticmethod
    def compute_metrics(predictions: np.ndarray,
                        targets: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算回归指标

        Args:
            predictions: 预测值
            targets: 真实值
            mask: 掩码（可选，True表示有效样本）

        Returns:
            回归指标字典
        """
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]

        if len(predictions) == 0:
            return {}

        # 计算基本指标
        mae = np.mean(np.abs(predictions - targets))  # 平均绝对误差
        mse = np.mean((predictions - targets) ** 2)  # 均方误差
        rmse = np.sqrt(mse)  # 均方根误差

        # 计算R²分数
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # 计算皮尔逊相关系数
        if len(predictions) > 1:
            pearson_corr, _ = stats.pearsonr(predictions.flatten(), targets.flatten())
            spearman_corr, _ = stats.spearmanr(predictions.flatten(), targets.flatten())
        else:
            pearson_corr = 0.0
            spearman_corr = 0.0

        # 计算对称平均绝对百分比误差 (SMAPE)
        denominator = (np.abs(predictions) + np.abs(targets)) / 2
        denominator[denominator == 0] = 1e-8  # 避免除零
        smape = np.mean(np.abs(predictions - targets) / denominator) * 100

        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson_corr': float(pearson_corr),
            'spearman_corr': float(spearman_corr),
            'smape': float(smape),
            'num_samples': int(len(predictions))
        }

        return metrics


class BoundingBoxMetrics:
    """边界框指标计算器"""

    @staticmethod
    def iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """
        计算交并比 (IoU)

        Args:
            box1: 边界框1 [N, 4] (x1, y1, x2, y2)
            box2: 边界框2 [M, 4] (x1, y1, x2, y2)

        Returns:
            IoU矩阵 [N, M]
        """
        # 确保坐标格式正确
        box1 = np.atleast_2d(box1)
        box2 = np.atleast_2d(box2)

        # 计算交集区域
        inter_x1 = np.maximum(box1[:, 0:1], box2[:, 0])
        inter_y1 = np.maximum(box1[:, 1:2], box2[:, 1])
        inter_x2 = np.minimum(box1[:, 2:3], box2[:, 2])
        inter_y2 = np.minimum(box1[:, 3:4], box2[:, 3])

        inter_width = np.maximum(inter_x2 - inter_x1, 0)
        inter_height = np.maximum(inter_y2 - inter_y1, 0)
        inter_area = inter_width * inter_height

        # 计算并集区域
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        union_area = area1[:, None] + area2 - inter_area

        # 计算IoU
        iou = inter_area / (union_area + 1e-8)

        return iou

    @staticmethod
    def compute_metrics(predictions: np.ndarray,
                        targets: np.ndarray,
                        iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        计算边界框检测指标

        Args:
            predictions: 预测边界框 [N, 4]
            targets: 真实边界框 [M, 4]
            iou_threshold: IoU阈值

        Returns:
            检测指标字典
        """
        if len(predictions) == 0 or len(targets) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'average_iou': 0.0,
                'num_predictions': 0,
                'num_targets': 0
            }

        # 计算IoU矩阵
        iou_matrix = BoundingBoxMetrics.iou(predictions, targets)

        # 为每个预测找到最佳匹配的真实框
        best_iou = np.max(iou_matrix, axis=1)
        best_match_idx = np.argmax(iou_matrix, axis=1)

        # 为每个真实框找到最佳匹配的预测
        best_iou_per_target = np.max(iou_matrix, axis=0)

        # 计算检测指标
        true_positives = np.sum(best_iou >= iou_threshold)
        false_positives = len(predictions) - true_positives
        false_negatives = len(targets) - np.sum(best_iou_per_target >= iou_threshold)

        # 精确率、召回率、F1分数
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # 平均IoU
        average_iou = np.mean(best_iou_per_target) if len(best_iou_per_target) > 0 else 0.0

        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'average_iou': float(average_iou),
            'num_predictions': int(len(predictions)),
            'num_targets': int(len(targets)),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }

        return metrics

    @staticmethod
    def _optimal_one_to_one_assignment(iou_matrix: np.ndarray) -> List[tuple[int, int, float]]:
        """单样本内做一对一最优匹配，最大化 IoU 总和。"""
        iou_matrix = np.asarray(iou_matrix, dtype=np.float32)
        if iou_matrix.size == 0:
            return []

        num_predictions, num_targets = iou_matrix.shape
        if num_predictions == 0 or num_targets == 0:
            return []

        if num_predictions <= num_targets:
            working_matrix = iou_matrix
            row_is_prediction = True
        else:
            working_matrix = iou_matrix.T
            row_is_prediction = False

        num_rows, num_cols = working_matrix.shape
        dp: Dict[int, tuple[float, List[tuple[int, int]]]] = {0: (0.0, [])}

        for row_idx in range(num_rows):
            next_dp: Dict[int, tuple[float, List[tuple[int, int]]]] = {}
            for used_mask, (score, pairs) in dp.items():
                for col_idx in range(num_cols):
                    if used_mask & (1 << col_idx):
                        continue
                    next_mask = used_mask | (1 << col_idx)
                    next_score = score + float(working_matrix[row_idx, col_idx])
                    best = next_dp.get(next_mask)
                    if best is None or next_score > best[0]:
                        next_dp[next_mask] = (next_score, pairs + [(row_idx, col_idx)])
            dp = next_dp

        _, best_pairs = max(dp.values(), key=lambda item: item[0])
        matched_pairs: List[tuple[int, int, float]] = []
        for row_idx, col_idx in best_pairs:
            if row_is_prediction:
                pred_idx, target_idx = row_idx, col_idx
            else:
                pred_idx, target_idx = col_idx, row_idx
            matched_pairs.append((pred_idx, target_idx, float(iou_matrix[pred_idx, target_idx])))
        return matched_pairs

    @staticmethod
    def compute_metrics_hungarian(predictions: np.ndarray,
                                  targets: np.ndarray,
                                  iou_threshold: float = 0.5) -> Dict[str, float]:
        """单样本内基于一对一最优匹配计算定位指标。"""
        predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 4)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, 4)

        num_predictions = int(len(predictions))
        num_targets = int(len(targets))
        if num_predictions == 0 or num_targets == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'average_iou': 0.0,
                'num_predictions': num_predictions,
                'num_targets': num_targets,
                'true_positives': 0,
                'false_positives': num_predictions,
                'false_negatives': num_targets,
                'matched_ious': [],
            }

        iou_matrix = BoundingBoxMetrics.iou(predictions, targets)
        matched_pairs = BoundingBoxMetrics._optimal_one_to_one_assignment(iou_matrix)
        matched_ious = np.asarray([pair[2] for pair in matched_pairs], dtype=np.float32)

        true_positives = int(np.sum(matched_ious >= iou_threshold))
        false_positives = int(num_predictions - true_positives)
        false_negatives = int(num_targets - true_positives)

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        target_ious = np.zeros(num_targets, dtype=np.float32)
        for _, target_idx, matched_iou in matched_pairs:
            target_ious[target_idx] = matched_iou
        average_iou = float(target_ious.mean()) if num_targets > 0 else 0.0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'average_iou': average_iou,
            'num_predictions': num_predictions,
            'num_targets': num_targets,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'matched_ious': matched_ious.tolist(),
        }

    @staticmethod
    def compute_metrics_per_sample(predictions_list: List[np.ndarray],
                                   targets_list: List[np.ndarray],
                                   iou_threshold: float = 0.5) -> Dict[str, float]:
        """按样本分别做一对一最优匹配，再汇总整个数据集的定位指标。"""
        if len(predictions_list) != len(targets_list):
            raise ValueError(
                f"predictions_list 与 targets_list 长度不一致: "
                f"{len(predictions_list)} vs {len(targets_list)}"
            )

        total_predictions = 0
        total_targets = 0
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        sample_average_ious: List[float] = []
        matched_ious_all: List[float] = []

        for predictions, targets in zip(predictions_list, targets_list):
            sample_metrics = BoundingBoxMetrics.compute_metrics_hungarian(
                predictions=predictions,
                targets=targets,
                iou_threshold=iou_threshold,
            )
            total_predictions += int(sample_metrics['num_predictions'])
            total_targets += int(sample_metrics['num_targets'])
            total_true_positives += int(sample_metrics['true_positives'])
            total_false_positives += int(sample_metrics['false_positives'])
            total_false_negatives += int(sample_metrics['false_negatives'])
            sample_average_ious.append(float(sample_metrics['average_iou']))
            matched_ious_all.extend(sample_metrics.get('matched_ious', []))

        precision = total_true_positives / (total_true_positives + total_false_positives + 1e-8)
        recall = total_true_positives / (total_true_positives + total_false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        average_iou = float(np.mean(sample_average_ious)) if sample_average_ious else 0.0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'average_iou': average_iou,
            'num_predictions': int(total_predictions),
            'num_targets': int(total_targets),
            'true_positives': int(total_true_positives),
            'false_positives': int(total_false_positives),
            'false_negatives': int(total_false_negatives),
            'num_samples': int(len(predictions_list)),
            'matched_ious': matched_ious_all,
        }


class MetricsTracker:
    """指标跟踪器（综合所有指标）"""

    def __init__(self, num_classes: int = 3, class_names: Optional[List[str]] = None):
        """
        初始化指标跟踪器

        Args:
            num_classes: 类别数量
            class_names: 类别名称
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]

        # 初始化各个计算器
        self.confusion_matrix = ConfusionMatrix(num_classes, class_names)
        self.roc_auc_calculator = ROC_AUC_Calculator(num_classes)
        self.pr_calculator = PrecisionRecallCalculator(num_classes)

        # 存储历史指标
        self.history = {
            'classification': [],
            'regression': [],
            'detection': [],
            'temporal': []
        }

    def reset(self) -> None:
        """重置所有指标"""
        self.confusion_matrix.reset()
        self.roc_auc_calculator.reset()
        self.pr_calculator.reset()

    def update_classification(self,
                              predictions: Union[np.ndarray, torch.Tensor],
                              targets: Union[np.ndarray, torch.Tensor],
                              probs: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
        """
        更新分类指标

        Args:
            predictions: 预测标签
            targets: 真实标签
            probs: 预测概率（可选）
        """
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        # 更新混淆矩阵
        self.confusion_matrix.update(predictions, targets)

        # 如果有概率，更新ROC和PR计算器
        if probs is not None:
            self.roc_auc_calculator.update(probs, targets)
            self.pr_calculator.update(probs, targets)

    def update_regression(self,
                          predictions: Union[np.ndarray, torch.Tensor],
                          targets: Union[np.ndarray, torch.Tensor],
                          metric_name: str = 'regression') -> Dict[str, float]:
        """
        更新回归指标

        Args:
            predictions: 预测值
            targets: 真实值
            metric_name: 指标名称

        Returns:
            回归指标
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        metrics = RegressionMetrics.compute_metrics(predictions, targets)

        # 存储到历史
        self.history[metric_name].append(metrics)

        return metrics

    def update_detection(self,
                         predictions: Union[np.ndarray, torch.Tensor],
                         targets: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        更新检测指标（边界框）

        Args:
            predictions: 预测边界框
            targets: 真实边界框

        Returns:
            检测指标
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        metrics = BoundingBoxMetrics.compute_metrics(predictions, targets)

        # 存储到历史
        self.history['detection'].append(metrics)

        return metrics

    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        计算所有指标

        Returns:
            包含所有指标的字典
        """
        results = {}

        # 分类指标
        confusion_metrics = self.confusion_matrix.compute_metrics()
        results['classification'] = confusion_metrics

        # ROC曲线和AUC
        roc_auc_metrics = self.roc_auc_calculator.compute_metrics()
        if roc_auc_metrics:
            results['roc_auc'] = roc_auc_metrics

        # 精确率-召回率曲线和AP
        pr_metrics = self.pr_calculator.compute_metrics()
        if pr_metrics:
            results['precision_recall'] = pr_metrics

        # 汇总关键指标
        results['summary'] = {
            'accuracy': confusion_metrics.get('overall_accuracy', 0.0),
            'macro_f1': confusion_metrics.get('macro_f1', 0.0),
            'weighted_f1': confusion_metrics.get('weighted_f1', 0.0),
            'macro_tss': confusion_metrics.get('macro_tss', 0.0),
            'weighted_tss': confusion_metrics.get('weighted_tss', 0.0),
        }

        if 'roc_auc' in results:
            if 'macro_auc' in results['roc_auc']:
                results['summary']['macro_auc'] = results['roc_auc']['macro_auc']

        return results

    def get_history(self, metric_type: str) -> List[Dict[str, float]]:
        """
        获取历史指标

        Args:
            metric_type: 指标类型 ('classification', 'regression', 'detection', 'temporal')

        Returns:
            历史指标列表
        """
        return self.history.get(metric_type, [])

    def get_average_metrics(self, metric_type: str) -> Dict[str, float]:
        """
        获取平均指标

        Args:
            metric_type: 指标类型

        Returns:
            平均指标字典
        """
        history = self.get_history(metric_type)
        if not history:
            return {}

        # 计算所有历史指标的平均值
        avg_metrics = {}
        for key in history[0].keys():
            values = [metrics[key] for metrics in history if key in metrics]
            if values:
                avg_metrics[key] = np.mean(values)

        return avg_metrics


def calculate_metrics(predictions: np.ndarray,
                      targets: np.ndarray,
                      num_classes: int = 3,
                      probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算分类指标（简化接口）

    Args:
        predictions: 预测标签
        targets: 真实标签
        num_classes: 类别数量
        probs: 预测概率（可选）

    Returns:
        分类指标字典
    """
    # 创建跟踪器
    tracker = MetricsTracker(num_classes)

    # 更新指标
    tracker.update_classification(predictions, targets, probs)

    # 计算指标
    results = tracker.compute_all_metrics()

    # 提取关键指标
    metrics = {
        'accuracy': results['classification']['overall_accuracy'],
        'macro_precision': results['classification']['macro_precision'],
        'macro_recall': results['classification']['macro_recall'],
        'macro_f1': results['classification']['macro_f1'],
        'macro_tss': results['classification']['macro_tss'],
        'weighted_precision': results['classification']['weighted_precision'],
        'weighted_recall': results['classification']['weighted_recall'],
        'weighted_f1': results['classification']['weighted_f1'],
        'weighted_tss': results['classification']['weighted_tss']
    }

    # 添加AUC指标（如果有）
    if 'roc_auc' in results:
        if 'macro_auc' in results['roc_auc']:
            metrics['macro_auc'] = results['roc_auc']['macro_auc']

    return metrics


if __name__ == '__main__':
    # 测试代码
    print("测试指标计算模块...")

    # 生成测试数据
    np.random.seed(42)
    num_samples = 1000
    num_classes = 3

    # 生成真实标签和预测标签
    targets = np.random.randint(0, num_classes, num_samples)
    predictions = np.random.randint(0, num_classes, num_samples)

    # 生成预测概率
    probs = np.random.rand(num_samples, num_classes)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # 测试简化接口
    metrics = calculate_metrics(predictions, targets, num_classes, probs)
    print("\n简化接口结果:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 测试完整跟踪器
    tracker = MetricsTracker(num_classes)
    tracker.update_classification(predictions, targets, probs)

    all_metrics = tracker.compute_all_metrics()
    print("\n完整跟踪器 - 分类指标:")
    for key, value in all_metrics['classification'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

    # 测试回归指标
    reg_pred = np.random.randn(100)
    reg_target = reg_pred + np.random.randn(100) * 0.1
    reg_metrics = RegressionMetrics.compute_metrics(reg_pred, reg_target)
    print("\n回归指标:")
    for key, value in reg_metrics.items():
        print(f"{key}: {value:.4f}")

    # 测试检测指标
    bbox_pred = np.random.rand(10, 4) * 100
    bbox_target = bbox_pred + np.random.randn(10, 4) * 10
    det_metrics = BoundingBoxMetrics.compute_metrics(bbox_pred, bbox_target)
    print("\n检测指标:")
    for key, value in det_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n测试完成!")
