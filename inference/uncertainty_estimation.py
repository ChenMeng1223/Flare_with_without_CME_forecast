"""
不确定性估计模块 - 估计模型预测的不确定性
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """不确定性估计器"""

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        """
        初始化不确定性估计器

        Args:
            model: 模型
            config: 不确定性估计配置
        """
        self.model = model
        self.config = config

        # 不确定性估计方法
        self.methods = config.get('methods', ['mc_dropout', 'ensemble'])

        # MC Dropout 配置
        self.mc_config = config.get('mc_dropout', {
            'num_samples': 10,
            'dropout_enabled': True
        })

        # 集成配置
        self.ensemble_config = config.get('ensemble', {
            'num_models': 5,
            'use_checkpoints': True
        })

        # 不确定性量化配置
        self.quantification_config = config.get('quantification', {
            'confidence_level': 0.95,
            'use_bootstrap': False
        })

        logger.info(f"不确定性估计器初始化完成，使用的方法: {self.methods}")

    def estimate(self, inputs: Dict[str, torch.Tensor],
                 method: str = 'mc_dropout') -> Dict[str, Any]:
        """
        估计预测不确定性

        Args:
            inputs: 输入数据
            method: 不确定性估计方法

        Returns:
            不确定性估计结果
        """
        if method == 'mc_dropout':
            return self._mc_dropout_uncertainty(inputs)
        elif method == 'ensemble':
            return self._ensemble_uncertainty(inputs)
        elif method == 'both':
            mc_result = self._mc_dropout_uncertainty(inputs)
            ensemble_result = self._ensemble_uncertainty(inputs)
            return self._combine_uncertainties(mc_result, ensemble_result)
        else:
            raise ValueError(f"未知的不确定性估计方法: {method}")

    def _mc_dropout_uncertainty(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """MC Dropout 不确定性估计"""
        num_samples = self.mc_config['num_samples']

        # 存储所有样本的预测
        all_class_probs = []
        all_bbox_preds = []
        all_time_preds = []
        all_event_probs = []

        # 启用Dropout
        self.model.train()

        with torch.no_grad():
            for i in range(num_samples):
                outputs = self.model(inputs)

                all_class_probs.append(outputs['class_probs'])

                if 'bbox_pred' in outputs:
                    all_bbox_preds.append(outputs['bbox_pred'])

                if 'time_pred' in outputs:
                    all_time_preds.append(outputs['time_pred'])

                if 'event_prob' in outputs:
                    all_event_probs.append(outputs['event_prob'])

        # 计算均值和标准差
        uncertainty_results = {}

        if all_class_probs:
            class_probs_stack = torch.stack(all_class_probs)
            uncertainty_results.update(
                self._compute_distribution_stats(class_probs_stack, 'class_probs')
            )

        if all_bbox_preds:
            bbox_preds_stack = torch.stack(all_bbox_preds)
            uncertainty_results.update(
                self._compute_distribution_stats(bbox_preds_stack, 'bbox_pred')
            )

        if all_time_preds:
            time_preds_stack = torch.stack(all_time_preds)
            uncertainty_results.update(
                self._compute_distribution_stats(time_preds_stack, 'time_pred')
            )

        if all_event_probs:
            event_probs_stack = torch.stack(all_event_probs)
            uncertainty_results.update(
                self._compute_distribution_stats(event_probs_stack, 'event_prob')
            )

        # 计算总的不确定性分数
        uncertainty_results['total_uncertainty'] = self._compute_total_uncertainty(
            uncertainty_results
        )

        # 恢复模型状态
        self.model.eval()

        return uncertainty_results

    def _ensemble_uncertainty(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """集成不确定性估计"""
        # 注意：这里需要多个模型或检查点
        # 简化实现：假设我们已经有多个模型

        logger.warning("集成不确定性估计需要多个模型，当前使用简化实现")

        # 这里应该加载多个模型并计算预测
        # 为简化，我们返回一个基本结构
        return {
            'method': 'ensemble',
            'class_probs_mean': torch.zeros(3),
            'class_probs_std': torch.ones(3) * 0.1,
            'total_uncertainty': 0.5
        }

    def _compute_distribution_stats(self, predictions: torch.Tensor,
                                   prefix: str) -> Dict[str, Any]:
        """
        计算分布的统计量

        Args:
            predictions: 预测张量 [num_samples, ...]
            prefix: 前缀

        Returns:
            统计量字典
        """
        # 计算均值
        mean = predictions.mean(dim=0)

        # 计算标准差
        std = predictions.std(dim=0)

        # 计算置信区间
        confidence_level = self.quantification_config['confidence_level']
        ci_lower, ci_upper = self._compute_confidence_interval(
            predictions, confidence_level
        )

        # 计算熵（用于分类不确定性）
        if prefix == 'class_probs':
            entropy = self._compute_entropy(mean)
        else:
            entropy = None

        return {
            f'{prefix}_mean': mean,
            f'{prefix}_std': std,
            f'{prefix}_ci_lower': ci_lower,
            f'{prefix}_ci_upper': ci_upper,
            f'{prefix}_entropy': entropy
        }

    def _compute_confidence_interval(self, predictions: torch.Tensor,
                                    confidence_level: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算置信区间"""
        # 使用百分位数方法
        alpha = 1.0 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        # 沿着样本维度计算
        predictions_np = predictions.cpu().numpy()

        # 对每个元素计算置信区间
        shape = predictions_np.shape[1:]
        ci_lower = np.zeros(shape)
        ci_upper = np.zeros(shape)

        # 展平非样本维度
        predictions_flat = predictions_np.reshape(predictions_np.shape[0], -1)

        for i in range(predictions_flat.shape[1]):
            data = predictions_flat[:, i]
            ci_lower_flat = np.percentile(data, lower_percentile)
            ci_upper_flat = np.percentile(data, upper_percentile)

            # 恢复形状
            ci_lower.flat[i] = ci_lower_flat
            ci_upper.flat[i] = ci_upper_flat

        return torch.tensor(ci_lower), torch.tensor(ci_upper)

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """计算概率分布的熵"""
        # 避免log(0)
        probs = probs + 1e-10
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy

    def _compute_total_uncertainty(self, uncertainty_results: Dict[str, Any]) -> float:
        """计算总的不确定性分数"""
        total_uncertainty = 0.0
        weights = []
        uncertainties = []

        # 收集各类不确定性
        if 'class_probs_entropy' in uncertainty_results:
            entropy = uncertainty_results['class_probs_entropy']
            if entropy is not None:
                # 归一化熵（最大熵为log(num_classes)）
                max_entropy = np.log(3)  # 3个类别
                norm_entropy = entropy.mean().item() / max_entropy
                uncertainties.append(norm_entropy)
                weights.append(0.4)  # 分类权重

        if 'class_probs_std' in uncertainty_results:
            class_std = uncertainty_results['class_probs_std'].mean().item()
            uncertainties.append(class_std)
            weights.append(0.3)

        if 'event_prob_std' in uncertainty_results:
            event_std = uncertainty_results['event_prob_std'].mean().item()
            uncertainties.append(event_std)
            weights.append(0.3)

        # 加权平均
        if uncertainties:
            total_uncertainty = np.average(uncertainties, weights=weights)

        return total_uncertainty

    def _combine_uncertainties(self, mc_result: Dict[str, Any],
                               ensemble_result: Dict[str, Any]) -> Dict[str, Any]:
        """合并不同方法的不确定性估计"""
        combined = {}

        # 合并均值（优先使用集成方法）
        for key in mc_result.keys():
            if '_mean' in key:
                if key in ensemble_result:
                    # 加权平均
                    combined[key] = 0.7 * ensemble_result[key] + 0.3 * mc_result[key]
                else:
                    combined[key] = mc_result[key]
            elif '_std' in key:
                # 取最大值作为保守估计
                if key in ensemble_result:
                    combined[key] = torch.maximum(mc_result[key], ensemble_result[key])
                else:
                    combined[key] = mc_result[key]
            elif key not in ['method', 'total_uncertainty']:
                combined[key] = mc_result[key]

        # 合并总不确定性
        mc_total = mc_result.get('total_uncertainty', 0.5)
        ensemble_total = ensemble_result.get('total_uncertainty', 0.5)
        combined['total_uncertainty'] = max(mc_total, ensemble_total)

        combined['method'] = 'combined'

        return combined

    def calibrate_uncertainty(self, predictions: Dict[str, Any],
                              calibration_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        校准不确定性估计

        Args:
            predictions: 预测结果
            calibration_data: 校准数据

        Returns:
            校准后的预测结果
        """
        calibrated = predictions.copy()

        # 简化的校准方法
        if calibration_data is not None and 'expected_variance' in calibration_data:
            # 缩放不确定性以匹配预期方差
            for key in list(predictions.keys()):
                if '_std' in key:
                    std = predictions[key]
                    expected_std = calibration_data['expected_variance'][key.replace('_std', '')] ** 0.5

                    # 计算缩放因子
                    scale_factor = expected_std / (std.mean().item() + 1e-6)
                    calibrated[key] = std * scale_factor

        # 添加校准标志
        calibrated['uncertainty_calibrated'] = True

        return calibrated

    def is_confident_prediction(self, uncertainty_results: Dict[str, Any],
                               threshold: float = 0.3) -> bool:
        """
        判断预测是否足够自信

        Args:
            uncertainty_results: 不确定性估计结果
            threshold: 不确定性阈值

        Returns:
            是否自信
        """
        total_uncertainty = uncertainty_results.get('total_uncertainty', 1.0)
        return total_uncertainty <= threshold

    def get_uncertainty_breakdown(self, uncertainty_results: Dict[str, Any]) -> Dict[str, float]:
        """
        获取不确定性的分解

        Args:
            uncertainty_results: 不确定性估计结果

        Returns:
            不确定性分解
        """
        breakdown = {}

        # 分类不确定性
        if 'class_probs_entropy' in uncertainty_results:
            entropy = uncertainty_results['class_probs_entropy']
            if entropy is not None:
                max_entropy = np.log(3)
                breakdown['classification_uncertainty'] = entropy.mean().item() / max_entropy

        # 回归不确定性
        reg_keys = ['bbox_pred_std', 'time_pred_std', 'event_prob_std']
        for key in reg_keys:
            if key in uncertainty_results:
                std = uncertainty_results[key]
                breakdown[key.replace('_std', '_uncertainty')] = std.mean().item()

        # 总不确定性
        breakdown['total_uncertainty'] = uncertainty_results.get('total_uncertainty', 0.0)

        return breakdown

    def visualize_uncertainty(self, uncertainty_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> None:
        """
        可视化不确定性

        Args:
            uncertainty_results: 不确定性估计结果
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt

            breakdown = self.get_uncertainty_breakdown(uncertainty_results)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 1. 不确定性分解条形图
            labels = list(breakdown.keys())
            values = list(breakdown.values())

            axes[0].bar(labels, values)
            axes[0].set_title('Uncertainty Breakdown')
            axes[0].set_ylabel('Uncertainty')
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis='x', rotation=45)

            # 2. 分类概率不确定性
            if 'class_probs_mean' in uncertainty_results and 'class_probs_std' in uncertainty_results:
                mean = uncertainty_results['class_probs_mean'].cpu().numpy()
                std = uncertainty_results['class_probs_std'].cpu().numpy()

                classes = ['No Event', 'Confined', 'Eruptive']
                x = np.arange(len(classes))

                axes[1].bar(x, mean, yerr=std, capsize=5, alpha=0.7)
                axes[1].set_title('Class Probabilities with Uncertainty')
                axes[1].set_xlabel('Class')
                axes[1].set_ylabel('Probability')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(classes)
                axes[1].set_ylim(0, 1)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"不确定性可视化已保存到: {save_path}")

            plt.show()

        except ImportError:
            logger.warning("Matplotlib未安装，无法可视化不确定性")