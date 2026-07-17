"""
可视化模块
用于可视化训练结果、模型性能、数据分布等
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import logging
import torch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import warnings

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def save_figure(fig: Figure,
                filepath: Union[str, Path],
                dpi: int = 300,
                bbox_inches: str = 'tight',
                format: str = 'png') -> None:
    """
    保存图形到文件

    Args:
        fig: matplotlib图形对象
        filepath: 文件路径
        dpi: 分辨率
        bbox_inches: 边界框设置
        format: 文件格式
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, format=format)
        logger.info(f"图形已保存到: {filepath}")
    except Exception as e:
        logger.error(f"保存图形失败 {filepath}: {e}")


def plot_confusion_matrix(confusion_matrix: np.ndarray,
                          class_names: List[str],
                          title: str = "混淆矩阵",
                          normalize: bool = True,
                          cmap: str = 'Blues',
                          figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    绘制混淆矩阵

    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
        title: 图形标题
        normalize: 是否归一化
        cmap: 颜色映射
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 归一化混淆矩阵
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        matrix = confusion_matrix / row_sums
        fmt = '.2f'
    else:
        matrix = confusion_matrix
        fmt = 'd'

    # 绘制热力图
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': '比例' if normalize else '数量'})

    # 设置标签
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_roc_curves(fpr_dict: Dict[str, np.ndarray],
                    tpr_dict: Dict[str, np.ndarray],
                    auc_dict: Dict[str, float],
                    title: str = "ROC曲线",
                    figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    绘制ROC曲线

    Args:
        fpr_dict: 假正率字典 {类别: fpr数组}
        tpr_dict: 真正率字典 {类别: tpr数组}
        auc_dict: AUC值字典 {类别: AUC值}
        title: 图形标题
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制每个类别的ROC曲线
    for class_name in fpr_dict.keys():
        fpr = fpr_dict[class_name]
        tpr = tpr_dict[class_name]
        auc = auc_dict.get(class_name, 0.0)

        ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {auc:.3f})')

    # 绘制随机猜测线
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')

    # 设置图形属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率 (FPR)')
    ax.set_ylabel('真正率 (TPR)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_precision_recall_curves(precision_dict: Dict[str, np.ndarray],
                                 recall_dict: Dict[str, np.ndarray],
                                 ap_dict: Dict[str, float],
                                 title: str = "精确率-召回率曲线",
                                 figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    绘制精确率-召回率曲线

    Args:
        precision_dict: 精确率字典 {类别: precision数组}
        recall_dict: 召回率字典 {类别: recall数组}
        ap_dict: 平均精确率字典 {类别: AP值}
        title: 图形标题
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制每个类别的PR曲线
    for class_name in precision_dict.keys():
        precision = precision_dict[class_name]
        recall = recall_dict[class_name]
        ap = ap_dict.get(class_name, 0.0)

        ax.plot(recall, precision, lw=2, label=f'{class_name} (AP = {ap:.3f})')

    # 设置图形属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('召回率')
    ax.set_ylabel('精确率')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_history(history: Dict[str, List[float]],
                          metrics: Optional[List[str]] = None,
                          title: str = "训练历史",
                          figsize: Tuple[int, int] = (12, 8)) -> Figure:
    """
    绘制训练历史

    Args:
        history: 训练历史字典 {指标名称: 值列表}
        metrics: 要绘制的指标列表（None表示绘制所有）
        title: 图形标题
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    if not history:
        logger.warning("训练历史为空")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, '无训练历史数据',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12)
        return fig

    # 确定要绘制的指标
    if metrics is None:
        metrics = list(history.keys())

    # 创建子图
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)

    if num_metrics == 1:
        axes = [axes]

    # 绘制每个指标
    for i, metric in enumerate(metrics):
        ax = axes[i]

        if metric in history:
            values = history[metric]
            epochs = range(1, len(values) + 1)

            # 绘制训练曲线
            ax.plot(epochs, values, 'b-', label=f'训练{metric}', linewidth=2)

            # 如果有验证指标
            val_metric = f'val_{metric}'
            if val_metric in history:
                val_values = history[val_metric]
                ax.plot(epochs, val_values, 'r-', label=f'验证{metric}', linewidth=2)

            # 标记最佳值
            if val_metric in history:
                best_idx = np.argmax(val_values) if 'acc' in metric or 'f1' in metric or 'auc' in metric else np.argmin(
                    val_values)
                best_value = val_values[best_idx]
                ax.plot(best_idx + 1, best_value, 'ro', markersize=8)
                ax.annotate(f'{best_value:.3f}',
                            xy=(best_idx + 1, best_value),
                            xytext=(best_idx + 1.5, best_value),
                            fontsize=10)

            ax.set_ylabel(metric)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

    # 设置x轴标签
    axes[-1].set_xlabel('训练轮次 (Epoch)')
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    return fig


def plot_feature_importance(feature_importances: np.ndarray,
                            feature_names: List[str],
                            top_n: int = 20,
                            title: str = "特征重要性",
                            figsize: Tuple[int, int] = (12, 8)) -> Figure:
    """
    绘制特征重要性条形图

    Args:
        feature_importances: 特征重要性数组
        feature_names: 特征名称列表
        top_n: 显示前N个特征
        title: 图形标题
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    # 创建DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })

    # 排序并选择前N个
    df = df.sort_values('importance', ascending=False).head(top_n)

    # 绘制条形图
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['importance'], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'])
    ax.invert_yaxis()  # 重要性从高到低显示

    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{width:.4f}', va='center', fontsize=9)

    ax.set_xlabel('重要性分数')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def visualize_attention(attention_weights: np.ndarray,
                        tokens: Optional[List[str]] = None,
                        title: str = "注意力权重热力图",
                        cmap: str = 'YlOrRd',
                        figsize: Tuple[int, int] = (12, 10)) -> Figure:
    """
    可视化注意力权重

    Args:
        attention_weights: 注意力权重矩阵 [seq_len, seq_len] 或 [heads, seq_len, seq_len]
        tokens: 令牌/单词列表
        title: 图形标题
        cmap: 颜色映射
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    if attention_weights.ndim == 3:
        # 多头的注意力，计算平均注意力
        attention_weights = attention_weights.mean(axis=0)

    seq_len = attention_weights.shape[0]

    # 如果没有提供tokens，使用位置索引
    if tokens is None:
        tokens = [f'Pos {i}' for i in range(seq_len)]

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    im = ax.imshow(attention_weights, cmap=cmap, aspect='auto')

    # 设置刻度
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('注意力权重', rotation=-90, va="bottom")

    # 添加数值标签
    threshold = attention_weights.max() / 2.
    for i in range(seq_len):
        for j in range(seq_len):
            color = "white" if attention_weights[i, j] > threshold else "black"
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                           ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_predictions_distribution(predictions: np.ndarray,
                                  targets: np.ndarray,
                                  class_names: Optional[List[str]] = None,
                                  title: str = "预测分布",
                                  figsize: Tuple[int, int] = (12, 8)) -> Figure:
    """
    绘制预测分布图

    Args:
        predictions: 预测标签
        targets: 真实标签
        class_names: 类别名称
        title: 图形标题
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    num_classes = len(np.unique(targets))
    if class_names is None:
        class_names = [f'类别 {i}' for i in range(num_classes)]

    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes))
    for pred, target in zip(predictions, targets):
        confusion_matrix[target][pred] += 1

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 子图1：预测分布条形图
    pred_counts = np.bincount(predictions, minlength=num_classes)
    target_counts = np.bincount(targets, minlength=num_classes)

    x = np.arange(num_classes)
    width = 0.35

    axes[0].bar(x - width / 2, target_counts, width, label='真实分布', alpha=0.8)
    axes[0].bar(x + width / 2, pred_counts, width, label='预测分布', alpha=0.8)
    axes[0].set_xlabel('类别')
    axes[0].set_ylabel('样本数量')
    axes[0].set_title('类别分布')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2：预测正确率
    accuracy_per_class = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    accuracy_per_class[np.isnan(accuracy_per_class)] = 0

    axes[1].bar(x, accuracy_per_class * 100, alpha=0.8, color='green')
    axes[1].set_xlabel('类别')
    axes[1].set_ylabel('准确率 (%)')
    axes[1].set_title('各类别准确率')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3, axis='y')

    # 添加准确率数值标签
    for i, acc in enumerate(accuracy_per_class):
        axes[1].text(i, acc * 100 + 1, f'{acc * 100:.1f}%',
                     ha='center', fontsize=10)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_calibration_curve(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           n_bins: int = 10,
                           title: str = "概率校准曲线",
                           figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    绘制概率校准曲线

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        n_bins: 分箱数量
        title: 图形标题
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    from sklearn.calibration import calibration_curve

    # 计算校准曲线
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制校准曲线
    ax.plot(prob_pred, prob_true, 's-', label='模型校准曲线', linewidth=2, markersize=8)

    # 绘制理想校准线
    ax.plot([0, 1], [0, 1], 'k--', label='完美校准', linewidth=2)

    # 设置图形属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('预测概率')
    ax.set_ylabel('真实概率')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 添加可靠性图
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.2, color='red')

    plt.tight_layout()
    return fig


def plot_uncertainty_distribution(predictions: np.ndarray,
                                  uncertainties: np.ndarray,
                                  correct_mask: np.ndarray,
                                  title: str = "不确定性分布",
                                  figsize: Tuple[int, int] = (12, 8)) -> Figure:
    """
    绘制不确定性分布图

    Args:
        predictions: 预测值
        uncertainties: 不确定性值
        correct_mask: 正确预测的布尔掩码
        title: 图形标题
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 子图1：正确和错误预测的不确定性分布
    correct_uncertainties = uncertainties[correct_mask]
    incorrect_uncertainties = uncertainties[~correct_mask]

    axes[0].hist(correct_uncertainties, bins=30, alpha=0.7,
                 label=f'正确预测 (N={len(correct_uncertainties)})',
                 color='green', density=True)
    axes[0].hist(incorrect_uncertainties, bins=30, alpha=0.7,
                 label=f'错误预测 (N={len(incorrect_uncertainties)})',
                 color='red', density=True)
    axes[0].set_xlabel('不确定性')
    axes[0].set_ylabel('密度')
    axes[0].set_title('不确定性分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2：不确定性随预测值的变化
    axes[1].scatter(predictions[correct_mask], uncertainties[correct_mask],
                    alpha=0.5, label='正确预测', s=20, color='green')
    axes[1].scatter(predictions[~correct_mask], uncertainties[~correct_mask],
                    alpha=0.5, label='错误预测', s=20, color='red')
    axes[1].set_xlabel('预测值')
    axes[1].set_ylabel('不确定性')
    axes[1].set_title('预测值与不确定性关系')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_multimodal_features(features_dict: Dict[str, np.ndarray],
                             sample_indices: Optional[List[int]] = None,
                             max_samples: int = 10,
                             title: str = "多模态特征可视化",
                             figsize: Tuple[int, int] = (15, 10)) -> Figure:
    """
    可视化多模态特征

    Args:
        features_dict: 特征字典 {模态名称: 特征数组}
        sample_indices: 样本索引列表
        max_samples: 最大样本数量
        title: 图形标题
        figsize: 图形大小

    Returns:
        matplotlib图形对象
    """
    modalities = list(features_dict.keys())
    num_modalities = len(modalities)

    # 确定要可视化的样本
    if sample_indices is None:
        total_samples = min(len(features_dict[modalities[0]]), max_samples)
        sample_indices = list(range(total_samples))
    else:
        sample_indices = sample_indices[:max_samples]

    # 创建图形
    fig, axes = plt.subplots(num_modalities, len(sample_indices),
                             figsize=figsize,
                             squeeze=False)

    for i, modality in enumerate(modalities):
        features = features_dict[modality]

        for j, sample_idx in enumerate(sample_indices):
            ax = axes[i, j]

            if sample_idx < len(features):
                # 获取样本特征
                sample_feature = features[sample_idx]

                # 根据特征维度选择可视化方式
                if sample_feature.ndim == 1:
                    # 1D特征：折线图
                    ax.plot(sample_feature)
                    ax.set_ylabel('特征值')
                    ax.set_xlabel('特征维度')

                elif sample_feature.ndim == 2:
                    # 2D特征：热力图
                    im = ax.imshow(sample_feature, aspect='auto', cmap='viridis')
                    if j == len(sample_indices) - 1:
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.set_ylabel('强度', rotation=-90, va="bottom")

                elif sample_feature.ndim == 3 and sample_feature.shape[0] in [1, 3]:
                    # 图像特征：如果是1或3通道，显示为图像
                    if sample_feature.shape[0] == 1:
                        # 单通道：灰度图
                        ax.imshow(sample_feature[0], cmap='gray')
                    else:
                        # 多通道：RGB图
                        # 需要将通道移到最后一个维度
                        img = np.transpose(sample_feature, (1, 2, 0))
                        ax.imshow(img)

                # 设置标题
                if i == 0:
                    ax.set_title(f'样本 {sample_idx}')

            # 设置y轴标签（只在第一列）
            if j == 0:
                ax.set_ylabel(modality)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def create_dashboard(metrics: Dict[str, Any],
                     save_path: Optional[Union[str, Path]] = None,
                     title: str = "模型性能仪表板") -> None:
    """
    创建综合性能仪表板

    Args:
        metrics: 指标字典
        save_path: 保存路径（可选）
        title: 仪表板标题
    """
    # 创建多个图形
    figs = []

    # 1. 混淆矩阵
    if 'confusion_matrix' in metrics and 'class_names' in metrics:
        fig = plot_confusion_matrix(
            metrics['confusion_matrix'],
            metrics['class_names'],
            title="混淆矩阵"
        )
        figs.append(('confusion_matrix', fig))

    # 2. ROC曲线
    if 'roc_curves' in metrics and 'auc_scores' in metrics:
        fig = plot_roc_curves(
            metrics['roc_curves'],
            metrics['roc_curves'],  # fpr和tpr在同一字典中
            metrics['auc_scores'],
            title="ROC曲线"
        )
        figs.append(('roc_curves', fig))

    # 3. 训练历史
    if 'training_history' in metrics:
        fig = plot_training_history(
            metrics['training_history'],
            title="训练历史"
        )
        figs.append(('training_history', fig))

    # 4. 预测分布
    if 'predictions' in metrics and 'targets' in metrics:
        class_names = metrics.get('class_names')
        fig = plot_predictions_distribution(
            metrics['predictions'],
            metrics['targets'],
            class_names,
            title="预测分布"
        )
        figs.append(('predictions_distribution', fig))

    # 保存所有图形
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        for fig_name, fig in figs:
            fig_path = save_path / f"{fig_name}.png"
            save_figure(fig, fig_path)
            plt.close(fig)

        logger.info(f"仪表板已保存到: {save_path}")
    else:
        # 显示图形
        plt.show()


def plot_solar_flare_prediction(images: Dict[str, np.ndarray],
                                predictions: Dict[str, Any],
                                timestamps: List[str],
                                save_path: Optional[Union[str, Path]] = None,
                                title: str = "太阳耀斑预测结果") -> None:
    """
    绘制太阳耀斑预测结果（专门针对太阳物理数据）

    Args:
        images: 图像字典 {模态: 图像序列}
        predictions: 预测结果字典
        timestamps: 时间戳列表
        save_path: 保存路径（可选）
        title: 图形标题
    """
    num_modalities = len(images)
    num_timepoints = min(len(timestamps), 5)  # 最多显示5个时间点

    # 创建图形
    fig, axes = plt.subplots(num_modalities, num_timepoints + 1,
                             figsize=(15, 3 * num_modalities))

    if num_modalities == 1:
        axes = axes.reshape(1, -1)

    # 绘制每个模态的图像
    for i, (modality, image_sequence) in enumerate(images.items()):
        for j in range(num_timepoints):
            ax = axes[i, j] if num_modalities > 1 else axes[j]

            if j < len(image_sequence):
                # 显示图像
                img = image_sequence[j]
                if img.ndim == 2:
                    ax.imshow(img, cmap='hot', origin='lower')
                elif img.ndim == 3 and img.shape[0] == 3:
                    # RGB图像
                    ax.imshow(np.transpose(img, (1, 2, 0)))

                ax.set_title(f"{timestamps[j]}\n{modality}")
                ax.axis('off')

        # 最后一列：预测结果
        ax_pred = axes[i, -1] if num_modalities > 1 else axes[-1]

        # 绘制预测概率条形图
        if 'class_probs' in predictions:
            class_probs = predictions['class_probs'][i] if len(predictions['class_probs'].shape) > 1 else predictions[
                'class_probs']
            classes = ['无事件', '爆发耀斑', '束缚耀斑']

            ax_pred.bar(classes, class_probs, color=['gray', 'red', 'orange'])
            ax_pred.set_ylim([0, 1])
            ax_pred.set_ylabel('概率')
            ax_pred.set_title('预测概率')
            ax_pred.grid(True, alpha=0.3, axis='y')

            # 添加概率值
            for k, prob in enumerate(class_probs):
                ax_pred.text(k, prob + 0.02, f'{prob:.2f}',
                             ha='center', fontsize=10)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    # 保存或显示
    if save_path:
        save_figure(fig, save_path)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    # 测试代码
    print("测试可视化模块...")

    # 设置随机种子
    np.random.seed(42)

    # 测试混淆矩阵
    confusion_mat = np.array([[50, 5, 2],
                              [3, 45, 7],
                              [1, 4, 55]])
    class_names = ['无事件', '爆发耀斑', '束缚耀斑']

    fig1 = plot_confusion_matrix(confusion_mat, class_names)
    fig1.savefig('test_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("✓ 混淆矩阵测试完成")

    # 测试ROC曲线
    fpr_dict = {
        'Class 0': np.linspace(0, 1, 100),
        'Class 1': np.linspace(0, 1, 100),
        'Class 2': np.linspace(0, 1, 100)
    }
    tpr_dict = {
        'Class 0': np.sqrt(np.linspace(0, 1, 100)),
        'Class 1': np.linspace(0, 1, 100) ** 0.7,
        'Class 2': np.linspace(0, 1, 100) ** 0.5
    }
    auc_dict = {
        'Class 0': 0.95,
        'Class 1': 0.85,
        'Class 2': 0.75
    }

    fig2 = plot_roc_curves(fpr_dict, tpr_dict, auc_dict)
    fig2.savefig('test_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("✓ ROC曲线测试完成")

    # 测试训练历史
    history = {
        'loss': [0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.09, 0.08],
        'val_loss': [0.6, 0.4, 0.25, 0.2, 0.18, 0.16, 0.15, 0.14],
        'accuracy': [0.7, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.93],
        'val_accuracy': [0.65, 0.75, 0.8, 0.83, 0.85, 0.86, 0.87, 0.88]
    }

    fig3 = plot_training_history(history)
    fig3.savefig('test_training_history.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("✓ 训练历史测试完成")

    # 测试特征重要性
    feature_names = [f'Feature_{i}' for i in range(20)]
    importances = np.random.rand(20)
    importances = importances / importances.sum()

    fig4 = plot_feature_importance(importances, feature_names, top_n=10)
    fig4.savefig('test_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("✓ 特征重要性测试完成")

    # 测试太阳耀斑预测可视化
    images = {
        'magnetogram': np.random.randn(5, 512, 512),
        'euv_94': np.random.rand(5, 512, 512) * 100
    }

    predictions = {
        'class_probs': np.array([0.1, 0.6, 0.3])
    }

    timestamps = ['2024-01-01T00:00:00', '2024-01-01T00:12:00',
                  '2024-01-01T00:24:00', '2024-01-01T00:36:00',
                  '2024-01-01T00:48:00']

    plot_solar_flare_prediction(images, predictions, timestamps,
                                save_path='test_solar_flare_prediction.png')
    print("✓ 太阳耀斑预测可视化测试完成")

    print("\n所有测试完成!")

    # 清理测试文件
    import os

    for file in ['test_confusion_matrix.png', 'test_roc_curves.png',
                 'test_training_history.png', 'test_feature_importance.png',
                 'test_solar_flare_prediction.png']:
        if os.path.exists(file):
            os.remove(file)

    print("已清理测试文件")