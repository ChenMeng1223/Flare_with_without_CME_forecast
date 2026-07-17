"""
物理约束模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class PhysicsConstraintModule(nn.Module):
    """物理约束模块"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化物理约束模块

        Args:
            config: 物理约束配置
        """
        super().__init__()

        self.config = config or {
            'magnetic_energy_weight': 0.1,
            'helicity_weight': 0.05,
            'energy_balance_weight': 0.1,
            'temporal_causality_weight': 0.05
        }

        self.weights = {
            'magnetic_energy': self.config.get('magnetic_energy_weight', 0.1),
            'helicity': self.config.get('helicity_weight', 0.05),
            'energy_balance': self.config.get('energy_balance_weight', 0.1),
            'temporal_causality': self.config.get('temporal_causality_weight', 0.05)
        }

    def compute_magnetic_energy_constraint(self, predictions: torch.Tensor,
                                           magnetogram: torch.Tensor) -> torch.Tensor:
        """
        计算磁自由能约束

        约束: 耀斑概率应与磁自由能正相关

        Args:
            predictions: 模型预测 (B, num_classes)
            magnetogram: 磁图数据 (B, T, H, W)

        Returns:
            磁自由能约束损失
        """
        # 计算磁梯度能量
        if magnetogram.dim() == 4:  # (B, T, H, W)
            # 计算空间梯度
            grad_x = torch.gradient(magnetogram, dim=2)[0]
            grad_y = torch.gradient(magnetogram, dim=3)[0]

            # 计算梯度能量
            gradient_energy = (grad_x ** 2 + grad_y ** 2).mean(dim=(1, 2, 3))  # (B,)

            # 使用预测的耀斑概率（类别1和2）
            flare_prob = predictions[:, 1:].sum(dim=1)  # (B,)

            # 约束：耀斑概率应与梯度能量正相关
            # 使用负相关惩罚
            pair = torch.stack([flare_prob, gradient_energy])
            if pair.shape[1] < 2:
                constraint = torch.tensor(0.0, device=predictions.device)
            else:
                corr = torch.corrcoef(pair)[0, 1]
                if torch.isnan(corr):
                    constraint = torch.tensor(0.0, device=predictions.device)
                else:
                    correlation = -corr
                    # 确保非负
                    constraint = F.relu(correlation)

        else:
            constraint = torch.tensor(0.0, device=predictions.device)

        return constraint

    def compute_magnetic_helicity_constraint(self, predictions: torch.Tensor,
                                             vector_magnetogram: Optional[torch.Tensor] = None,
                                             magnetogram: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算磁螺度约束

        约束: 爆发耀斑（类别1）应有更高的磁螺度

        Args:
            predictions: 模型预测 (B, num_classes)
            vector_magnetogram: 矢量磁图 (B, T, 3, H, W)，可选
            magnetogram: 标量磁图 (B, T, H, W)，可选

        Returns:
            磁螺度约束损失
        """
        # 如果没有矢量磁图，使用标量磁图近似
        if vector_magnetogram is not None:
            # 计算电流密度 J = ∇ × B
            # 简化计算：使用差分近似
            Bx, By, Bz = vector_magnetogram[:, :, 0], vector_magnetogram[:, :, 1], vector_magnetogram[:, :, 2]

            # 计算电流密度的z分量 Jz = ∂By/∂x - ∂Bx/∂y
            grad_By_x = torch.gradient(By, dim=3)[0]
            grad_Bx_y = torch.gradient(Bx, dim=2)[0]
            Jz = grad_By_x - grad_Bx_y

            # 计算电流螺度 Hc = Bz * Jz
            helicity = (Bz * Jz).mean(dim=(1, 2, 3))  # (B,)

        elif magnetogram is not None:
            # 使用标量磁图近似螺度
            # 计算磁梯度
            grad_x = torch.gradient(magnetogram, dim=3)[0]
            grad_y = torch.gradient(magnetogram, dim=2)[0]

            # 近似电流螺度
            helicity = torch.sqrt(grad_x ** 2 + grad_y ** 2).mean(dim=(1, 2, 3))  # (B,)
        else:
            # 没有磁图数据，返回0约束
            return torch.tensor(0.0, device=predictions.device)

        # 获取爆发耀斑概率
        eruptive_prob = predictions[:, 1]  # 类别1的概率
        confined_prob = predictions[:, 2]  # 类别2的概率

        # 约束：爆发耀斑应有更高螺度
        # 使用对比损失
        margin = 0.1
        constraint = F.relu(confined_prob - eruptive_prob * (1 + margin * helicity))

        return constraint.mean()

    def compute_energy_balance_constraint(self, predictions: torch.Tensor,
                                          observations: Dict) -> torch.Tensor:
        """
        计算能量平衡约束

        约束: 预测的耀斑能量应与观测值一致

        Args:
            predictions: 模型预测
            observations: 观测数据字典

        Returns:
            能量平衡约束损失
        """
        # 这里需要根据具体数据实现
        # 示例：使用预测的耀斑类别估计能量
        flare_classes = predictions.argmax(dim=1)

        # 简化的能量估计（实际应根据耀斑级别估计）
        # C-class: 1e-6, M-class: 1e-5, X-class: 1e-4
        energy_scale = {
            0: 0.0,  # 无耀斑
            1: 1e-5,  # 爆发耀斑（M-class）
            2: 1e-6  # 束缚耀斑（C-class）
        }

        # 估计能量
        estimated_energy = torch.tensor(
            [energy_scale[c.item()] for c in flare_classes],
            device=predictions.device
        )

        # 如果有观测能量，计算差异
        if 'observed_energy' in observations:
            observed_energy = observations['observed_energy']
            if observed_energy.shape[0] == predictions.shape[0]:
                # 计算相对误差
                energy_diff = torch.abs(estimated_energy - observed_energy) / (observed_energy + 1e-8)
                return energy_diff.mean()

        return torch.tensor(0.0, device=predictions.device)

    def compute_temporal_causality_constraint(self, predictions: torch.Tensor,
                                              sequence_features: torch.Tensor) -> torch.Tensor:
        """
        计算时间因果约束

        约束: 事件发展应符合因果顺序

        Args:
            predictions: 模型预测 (B, num_classes)
            sequence_features: 序列特征 (B, T, D)

        Returns:
            时间因果约束损失
        """
        if sequence_features.dim() != 3:
            return torch.tensor(0.0, device=predictions.device)

        B, T, D = sequence_features.shape

        # 计算时间梯度的平滑度
        time_grad = torch.diff(sequence_features, dim=1)  # (B, T-1, D)

        # 计算梯度变化的平滑度（二阶梯度）
        if T > 2:
            second_grad = torch.diff(time_grad, dim=1)  # (B, T-2, D)

            # 约束：二阶梯度应较小（变化应平滑）
            smoothness_loss = second_grad.norm(dim=2).mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)

        return smoothness_loss

    def forward(self, features: torch.Tensor, predictions: torch.Tensor,
                physics_inputs: Optional[Dict] = None) -> torch.Tensor:
        """
        计算总物理约束损失

        Args:
            features: 模型特征
            predictions: 模型预测
            physics_inputs: 物理输入数据

        Returns:
            总物理约束损失
        """
        total_constraint = torch.tensor(0.0, device=features.device)
        constraint_components = {}

        # 检查是否有物理输入
        if physics_inputs is None:
            return total_constraint

        # 1. 磁自由能约束
        if 'magnetogram' in physics_inputs:
            magnetic_constraint = self.compute_magnetic_energy_constraint(
                predictions, physics_inputs['magnetogram']
            )
            total_constraint += self.weights['magnetic_energy'] * magnetic_constraint
            constraint_components['magnetic_energy'] = magnetic_constraint.item()

        # 2. 磁螺度约束
        if 'vector_magnetogram' in physics_inputs:
            helicity_constraint = self.compute_magnetic_helicity_constraint(
                predictions, physics_inputs['vector_magnetogram']
            )
        elif 'magnetogram' in physics_inputs:
            helicity_constraint = self.compute_magnetic_helicity_constraint(
                predictions, magnetogram=physics_inputs['magnetogram']
            )
        else:
            helicity_constraint = torch.tensor(0.0, device=features.device)

        total_constraint += self.weights['helicity'] * helicity_constraint
        constraint_components['helicity'] = helicity_constraint.item()

        # 3. 能量平衡约束
        energy_constraint = self.compute_energy_balance_constraint(
            predictions, physics_inputs
        )
        total_constraint += self.weights['energy_balance'] * energy_constraint
        constraint_components['energy_balance'] = energy_constraint.item()

        # 4. 时间因果约束
        temporal_constraint = self.compute_temporal_causality_constraint(
            predictions, features
        )
        total_constraint += self.weights['temporal_causality'] * temporal_constraint
        constraint_components['temporal_causality'] = temporal_constraint.item()

        # 添加约束信息到返回字典
        self.constraint_components = constraint_components

        return total_constraint