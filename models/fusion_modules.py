"""
多模态融合模块
实现多种多模态融合策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math


class AdaptiveModalityFusion(nn.Module):
    """自适应多模态融合"""

    def __init__(self, modality_dims: Dict[str, int],
                 hidden_dim: int = 256,
                 fusion_method: str = 'attention'):
        """
        Args:
            modality_dims: 模态维度字典 {模态名称: 特征维度}
            hidden_dim: 隐藏层维度
            fusion_method: 融合方法 ('attention', 'concatenation', 'weighted_sum')
        """
        super().__init__()

        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method

        # 模态编码器
        self.modality_encoders = nn.ModuleDict()
        for modality, input_dim in modality_dims.items():
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        if fusion_method == 'attention':
            # 注意力融合
            self.attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads=8,
                batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(0.1)

            # 注意力权重学习
            self.weight_network = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        elif fusion_method == 'weighted_sum':
            # 加权求和融合
            self.modality_weights = nn.ParameterDict()
            for modality in self.modalities:
                self.modality_weights[modality] = nn.Parameter(torch.ones(1))

        # 输出投影
        self.output_projection = nn.Linear(
            hidden_dim * len(self.modalities) if fusion_method == 'concatenation' else hidden_dim,
            hidden_dim
        )

    def forward(self, modality_features: Dict[str, torch.Tensor],
                modality_mask: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            modality_features: 模态特征字典
            modality_mask: 模态掩码（True表示可用）

        Returns:
            融合后的特征
        """
        batch_size = next(iter(modality_features.values())).shape[0]
        device = next(iter(modality_features.values())).device

        # 编码每个模态特征
        encoded_features = {}
        for modality in self.modalities:
            if modality in modality_features and modality_features[modality] is not None:
                feature = modality_features[modality]
                encoded = self.modality_encoders[modality](feature)
                encoded_features[modality] = encoded
            else:
                # 模态缺失，用零向量填充
                encoded_features[modality] = torch.zeros(
                    batch_size, self.hidden_dim, device=device
                )

        # 根据融合方法进行融合
        if self.fusion_method == 'attention':
            # 注意力融合
            features_list = [encoded_features[modality].unsqueeze(1)
                             for modality in self.modalities]
            stacked_features = torch.cat(features_list, dim=1)  # (B, M, D)

            # 创建注意力掩码
            attention_mask = None
            if modality_mask is not None:
                mask_list = []
                for modality in self.modalities:
                    if modality in modality_mask:
                        mask_list.append(~modality_mask[modality])
                    else:
                        mask_list.append(torch.zeros(batch_size, device=device, dtype=torch.bool))
                attention_mask = torch.stack(mask_list, dim=1)  # (B, M)

            # 自注意力
            attn_output, _ = self.attention(
                stacked_features, stacked_features, stacked_features,
                key_padding_mask=attention_mask
            )

            # 残差连接和归一化
            attended_features = self.norm(stacked_features + self.dropout(attn_output))

            # 学习每个模态的重要性权重
            weights = self.weight_network(attended_features)  # (B, M, 1)
            weights = F.softmax(weights, dim=1)

            # 加权求和
            fused_features = (attended_features * weights).sum(dim=1)  # (B, D)

        elif self.fusion_method == 'concatenation':
            # 拼接融合
            features_list = [encoded_features[modality] for modality in self.modalities]
            fused_features = torch.cat(features_list, dim=-1)  # (B, M*D)

        elif self.fusion_method == 'weighted_sum':
            # 加权求和融合
            weighted_features = []
            for modality in self.modalities:
                weight = self.modality_weights[modality]
                weighted_features.append(encoded_features[modality] * weight)

            fused_features = sum(weighted_features) / len(weighted_features)

        else:
            raise ValueError(f"未知的融合方法: {self.fusion_method}")

        # 输出投影
        output = self.output_projection(fused_features)

        return output


class CrossModalAttentionFusion(nn.Module):
    """跨模态注意力融合"""

    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int = 256):
        super().__init__()

        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.hidden_dim = hidden_dim

        # 查询、键、值投影
        self.query_projections = nn.ModuleDict()
        self.key_projections = nn.ModuleDict()
        self.value_projections = nn.ModuleDict()

        for modality, input_dim in modality_dims.items():
            self.query_projections[modality] = nn.Linear(input_dim, hidden_dim)
            self.key_projections[modality] = nn.Linear(input_dim, hidden_dim)
            self.value_projections[modality] = nn.Linear(input_dim, hidden_dim)

        # 多头注意力
        self.multihead_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 模态特征字典

        Returns:
            融合后的特征
        """
        # 准备查询、键、值
        queries, keys, values = [], [], []

        for modality in self.modalities:
            if modality in modality_features:
                feature = modality_features[modality]
                queries.append(self.query_projections[modality](feature))
                keys.append(self.key_projections[modality](feature))
                values.append(self.value_projections[modality](feature))

        # 堆叠
        stacked_queries = torch.stack(queries, dim=1)  # (B, M, D)
        stacked_keys = torch.stack(keys, dim=1)
        stacked_values = torch.stack(values, dim=1)

        # 跨模态注意力
        attention_output, _ = self.multihead_attention(
            stacked_queries, stacked_keys, stacked_values
        )

        # 残差连接和归一化
        fused_features = self.norm(stacked_queries + self.dropout(attention_output))

        # 平均池化
        fused_features = fused_features.mean(dim=1)  # (B, D)

        # 输出投影
        output = self.output_projection(fused_features)

        return output


class HierarchicalFusion(nn.Module):
    """层次化融合"""

    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int = 256):
        super().__init__()

        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.hidden_dim = hidden_dim

        # 第一层：模态内融合（如果需要）
        self.intra_modality_fusion = nn.ModuleDict()
        for modality, input_dim in modality_dims.items():
            if isinstance(input_dim, list):  # 如果模态有多个特征
                self.intra_modality_fusion[modality] = nn.Sequential(
                    nn.Linear(sum(input_dim), hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )

        # 第二层：跨模态融合
        self.cross_modality_fusion = AdaptiveModalityFusion(
            {modality: hidden_dim for modality in self.modalities},
            hidden_dim,
            fusion_method='attention'
        )

        # 第三层：特征增强
        self.feature_enhancement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 模态特征字典

        Returns:
            融合后的特征
        """
        # 第一层：模态内融合
        intra_fused = {}

        for modality in self.modalities:
            if modality in modality_features:
                feature = modality_features[modality]

                # 如果模态有多个特征，先进行模态内融合
                if modality in self.intra_modality_fusion:
                    # 假设特征是列表形式
                    if isinstance(feature, list):
                        concatenated = torch.cat(feature, dim=-1)
                        fused = self.intra_modality_fusion[modality](concatenated)
                    else:
                        fused = feature
                else:
                    fused = feature

                intra_fused[modality] = fused

        # 第二层：跨模态融合
        cross_fused = self.cross_modality_fusion(intra_fused)

        # 第三层：特征增强
        enhanced = self.feature_enhancement(cross_fused)

        return enhanced


class GatedFusion(nn.Module):
    """门控融合"""

    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int = 256):
        super().__init__()

        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.hidden_dim = hidden_dim

        # 模态编码器
        self.modality_encoders = nn.ModuleDict()
        for modality, input_dim in modality_dims.items():
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )

        # 门控机制
        self.gate_networks = nn.ModuleDict()
        for modality in self.modalities:
            self.gate_networks[modality] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )

        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * len(self.modalities), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 模态特征字典

        Returns:
            融合后的特征
        """
        # 编码每个模态
        encoded_features = []

        for modality in self.modalities:
            if modality in modality_features:
                feature = modality_features[modality]
                encoded = self.modality_encoders[modality](feature)

                # 应用门控
                gate = self.gate_networks[modality](encoded)
                gated_feature = encoded * gate

                encoded_features.append(gated_feature)
            else:
                # 缺失模态，用零向量填充
                batch_size = next(iter(modality_features.values())).shape[0]
                device = next(iter(modality_features.values())).device
                encoded_features.append(
                    torch.zeros(batch_size, self.hidden_dim, device=device)
                )

        # 拼接所有门控特征
        concatenated = torch.cat(encoded_features, dim=-1)

        # 融合
        fused = self.fusion_network(concatenated)

        return fused


class SimpleFusion(nn.Module):
    """简单的多模态融合（用于已编码特征）"""

    def __init__(self, fusion_method: str = 'attention', hidden_dim: int = 256,
                 num_modalities: int = 5):
        """
        Args:
            fusion_method: 融合方法 ('attention', 'concatenation', 'mean')
            hidden_dim: 隐藏层维度
            num_modalities: 模态数量
        """
        super().__init__()

        self.fusion_method = fusion_method
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        if fusion_method == 'attention':
            # 简单的注意力融合
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads=4, batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(0.1)

        elif fusion_method == 'concatenation':
            # 拼接后投影
            self.output_projection = nn.Linear(hidden_dim * num_modalities, hidden_dim)

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_features: 模态特征字典 {modality_name: (B, T, D)}

        Returns:
            融合后的特征 (B, T, D)
        """
        # 获取所有可用模态的特征
        features_list = []
        available_modalities = []

        for modality, features in modality_features.items():
            if features is not None:
                features_list.append(features)
                available_modalities.append(modality)

        if not features_list:
            raise ValueError("没有可用的模态特征")

        if self.fusion_method == 'attention':
            # 注意力融合
            if len(features_list) == 1:
                # 只有一个模态，直接返回
                return features_list[0]

            # 堆叠特征用于注意力
            stacked_features = torch.stack(features_list, dim=1)  # (B, M, T, D)

            # 为了使用注意力，需要重塑
            B, M, T, D = stacked_features.shape
            reshaped = stacked_features.view(B * T, M, D)  # (B*T, M, D)

            # 应用注意力
            attn_output, _ = self.attention(reshaped, reshaped, reshaped)

            # 残差连接
            attended = self.norm(reshaped + self.dropout(attn_output))

            # 平均池化模态维度
            fused = attended.mean(dim=1)  # (B*T, D)

            # 恢复原始形状
            fused = fused.view(B, T, D)  # (B, T, D)

        elif self.fusion_method == 'concatenation':
            # 拼接融合
            concatenated = torch.cat(features_list, dim=-1)  # (B, T, M*D)
            fused = self.output_projection(concatenated)  # (B, T, D)

        elif self.fusion_method == 'mean':
            # 平均融合
            stacked = torch.stack(features_list, dim=0)  # (M, B, T, D)
            fused = stacked.mean(dim=0)  # (B, T, D)

        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")

        return fused