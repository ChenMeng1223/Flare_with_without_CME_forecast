"""
注意力机制模块
实现各种注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """标准的多头注意力机制"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数量
            dropout: dropout概率
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 查询、键、值投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: 查询张量 (B, T_q, D)
            key: 键张量 (B, T_k, D)
            value: 值张量 (B, T_k, D)
            key_padding_mask: 键填充掩码 (B, T_k)
            need_weights: 是否返回注意力权重

        Returns:
            注意力输出和权重
        """
        batch_size = query.shape[0]

        # 线性投影并重塑为多头形式
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T_q, T_k)

        # 应用掩码
        if key_padding_mask is not None:
            # 将掩码扩展为注意力头维度
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_k)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到值上
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T_q, D_h)

        # 重塑回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )  # (B, T_q, D)

        # 输出投影
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # 返回平均后的注意力权重 (B, T_q, T_k)
            avg_weights = attn_weights.mean(dim=1)
            return attn_output, avg_weights

        return attn_output, None


class SelfAttention(MultiHeadAttention):
    """自注意力机制"""

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: 输入张量 (B, T, D)
            padding_mask: 填充掩码
            need_weights: 是否返回注意力权重

        Returns:
            自注意力输出和权重
        """
        return super().forward(x, x, x, padding_mask, need_weights)


class CrossAttention(MultiHeadAttention):
    """交叉注意力机制"""

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: 查询张量
            context: 上下文张量
            context_mask: 上下文掩码
            need_weights: 是否返回注意力权重

        Returns:
            交叉注意力输出和权重
        """
        return super().forward(query, context, context, context_mask, need_weights)


class TemporalAttention(nn.Module):
    """时间注意力机制"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, T, D)
            padding_mask: 填充掩码

        Returns:
            时间注意力输出
        """
        # 自注意力
        attended, _ = self.attention(x, x, x, padding_mask)

        # 残差连接和归一化
        output = self.norm(x + self.dropout(attended))

        return output


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.channels = channels
        self.reduction = reduction

        # 空间注意力模块
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // reduction, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, C, H, W)

        Returns:
            空间注意力加权后的特征
        """
        # 计算空间注意力权重
        attention = self.conv1(x)
        attention = self.bn1(attention)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)  # (B, 1, H, W)

        # 应用注意力权重
        weighted_x = x * attention

        return weighted_x


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.channels = channels
        self.reduction = reduction

        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, C, H, W)

        Returns:
            通道注意力加权后的特征
        """
        # 平均池化和最大池化
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        # 合并
        attention = avg_out + max_out
        attention = self.sigmoid(attention)  # (B, C, 1, 1)

        # 应用注意力权重
        weighted_x = x * attention

        return weighted_x


class CBAM(nn.Module):
    """卷积块注意力模块 (CBAM)"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, C, H, W)

        Returns:
            CBAM处理后的特征
        """
        # 应用通道注意力
        x = self.channel_attention(x)

        # 应用空间注意力
        x = self.spatial_attention(x)

        return x


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, T, D)

        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class RelativePositionEncoding(nn.Module):
    """相对位置编码"""

    def __init__(self, d_model: int, max_relative_position: int = 64):
        super().__init__()

        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.zeros(2 * max_relative_position + 1, d_model))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q: int, length_k: int) -> torch.Tensor:
        """
        Args:
            length_q: 查询序列长度
            length_k: 键序列长度

        Returns:
            相对位置编码
        """
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)

        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position,
                                           self.max_relative_position)

        final_mat = distance_mat_clipped + self.max_relative_position
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class MultiScaleAttention(nn.Module):
    """多尺度注意力"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 多尺度卷积
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)

        # 注意力
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, T, D)

        Returns:
            多尺度注意力输出
        """
        B, T, D = x.shape

        # 多尺度特征提取
        x_t = x.transpose(1, 2)  # (B, D, T)

        feat1 = self.conv1(x_t).transpose(1, 2)  # (B, T, D)
        feat3 = self.conv3(x_t).transpose(1, 2)
        feat5 = self.conv5(x_t).transpose(1, 2)

        # 门控融合
        combined = torch.cat([feat1, feat3, feat5], dim=-1)  # (B, T, 3*D)
        gate = self.gate(combined)  # (B, T, D)

        multi_scale_feat = (feat1 + feat3 + feat5) / 3 * gate

        # 注意力
        attended, _ = self.attention(multi_scale_feat, multi_scale_feat, multi_scale_feat)

        # 残差连接
        output = self.norm(x + self.dropout(attended))

        return output