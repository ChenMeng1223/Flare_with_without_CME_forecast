from .physical_constraints import PhysicsConstraintModule
from .multimodal_transformer import MultimodalTransformer
from .spatial_temporal_cnn import SpatioTemporalCNN, ResNet3D, ConvLSTM
from .fusion_modules import (
    AdaptiveModalityFusion,
    CrossModalAttentionFusion,
    HierarchicalFusion,
    GatedFusion,
    SimpleFusion
)
from .attention_mechanisms import (
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    TemporalAttention,
    SpatialAttention,
    ChannelAttention,
    CBAM,
    PositionalEncoding,
    RelativePositionEncoding,
    MultiScaleAttention
)

__all__ = [
    # 物理约束
    'PhysicsConstraintModule',

    # 主模型
    'MultimodalTransformer',

    # 时空CNN模型
    'SpatioTemporalCNN',
    'ResNet3D',
    'ConvLSTM',

    # 融合模块
    'AdaptiveModalityFusion',
    'CrossModalAttentionFusion',
    'HierarchicalFusion',
    'GatedFusion',

    # 注意力机制
    'MultiHeadAttention',
    'SelfAttention',
    'CrossAttention',
    'TemporalAttention',
    'SpatialAttention',
    'ChannelAttention',
    'CBAM',
    'PositionalEncoding',
    'RelativePositionEncoding',
    'MultiScaleAttention'
]

'''
1.spatial_temporal_cnn.py:

时空CNN模型作为Transformer的替代方案

支持ConvLSTM进行时间建模

3D残差块实现

2.fusion_modules.py:

多种多模态融合策略

自适应注意力融合

跨模态注意力融合

层次化融合

门控融合

3.attention_mechanisms.py:

标准的多头注意力机制

自注意力和交叉注意力

时间和空间注意力

通道注意力（CBAM）

位置编码

多尺度注意力
'''