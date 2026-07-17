"""
时空CNN模型
用于太阳耀斑预测的时空特征提取
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math


class ConvLSTM(nn.Module):
    """ConvLSTM单元"""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 门控卷积
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, x: torch.Tensor,
                prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            x: 输入张量 (B, C, H, W)
            prev_state: 前一个状态 (h, c)

        Returns:
            输出张量和新的状态
        """
        batch_size, _, height, width = x.shape

        # 初始化隐藏状态
        if prev_state is None:
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
        else:
            h, c = prev_state

        # 连接输入和隐藏状态
        combined = torch.cat([x, h], dim=1)

        # 计算所有门
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)

        # 应用激活函数
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 更新记忆单元和隐藏状态
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class SpatioTemporalCNN(nn.Module):
    """时空CNN模型"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 模型配置字典
        """
        super().__init__()

        # 获取配置
        self.input_channels = config.get('input_channels', 1)
        self.num_classes = config.get('num_classes', 3)
        self.hidden_dims = config.get('hidden_dims', [32, 64, 128, 256])
        self.kernel_sizes = config.get('kernel_sizes', [3, 3, 3, 3])
        self.use_convlstm = config.get('use_convlstm', True)
        self.lstm_hidden_dim = config.get('lstm_hidden_dim', 256)

        # 空间编码器 (3D CNN)
        self.spatial_encoder = self._build_spatial_encoder()

        # 时间编码器 (ConvLSTM或1D CNN)
        if self.use_convlstm:
            self.temporal_encoder = ConvLSTM(
                input_dim=self.hidden_dims[-1],
                hidden_dim=self.lstm_hidden_dim
            )
            temporal_output_dim = self.lstm_hidden_dim
        else:
            # 使用1D CNN进行时间编码
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(self.hidden_dims[-1], self.lstm_hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(self.lstm_hidden_dim),
                nn.ReLU(),
                nn.Conv1d(self.lstm_hidden_dim, self.lstm_hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(self.lstm_hidden_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            temporal_output_dim = self.lstm_hidden_dim

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(temporal_output_dim, temporal_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(temporal_output_dim // 2, self.num_classes)
        )

        # 边界框回归头
        self.bbox_predictor = nn.Sequential(
            nn.Linear(temporal_output_dim, temporal_output_dim // 2),
            nn.ReLU(),
            nn.Linear(temporal_output_dim // 2, 4)  # x_min, y_min, x_max, y_max
        )

        # 时间预测头
        self.time_predictor = nn.Sequential(
            nn.Linear(temporal_output_dim, temporal_output_dim // 2),
            nn.ReLU(),
            nn.Linear(temporal_output_dim // 2, 3)  # start, peak, duration
        )

        # 事件概率头
        self.event_prob_predictor = nn.Sequential(
            nn.Linear(temporal_output_dim, temporal_output_dim // 2),
            nn.ReLU(),
            nn.Linear(temporal_output_dim // 2, 1),
            nn.Sigmoid()
        )

    def _build_spatial_encoder(self) -> nn.Module:
        """构建空间编码器 (3D CNN)"""
        layers = []
        in_channels = self.input_channels

        for i, (hidden_dim, kernel_size) in enumerate(zip(self.hidden_dims, self.kernel_sizes)):
            # 3D卷积层
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim,
                          kernel_size=(1, kernel_size, kernel_size),
                          padding=(0, kernel_size // 2, kernel_size // 2)),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) if i < len(
                    self.hidden_dims) - 1 else nn.Identity()
            ])
            in_channels = hidden_dim

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入张量 (B, T, C, H, W) 或 (B, T, H, W)

        Returns:
            预测结果字典
        """
        batch_size, timesteps = x.shape[0], x.shape[1]

        # 添加通道维度（如果输入是灰度图）
        if x.dim() == 4:  # (B, T, H, W)
            x = x.unsqueeze(2)  # (B, T, 1, H, W)

        # 重塑为 (B, C, T, H, W) 用于3D卷积
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # 空间编码
        spatial_features = self.spatial_encoder(x)  # (B, hidden_dim, T, H', W')

        # 获取特征图大小
        B, C, T, H, W = spatial_features.shape

        # 时间编码
        if self.use_convlstm:
            # 使用ConvLSTM处理时间序列
            h_state = None
            temporal_features = []

            for t in range(T):
                spatial_frame = spatial_features[:, :, t, :, :]  # (B, C, H', W')
                h_state, _ = self.temporal_encoder(spatial_frame, h_state)
                temporal_features.append(h_state)

            # 取最后一个时间步的特征
            temporal_output = temporal_features[-1]  # (B, hidden_dim, H', W')

            # 全局平均池化
            pooled_features = F.adaptive_avg_pool2d(temporal_output, (1, 1)).squeeze(-1).squeeze(-1)  # (B, hidden_dim)
        else:
            # 使用1D CNN处理时间序列
            # 重塑为 (B, C, T)
            spatial_features_reshaped = spatial_features.mean(dim=[-2, -1])  # (B, C, T)
            temporal_output = self.temporal_encoder(spatial_features_reshaped)  # (B, hidden_dim, 1)
            pooled_features = temporal_output.squeeze(-1)  # (B, hidden_dim)

        # 多任务预测
        class_logits = self.classifier(pooled_features)
        class_probs = F.softmax(class_logits, dim=-1)
        bbox_pred = self.bbox_predictor(pooled_features)
        time_pred = self.time_predictor(pooled_features)  # 语义：[start_offset, peak_offset, duration]
        # duration 必须为正：用 softplus 提供正值约束
        if time_pred.dim() == 2 and time_pred.shape[1] == 3:
            duration = F.softplus(time_pred[:, 2:3])
            time_pred = torch.cat([time_pred[:, 0:2], duration], dim=1)
        event_prob = self.event_prob_predictor(pooled_features)

        return {
            'class_logits': class_logits,
            'class_probs': class_probs,
            'bbox_pred': bbox_pred,
            'time_pred': time_pred,
            'event_prob': event_prob,
            'features': pooled_features
        }


class ResidualBlock3D(nn.Module):
    """3D残差块"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 如果输入输出维度不匹配，使用1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(SpatioTemporalCNN):
    """3D ResNet变体"""

    def __init__(self, config: Dict):
        super().__init__(config)

        # 覆盖空间编码器
        self.spatial_encoder = self._build_resnet_encoder()

    def _build_resnet_encoder(self) -> nn.Module:
        """构建ResNet风格的3D编码器"""
        layers = []

        # 初始卷积层
        layers.append(nn.Conv3d(self.input_channels, 64,
                                kernel_size=(1, 7, 7),
                                stride=(1, 2, 2),
                                padding=(0, 3, 3),
                                bias=False))
        layers.append(nn.BatchNorm3d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

        # 残差块
        in_channels = 64
        for i, (hidden_dim, num_blocks) in enumerate([(64, 2), (128, 2), (256, 2), (512, 2)]):
            stride = 2 if i > 0 else 1
            layers.append(self._make_residual_layer(in_channels, hidden_dim, num_blocks, stride))
            in_channels = hidden_dim

        return nn.Sequential(*layers)

    def _make_residual_layer(self, in_channels: int, out_channels: int,
                             num_blocks: int, stride: int = 1) -> nn.Module:
        """创建残差层"""
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, 1))

        return nn.Sequential(*layers)