"""多模态Transformer模型"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any


class SpatialModalityEncoder(nn.Module):
    """保留空间分辨率的模态编码器。"""

    def __init__(self, input_channels: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, mid_dim, 3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(mid_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:                 # (B, T, H, W) -> 缺少通道维度
            x = x.unsqueeze(2)           # (B, T, 1, H, W)
        bsz, steps = x.shape[:2]         # bsz = batch size, steps = 时间步数,[:2]别忘了左闭右开
        # 合并 batch 和 steps 维度，使每一帧独立通过卷积网络
        feat = self.encoder(x.reshape(bsz * steps, *x.shape[2:]))   # 将 (B, T, C, H, W) 重塑为 (B*T, C, H, W)，*是解包操作符，将可迭代对象（如列表、元组）中的元素逐个取出，作为独立的参数传递给函数
        # feat 形状: (bsz*steps, hidden_dim, H_out, W_out)
        _, channels, height, width = feat.shape
        # 恢复 batch 和 steps 维度，并保留空间尺寸
        return feat.reshape(bsz, steps, channels, height, width)


class CrossModalAttention(nn.Module):
    """跨模态注意力。"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, key, value)
        return self.norm(query + self.dropout(out))


class LearnablePositionalEncoding(nn.Module):
    """简单可学习位置编码。"""

    def __init__(self, max_length: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = int(max_length)
        self.embedding = nn.Parameter(torch.randn(1, self.max_length, hidden_dim) * 0.02)

    def _expand_embedding(self, required_length: int) -> None:
        new_max_length = max(required_length, self.max_length * 2)
        new_embedding = torch.randn(
            1,
            new_max_length,
            self.hidden_dim,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        ) * 0.02
        new_embedding[:, :self.max_length] = self.embedding.data
        self.embedding = nn.Parameter(new_embedding)
        self.max_length = new_max_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.max_length:
            self._expand_embedding(length)
        return x + self.embedding[:, :length]


class MultimodalSpatiotemporalFusion(nn.Module):
    """融合多模态空间特征，并保留时空特征图。"""

    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float, fusion_method: str = 'attention'):
        super().__init__()
        self.fusion_method = fusion_method
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads, dropout)
        layer = nn.TransformerEncoderLayer( #定义transformer编码器单层
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers) # 传入单层编码器和编码器堆叠层数

    def forward(self, modality_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not modality_maps:
            raise ValueError("没有可用的模态特征")

        pooled = {name: fmap.mean(dim=(-1, -2)) for name, fmap in modality_maps.items()}    # 把二维图像压成一个特征值，全局平均池化    ？？？？？？？？会不会太草率？？？？？？？
        names = list(modality_maps.keys())

        if len(names) == 1:
            fused_sequence = pooled[names[0]]
            fused_maps = modality_maps[names[0]]
        elif self.fusion_method == 'attention':
            enhanced = []
            for name in names:
                others = [pooled[n] for n in names if n != name]    # 列表推导式
                key_value = torch.stack(others, dim=0).mean(dim=0)  #堆叠再压缩，得到其他模态的压缩特征信息
                enhanced.append(self.cross_attention(pooled[name], key_value, key_value))   # 把进行跨模态融合后的信息存在enhanced中    跨模态注意力：当前模态特征(Q) + 其他模态(KV) → 增强特征
            fused_sequence = torch.stack(enhanced, dim=0).mean(dim=0)   # 所有增强后的模态特征 → 堆叠求平均 → 最终融合时序特征
            fused_maps = torch.stack(list(modality_maps.values()), dim=0).mean(dim=0)   # 多模态空间特征图 → 直接平均融合   ？？？？？？？？？这好吗？？？？？？？
        else:
            fused_sequence = torch.stack(list(pooled.values()), dim=0).mean(dim=0)
            fused_maps = torch.stack(list(modality_maps.values()), dim=0).mean(dim=0)

        # 时间编码：把融合后的时序特征输入Transformer，提取时间动态信息
        temporal_features = self.temporal_encoder(fused_sequence)   
        # 时空融合：把时间特征 加回到 空间特征图上
        fused_maps = fused_maps + temporal_features.unsqueeze(-1).unsqueeze(-1) # unsqueeze(-1) → 增加1个维度 → 让时间特征适配空间特征图的形状
        # 全局特征：对时间维度求平均 → 得到整个序列的全局表征
        global_feature = temporal_features.mean(dim=1)  # ？？？？？？？？？？？？？？？？？？？？？？？？？？直接平均？？？？？？？？？？？？？？？？？？？？？
        return {
            'fused_maps': fused_maps,
            'temporal_features': temporal_features,
            'global_feature': global_feature,
        }


class ProposalDecoder(nn.Module):
    """读取时空 token 的 proposal 解码器。"""

    def __init__(self, hidden_dim: int, num_queries: int, num_heads: int, dropout: float = 0.1, proposal_init_size_bias: float = -2.2, max_spatiotemporal_tokens: int = 4096):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.token_pos_encoding = LearnablePositionalEncoding(max_spatiotemporal_tokens, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        with torch.no_grad():
            self.box_head[-1].bias[2:].fill_(proposal_init_size_bias)

    def forward(self, fused_maps: torch.Tensor, global_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not torch.isfinite(fused_maps).all():
            bad_count = int((~torch.isfinite(fused_maps)).sum().item())
            raise ValueError(f"proposal decoder 输入 fused_maps 出现非有限值，bad_count={bad_count}")
        if not torch.isfinite(global_feature).all():
            bad_count = int((~torch.isfinite(global_feature)).sum().item())
            raise ValueError(f"proposal decoder 输入 global_feature 出现非有限值，bad_count={bad_count}")

        bsz, steps, channels, height, width = fused_maps.shape
        spatial_temporal_tokens = fused_maps.permute(0, 1, 3, 4, 2).reshape(bsz, steps * height * width, channels)
        spatial_temporal_tokens = self.token_pos_encoding(spatial_temporal_tokens)
        queries = self.query_embed.unsqueeze(0).expand(bsz, -1, -1)
        attended, _ = self.attn(queries, spatial_temporal_tokens, spatial_temporal_tokens)

        if not torch.isfinite(attended).all():
            bad_count = int((~torch.isfinite(attended)).sum().item())
            raise ValueError(f"proposal decoder attention 输出出现非有限值，bad_count={bad_count}")

        proposal_features = self.norm(attended + queries + global_feature.unsqueeze(1))

        if not torch.isfinite(proposal_features).all():
            bad_count = int((~torch.isfinite(proposal_features)).sum().item())
            raise ValueError(f"proposal_features 出现非有限值，bad_count={bad_count}")

        proposal_scores = self.score_head(proposal_features)
        box_raw = self.box_head(proposal_features)

        if not torch.isfinite(box_raw).all():
            bad_count = int((~torch.isfinite(box_raw)).sum().item())
            raise ValueError(f"proposal box_raw 出现非有限值，bad_count={bad_count}")

        center = torch.sigmoid(box_raw[..., :2])
        size = torch.sigmoid(box_raw[..., 2:])
        half = size * 0.5
        proposal_boxes = torch.cat([
            torch.clamp(center - half, 0.0, 1.0),
            torch.clamp(center + half, 0.0, 1.0),
        ], dim=-1)
        if not torch.isfinite(proposal_boxes).all():
            bad_count = int((~torch.isfinite(proposal_boxes)).sum().item())
            raise ValueError(f"proposal_boxes 出现非有限值，bad_count={bad_count}")
        return {
            'proposal_boxes': proposal_boxes,
            'proposal_scores': proposal_scores,
            'proposal_features': proposal_features,
            'spatiotemporal_tokens': spatial_temporal_tokens,
            'external_proposals': False,
        }


class RoiTemporalEncoder(nn.Module):
    """对 ROI 时序特征做编码，并保留时序输出。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, max_steps: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.time_pos_encoding = LearnablePositionalEncoding(max_steps, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, roi_maps: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, num_slots, steps, channels, _, _ = roi_maps.shape
        pooled = self.pool(roi_maps.reshape(bsz * num_slots * steps, channels, roi_maps.size(-2), roi_maps.size(-1)))
        pooled = pooled.reshape(bsz * num_slots, steps, channels)
        encoded = self.temporal_encoder(self.time_pos_encoding(pooled))
        temporal_sequence = encoded.reshape(bsz, num_slots, steps, channels)
        temporal_feature = temporal_sequence.mean(dim=2)
        return {
            'sequence': temporal_sequence,
            'pooled_feature': temporal_feature,
        }


class StageTwoPredictor(nn.Module):
    """融合 local/context/global/proposal 特征并输出第二阶段结果。"""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1, head_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        head_config = head_config or {}
        time_cfg = head_config.get('time', {})
        bbox_cfg = head_config.get('bbox', {})
        self.enable_time_prediction = bool(time_cfg.get('enabled', True))
        self.time_offset_range = float(time_cfg.get('offset_range_days', 2.0))
        self.max_duration_days = float(time_cfg.get('max_duration_days', bbox_cfg.get('max_duration_days', time_cfg.get('max_duration_hours', 1.0) / 24.0)))
        self.min_duration_days = float(time_cfg.get('min_duration_days', 1e-3))
        self.refine_center_scale = float(bbox_cfg.get('refine_center_scale', 0.15))
        self.refine_size_log_scale = float(bbox_cfg.get('refine_size_log_scale', 0.35))
        self.min_gated_size = float(bbox_cfg.get('min_gated_size', 1e-4))
        self.gate_bbox_size = bool(bbox_cfg.get('gate_bbox_size', True))
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        if self.enable_time_prediction:
            self.time_sequence_attn = nn.MultiheadAttention(hidden_dim, max(1, min(4, hidden_dim // 32)), dropout=dropout, batch_first=True)
            self.time_sequence_norm = nn.LayerNorm(hidden_dim)
            self.time_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),
            )
        else:
            self.time_sequence_attn = None
            self.time_sequence_norm = None
            self.time_predictor = None
        self.event_prob_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.bbox_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4),
        )

    def _predict_time(self, slot_features: torch.Tensor, local_sequence: torch.Tensor, context_sequence: torch.Tensor, event_gate: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.enable_time_prediction:
            bsz, num_slots = slot_features.shape[:2]
            zeros = torch.zeros(bsz, num_slots, 3, device=slot_features.device, dtype=slot_features.dtype)
            zeros_scalar = torch.zeros(bsz, num_slots, 1, device=slot_features.device, dtype=slot_features.dtype)
            return {
                'time_pred': zeros,
                'time_params': torch.cat([zeros_scalar, zeros_scalar, zeros_scalar], dim=-1),
                'time_center': zeros_scalar,
                'time_duration': zeros_scalar,
                'time_duration_raw': zeros_scalar,
            }
        bsz, num_slots, steps, channels = local_sequence.shape
        time_tokens = 0.5 * (local_sequence + context_sequence).reshape(bsz * num_slots, steps, channels)
        query = slot_features.reshape(bsz * num_slots, 1, channels)
        attended, _ = self.time_sequence_attn(query, time_tokens, time_tokens)
        attended = self.time_sequence_norm(attended + query).squeeze(1)
        time_feature = torch.cat([slot_features.reshape(bsz * num_slots, channels), attended], dim=-1)
        time_raw = self.time_predictor(time_feature).reshape(bsz, num_slots, 3)

        time_offsets = torch.tanh(time_raw) * self.time_offset_range
        start_peak_end, _ = torch.sort(time_offsets, dim=-1)
        start_time = start_peak_end[..., 0:1]
        peak_time = start_peak_end[..., 1:2]
        end_time = start_peak_end[..., 2:3]

        # 负槽位时将时间区间压缩到参考时刻附近，减少伪事件时间跨度
        gated_peak = peak_time * event_gate
        start_time = start_time * event_gate
        end_time = end_time * event_gate
        start_peak_end = torch.cat([start_time, gated_peak, end_time], dim=-1)

        center_time = 0.5 * (start_time + end_time)
        duration_days = (end_time - start_time).clamp(min=self.min_duration_days)
        raw_duration_days = duration_days.clone()
        log_duration = torch.log(duration_days)
        time_params = torch.cat([center_time, log_duration, duration_days], dim=-1)
        return {
            'time_pred': start_peak_end,
            'time_params': time_params,
            'time_center': center_time,
            'time_duration': duration_days,
            'time_duration_raw': raw_duration_days,
        }

    def forward(self, local_feature: torch.Tensor, context_feature: torch.Tensor, proposal_feature: torch.Tensor, global_feature: torch.Tensor, proposal_scores: torch.Tensor, proposal_boxes: torch.Tensor, local_sequence: torch.Tensor, context_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_repeated = global_feature.unsqueeze(1).expand(-1, proposal_feature.size(1), -1)
        slot_features = self.fusion(torch.cat([local_feature, context_feature, proposal_feature, global_repeated, proposal_scores], dim=-1))
        class_logits = self.classifier(slot_features)
        event_logit = self.event_prob_predictor(slot_features)
        event_gate = torch.sigmoid(event_logit)
        time_outputs = self._predict_time(slot_features, local_sequence, context_sequence, event_gate)

        proposal_center = (proposal_boxes[..., :2] + proposal_boxes[..., 2:]) * 0.5
        proposal_size = (proposal_boxes[..., 2:] - proposal_boxes[..., :2]).clamp(min=1e-4)
        refine_raw = self.bbox_refiner(slot_features)
        delta_center = torch.tanh(refine_raw[..., :2]) * (proposal_size * self.refine_center_scale)
        delta_log_size = torch.tanh(refine_raw[..., 2:]) * self.refine_size_log_scale
        refined_center = torch.clamp(proposal_center + delta_center, 0.0, 1.0)
        raw_refined_size = torch.clamp(proposal_size * torch.exp(delta_log_size), min=1e-4, max=1.0)
        if self.gate_bbox_size:
            gated_refined_size = torch.clamp(raw_refined_size * event_gate + self.min_gated_size * (1.0 - event_gate), min=self.min_gated_size, max=1.0)
        else:
            gated_refined_size = raw_refined_size
        half = gated_refined_size * 0.5
        refined_boxes = torch.cat([
            torch.clamp(refined_center - half, 0.0, 1.0),
            torch.clamp(refined_center + half, 0.0, 1.0),
        ], dim=-1)
        refined_boxes = torch.stack([
            torch.minimum(refined_boxes[..., 0], refined_boxes[..., 2]),
            torch.minimum(refined_boxes[..., 1], refined_boxes[..., 3]),
            torch.maximum(refined_boxes[..., 0], refined_boxes[..., 2]),
            torch.maximum(refined_boxes[..., 1], refined_boxes[..., 3]),
        ], dim=-1)
        refine_delta = torch.cat([delta_center, delta_log_size], dim=-1)
        return {
            'slot_features': slot_features,
            'class_logits': class_logits,
            'class_probs': F.softmax(class_logits, dim=-1),
            'time_pred': time_outputs['time_pred'],
            'time_params': time_outputs['time_params'],
            'time_center': time_outputs['time_center'],
            'time_duration': time_outputs['time_duration'],
            'time_duration_raw': time_outputs['time_duration_raw'],
            'event_prob': event_logit,
            'event_gate': event_gate,
            'bbox_pred': refined_boxes,
            'bbox_size_raw': raw_refined_size,
            'bbox_size_gated': gated_refined_size,
            'refine_delta': refine_delta,
        }


class MultimodalTransformer(nn.Module):
    """两阶段蓝图的最小可落地版本。"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.modalities = list(config['modalities'].keys())
        self.hidden_dim = config['transformer']['hidden_dim']
        self.num_heads = config['transformer']['num_heads']
        self.num_layers = config['transformer']['num_layers']
        self.dropout = config['transformer'].get('dropout', 0.1)
        self.num_classes = config['num_classes']
        self.max_activities = config.get('max_activities', 5)
        stage2_cfg = config.get('stage2', {})
        head_cfg = config.get('prediction_heads', {})
        time_head_cfg = head_cfg.get('time', {})
        self.enable_time_prediction = bool(time_head_cfg.get('enabled', True))
        self.context_scale = float(stage2_cfg.get('context_scale', 2.0))
        self.roi_output_size = int(stage2_cfg.get('roi_output_size', 4))
        self.roi_source_mix = self._normalize_roi_source_mix(stage2_cfg.get('roi_source_mix', {'predicted': 1.0}))

        sequence_length = int(config.get('sequence_length', config.get('max_sequence_length', 16)))
        input_height, input_width = self._resolve_input_size(config)
        spatial_cfg = config.get('spatial', {})
        encoder_downsample_factor = int(spatial_cfg.get('encoder_downsample_factor', config.get('encoder_downsample_factor', 4)))
        feature_height = max(1, input_height // encoder_downsample_factor)
        feature_width = max(1, input_width // encoder_downsample_factor)
        max_spatiotemporal_tokens = sequence_length * feature_height * feature_width

        self.modality_encoders = nn.ModuleDict({   # 自动给每一种数据模态创建独立的编码器
            name: SpatialModalityEncoder(cfg['channels'], self.hidden_dim, self.dropout)    # {键: 值 for 变量 in 可迭代对象}，字典推导式，不用写繁琐的 for 循环 + 赋值
            for name, cfg in config['modalities'].items()
        })
        fusion_method = config.get('fusion', {}).get('method', 'attention')
        self.fusion_module = MultimodalSpatiotemporalFusion(
            self.hidden_dim, self.num_heads, self.num_layers, self.dropout, fusion_method
        )
        self.proposal_decoder = ProposalDecoder(
            self.hidden_dim,
            self.max_activities,
            self.num_heads,
            self.dropout,
            proposal_init_size_bias=float(head_cfg.get('bbox', {}).get('proposal_init_size_bias', -2.2)),
            max_spatiotemporal_tokens=max_spatiotemporal_tokens,
        )
        self.roi_encoder = RoiTemporalEncoder(self.hidden_dim, self.num_heads, self.dropout, max_steps=sequence_length)
        self.stage_two_predictor = StageTwoPredictor(self.hidden_dim, self.num_classes, self.dropout, head_config=head_cfg)

    def _resolve_input_size(self, config: Dict) -> tuple[int, int]:
        input_size = config.get('input_size')
        if input_size is not None:
            return int(input_size[0]), int(input_size[1])

        first_modality_cfg = next(iter(config['modalities'].values()), None)
        if not first_modality_cfg or 'resolution' not in first_modality_cfg:
            raise ValueError("模型配置缺少 input_size，且 modalities 中未提供 resolution")
        resolution = first_modality_cfg['resolution']
        return int(resolution[0]), int(resolution[1])

    def _normalize_roi_source_mix(self, mix_cfg: Optional[Dict[str, float]]) -> Dict[str, float]:
        mix_cfg = mix_cfg or {'predicted': 1.0}
        normalized = {
            'gt': float(mix_cfg.get('gt', 0.0)),
            'jittered_gt': float(mix_cfg.get('jittered_gt', 0.0)),
            'predicted': float(mix_cfg.get('predicted', 1.0)),
        }
        total = sum(max(value, 0.0) for value in normalized.values())
        if total <= 0:
            return {'gt': 0.0, 'jittered_gt': 0.0, 'predicted': 1.0}
        return {key: max(value, 0.0) / total for key, value in normalized.items()}

    def _mix_boxes_for_stage2(self, proposal_boxes: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if (not self.training) or (not targets) or 'bbox' not in targets or 'label' not in targets:
            return proposal_boxes

        region_boxes = targets.get('region_bbox', targets['bbox']).to(device=proposal_boxes.device, dtype=proposal_boxes.dtype)
        region_mask = targets.get('region_mask')
        if region_mask is None:
            labels = targets['label'].to(device=proposal_boxes.device)
            slot_mask = targets.get('activity_mask')
            if slot_mask is None:
                slot_mask = torch.ones_like(labels, dtype=torch.bool)
            else:
                slot_mask = slot_mask.to(device=proposal_boxes.device, dtype=torch.bool)
            positive_slot_mask = (labels > 0) & slot_mask
        else:
            positive_slot_mask = region_mask.to(device=proposal_boxes.device, dtype=torch.bool)
        if not positive_slot_mask.any():
            return proposal_boxes

        mixed_boxes = proposal_boxes.clone()
        selector = torch.rand_like(proposal_boxes[..., 0])
        gt_thr = self.roi_source_mix.get('gt', 0.0)
        jitter_thr = gt_thr + self.roi_source_mix.get('jittered_gt', 0.0)

        gt_mask = positive_slot_mask & (selector < gt_thr)
        jitter_mask = positive_slot_mask & (selector >= gt_thr) & (selector < jitter_thr)

        if gt_mask.any():
            mixed_boxes[gt_mask] = region_boxes[gt_mask]

        if jitter_mask.any():
            jitter_noise = (torch.rand_like(region_boxes) - 0.5) * 0.08
            jittered_boxes = torch.clamp(region_boxes + jitter_noise, 0.0, 1.0)
            jittered_boxes = torch.stack([
                torch.minimum(jittered_boxes[..., 0], jittered_boxes[..., 2]),
                torch.minimum(jittered_boxes[..., 1], jittered_boxes[..., 3]),
                torch.maximum(jittered_boxes[..., 0], jittered_boxes[..., 2]),
                torch.maximum(jittered_boxes[..., 1], jittered_boxes[..., 3]),
            ], dim=-1)
            mixed_boxes[jitter_mask] = jittered_boxes[jitter_mask]

        return mixed_boxes

    def _resolve_external_proposals(
            self,
            inputs: Dict[str, torch.Tensor],
            proposal_outputs: Dict[str, torch.Tensor],
            global_feature: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        proposal_boxes = inputs.get('proposal_boxes')
        proposal_scores = inputs.get('proposal_scores')

        if proposal_boxes is None:
            return proposal_outputs

        proposal_boxes = proposal_boxes.to(device=global_feature.device, dtype=global_feature.dtype)
        if proposal_boxes.dim() != 3 or proposal_boxes.size(-1) != 4:
            raise ValueError(f"external proposal_boxes shape invalid: {tuple(proposal_boxes.shape)}")

        if proposal_scores is None:
            proposal_scores = torch.ones(
                proposal_boxes.size(0),
                proposal_boxes.size(1),
                1,
                device=global_feature.device,
                dtype=global_feature.dtype,
            )
        else:
            proposal_scores = proposal_scores.to(device=global_feature.device, dtype=global_feature.dtype)
            if proposal_scores.dim() == 2:
                proposal_scores = proposal_scores.unsqueeze(-1)
            if proposal_scores.dim() != 3:
                raise ValueError(f"external proposal_scores shape invalid: {tuple(proposal_scores.shape)}")

        num_slots = proposal_boxes.size(1)
        proposal_features = global_feature.unsqueeze(1).expand(-1, num_slots, -1).contiguous()

        return {
            'proposal_boxes': proposal_boxes,
            'proposal_scores': proposal_scores,
            'proposal_features': proposal_features,
            'spatiotemporal_tokens': proposal_outputs['spatiotemporal_tokens'],
            'external_proposals': True,
        }

    def _expand_boxes(self, boxes: torch.Tensor, scale: float) -> torch.Tensor:
        center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        size = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4) * scale * 0.5
        boxes = torch.clamp(torch.cat([center - size, center + size], dim=-1), 0.0, 1.0)
        return torch.stack([
            torch.minimum(boxes[..., 0], boxes[..., 2]),
            torch.minimum(boxes[..., 1], boxes[..., 3]),
            torch.maximum(boxes[..., 0], boxes[..., 2]),
            torch.maximum(boxes[..., 1], boxes[..., 3]),
        ], dim=-1)

    def _build_roi_grid(self, boxes: torch.Tensor, output_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        widths = (x2 - x1).clamp(min=1e-4)
        heights = (y2 - y1).clamp(min=1e-4)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, output_size, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, output_size, device=device, dtype=dtype),
            indexing='ij'
        )
        grid_x = x1[:, None, None] + widths[:, None, None] * grid_x[None, :, :]
        grid_y = y1[:, None, None] + heights[:, None, None] * grid_y[None, :, :]
        return torch.stack([grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0], dim=-1)

    def _extract_roi_maps(self, fused_maps: torch.Tensor, boxes: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:
        output_size = output_size or self.roi_output_size
        bsz, steps, channels, _, _ = fused_maps.shape
        num_slots = boxes.size(1)
        flat_maps = fused_maps.reshape(bsz * steps, channels, fused_maps.size(-2), fused_maps.size(-1))
        expanded_boxes = boxes[:, :, None, :].expand(-1, -1, steps, -1).reshape(bsz * num_slots * steps, 4)
        grid = self._build_roi_grid(expanded_boxes, output_size, fused_maps.device, fused_maps.dtype)
        sampled = F.grid_sample(
            flat_maps.repeat_interleave(num_slots, dim=0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
        return sampled.reshape(bsz, num_slots, steps, channels, output_size, output_size)

    def forward(self, inputs: Dict[str, torch.Tensor], physics_inputs: Optional[Dict] = None, targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        # targets数据的由来：创建model完成，并传入trainer之后，在trainer中前向传播中会传入targets数据
        modality_maps = {}
        available_modalities = []
        for modality in self.modalities:
            if modality in inputs and inputs[modality] is not None and inputs[modality].numel() > 0:
                modality_maps[modality] = self.modality_encoders[modality](inputs[modality])
                available_modalities.append(modality)
        if not available_modalities:
            raise ValueError("没有可用的模态数据")

        fusion_outputs = self.fusion_module(modality_maps)

        if not torch.isfinite(fusion_outputs['fused_maps']).all():  # 非有限值检查
            bad_count = int((~torch.isfinite(fusion_outputs['fused_maps'])).sum().item())
            raise ValueError(f"fusion_outputs['fused_maps'] 出现非有限值，bad_count={bad_count}")
        if not torch.isfinite(fusion_outputs['global_feature']).all():
            bad_count = int((~torch.isfinite(fusion_outputs['global_feature'])).sum().item())
            raise ValueError(f"fusion_outputs['global_feature'] 出现非有限值，bad_count={bad_count}")
        
        proposal_outputs = self.proposal_decoder(fusion_outputs['fused_maps'], fusion_outputs['global_feature'])
        if 'proposal_boxes' in inputs and inputs['proposal_boxes'] is not None:
            proposal_outputs = self._resolve_external_proposals(inputs, proposal_outputs, fusion_outputs['global_feature'])
        proposal_boxes = proposal_outputs['proposal_boxes']
        stage2_boxes = self._mix_boxes_for_stage2(proposal_boxes, targets)

        if not torch.isfinite(stage2_boxes).all():
            bad_count = int((~torch.isfinite(stage2_boxes)).sum().item())
            raise ValueError(f"stage2_boxes 出现非有限值，bad_count={bad_count}")
        context_boxes = self._expand_boxes(stage2_boxes, self.context_scale)
        if not torch.isfinite(context_boxes).all():
            bad_count = int((~torch.isfinite(context_boxes)).sum().item())
            raise ValueError(f"context_boxes 出现非有限值，bad_count={bad_count}")
        
        local_roi_maps = self._extract_roi_maps(fusion_outputs['fused_maps'], stage2_boxes)
        context_roi_maps = self._extract_roi_maps(fusion_outputs['fused_maps'], context_boxes)
        local_roi_outputs = self.roi_encoder(local_roi_maps)
        context_roi_outputs = self.roi_encoder(context_roi_maps)

        stage2_outputs = self.stage_two_predictor(
            local_roi_outputs['pooled_feature'],
            context_roi_outputs['pooled_feature'],
            proposal_outputs['proposal_features'],
            fusion_outputs['global_feature'],
            proposal_outputs['proposal_scores'],
            stage2_boxes,
            local_roi_outputs['sequence'],
            context_roi_outputs['sequence'],
        )

        physics_loss = torch.tensor(0.0, device=stage2_outputs['class_logits'].device)
        if self.config.get('use_physical_constraints', False) and physics_inputs is not None:
            from .physical_constraints import PhysicsConstraintModule
            physics_module = PhysicsConstraintModule(self.config.get('physics', {})).to(stage2_outputs['class_logits'].device)
            physics_loss = physics_module(fusion_outputs['global_feature'], stage2_outputs['class_probs'], physics_inputs)
            if not isinstance(physics_loss, torch.Tensor):
                physics_loss = torch.tensor(physics_loss, device=stage2_outputs['class_logits'].device, dtype=torch.float32)

        outputs = {
            'class_logits': stage2_outputs['class_logits'],
            'class_probs': stage2_outputs['class_probs'],
            'bbox_pred': stage2_outputs['bbox_pred'],
            'time_pred': stage2_outputs['time_pred'],
            'time_params': stage2_outputs['time_params'],
            'time_center': stage2_outputs['time_center'],
            'time_duration': stage2_outputs['time_duration'],
            'time_duration_raw': stage2_outputs['time_duration_raw'],
            'event_prob': stage2_outputs['event_prob'],
            'event_gate': stage2_outputs['event_gate'],
            'bbox_size_raw': stage2_outputs['bbox_size_raw'],
            'bbox_size_gated': stage2_outputs['bbox_size_gated'],
            'physics_loss': physics_loss,
            'available_modalities': available_modalities,
            'features': fusion_outputs['global_feature'],
            'global_feature': fusion_outputs['global_feature'],
            'temporal_features': fusion_outputs['temporal_features'],
            'fused_maps': fusion_outputs['fused_maps'],
            'proposal_boxes': proposal_boxes,
            'stage2_input_boxes': stage2_boxes,
            'proposal_scores': proposal_outputs['proposal_scores'],
            'proposal_features': proposal_outputs['proposal_features'],
            'spatiotemporal_tokens': proposal_outputs['spatiotemporal_tokens'],
            'external_proposals': bool(proposal_outputs.get('external_proposals', False)),
            'slot_features': stage2_outputs['slot_features'],
            'local_roi_feature': local_roi_outputs['pooled_feature'],
            'context_roi_feature': context_roi_outputs['pooled_feature'],
            'local_roi_sequence': local_roi_outputs['sequence'],
            'context_roi_sequence': context_roi_outputs['sequence'],
            'local_roi_maps': local_roi_maps,
            'context_roi_maps': context_roi_maps,
            'context_boxes': context_boxes,
            'refine_delta': stage2_outputs['refine_delta'],
        }
        for name in ['class_logits', 'bbox_pred', 'time_pred', 'event_prob', 'physics_loss']:
            if not torch.isfinite(outputs[name]).all():
                raise ValueError(f"模型输出 {name} 出现非有限值")
        return outputs

    def predict_with_uncertainty(self, inputs: Dict[str, torch.Tensor], num_samples: int = 10) -> Dict[str, torch.Tensor]:
        self.train()
        predictions = {
            'class_probs_samples': [],
            'class_logits_samples': [],
            'bbox_samples': [],
            'time_samples': [],
            'event_prob_samples': [],
        }
        deterministic_outputs: Optional[Dict[str, torch.Tensor]] = None
        for _ in range(num_samples):
            outputs = self.forward(inputs)
            if deterministic_outputs is None:
                deterministic_outputs = {
                    'proposal_boxes': outputs.get('proposal_boxes'),
                    'context_boxes': outputs.get('context_boxes'),
                    'proposal_scores': outputs.get('proposal_scores'),
                    'proposal_features': outputs.get('proposal_features'),
                    'global_feature': outputs.get('global_feature'),
                    'bbox_size_gated': outputs.get('bbox_size_gated'),
                    'refine_delta': outputs.get('refine_delta'),
                }
            predictions['class_probs_samples'].append(outputs['class_probs'])
            predictions['class_logits_samples'].append(outputs['class_logits'])
            predictions['bbox_samples'].append(outputs['bbox_pred'])
            predictions['time_samples'].append(outputs['time_pred'])
            predictions['event_prob_samples'].append(outputs['event_prob'])

        class_probs_samples = torch.stack(predictions['class_probs_samples'], dim=0)
        class_logits_samples = torch.stack(predictions['class_logits_samples'], dim=0)
        bbox_samples = torch.stack(predictions['bbox_samples'], dim=0)
        time_samples = torch.stack(predictions['time_samples'], dim=0)
        event_prob_samples = torch.stack(predictions['event_prob_samples'], dim=0)
        result = {
            'class_probs_mean': class_probs_samples.mean(dim=0),
            'class_probs_std': class_probs_samples.std(dim=0),
            'class_logits_mean': class_logits_samples.mean(dim=0),
            'class_logits_std': class_logits_samples.std(dim=0),
            'bbox_pred_mean': bbox_samples.mean(dim=0),
            'bbox_pred_std': bbox_samples.std(dim=0),
            'time_pred_mean': time_samples.mean(dim=0),
            'time_pred_std': time_samples.std(dim=0),
            'event_prob_mean': event_prob_samples.mean(dim=0),
            'event_prob_std': event_prob_samples.std(dim=0),
            'predicted_class': class_probs_samples.mean(dim=0).argmax(dim=-1),
        }
        if deterministic_outputs:
            result.update(deterministic_outputs)
        return result
