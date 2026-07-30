"""
多模态时序样本变换模块。
"""
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as torch_functional


logger = logging.getLogger(__name__)


class ComposeSampleTransforms:
    """顺序执行 sample 级变换。"""

    def __init__(self, transforms: Iterable[Any]):
        self.transforms = list(transforms)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class NormalizeTransform:
    """使用统一均值和标准差归一化全部模态。"""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        if std == 0:
            raise ValueError("NormalizeTransform 的 std 不能为 0")
        self.mean = float(mean)
        self.std = float(std)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'data' not in sample:
            return sample

        sample['data'] = {
            modality: (data - self.mean) / self.std
            if isinstance(data, torch.Tensor) else data
            for modality, data in sample['data'].items()
        }
        return sample


class RandomSynchronizedAffine:
    """同步变换所有模态、时间帧和空间框字段。"""

    BOX_FIELDS = {
        'bbox': 'activity_mask',
        'region_bbox': 'region_mask',
        'proposal_boxes': None,
    }

    def __init__(self, config: Dict[str, Any]):
        self.probability = float(config.get('probability', 1.0))
        self.rotation_range = self._symmetric_or_range(
            config.get('rotation_degrees', config.get('rotation_range', 0.0)),
            default=(0.0, 0.0),
        )
        self.translate_fraction = self._pair(
            config.get('translate_fraction', (0.0, 0.0)),
            default=(0.0, 0.0),
        )
        self.scale_range = self._ordered_range(
            config.get('scale_range', (1.0, 1.0)),
            default=(1.0, 1.0),
        )
        self.horizontal_flip_probability = float(
            config.get('horizontal_flip_probability', 0.0)
        )
        self.vertical_flip_probability = float(
            config.get('vertical_flip_probability', 0.0)
        )
        self.fill_value = float(config.get('fill_value', 0.0))
        self.interpolation = str(config.get('interpolation', 'bilinear')).lower()
        self.min_visibility = float(config.get('min_visibility', 0.6))
        self.min_box_size_pixels = float(config.get('min_box_size_pixels', 4.0))
        self.max_resample_attempts = max(
            1,
            int(config.get('max_resample_attempts', 10)),
        )
        self.record_parameters = bool(config.get('record_parameters', False))
        self.validate_every_sample = bool(config.get('validate_every_sample', False))

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("spatial.probability 必须位于 [0, 1]")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability 必须位于 [0, 1]")
        if not 0.0 <= self.vertical_flip_probability <= 1.0:
            raise ValueError("vertical_flip_probability 必须位于 [0, 1]")
        if self.scale_range[0] <= 0:
            raise ValueError("scale_range 必须为正数")
        if self.interpolation not in {'bilinear', 'nearest'}:
            raise ValueError("interpolation 仅支持 bilinear 或 nearest")

    @staticmethod
    def _pair(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            scalar = float(value)
            return scalar, scalar
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"期望长度为 2 的配置，实际得到 {value}")
        return float(values[0]), float(values[1])

    @classmethod
    def _ordered_range(
            cls,
            value: Any,
            default: Tuple[float, float],
    ) -> Tuple[float, float]:
        low, high = cls._pair(value, default)
        return min(low, high), max(low, high)

    @classmethod
    def _symmetric_or_range(
            cls,
            value: Any,
            default: Tuple[float, float],
    ) -> Tuple[float, float]:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            magnitude = abs(float(value))
            return -magnitude, magnitude
        return cls._ordered_range(value, default)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        if low == high:
            return float(low)
        return float(torch.empty((), dtype=torch.float32).uniform_(low, high).item())

    @staticmethod
    def _bernoulli(probability: float) -> bool:
        if probability <= 0:
            return False
        if probability >= 1:
            return True
        return bool(torch.rand((), dtype=torch.float32).item() < probability)

    def _sample_parameters(self) -> Dict[str, Any]:
        return {
            'angle': self._uniform(*self.rotation_range),
            'translate_x_fraction': self._uniform(
                -abs(self.translate_fraction[0]),
                abs(self.translate_fraction[0]),
            ),
            'translate_y_fraction': self._uniform(
                -abs(self.translate_fraction[1]),
                abs(self.translate_fraction[1]),
            ),
            'scale': self._uniform(*self.scale_range),
            'flip_horizontal': self._bernoulli(
                self.horizontal_flip_probability
            ),
            'flip_vertical': self._bernoulli(
                self.vertical_flip_probability
            ),
        }

    @staticmethod
    def _translation(x: float, y: float) -> torch.Tensor:
        return torch.tensor(
            [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )

    def _build_forward_edge_matrix(
            self,
            height: int,
            width: int,
            parameters: Dict[str, Any],
    ) -> torch.Tensor:
        center_x = width / 2.0
        center_y = height / 2.0
        angle_radians = math.radians(float(parameters['angle']))
        cos_value = math.cos(angle_radians)
        sin_value = math.sin(angle_radians)
        flip_x = -1.0 if parameters['flip_horizontal'] else 1.0
        flip_y = -1.0 if parameters['flip_vertical'] else 1.0
        scale = float(parameters['scale'])

        rotation = torch.tensor(
            [
                [cos_value, sin_value, 0.0],
                [-sin_value, cos_value, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        scale_and_flip = torch.tensor(
            [
                [scale * flip_x, 0.0, 0.0],
                [0.0, scale * flip_y, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        translate_x = float(parameters['translate_x_fraction']) * width
        translate_y = float(parameters['translate_y_fraction']) * height

        return (
            self._translation(center_x + translate_x, center_y + translate_y)
            @ rotation
            @ scale_and_flip
            @ self._translation(-center_x, -center_y)
        )

    @staticmethod
    def _pixel_to_normalized_matrix(height: int, width: int) -> torch.Tensor:
        return torch.tensor(
            [
                [2.0 / width, 0.0, (1.0 / width) - 1.0],
                [0.0, 2.0 / height, (1.0 / height) - 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

    @staticmethod
    def _normalized_to_pixel_matrix(height: int, width: int) -> torch.Tensor:
        return torch.tensor(
            [
                [width / 2.0, 0.0, (width - 1.0) / 2.0],
                [0.0, height / 2.0, (height - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

    def _build_grid_theta(
            self,
            forward_edge_matrix: torch.Tensor,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        half_pixel_to_edge = self._translation(0.5, 0.5)
        edge_to_half_pixel = self._translation(-0.5, -0.5)
        forward_pixel_matrix = (
            edge_to_half_pixel
            @ forward_edge_matrix
            @ half_pixel_to_edge
        )
        inverse_pixel_matrix = torch.linalg.inv(forward_pixel_matrix)
        normalized_matrix = (
            self._pixel_to_normalized_matrix(height, width)
            @ inverse_pixel_matrix
            @ self._normalized_to_pixel_matrix(height, width)
        )
        return normalized_matrix[:2].to(device=device, dtype=dtype)

    def _warp_tensor(
            self,
            data: torch.Tensor,
            parameters: Dict[str, Any],
    ) -> torch.Tensor:
        if data.ndim != 4:
            return data
        if not data.is_floating_point():
            raise TypeError("空间增强要求模态 tensor 为浮点类型")

        time_steps, channels, height, width = data.shape
        forward_edge_matrix = self._build_forward_edge_matrix(
            height,
            width,
            parameters,
        )
        theta = self._build_grid_theta(
            forward_edge_matrix,
            height,
            width,
            data.dtype,
            data.device,
        ).unsqueeze(0).expand(time_steps, -1, -1)
        grid = torch_functional.affine_grid(
            theta,
            size=(time_steps, channels, height, width),
            align_corners=False,
        )
        warped = torch_functional.grid_sample(
            data,
            grid,
            mode=self.interpolation,
            padding_mode='zeros',
            align_corners=False,
        )
        if self.fill_value != 0.0:
            valid_source = torch.ones(
                (time_steps, 1, height, width),
                dtype=data.dtype,
                device=data.device,
            )
            valid_warp = torch_functional.grid_sample(
                valid_source,
                grid,
                mode='nearest',
                padding_mode='zeros',
                align_corners=False,
            )
            warped = torch.where(
                valid_warp.expand_as(warped) > 0.5,
                warped,
                torch.as_tensor(
                    self.fill_value,
                    dtype=data.dtype,
                    device=data.device,
                ),
            )
        return warped

    @staticmethod
    def _proposal_mask(sample: Dict[str, Any], boxes: torch.Tensor) -> torch.Tensor:
        scores = sample.get('proposal_scores')
        if isinstance(scores, torch.Tensor):
            if scores.ndim > 1 and scores.shape[-1] == 1:
                scores = scores.squeeze(-1)
            if scores.shape == boxes.shape[:-1]:
                return scores > 0
        return torch.ones(boxes.shape[:-1], dtype=torch.bool, device=boxes.device)

    @staticmethod
    def _field_mask(
            sample: Dict[str, Any],
            field_name: str,
            mask_name: Optional[str],
            boxes: torch.Tensor,
    ) -> torch.Tensor:
        if field_name == 'proposal_boxes':
            return RandomSynchronizedAffine._proposal_mask(sample, boxes)
        mask = sample.get(mask_name) if mask_name else None
        if isinstance(mask, torch.Tensor) and mask.shape == boxes.shape[:-1]:
            return mask.to(device=boxes.device, dtype=torch.bool)
        return torch.ones(boxes.shape[:-1], dtype=torch.bool, device=boxes.device)

    def _transform_boxes(
            self,
            boxes: torch.Tensor,
            mask: torch.Tensor,
            forward_edge_matrix: torch.Tensor,
            height: int,
            width: int,
    ) -> Dict[str, torch.Tensor]:
        original_shape = boxes.shape
        flat_boxes = boxes.reshape(-1, 4)
        flat_mask = mask.reshape(-1)
        finite = torch.isfinite(flat_boxes).all(dim=-1)
        ordered = (
            (flat_boxes[:, 2] > flat_boxes[:, 0])
            & (flat_boxes[:, 3] > flat_boxes[:, 1])
        )
        valid_original = flat_mask & finite & ordered

        x1 = flat_boxes[:, 0] * width
        y1 = flat_boxes[:, 1] * height
        x2 = flat_boxes[:, 2] * width
        y2 = flat_boxes[:, 3] * height
        corners = torch.stack(
            [
                torch.stack([x1, y1], dim=-1),
                torch.stack([x2, y1], dim=-1),
                torch.stack([x2, y2], dim=-1),
                torch.stack([x1, y2], dim=-1),
            ],
            dim=1,
        )
        homogeneous_corners = torch.cat(
            [
                corners,
                torch.ones(
                    (*corners.shape[:-1], 1),
                    dtype=boxes.dtype,
                    device=boxes.device,
                ),
            ],
            dim=-1,
        )
        matrix = forward_edge_matrix.to(
            device=boxes.device,
            dtype=boxes.dtype,
        )
        transformed_corners = homogeneous_corners @ matrix.transpose(0, 1)
        transformed_xy = transformed_corners[..., :2]

        transformed_min = transformed_xy.amin(dim=1)
        transformed_max = transformed_xy.amax(dim=1)
        unclipped_width = (transformed_max[:, 0] - transformed_min[:, 0]).clamp(min=0)
        unclipped_height = (transformed_max[:, 1] - transformed_min[:, 1]).clamp(min=0)
        unclipped_area = unclipped_width * unclipped_height

        clipped_min_x = transformed_min[:, 0].clamp(0.0, float(width))
        clipped_min_y = transformed_min[:, 1].clamp(0.0, float(height))
        clipped_max_x = transformed_max[:, 0].clamp(0.0, float(width))
        clipped_max_y = transformed_max[:, 1].clamp(0.0, float(height))
        clipped_width = (clipped_max_x - clipped_min_x).clamp(min=0)
        clipped_height = (clipped_max_y - clipped_min_y).clamp(min=0)
        clipped_area = clipped_width * clipped_height
        visibility = clipped_area / unclipped_area.clamp(min=1e-8)
        transformed_finite = torch.isfinite(transformed_xy).reshape(
            flat_boxes.shape[0],
            -1,
        ).all(dim=1)
        valid_after = (
            valid_original
            & transformed_finite
            & (clipped_width >= self.min_box_size_pixels)
            & (clipped_height >= self.min_box_size_pixels)
            & (visibility >= self.min_visibility)
        )

        transformed_boxes = torch.zeros_like(flat_boxes)
        transformed_boxes[:, 0] = clipped_min_x / width
        transformed_boxes[:, 1] = clipped_min_y / height
        transformed_boxes[:, 2] = clipped_max_x / width
        transformed_boxes[:, 3] = clipped_max_y / height
        transformed_boxes[~valid_after] = 0.0

        return {
            'boxes': transformed_boxes.reshape(original_shape),
            'valid_original': valid_original.reshape(mask.shape),
            'valid_after': valid_after.reshape(mask.shape),
            'visibility': visibility.reshape(mask.shape),
        }

    def _prepare_box_results(
            self,
            sample: Dict[str, Any],
            parameters: Dict[str, Any],
            height: int,
            width: int,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], bool]:
        forward_edge_matrix = self._build_forward_edge_matrix(
            height,
            width,
            parameters,
        )
        results = {}
        required_fields_valid = True

        for field_name, mask_name in self.BOX_FIELDS.items():
            boxes = sample.get(field_name)
            if not isinstance(boxes, torch.Tensor):
                continue
            if boxes.shape[-1] != 4:
                raise ValueError(
                    f"{field_name} 的最后一维应为 4，实际为 {tuple(boxes.shape)}"
                )
            mask = self._field_mask(
                sample,
                field_name,
                mask_name,
                boxes,
            )
            result = self._transform_boxes(
                boxes,
                mask,
                forward_edge_matrix,
                height,
                width,
            )
            results[field_name] = result
            if field_name != 'proposal_boxes':
                valid_original = result['valid_original']
                if valid_original.any():
                    required_fields_valid = (
                        required_fields_valid
                        and bool(result['valid_after'][valid_original].all().item())
                    )

        return results, required_fields_valid

    @staticmethod
    def _common_image_size(sample: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        sizes = {
            tuple(data.shape[-2:])
            for data in sample.get('data', {}).values()
            if isinstance(data, torch.Tensor) and data.ndim == 4
        }
        if not sizes:
            return None
        if len(sizes) != 1:
            raise ValueError(f"同步空间增强要求各模态尺寸一致，实际为 {sorted(sizes)}")
        return next(iter(sizes))

    def _record(
            self,
            sample: Dict[str, Any],
            applied: bool,
            parameters: Optional[Dict[str, Any]] = None,
            reason: Optional[str] = None,
    ) -> None:
        if not self.record_parameters:
            return
        metadata = sample.get('metadata')
        if not isinstance(metadata, dict):
            return
        augmentation = metadata.setdefault('augmentation', {})
        augmentation['spatial'] = {
            'applied': applied,
            'reason': reason,
            **(parameters or {}),
        }

    def _validate_output(self, sample: Dict[str, Any]) -> None:
        for modality, data in sample.get('data', {}).items():
            if isinstance(data, torch.Tensor) and not torch.isfinite(data).all():
                raise ValueError(f"增强后模态 {modality} 出现非有限值")

        for field_name, mask_name in self.BOX_FIELDS.items():
            boxes = sample.get(field_name)
            if not isinstance(boxes, torch.Tensor):
                continue
            mask = self._field_mask(
                sample,
                field_name,
                mask_name,
                boxes,
            )
            finite = torch.isfinite(boxes).all(dim=-1)
            in_bounds = ((boxes >= 0.0) & (boxes <= 1.0)).all(dim=-1)
            ordered_or_zero = (
                (
                    (boxes[..., 2] > boxes[..., 0])
                    & (boxes[..., 3] > boxes[..., 1])
                )
                | (boxes == 0).all(dim=-1)
            )
            if not bool((finite & in_bounds & ordered_or_zero).all().item()):
                raise ValueError(f"增强后 {field_name} 不满足归一化 xyxy 约束")
            if field_name != 'proposal_boxes' and not bool(
                    ordered_or_zero[mask].all().item()
            ):
                raise ValueError(f"增强后有效 {field_name} 出现非法框")

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_size = self._common_image_size(sample)
        if image_size is None:
            self._record(sample, False, reason='missing_image_tensor')
            return sample
        if not self._bernoulli(self.probability):
            self._record(sample, False, reason='probability')
            return sample

        height, width = image_size
        selected_parameters = None
        selected_box_results = None

        for _ in range(self.max_resample_attempts):
            parameters = self._sample_parameters()
            box_results, required_fields_valid = self._prepare_box_results(
                sample,
                parameters,
                height,
                width,
            )
            if required_fields_valid:
                selected_parameters = parameters
                selected_box_results = box_results
                break

        if selected_parameters is None or selected_box_results is None:
            self._record(sample, False, reason='gt_visibility_rejected')
            return sample

        missing_modalities = set(
            sample.get('metadata', {}).get('missing_modalities', [])
            if isinstance(sample.get('metadata'), dict) else []
        )
        sample['data'] = {
            modality: self._warp_tensor(data, selected_parameters)
            if isinstance(data, torch.Tensor) and modality not in missing_modalities
            else data
            for modality, data in sample['data'].items()
        }

        for field_name, result in selected_box_results.items():
            sample[field_name] = result['boxes']

        proposal_result = selected_box_results.get('proposal_boxes')
        proposal_scores = sample.get('proposal_scores')
        if proposal_result is not None and isinstance(proposal_scores, torch.Tensor):
            removed = (
                self._proposal_mask(sample, sample['proposal_boxes'])
                & ~proposal_result['valid_after']
            )
            if proposal_scores.ndim > removed.ndim:
                removed = removed.unsqueeze(-1)
            sample['proposal_scores'] = proposal_scores.masked_fill(removed, 0.0)

        self._record(sample, True, selected_parameters)
        if self.validate_every_sample:
            self._validate_output(sample)
        return sample


class RandomRotate(RandomSynchronizedAffine):
    """兼容旧接口的同步随机旋转。"""

    def __init__(self, degrees: float = 30.0, p: float = 0.5):
        super().__init__({
            'probability': p,
            'rotation_degrees': degrees,
            'min_visibility': 0.0,
            'min_box_size_pixels': 0.0,
        })


class RandomFlip(RandomSynchronizedAffine):
    """兼容旧接口的同步随机翻转。"""

    def __init__(
            self,
            horizontal: bool = True,
            vertical: bool = False,
            p: float = 0.5,
    ):
        super().__init__({
            'probability': 1.0,
            'horizontal_flip_probability': p if horizontal else 0.0,
            'vertical_flip_probability': p if vertical else 0.0,
            'min_visibility': 0.0,
            'min_box_size_pixels': 0.0,
        })


class RandomModalityIntensity:
    """按模态执行物理量范围受控的强度扰动。"""

    def __init__(
            self,
            config: Dict[str, Any],
            modalities_config: Optional[Dict[str, Any]] = None,
    ):
        self.probability = float(config.get('probability', 0.0))
        self.per_modality = config.get('per_modality', {}) or {}
        self.modalities_config = modalities_config or {}
        self.record_parameters = bool(config.get('record_parameters', False))

    @staticmethod
    def _uniform(value_range: Any, default: Tuple[float, float]) -> float:
        low, high = RandomSynchronizedAffine._ordered_range(
            value_range,
            default,
        )
        return RandomSynchronizedAffine._uniform(low, high)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if not RandomSynchronizedAffine._bernoulli(self.probability):
            return sample

        missing_modalities = set(
            sample.get('metadata', {}).get('missing_modalities', [])
            if isinstance(sample.get('metadata'), dict) else []
        )
        applied_parameters = {}
        transformed_data = {}

        for modality, data in sample.get('data', {}).items():
            modality_config = self.per_modality.get(modality)
            if (
                    modality in missing_modalities
                    or not isinstance(data, torch.Tensor)
                    or not modality_config
            ):
                transformed_data[modality] = data
                continue

            gain = self._uniform(
                modality_config.get('gain_range', (1.0, 1.0)),
                (1.0, 1.0),
            )
            normalization = self.modalities_config.get(
                modality,
                {},
            ).get('normalization')
            if (
                    isinstance(normalization, (list, tuple))
                    and len(normalization) == 2
            ):
                lower = float(normalization[0])
                upper = float(normalization[1])
                value_span = max(upper - lower, 1e-8)
            else:
                lower = None
                upper = None
                value_span = float(
                    (data.detach().amax() - data.detach().amin()).clamp(min=1e-8).item()
                )

            bias_fraction = float(modality_config.get('bias_fraction', 0.0))
            noise_std_fraction = float(
                modality_config.get('noise_std_fraction', 0.0)
            )
            bias = self._uniform(
                (-abs(bias_fraction), abs(bias_fraction)),
                (0.0, 0.0),
            ) * value_span
            noise_std = abs(noise_std_fraction) * value_span

            transformed = data * gain + bias
            if noise_std > 0:
                transformed = transformed + torch.randn_like(data) * noise_std
            if lower is not None and upper is not None:
                transformed = transformed.clamp(lower, upper)
            transformed_data[modality] = transformed
            applied_parameters[modality] = {
                'gain': gain,
                'bias': bias,
                'noise_std': noise_std,
            }

        sample['data'] = transformed_data
        if self.record_parameters and isinstance(sample.get('metadata'), dict):
            augmentation = sample['metadata'].setdefault('augmentation', {})
            augmentation['intensity'] = applied_parameters
        return sample


class ResizeTransform:
    """调整全部模态空间尺寸，归一化 bbox 无需改变。"""

    def __init__(self, size: Tuple[int, int]):
        self.size = tuple(int(value) for value in size)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'data' not in sample:
            return sample

        resized_data = {}
        for modality, data in sample['data'].items():
            if isinstance(data, torch.Tensor) and data.ndim == 4:
                resized_data[modality] = torch_functional.interpolate(
                    data,
                    size=self.size,
                    mode='bilinear',
                    align_corners=False,
                )
            else:
                resized_data[modality] = data
        sample['data'] = resized_data
        return sample


def _legacy_spatial_config(augmentation_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'enabled': True,
        'probability': 1.0,
        'rotation_degrees': augmentation_config.get('rotation_range', 0.0),
        'horizontal_flip_probability': (
            0.5 if augmentation_config.get('flip_horizontal', False) else 0.0
        ),
        'vertical_flip_probability': (
            0.5 if augmentation_config.get('flip_vertical', False) else 0.0
        ),
    }


def create_transforms(config: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    """根据 data config 创建 train/val/test sample 级变换。"""
    data_config = config.get('data', config)
    train_transforms: List[Any] = []
    validation_transforms: List[Any] = []
    augmentation_config = data_config.get('augmentation', {}) or {}

    if augmentation_config.get('enabled', False):
        spatial_config = augmentation_config.get('spatial')
        if spatial_config is None:
            spatial_config = _legacy_spatial_config(augmentation_config)
        spatial_config = dict(spatial_config)
        debug_config = augmentation_config.get('debug', {}) or {}
        spatial_config.setdefault(
            'record_parameters',
            debug_config.get('record_parameters', False),
        )
        spatial_config.setdefault(
            'validate_every_sample',
            debug_config.get('validate_every_sample', False),
        )
        if spatial_config.get('enabled', True):
            train_transforms.append(RandomSynchronizedAffine(spatial_config))

        intensity_config = augmentation_config.get('intensity', {}) or {}
        if intensity_config.get('enabled', False):
            intensity_config = dict(intensity_config)
            intensity_config.setdefault(
                'record_parameters',
                debug_config.get('record_parameters', False),
            )
            train_transforms.append(
                RandomModalityIntensity(
                    intensity_config,
                    data_config.get('modalities', {}),
                )
            )

    normalize_config = data_config.get('normalize', {}) or {}
    if normalize_config.get('enabled', False):
        normalize_transform = NormalizeTransform(
            mean=normalize_config.get('mean', 0.0),
            std=normalize_config.get('std', 1.0),
        )
        train_transforms.append(normalize_transform)
        validation_transforms.append(normalize_transform)

    if 'target_size' in data_config:
        resize_transform = ResizeTransform(tuple(data_config['target_size']))
        train_transforms.append(resize_transform)
        validation_transforms.append(resize_transform)

    def compose(transforms: List[Any]) -> Optional[ComposeSampleTransforms]:
        return ComposeSampleTransforms(transforms) if transforms else None

    transforms = {
        'train': compose(train_transforms),
        'val': compose(validation_transforms),
        'test': compose(validation_transforms),
    }
    logger.info(
        "创建数据变换: augmentation=%s, train_steps=%d, val_steps=%d",
        bool(augmentation_config.get('enabled', False)),
        len(train_transforms),
        len(validation_transforms),
    )
    return transforms


__all__ = [
    'ComposeSampleTransforms',
    'NormalizeTransform',
    'RandomSynchronizedAffine',
    'RandomRotate',
    'RandomFlip',
    'RandomModalityIntensity',
    'ResizeTransform',
    'create_transforms',
]
