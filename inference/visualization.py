"""
推理结果可视化模块 - 在原图上绘制预测的 bbox 与标注信息

坐标约定（与训练 / HDF5 一致）:
- bbox 为归一化坐标 [x1,y1,x2,y2]，取值约 [0,1]，相对 **图像宽高** 线性缩放；
- **(0,0) 对应图像左上角**（与 matplotlib imshow 默认 origin='upper' 一致），x 向右、y 向下；
  不是日面中心坐标系。
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _parse_iso_datetime(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    s = str(ts).strip()
    try:
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _to_numpy_cpu(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """将 tensor / list / ndarray 安全转为 CPU numpy。"""
    if value is None:
        return np.asarray([], dtype=dtype)
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, list):
        converted = []
        for item in value:
            if isinstance(item, torch.Tensor):
                converted.append(item.detach().cpu().numpy())
            else:
                converted.append(item)
        arr = np.asarray(converted, dtype=dtype)
    else:
        arr = np.asarray(value, dtype=dtype)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _bboxes_to_xyxy_list(bboxes: Any) -> List[np.ndarray]:
    """将 processed_bboxes 转为 List[ length-4 float ndarray ]。"""
    if bboxes is None:
        return []
    arr = _to_numpy_cpu(bboxes, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.ndim == 1 and arr.shape[0] == 4:
        return [arr]
    if arr.ndim == 2 and arr.shape[1] == 4:
        return [arr[i] for i in range(arr.shape[0])]
    return []


def _ref_dt_from_window(metadata: Dict[str, Any], reference_frame: Union[str, int]) -> Optional[datetime]:
    """与底图同一帧对应的时间戳（window_timestamps 与窗口对齐）。"""
    ts_list = metadata.get('window_timestamps', [])
    if isinstance(ts_list, np.ndarray):
        ts_list = ts_list.tolist()
    if not ts_list:
        return None
    n = len(ts_list)
    if isinstance(reference_frame, int):
        idx = max(0, min(int(reference_frame), n - 1))
    elif reference_frame == 'first':
        idx = 0
    elif reference_frame == 'middle':
        idx = n // 2
    else:
        idx = n - 1
    return _parse_iso_datetime(str(ts_list[idx]))


def _fmt_utc(dt: Optional[datetime]) -> str:
    if dt is None:
        return 'N/A'
    return dt.strftime('%Y-%m-%d %H:%M:%S') + ' UTC'


def _compute_start_peak_end_utc(
    ref_dt: Optional[datetime],
    tp_proc: List[float],
) -> Tuple[str, str, str]:
    """由参考时刻 + 后处理时间向量得到 start / peak / end 的 UTC 字符串。
    tp_proc 格式：[start_offset_hours, peak_offset_hours, end_offset_hours]
    """
    if ref_dt is None or not tp_proc or len(tp_proc) < 3:
        return 'N/A', 'N/A', 'N/A'
    try:
        t0, t1, t2 = float(tp_proc[0]), float(tp_proc[1]), float(tp_proc[2])
        abs_start = ref_dt + timedelta(hours=t0)
        abs_peak = ref_dt + timedelta(hours=t1)
        abs_end = ref_dt + timedelta(hours=t2)
        return _fmt_utc(abs_start), _fmt_utc(abs_peak), _fmt_utc(abs_end)
    except Exception:
        return 'N/A', 'N/A', 'N/A'


def _flare_category_line(cls: int) -> str:
    """仅 1/2 为耀斑类型；0 为无事件。"""
    if cls == 1:
        return '1: Eruptive Flare (CME)'
    if cls == 2:
        return '2: Confined Flare'
    return '0: No Event'


class PredictionVisualizer:
    """预测结果可视化器"""

    def __init__(self, image_size: Tuple[int, int] = (256, 256), time_head_enabled: bool = True):
        """
        初始化可视化器
        
        Args:
            image_size: 默认图像大小 (H, W)；若实际图像尺寸不同，绘制时会以实际 ndarray 形状为准
        """
        self.image_size = image_size
        self.time_head_enabled = bool(time_head_enabled)
        self.class_names = {0: 'No Event', 1: 'Eruptive (CME)', 2: 'Confined'}
        self.class_colors = {0: 'green', 1: 'red', 2: 'yellow'}

    def visualize_prediction(self,
                            modality_images: Dict[str, np.ndarray],
                            predictions: Dict[str, Any],
                            event_id: str,
                            window_info: Optional[Dict] = None,
                            gt_sample: Optional[Dict[str, Any]] = None,
                            output_dir: Optional[Path] = None,
                            reference_frame: Union[str, int] = 'last') -> Optional[plt.Figure]:
        """
        可视化单个窗口的预测结果

        Args:
            modality_images: 模态图像字典 {modality: (T, H, W)}
            predictions: 预测结果字典
            event_id: 事件ID
            window_info: 窗口信息（时间戳等）
            output_dir: 输出目录
            reference_frame: 使用哪一帧作底图：'last'（默认，与序列末端对齐）| 'first' | 'middle' | 非负整数索引

        Returns:
            matplotlib Figure 对象
        """
        try:
            # 获取第一个可用的模态作为背景
            background_image, modality_used = self._get_background_image(modality_images, reference_frame)
            if background_image is None:
                logger.warning("没有可用的背景图像")
                return None

            H, W = int(background_image.shape[0]), int(background_image.shape[1])

            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 10))

            # 显示背景图像（origin=upper，左上角为 (0,0)）
            ax.imshow(background_image, cmap='gray', origin='upper', aspect='equal')
            if gt_sample is not None:
                self._draw_gt_bboxes(ax, gt_sample, H, W)

            # 获取预测信息：优先使用上游统一构建好的 canonical detections，
            # 保证可视化 / JSON / detailed CSV 使用同一份检测结果。
            detections = predictions.get('detections', None)
            if detections is None:
                detections = []
            if not detections:
                kept_slot_indices = predictions.get('kept_bbox_slot_indices', [])
                if isinstance(kept_slot_indices, torch.Tensor):
                    kept_slot_indices = kept_slot_indices.detach().cpu().numpy().reshape(-1).tolist()
                else:
                    kept_slot_indices = _to_numpy_cpu(kept_slot_indices).reshape(-1).tolist() if kept_slot_indices is not None else []

                bbox_list_raw = _bboxes_to_xyxy_list(predictions.get('processed_bboxes', []))
                proposal_list_raw = _bboxes_to_xyxy_list(predictions.get('slot_proposal_boxes', []))
                context_list_raw = _bboxes_to_xyxy_list(predictions.get('slot_context_boxes', []))
                proposal_scores = predictions.get('slot_proposal_scores', [])
                if isinstance(proposal_scores, torch.Tensor):
                    proposal_scores = proposal_scores.detach().cpu().numpy().reshape(-1).tolist()
                else:
                    proposal_scores = _to_numpy_cpu(proposal_scores).reshape(-1).tolist() if proposal_scores is not None else []

                class_ids = predictions.get('final_classes', predictions.get('predicted_class', []))
                if isinstance(class_ids, torch.Tensor):
                    class_ids = class_ids.detach().cpu().numpy().reshape(-1).tolist()
                else:
                    class_ids = _to_numpy_cpu(class_ids).reshape(-1).tolist() if class_ids is not None else []

                class_confidences = predictions.get('classification_confidences', [])
                if isinstance(class_confidences, torch.Tensor):
                    class_confidences = class_confidences.detach().cpu().numpy().reshape(-1).tolist()
                else:
                    class_confidences = _to_numpy_cpu(class_confidences).reshape(-1).tolist() if class_confidences is not None else []

                event_probs = predictions.get('processed_event_probs', predictions.get('event_prob_mean', []))
                if isinstance(event_probs, torch.Tensor):
                    event_probs = event_probs.detach().cpu().numpy().reshape(-1).tolist()
                else:
                    event_probs = _to_numpy_cpu(event_probs).reshape(-1).tolist() if event_probs is not None else []

                time_preds = np.asarray([])
                if self.time_head_enabled:
                    time_preds = predictions.get('processed_time_pred', [])
                    if isinstance(time_preds, torch.Tensor):
                        time_preds = time_preds.detach().cpu().numpy()
                    else:
                        time_preds = _to_numpy_cpu(time_preds) if time_preds is not None else np.asarray([])
                    if time_preds.ndim == 1 and time_preds.size > 0:
                        time_preds = time_preds.reshape(1, -1)

                if len(bbox_list_raw) == 0:
                    slot_bbox_candidates = predictions.get('slot_processed_bboxes', [])
                    if isinstance(slot_bbox_candidates, torch.Tensor):
                        slot_bbox_candidates = slot_bbox_candidates.detach().cpu().numpy()

                    bbox_list_raw = []
                    fallback_slot_indices = []
                    for slot_idx, slot_box in enumerate(slot_bbox_candidates):
                        if slot_box is None:
                            continue
                        arr = _to_numpy_cpu(slot_box, dtype=np.float32).reshape(-1)
                        if arr.size < 4 or not np.isfinite(arr[:4]).all():
                            continue
                        slot_cls = int(class_ids[slot_idx]) if slot_idx < len(class_ids) else 0
                        if slot_cls == 0:
                            continue
                        bbox_list_raw.append(arr[:4])
                        fallback_slot_indices.append(slot_idx)
                    kept_slot_indices = fallback_slot_indices

                detections = []
                for det_idx, bbox in enumerate(bbox_list_raw):
                    slot_idx = int(kept_slot_indices[det_idx]) if det_idx < len(kept_slot_indices) else det_idx
                    # 这些数组在后处理后已按最终保留 detection 顺序对齐，优先按 det_idx 读取；
                    # slot_idx 仅用于记录原始槽位编号。
                    class_id = int(class_ids[det_idx]) if det_idx < len(class_ids) else 0
                    if class_id == 0:
                        continue
                    proposal_bbox = proposal_list_raw[det_idx] if det_idx < len(proposal_list_raw) else []
                    context_bbox = context_list_raw[det_idx] if det_idx < len(context_list_raw) else []
                    if proposal_bbox is None:
                        proposal_bbox = []
                    if context_bbox is None:
                        context_bbox = []
                    det = {
                        'detection_idx': det_idx,
                        'slot_id': slot_idx,
                        'class_id': class_id,
                        'class_confidence': float(class_confidences[det_idx]) if det_idx < len(class_confidences) else 0.0,
                        'event_probability': float(event_probs[det_idx]) if det_idx < len(event_probs) else 0.0,
                        'proposal_score': float(proposal_scores[det_idx]) if det_idx < len(proposal_scores) else 0.0,
                        'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox),
                        'proposal_bbox': proposal_bbox.tolist() if hasattr(proposal_bbox, 'tolist') else list(proposal_bbox),
                        'context_bbox': context_bbox.tolist() if hasattr(context_bbox, 'tolist') else list(context_bbox),
                        'linked_events': [],
                    }
                    if self.time_head_enabled and det_idx < len(time_preds):
                        slot_time = _to_numpy_cpu(time_preds[det_idx]).reshape(-1)
                        if slot_time.size >= 3:
                            det['time_prediction'] = {
                                'start_offset_hours': float(slot_time[0]),
                                'peak_offset_hours': float(slot_time[1]),
                                'end_offset_hours': float(slot_time[2]),
                            }
                    detections.append(det)

            metadata = predictions.get('metadata', {})
            # 时间偏移的训练目标参考时刻固定为“窗口最后一帧”，
            # 因此时间换算不随 reference_frame（底图选择）变化。
            ref_dt = _ref_dt_from_window(metadata, 'last')

            # 绘制 bbox（保证每个框使用自己的 class/time）
            n_box = len(detections)
            for i, det in enumerate(detections):
                bbox = _to_numpy_cpu(det.get('bbox', []), dtype=np.float32)
                if bbox.size < 4:
                    continue
                proposal_bbox = _to_numpy_cpu(det.get('proposal_bbox', []), dtype=np.float32)
                context_bbox = _to_numpy_cpu(det.get('context_bbox', []), dtype=np.float32)
                cls_id = int(det.get('class_id', 0) or 0)
                time_pred = det.get('time_prediction', {}) or {}
                if self.time_head_enabled:
                    tp_proc = [
                        float(time_pred.get('start_offset_hours', np.nan)),
                        float(time_pred.get('peak_offset_hours', np.nan)),
                        float(time_pred.get('end_offset_hours', np.nan)),
                    ]
                    start_s, peak_s, end_s = _compute_start_peak_end_utc(ref_dt, tp_proc)
                else:
                    start_s = peak_s = end_s = 'N/A'
                proposal_score = float(det.get('proposal_score', 0.0) or 0.0)
                linked_events = det.get('linked_events', []) or []
                self._draw_bbox_with_label(
                    ax, bbox, H, W, cls_id, i, n_box,
                    start_s, peak_s, end_s,
                    ref_dt=ref_dt,
                    proposal_bbox=proposal_bbox,
                    context_bbox=context_bbox,
                    proposal_score=proposal_score,
                    linked_events=linked_events,
                )

            # 标题（简要）
            coord_note = (
                'bbox: [0,1] normalized; (0,0)=top-left; GT: magenta dash-dot'
            )
            title = (
                f'Event: {event_id}  |  {modality_used}  |  frame: {reference_frame}\n'
                f'{coord_note}'
            )
            ax.set_title(title, fontsize=10, fontweight='bold')

            # 移除轴
            ax.set_xticks([])
            ax.set_yticks([])

            plt.tight_layout()

            # 保存图表
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                window_start = metadata.get('window_start_idx', 0)
                safe_event_id = str(event_id).replace('/', '_').replace('\\', '_').replace(':', '_')
                save_path = output_dir / f'{safe_event_id}_prediction_window_{window_start}.png'
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"预测可视化已保存: {save_path}")

            return fig

        except Exception as e:
            logger.error(f"可视化预测失败: {e}")
            return None

    def visualize_event_prediction_summary(
        self,
        modality_images: Dict[str, np.ndarray],
        predictions_list: List[Dict[str, Any]],
        event_id: str,
        gt_samples: Optional[List[Optional[Dict[str, Any]]]] = None,
        output_dir: Optional[Path] = None,
        reference_frame: Union[str, int] = 'last',
        show_gt_proposal_prediction_overlay: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Event-level summary figure.

        Keep this method isolated so the GT/proposal/final overlay can be
        removed later by deleting one call site or toggling the boolean flag.
        """
        try:
            background_image, modality_used = self._get_background_image(modality_images, reference_frame)
            if background_image is None:
                logger.warning("No usable event-level background image")
                return None

            H, W = int(background_image.shape[0]), int(background_image.shape[1])
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(background_image, cmap='gray', origin='upper', aspect='equal')

            if show_gt_proposal_prediction_overlay and gt_samples:
                for gt_sample in gt_samples:
                    if gt_sample is not None:
                        self._draw_gt_bboxes(ax, gt_sample, H, W)

            total_detections = sum(len(pred.get('detections', []) or []) for pred in predictions_list)
            ref_dt = None
            if predictions_list:
                ref_dt = _ref_dt_from_window(predictions_list[-1].get('metadata', {}), 'last')

            det_counter = 0
            for pred in predictions_list:
                detections = pred.get('detections', []) or []
                for det in detections:
                    bbox = _to_numpy_cpu(det.get('bbox', []), dtype=np.float32)
                    if bbox.size < 4:
                        continue
                    proposal_bbox = _to_numpy_cpu(det.get('proposal_bbox', []), dtype=np.float32)
                    context_bbox = _to_numpy_cpu(det.get('context_bbox', []), dtype=np.float32)
                    if not show_gt_proposal_prediction_overlay:
                        proposal_bbox = np.asarray([])
                        context_bbox = np.asarray([])

                    cls_id = int(det.get('class_id', 0) or 0)
                    time_pred = det.get('time_prediction', {}) or {}
                    if self.time_head_enabled:
                        tp_proc = [
                            float(time_pred.get('start_offset_hours', np.nan)),
                            float(time_pred.get('peak_offset_hours', np.nan)),
                            float(time_pred.get('end_offset_hours', np.nan)),
                        ]
                        start_s, peak_s, end_s = _compute_start_peak_end_utc(ref_dt, tp_proc)
                    else:
                        start_s = peak_s = end_s = 'N/A'

                    self._draw_bbox_with_label(
                        ax,
                        bbox,
                        H,
                        W,
                        cls_id,
                        det_counter,
                        max(total_detections, 1),
                        start_s,
                        peak_s,
                        end_s,
                        ref_dt=ref_dt,
                        proposal_bbox=proposal_bbox,
                        context_bbox=context_bbox,
                        proposal_score=float(det.get('proposal_score', 0.0) or 0.0),
                        linked_events=det.get('linked_events', []) or [],
                    )
                    det_counter += 1

            overlay_note = (
                'GT: magenta dash-dot | proposal: orange dashed | final prediction: solid'
                if show_gt_proposal_prediction_overlay
                else 'final prediction only'
            )
            ax.set_title(
                f'Event Summary: {event_id}  |  {modality_used}  |  frame: {reference_frame}\n{overlay_note}',
                fontsize=10,
                fontweight='bold',
            )
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f'event_summary_overlay_{event_id}.png'
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Event-level overlay saved: {save_path}")

            return fig
        except Exception as e:
            logger.error(f"Failed to render event-level overlay: {e}")
            return None

    def _draw_gt_bboxes(
        self,
        ax: plt.Axes,
        gt_sample: Dict[str, Any],
        H: int,
        W: int,
    ) -> None:
        """缁樺埗褰撳墠绐楀彛鐨?GT bbox銆?"""
        try:
            gt_boxes = _to_numpy_cpu(gt_sample.get('bbox'), dtype=np.float32)
            gt_labels = _to_numpy_cpu(gt_sample.get('label')).reshape(-1)
            gt_mask = _to_numpy_cpu(gt_sample.get('activity_mask')).reshape(-1).astype(bool)

            if gt_boxes.ndim == 1 and gt_boxes.size >= 4:
                gt_boxes = gt_boxes.reshape(1, -1)

            num_gt = min(len(gt_boxes), len(gt_labels), len(gt_mask))
            for idx in range(num_gt):
                if not gt_mask[idx]:
                    continue
                box = np.asarray(gt_boxes[idx], dtype=np.float32).reshape(-1)
                if box.size < 4 or not np.isfinite(box[:4]).all():
                    continue
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                if x2 <= x1 or y2 <= y1:
                    continue

                rect = patches.Rectangle(
                    (x1 * W, y1 * H),
                    (x2 - x1) * W,
                    (y2 - y1) * H,
                    linewidth=2.2,
                    edgecolor='magenta',
                    facecolor='none',
                    linestyle='-.',
                    alpha=0.95,
                )
                ax.add_patch(rect)

                gt_class = int(gt_labels[idx]) if idx < len(gt_labels) else -1
                label_text = f"GT {idx + 1}: {self.class_names.get(gt_class, str(gt_class))}"
                ax.text(
                    x1 * W,
                    max(2.0, y1 * H - 2.0),
                    label_text,
                    color='white',
                    fontsize=8,
                    ha='left',
                    va='top',
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        facecolor='magenta',
                        edgecolor='magenta',
                        alpha=0.8,
                    ),
                )
        except Exception as e:
            logger.debug(f"缁樺埗 GT bbox 澶辫触: {e}")

    def _get_background_image(
        self,
        modality_images: Dict[str, np.ndarray],
        reference_frame: Union[str, int] = 'last',
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        获取背景图像，优先级: magnetogram > euv_* > halpha
        
        Args:
            modality_images: 模态图像字典
            reference_frame: 帧选择策略

        Returns:
            (灰度图像 (H, W) 或 None, 使用的模态名)
        """
        priority = ['magnetogram', 'euv_193', 'euv_171', 'euv_94', 'halpha']

        for modality in priority:
            if modality in modality_images:
                images = modality_images[modality]
                if images is not None and len(images) > 0:
                    image = self._pick_frame(images, reference_frame)

                    if isinstance(image, torch.Tensor):
                        image = image.detach().cpu().numpy()
                    image = np.asarray(image, dtype=np.float32)

                    if image.ndim == 3:
                        image = image.mean(axis=0)
                    elif image.ndim == 2:
                        pass
                    else:
                        continue

                    img_max = float(image.max())
                    if img_max > 1.0:
                        img_min = float(image.min())
                        image = (image - img_min) / (img_max - img_min + 1e-8)

                    return image, modality

        return None, ''

    def _pick_frame(self, images: np.ndarray, reference_frame: Union[str, int]) -> np.ndarray:
        """从 (T, H, W) 或 (T, C, H, W) 选取一帧。"""
        t = images.shape[0]
        if isinstance(reference_frame, int):
            idx = max(0, min(reference_frame, t - 1))
        elif reference_frame == 'first':
            idx = 0
        elif reference_frame == 'middle':
            idx = t // 2
        else:
            idx = t - 1
        return images[idx]

    def _draw_bbox_with_label(
        self,
        ax: plt.Axes,
        bbox: np.ndarray,
        H: int,
        W: int,
        final_class: int,
        bbox_idx: int,
        num_bboxes: int,
        start_s: str,
        peak_s: str,
        end_s: str,
        ref_dt: Optional[datetime] = None,
        proposal_bbox: Optional[np.ndarray] = None,
        context_bbox: Optional[np.ndarray] = None,
        proposal_score: float = 0.0,
        linked_events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        绘制 bbox 及标注信息：类别(1/2)、start/peak/end UTC 时间。
        信息框至少一条边与 bbox 的一条边对齐，并尽量放在 bbox 外侧以减少遮挡。
        """
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox.reshape(-1)[:4]]

            x1_pixel = x1 * W
            y1_pixel = y1 * H
            x2_pixel = x2 * W
            y2_pixel = y2 * H

            width = x2_pixel - x1_pixel
            height = y2_pixel - y1_pixel

            color = self.class_colors.get(final_class, 'white')

            if context_bbox is not None and np.asarray(context_bbox).size >= 4:
                cx1, cy1, cx2, cy2 = [float(v) for v in np.asarray(context_bbox).reshape(-1)[:4]]
                context_rect = patches.Rectangle(
                    (cx1 * W, cy1 * H), (cx2 - cx1) * W, (cy2 - cy1) * H,
                    linewidth=1.2, edgecolor='cyan', facecolor='none', linestyle=':', alpha=0.8
                )
                ax.add_patch(context_rect)

            if proposal_bbox is not None and np.asarray(proposal_bbox).size >= 4:
                px1, py1, px2, py2 = [float(v) for v in np.asarray(proposal_bbox).reshape(-1)[:4]]
                proposal_rect = patches.Rectangle(
                    (px1 * W, py1 * H), (px2 - px1) * W, (py2 - py1) * H,
                    linewidth=1.5, edgecolor='orange', facecolor='none', linestyle='--', alpha=0.9
                )
                ax.add_patch(proposal_rect)

            rect = patches.Rectangle(
                (x1_pixel, y1_pixel), width, height,
                linewidth=2.5, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            cat_line = _flare_category_line(final_class)
            linked_events = linked_events or []
            if linked_events:
                event_lines = []
                for evt_idx, evt in enumerate(linked_events, start=1):
                    evt_cls = int(evt.get('class_id', final_class))
                    evt_time = evt.get('time_prediction') or []
                    if self.time_head_enabled and isinstance(evt_time, list) and len(evt_time) >= 3:
                        evt_start_s, evt_peak_s, evt_end_s = _compute_start_peak_end_utc(ref_dt, evt_time)
                        event_lines.append(
                            f'evt{evt_idx}: {_flare_category_line(evt_cls)}\n'
                            f'  start={evt_start_s}\n'
                            f'  peak= {evt_peak_s}\n'
                            f'  end=  {evt_end_s}'
                        )
                    else:
                        event_lines.append(f'evt{evt_idx}: {_flare_category_line(evt_cls)}')
                event_text = '\n'.join(event_lines[:4])
            else:
                event_text = (
                    f'start_time: {start_s}\n'
                    f'peak_time:  {peak_s}\n'
                    f'end_time:   {end_s}'
                ) if self.time_head_enabled else 'time_prediction: disabled'
            label_text = (
                f'{cat_line}\n'
                f'proposal_score: {proposal_score:.3f}\n'
                f'final_box: solid / proposal: dashed / context: dotted\n'
                f'{event_text}'
            )

            # 依次尝试 4 个锚点：上、下、右、左。
            # 每个锚点都保证：信息框的一条边与 bbox 边基本对齐，并尽量在 bbox 外侧。
            candidates = []
            margin = 2.0

            # 顶部：文字框底边贴着 bbox 顶边（y=y1_pixel - margin, va='top'）
            candidates.append({
                'x': float(np.clip(x1_pixel, 4.0, W - 4.0)),
                'y': float(y1_pixel - margin),
                'va': 'top',
                'ha': 'left',
                'score': 0,  # 优先级最高
            })
            # 底部：文字框顶边贴着 bbox 底边（y=y2_pixel + margin, va='bottom'）
            candidates.append({
                'x': float(np.clip(x1_pixel, 4.0, W - 4.0)),
                'y': float(y2_pixel + margin),
                'va': 'bottom',
                'ha': 'left',
                'score': 1,
            })
            # 右侧：文字框左边贴着 bbox 右边（x=x2_pixel + margin, ha='left'）
            candidates.append({
                'x': float(x2_pixel + margin),
                'y': float(np.clip(y1_pixel, 4.0, H - 4.0)),
                'va': 'top',
                'ha': 'left',
                'score': 2,
            })
            # 左侧：文字框右边贴着 bbox 左边（x=x1_pixel - margin, ha='right'）
            candidates.append({
                'x': float(x1_pixel - margin),
                'y': float(np.clip(y1_pixel, 4.0, H - 4.0)),
                'va': 'top',
                'ha': 'right',
                'score': 3,
            })

            # 简单过滤：避免 y 太接近图像边缘（以减少被裁剪），并对多框做轻微错开
            chosen = None
            for cand in sorted(candidates, key=lambda c: c['score']):
                x = cand['x'] + (bbox_idx % 3) * min(0.05 * W, 16.0)
                y = cand['y'] - (bbox_idx // 3) * min(0.04 * H, 12.0)
                if 0 <= y <= H:
                    chosen = {**cand, 'x': x, 'y': y}
                    break
            if chosen is None:
                chosen = candidates[0]

            bbox_props = dict(
                boxstyle='round,pad=0.45',
                facecolor='black',
                alpha=0.72,
                edgecolor=color,
                linewidth=1.8,
            )
            ax.text(
                chosen['x'],
                chosen['y'],
                label_text,
                fontsize=7.5,
                color='white',
                bbox=bbox_props,
                verticalalignment=chosen['va'],
                horizontalalignment=chosen['ha'],
                family='monospace',
                clip_on=False,
            )

        except Exception as e:
            logger.error(f"绘制bbox失败: {e}")

    def visualize_batch_predictions(self,
                                   predictions_list: List[Dict[str, Any]],
                                   output_dir: Path,
                                   max_cols: int = 3) -> None:
        """
        批量可视化多个预测结果

        Args:
            predictions_list: 预测结果列表
            output_dir: 输出目录
            max_cols: 每行最多显示的列数
        """
        try:
            if not predictions_list:
                logger.warning("预测结果列表为空")
                return

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 计算行列数
            total = len(predictions_list)
            num_cols = min(max_cols, total)
            num_rows = (total + num_cols - 1) // num_cols

            # 创建大图表
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
            if num_rows == 1 and num_cols == 1:
                axes = np.array([[axes]])
            elif num_rows == 1 or num_cols == 1:
                axes = axes.reshape(num_rows, num_cols)

            axes = axes.flatten()

            # 绘制每个预测
            for idx, pred in enumerate(predictions_list):
                ax = axes[idx]

                try:
                    # 获取事件ID和元数据
                    metadata = pred.get('metadata', {})
                    event_id = metadata.get('event_id', f'Event_{idx}')

                    # 获取预测信息
                    final_class = pred.get('final_class', 0)
                    class_prob = pred.get('classification_confidence', 0.0)
                    event_prob = pred.get('processed_event_prob', 0.0)
                    bboxes = pred.get('processed_bboxes', [])

                    # 绘制
                    class_name = self.class_names.get(final_class, 'Unknown')
                    color = self.class_colors.get(final_class, 'white')

                    # 创建虚拟图像（纯色背景）
                    dummy_image = np.ones(self.image_size) * 0.3

                    ax.imshow(dummy_image, cmap='gray')
                    ax.text(0.5, 0.5, f'{class_name}\n{class_prob:.2%}',
                           transform=ax.transAxes,
                           fontsize=12, fontweight='bold',
                           ha='center', va='center',
                           color=color,
                           bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))

                    # 标题
                    ax.set_title(f'{event_id}\nProb: {event_prob:.2%}',
                                fontsize=10, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])

                except Exception as e:
                    logger.error(f"绘制预测 {idx} 失败: {e}")

            # 隐藏未使用的子图
            for idx in range(total, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()

            # 保存
            save_path = output_dir / 'batch_predictions_summary.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"批量预测可视化已保存: {save_path}")

            plt.close(fig)

        except Exception as e:
            logger.error(f"批量可视化失败: {e}")

    def create_summary_report(self,
                             predictions_list: List[Dict[str, Any]],
                             output_dir: Path,
                             event_id: str) -> None:
        """
        创建事件预测摘要报告
        
        Args:
            predictions_list: 预测结果列表
            output_dir: 输出目录
            event_id: 事件ID
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if not predictions_list:
                logger.warning("预测结果列表为空")
                return

            # 统计信息
            class_counts = {0: 0, 1: 0, 2: 0}
            total_event_prob = 0
            max_event_prob = 0
            max_event_prob_idx = -1
            time_predictions = [] if self.time_head_enabled else None

            for idx, pred in enumerate(predictions_list):
                final_class = pred.get('final_class', 0)
                class_counts[final_class] += 1

                event_prob = pred.get('processed_event_prob', 0.0)
                total_event_prob += event_prob

                if event_prob > max_event_prob:
                    max_event_prob = event_prob
                    max_event_prob_idx = idx

                time_pred = pred.get('processed_time_pred', []) if self.time_head_enabled else []
                if isinstance(time_pred, torch.Tensor):
                    time_pred = time_pred.detach().cpu().numpy()
                time_pred = np.asarray(time_pred, dtype=np.float32) if time_pred is not None else np.empty((0, 3))
                # 展平为 (N, 3)，每行是一个 detection 的 [start, end, duration]
                if time_pred.ndim == 1 and time_pred.size >= 3:
                    time_pred = time_pred.reshape(1, -1)
                if time_pred.ndim == 2 and time_pred.shape[1] >= 3:
                    for row in time_pred:
                        time_predictions.append(row[:3].tolist())

            # 创建报告
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. 类别分布
            ax = axes[0, 0]
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            class_labels = [self.class_names[c] for c in classes]
            colors_list = [self.class_colors.get(c, 'gray') for c in classes]

            ax.bar(class_labels, counts, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # 2. 事件概率分布
            ax = axes[0, 1]
            event_probs = [p.get('processed_event_prob', 0.0) for p in predictions_list]
            ax.hist(event_probs, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(total_event_prob / len(predictions_list) if predictions_list else 0,
                      color='red', linestyle='--', linewidth=2, label='Mean')
            ax.set_xlabel('Event Probability', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Event Probability Distribution', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

            # 3. 时间预测统计
            ax = axes[1, 0]
            if self.time_head_enabled and time_predictions:
                time_predictions = np.array(time_predictions)
                time_starts = time_predictions[:, 0]
                time_peaks = time_predictions[:, 1]
                time_durations = time_predictions[:, 2]

                ax.scatter(time_starts, time_durations, alpha=0.6, s=100, c=time_peaks, cmap='viridis')
                ax.set_xlabel('Start Time Offset (hours)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Duration (hours)', fontsize=12, fontweight='bold')
                ax.set_title('Time Predictions', fontsize=14, fontweight='bold')
                cbar = plt.colorbar(ax.collections[0], ax=ax)
                cbar.set_label('Peak Offset (hours)', fontsize=11)
                ax.grid(alpha=0.3)
            else:
                ax.axis('off')

            # 4. 文本信息框
            ax = axes[1, 1]
            ax.axis('off')

            summary_text = f"""
Event Summary Report
Event ID: {event_id}
Total Windows: {len(predictions_list)}

Class Statistics:
  • No Event: {class_counts[0]} ({100*class_counts[0]/len(predictions_list):.1f}%)
  • Eruptive (CME): {class_counts[1]} ({100*class_counts[1]/len(predictions_list):.1f}%)
  • Confined: {class_counts[2]} ({100*class_counts[2]/len(predictions_list):.1f}%)

Event Probability:
  • Mean: {total_event_prob/len(predictions_list):.3f}
  • Max: {max_event_prob:.3f} (Window {max_event_prob_idx})
  • Min: {min(event_probs):.3f}

Time Prediction Unit: Hours
Time Prediction: disabled
Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """ if not self.time_head_enabled else f"""
Event Summary Report
Event ID: {event_id}
Total Windows: {len(predictions_list)}

Class Statistics:
  • No Event: {class_counts[0]} ({100*class_counts[0]/len(predictions_list):.1f}%)
  • Eruptive (CME): {class_counts[1]} ({100*class_counts[1]/len(predictions_list):.1f}%)
  • Confined: {class_counts[2]} ({100*class_counts[2]/len(predictions_list):.1f}%)

Event Probability:
  • Mean: {total_event_prob/len(predictions_list):.3f}
  • Max: {max_event_prob:.3f} (Window {max_event_prob_idx})
  • Min: {min(event_probs):.3f}

Time Prediction Unit: Hours
Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

            ax.text(0.1, 0.5, summary_text.strip(),
                   fontsize=11, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, pad=1))

            plt.tight_layout()

            # 保存报告
            report_path = output_dir / f'summary_report_{event_id}.png'
            fig.savefig(report_path, dpi=150, bbox_inches='tight')
            logger.info(f"摘要报告已保存: {report_path}")

            plt.close(fig)

        except Exception as e:
            logger.error(f"创建摘要报告失败: {e}")
