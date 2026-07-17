import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _bbox_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = box_a.tolist()
    bx1, by1, bx2, by2 = box_b.tolist()
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / (union + 1e-6)) if union > 0 else 0.0


def _score_to_probability(score: torch.Tensor) -> float:
    value = float(score.reshape(-1)[0].item())
    if 0.0 <= value <= 1.0:
        return value
    return float(torch.sigmoid(score.reshape(-1)[0]).item())


class PostProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.time_head_enabled = bool(config.get('prediction_heads', {}).get('time', {}).get('enabled', True))
        self.class_thresholds = config.get('thresholds', {'class0': 0.3, 'class1': 0.5, 'class2': 0.7})
        self.bbox_config = config.get(
            'bbox_processing',
            {
                'min_area': 10,
                'max_area_ratio': 0.2,
                'min_confidence': 0.3,
                'min_event_probability': 0.35,
                'min_proposal_score': 0.35,
                'nms_threshold': 0.35,
                'merge_iou_threshold': 0.6,
                'max_detections': 10,
            }
        )
        self.detection_box_source = str(self.bbox_config.get('detection_box_source', 'refined')).lower()
        self.allow_background_fallback = bool(self.bbox_config.get('allow_background_fallback', False))
        self.min_positive_confidence = float(
            self.bbox_config.get('min_positive_confidence', self.bbox_config.get('min_confidence', 0.3))
        )
        self.time_config = config.get(
            'time_processing',
            {'min_prediction_interval': 1.0 / 60.0, 'max_prediction_horizon': 48.0, 'max_duration_hours': 1.0}
        )
        logger.info("后处理器初始化完成")

    def process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        out = predictions.copy()
        out = self._squeeze_batch_outputs(out)
        out = self._process_classification(out)
        out = self._process_time_predictions(out)
        out = self._process_event_probability(out)
        out = self._process_bboxes(out)
        out = self._apply_consistency_checks(out)
        return out

    def _squeeze_batch_outputs(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        for key in (
            'class_probs_mean', 'class_probs_std', 'class_logits_mean', 'class_logits_std',
            'time_pred_mean', 'time_pred_std', 'event_prob_mean', 'event_prob_std',
        ):
            if key in predictions:
                val = predictions[key]
                if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[0] == 1:
                    predictions[key] = val.squeeze(0)
        for key in ('bbox_pred_mean', 'bbox_pred_std', 'proposal_boxes', 'context_boxes'):
            if key in predictions:
                val = predictions[key]
                if isinstance(val, torch.Tensor) and val.dim() >= 3 and val.shape[0] == 1:
                    predictions[key] = val.squeeze(0)
        for key in ('proposal_scores',):
            if key in predictions:
                val = predictions[key]
                if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[0] == 1:
                    predictions[key] = val.squeeze(0)
        if 'predicted_class' in predictions:
            val = predictions['predicted_class']
            if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == 1:
                predictions['predicted_class'] = val.squeeze(0)
        return predictions

    def _process_classification(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        if 'class_probs_mean' not in predictions:
            return predictions
        probs = predictions['class_probs_mean']
        if not isinstance(probs, torch.Tensor):
            probs = torch.as_tensor(probs)
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        processed, final_classes, confs = [], [], []
        for slot_probs in probs:
            slot_probs = slot_probs.reshape(-1)
            if slot_probs.numel() == 0:
                slot_probs = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            if slot_probs.sum() > 0:
                slot_probs = slot_probs / slot_probs.sum()
            final_class = self._apply_class_thresholds(slot_probs)
            processed.append(slot_probs)
            final_classes.append(final_class)
            confs.append(float(slot_probs[final_class].item()) if slot_probs.numel() > final_class else 0.0)
        predictions['processed_class_probs'] = torch.stack(processed) if processed else torch.empty((0, 3))
        predictions['final_classes'] = torch.tensor(final_classes, dtype=torch.long)
        predictions['classification_confidences'] = torch.tensor(confs, dtype=torch.float32)
        predictions['predicted_class'] = predictions['final_classes']
        if final_classes:
            best_idx = int(np.argmax(confs))
            predictions['final_class'] = int(final_classes[best_idx])
            predictions['classification_confidence'] = float(confs[best_idx])
        else:
            predictions['final_class'] = 0
            predictions['classification_confidence'] = 0.0
        return predictions

    def _apply_class_thresholds(self, class_probs: torch.Tensor) -> int:
        probs = class_probs.detach().cpu().numpy()
        idx = int(np.argmax(probs))
        return idx if float(probs[idx]) >= self.class_thresholds.get(f'class{idx}', 0.3) else 0

    def _process_bboxes(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        if 'bbox_pred_mean' not in predictions:
            return predictions
        bbox_pred = predictions['bbox_pred_mean']
        bbox_size_gated = predictions.get('bbox_size_gated')
        proposal_boxes = predictions.get('proposal_boxes')
        context_boxes = predictions.get('context_boxes')
        proposal_scores = predictions.get('proposal_scores')
        if not isinstance(bbox_pred, torch.Tensor):
            bbox_pred = torch.as_tensor(bbox_pred, dtype=torch.float32)
        if bbox_pred.dim() == 2:
            bbox_pred = bbox_pred.unsqueeze(0)
        if bbox_size_gated is not None and not isinstance(bbox_size_gated, torch.Tensor):
            bbox_size_gated = torch.as_tensor(bbox_size_gated, dtype=torch.float32)
        if bbox_size_gated is not None and bbox_size_gated.dim() == 2:
            bbox_size_gated = bbox_size_gated.unsqueeze(0)
        if proposal_boxes is not None and not isinstance(proposal_boxes, torch.Tensor):
            proposal_boxes = torch.as_tensor(proposal_boxes, dtype=torch.float32)
        if proposal_boxes is not None and proposal_boxes.dim() == 2:
            proposal_boxes = proposal_boxes.unsqueeze(0)
        if context_boxes is not None and not isinstance(context_boxes, torch.Tensor):
            context_boxes = torch.as_tensor(context_boxes, dtype=torch.float32)
        if context_boxes is not None and context_boxes.dim() == 2:
            context_boxes = context_boxes.unsqueeze(0)
        if proposal_scores is not None and not isinstance(proposal_scores, torch.Tensor):
            proposal_scores = torch.as_tensor(proposal_scores, dtype=torch.float32)
        if proposal_scores is not None and proposal_scores.dim() == 1:
            proposal_scores = proposal_scores.reshape(1, -1, 1)
        elif proposal_scores is not None and proposal_scores.dim() == 2:
            if proposal_scores.shape[-1] == 1:
                proposal_scores = proposal_scores.unsqueeze(0)
            else:
                proposal_scores = proposal_scores.unsqueeze(-1)

        pred_classes = predictions.get('predicted_class')
        class_confs = predictions.get('classification_confidences')
        event_probs = predictions.get('processed_event_probs', predictions.get('event_prob_mean'))
        time_preds = predictions.get('processed_time_pred', predictions.get('time_pred_mean')) if self.time_head_enabled else None
        if event_probs is not None and not isinstance(event_probs, torch.Tensor):
            event_probs = torch.as_tensor(event_probs, dtype=torch.float32)
        if time_preds is not None and not isinstance(time_preds, torch.Tensor):
            time_preds = torch.as_tensor(time_preds, dtype=torch.float32)

        slot_boxes, slot_confidences = [], []
        slot_proposal_boxes, slot_context_boxes, slot_proposal_scores = [], [], []
        candidate_detections: List[Dict[str, Any]] = []

        min_cls_conf = float(self.bbox_config.get('min_confidence', 0.3))
        min_event_prob = float(self.bbox_config.get('min_event_probability', min_cls_conf))
        min_proposal_score = float(self.bbox_config.get('min_proposal_score', min_cls_conf))
        min_effective_size = float(self.bbox_config.get('min_effective_size', 0.01))
        merge_iou = float(self.bbox_config.get('merge_iou_threshold', self.bbox_config.get('nms_threshold', 0.5)))

        for i in range(bbox_pred.shape[1]):
            refined_bbox = self._clip_and_validate_bbox(bbox_pred[0, i])
            proposal_bbox = None
            context_bbox = None
            if proposal_boxes is not None and proposal_boxes.shape[1] > i:
                proposal_bbox = self._clip_and_validate_bbox(proposal_boxes[0, i])
            if context_boxes is not None and context_boxes.shape[1] > i:
                context_bbox = self._clip_and_validate_bbox(context_boxes[0, i])

            slot_boxes.append(refined_bbox if refined_bbox is not None else None)
            slot_proposal_boxes.append(proposal_bbox if proposal_bbox is not None else None)
            slot_context_boxes.append(context_bbox if context_bbox is not None else None)

            slot_class = 0
            if pred_classes is not None and len(pred_classes) > i:
                slot_class = int(pred_classes[i].item() if hasattr(pred_classes[i], 'item') else pred_classes[i])
            slot_conf = 0.5
            if class_confs is not None and len(class_confs) > i:
                slot_conf = float(class_confs[i].item() if hasattr(class_confs[i], 'item') else class_confs[i])
            proposal_conf = slot_conf
            if proposal_scores is not None and proposal_scores.shape[1] > i:
                proposal_conf = _score_to_probability(proposal_scores[0, i])
            event_prob = proposal_conf
            if event_probs is not None and len(event_probs.reshape(-1)) > i:
                event_prob = float(event_probs.reshape(-1)[i].item() if hasattr(event_probs.reshape(-1)[i], 'item') else event_probs.reshape(-1)[i])

            slot_confidences.append(slot_conf)
            slot_proposal_scores.append(proposal_conf)

            gated_size_valid = True
            if bbox_size_gated is not None and bbox_size_gated.shape[1] > i:
                size_vec = bbox_size_gated[0, i].reshape(-1)
                gated_size_valid = bool(torch.all(size_vec >= min_effective_size).item())

            if slot_class == 0 and self.allow_background_fallback and 'processed_class_probs' in predictions:
                class_probs = predictions['processed_class_probs']
                if isinstance(class_probs, torch.Tensor) and class_probs.dim() >= 2 and class_probs.shape[0] > i:
                    positive_probs = class_probs[i, 1:3]
                    if positive_probs.numel() == 2:
                        positive_conf, positive_idx = torch.max(positive_probs, dim=0)
                        if float(positive_conf.item()) >= self.min_positive_confidence:
                            slot_class = int(positive_idx.item()) + 1
                            slot_conf = float(positive_conf.item())

            detection_bbox = refined_bbox
            if self.detection_box_source == 'proposal' and proposal_bbox is not None:
                detection_bbox = proposal_bbox
            elif self.detection_box_source == 'context' and context_bbox is not None:
                detection_bbox = context_bbox

            if detection_bbox is None or slot_class == 0 or (not gated_size_valid):
                continue
            if slot_conf < min_cls_conf or event_prob < min_event_prob or proposal_conf < min_proposal_score:
                continue

            time_vec = None
            if time_preds is not None and time_preds.numel() > 0:
                flat_time = time_preds if time_preds.dim() > 1 else time_preds.reshape(1, -1)
                if flat_time.shape[0] > i:
                    time_vec = flat_time[i].detach().clone()

            candidate_detections.append({
                'slot_id': i,
                'bbox': detection_bbox,
                'refined_bbox': refined_bbox,
                'class_id': slot_class,
                'class_confidence': slot_conf,
                'event_probability': event_prob,
                'proposal_score': proposal_conf,
                'proposal_bbox': proposal_bbox,
                'context_bbox': context_bbox,
                'time_prediction': time_vec if self.time_head_enabled else None,
            })

        candidate_detections.sort(
            key=lambda det: (det['event_probability'] + det['class_confidence'] + det['proposal_score']),
            reverse=True,
        )

        if candidate_detections:
            nms_boxes = torch.stack([det['bbox'] for det in candidate_detections])
            nms_scores = torch.tensor(
                [det['event_probability'] + det['class_confidence'] + det['proposal_score'] for det in candidate_detections],
                dtype=torch.float32,
            )
            keep = self._non_max_suppression(
                nms_boxes,
                nms_scores,
                float(self.bbox_config.get('nms_threshold', 0.35)),
            )
            candidate_detections = [candidate_detections[idx] for idx in keep]

        merged_detections: List[Dict[str, Any]] = []
        for det in candidate_detections:
            matched = None
            for existing in merged_detections:
                if _bbox_iou(det['bbox'], existing['bbox']) >= merge_iou:
                    matched = existing
                    break
            if matched is None:
                merged_detections.append({**det, 'linked_events': [{
                    'slot_id': det['slot_id'],
                    'class_id': det['class_id'],
                    'class_confidence': det['class_confidence'],
                    'event_probability': det['event_probability'],
                    'proposal_score': det['proposal_score'],
                    'time_prediction': det['time_prediction'],
                }]})
                continue

            matched['linked_events'].append({
                'slot_id': det['slot_id'],
                'class_id': det['class_id'],
                'class_confidence': det['class_confidence'],
                'event_probability': det['event_probability'],
                'proposal_score': det['proposal_score'],
                'time_prediction': det['time_prediction'],
            })
            current_score = matched['event_probability'] + matched['class_confidence'] + matched['proposal_score']
            new_score = det['event_probability'] + det['class_confidence'] + det['proposal_score']
            if new_score > current_score:
                matched.update({
                    'slot_id': det['slot_id'],
                    'bbox': det['bbox'],
                    'class_id': det['class_id'],
                    'class_confidence': det['class_confidence'],
                    'event_probability': det['event_probability'],
                    'proposal_score': det['proposal_score'],
                    'proposal_bbox': det['proposal_bbox'],
                    'context_bbox': det['context_bbox'],
                    'time_prediction': det['time_prediction'],
                })

        max_det = self.bbox_config['max_detections']
        merged_detections = merged_detections[:max_det]

        if merged_detections:
            boxes = torch.stack([det['bbox'] for det in merged_detections])
            confs = torch.tensor([det['class_confidence'] for det in merged_detections], dtype=torch.float32)
            kept_slot_indices = [det['slot_id'] for det in merged_detections]
            linked_events = []
            for det in merged_detections:
                if self.time_head_enabled:
                    linked_events.append([
                        {
                            'slot_id': int(evt['slot_id']),
                            'class_id': int(evt['class_id']),
                            'class_confidence': float(evt['class_confidence']),
                            'event_probability': float(evt['event_probability']),
                            'proposal_score': float(evt['proposal_score']),
                            'time_prediction': evt['time_prediction'].detach().cpu().tolist() if isinstance(evt['time_prediction'], torch.Tensor) else None,
                        }
                        for evt in sorted(
                            det['linked_events'],
                            key=lambda item: (item['time_prediction'][0].item() if isinstance(item['time_prediction'], torch.Tensor) and item['time_prediction'].numel() > 0 else float('inf'))
                        )
                    ])
                else:
                    linked_events.append([
                        {
                            'slot_id': int(evt['slot_id']),
                            'class_id': int(evt['class_id']),
                            'class_confidence': float(evt['class_confidence']),
                            'event_probability': float(evt['event_probability']),
                            'proposal_score': float(evt['proposal_score']),
                        }
                        for evt in det['linked_events']
                    ])
            final_classes = torch.tensor([det['class_id'] for det in merged_detections], dtype=torch.long)
            final_class_confidences = torch.tensor([det['class_confidence'] for det in merged_detections], dtype=torch.float32)
            final_event_probs = torch.tensor([det['event_probability'] for det in merged_detections], dtype=torch.float32)
            final_proposal_boxes = []
            final_context_boxes = []
            final_proposal_scores = []
            for det in merged_detections:
                final_proposal_boxes.append(det['proposal_bbox'])
                final_context_boxes.append(det['context_bbox'])
                final_proposal_scores.append(det['proposal_score'])
            if self.time_head_enabled:
                final_times = []
                for det in merged_detections:
                    tp = det['time_prediction']
                    final_times.append(tp.detach().cpu() if isinstance(tp, torch.Tensor) else torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
                predictions['processed_time_pred'] = torch.stack(final_times) if final_times else torch.empty((0, 3), dtype=torch.float32)
            else:
                predictions.pop('processed_time_pred', None)
            predictions['processed_event_probs'] = final_event_probs
            predictions['final_classes'] = final_classes
            predictions['classification_confidences'] = final_class_confidences
            predictions['slot_proposal_boxes'] = final_proposal_boxes
            predictions['slot_context_boxes'] = final_context_boxes
            predictions['slot_proposal_scores'] = torch.tensor(final_proposal_scores, dtype=torch.float32)
            predictions['linked_events'] = linked_events
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            confs = torch.empty((0,), dtype=torch.float32)
            kept_slot_indices = []
            predictions.pop('processed_time_pred', None)
            predictions['processed_event_probs'] = torch.empty((0,), dtype=torch.float32)
            predictions['final_classes'] = torch.empty((0,), dtype=torch.long)
            predictions['classification_confidences'] = torch.empty((0,), dtype=torch.float32)
            predictions['slot_proposal_boxes'] = []
            predictions['slot_context_boxes'] = []
            predictions['slot_proposal_scores'] = torch.empty((0,), dtype=torch.float32)
            predictions['linked_events'] = []

        predictions['processed_bboxes'] = boxes
        predictions['bbox_confidences'] = confs
        predictions['slot_processed_bboxes'] = slot_boxes
        predictions['slot_bbox_confidences'] = torch.tensor(slot_confidences, dtype=torch.float32) if slot_confidences else torch.empty((0,), dtype=torch.float32)
        predictions['kept_bbox_slot_indices'] = torch.tensor(kept_slot_indices, dtype=torch.long) if kept_slot_indices else torch.empty((0,), dtype=torch.long)
        return predictions

    def _clip_and_validate_bbox(self, bbox: torch.Tensor) -> Optional[torch.Tensor]:
        bbox = torch.clamp(bbox, 0.0, 1.0)
        x1, y1, x2, y2 = bbox

        # 保险修复：若坐标顺序异常，自动重排为合法角点框
        x_min = torch.minimum(x1, x2)
        x_max = torch.maximum(x1, x2)
        y_min = torch.minimum(y1, y2)
        y_max = torch.maximum(y1, y2)
        repaired_bbox = torch.stack([x_min, y_min, x_max, y_max])

        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            return None

        area = width * height
        max_area_ratio = float(self.bbox_config.get('max_area_ratio', 1.0))
        if area > max_area_ratio:
            return None
        return None if area < self.bbox_config['min_area'] / 10000.0 else repaired_bbox

    def _non_max_suppression(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
        if len(boxes) == 0:
            return []
        boxes_np, scores_np = boxes.cpu().numpy(), scores.cpu().numpy()
        order, keep = scores_np.argsort()[::-1], []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(boxes_np[i, 0], boxes_np[order[1:], 0])
            yy1 = np.maximum(boxes_np[i, 1], boxes_np[order[1:], 1])
            xx2 = np.minimum(boxes_np[i, 2], boxes_np[order[1:], 2])
            yy2 = np.minimum(boxes_np[i, 3], boxes_np[order[1:], 3])
            w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (boxes_np[i, 2] - boxes_np[i, 0]) * (boxes_np[i, 3] - boxes_np[i, 1])
            area_j = (boxes_np[order[1:], 2] - boxes_np[order[1:], 0]) * (boxes_np[order[1:], 3] - boxes_np[order[1:], 1])
            iou = inter / (area_i + area_j - inter + 1e-6)
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        return keep

    def _process_time_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        if (not self.time_head_enabled) or ('time_pred_mean' not in predictions):
            predictions.pop('processed_time_pred', None)
            return predictions
        time_pred = predictions['time_pred_mean']
        if not isinstance(time_pred, torch.Tensor):
            time_pred = torch.as_tensor(time_pred, dtype=torch.float32)
        if time_pred.dim() == 1:
            time_pred = time_pred.reshape(1, 1, -1)
        elif time_pred.dim() == 2:
            time_pred = time_pred.unsqueeze(0)

        processed_slots = []
        scale = 24.0
        min_duration_hours = float(self.time_config.get('min_prediction_interval', 1.0 / 60.0))
        horizon = float(self.time_config.get('max_prediction_horizon', 48.0))
        for slot_time in time_pred[0]:
            slot_time = slot_time.reshape(-1)
            if slot_time.numel() < 3:
                processed_slots.append([0.0, 0.0, 0.0])
                continue
            # 模型输出格式：[start_offset_days, peak_offset_days, end_offset_days]
            start_hours = float(slot_time[0]) * scale
            peak_hours  = float(slot_time[1]) * scale
            end_hours   = float(slot_time[2]) * scale

            start_hours = max(-horizon, min(horizon, start_hours))
            peak_hours  = max(-horizon, min(horizon, peak_hours))
            end_hours   = max(-horizon, min(horizon, end_hours))

            ordered = sorted([start_hours, peak_hours, end_hours])
            start_hours, peak_hours, end_hours = ordered[0], ordered[1], ordered[2]
            if end_hours - start_hours < min_duration_hours:
                end_hours = start_hours + min_duration_hours
                peak_hours = 0.5 * (start_hours + end_hours)

            processed_slots.append([start_hours, peak_hours, end_hours])
        predictions['processed_time_pred'] = torch.tensor(processed_slots, dtype=torch.float32)
        return predictions

    def _process_event_probability(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        if 'event_prob_mean' not in predictions:
            return predictions
        event_prob = predictions['event_prob_mean']
        if isinstance(event_prob, torch.Tensor):
            event_prob = torch.sigmoid(event_prob.detach())
        else:
            event_prob = torch.as_tensor(event_prob, dtype=torch.float32)
            event_prob = torch.sigmoid(event_prob)
        if event_prob.dim() == 0:
            event_prob = event_prob.reshape(1)
        event_prob = torch.clamp(event_prob.reshape(-1), 0.0, 1.0)
        if 'event_prob_calibration' in self.config:
            calib = self.config['event_prob_calibration']
            event_prob = torch.clamp(calib.get('a', 1.0) * event_prob + calib.get('b', 0.0), 0.0, 1.0)
        predictions['processed_event_probs'] = event_prob
        predictions['processed_event_prob'] = float(event_prob.max().item()) if event_prob.numel() > 0 else 0.0
        return predictions

    def _calibrate_probability(self, prob: float, calibration: Dict) -> float:
        calibrated = calibration.get('a', 1.0) * prob + calibration.get('b', 0.0)
        return max(0.0, min(1.0, calibrated))

    def _apply_consistency_checks(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        if predictions.get('final_class') == 0 and predictions.get('processed_event_prob', 0.0) > 0.7:
            predictions['processed_event_prob'] *= 0.5
        if predictions.get('processed_event_prob', 1.0) < 0.3 and predictions.get('final_class') == 2:
            predictions['final_class'] = 1
        predictions['consistency_check_passed'] = True
        return predictions

    def filter_by_confidence(self, predictions: Dict[str, Any], min_confidence: float = 0.5) -> Dict[str, Any]:
        filtered = predictions.copy()
        if filtered.get('classification_confidence', 0.0) < min_confidence:
            filtered['final_class'] = 0
            filtered['classification_confidence'] = 0.0
        if 'bbox_confidences' in predictions:
            keep = [i for i, conf in enumerate(predictions['bbox_confidences']) if conf >= min_confidence]
            filtered['processed_bboxes'] = predictions['processed_bboxes'][keep] if keep else torch.tensor([])
            filtered['bbox_confidences'] = predictions['bbox_confidences'][keep] if keep else torch.tensor([])
        return filtered

    def to_visualization_format(self, predictions: Dict[str, Any], image_size: Tuple[int, int] = (512, 512)) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        class_names = ['无事件', '爆发耀斑', '束缚耀斑']
        if 'final_class' in predictions:
            out['classification'] = {'class_id': int(predictions['final_class']), 'class_name': class_names[predictions['final_class']], 'confidence': float(predictions.get('classification_confidence', 0.0))}
        if 'final_classes' in predictions:
            slot_confidences = predictions.get('classification_confidences', [])
            out['slot_classifications'] = []
            for idx, cls in enumerate(predictions['final_classes']):
                cls_id = int(cls.item() if hasattr(cls, 'item') else cls)
                conf = float(slot_confidences[idx].item() if len(slot_confidences) > idx and hasattr(slot_confidences[idx], 'item') else (slot_confidences[idx] if len(slot_confidences) > idx else 0.0))
                out['slot_classifications'].append({'slot_id': idx, 'class_id': cls_id, 'class_name': class_names[cls_id], 'confidence': conf})
        if 'processed_event_prob' in predictions:
            out['event_probability'] = float(predictions['processed_event_prob'])
        if 'processed_bboxes' in predictions and len(predictions['processed_bboxes']) > 0:
            h, w = image_size
            out['bounding_boxes'] = []
            for bbox, conf in zip(predictions['processed_bboxes'], predictions['bbox_confidences']):
                x1, y1, x2, y2 = bbox.tolist()
                out['bounding_boxes'].append({'bbox': [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)], 'confidence': float(conf), 'area': int((x2 - x1) * w * (y2 - y1) * h)})
        if 'processed_time_pred' in predictions:
            slot_times = predictions['processed_time_pred']
            if isinstance(slot_times, torch.Tensor):
                slot_times = slot_times.detach().cpu().numpy()
            out['time_predictions'] = []
            for idx, slot_time in enumerate(np.asarray(slot_times)):
                slot_time = np.asarray(slot_time).reshape(-1)
                if slot_time.size < 3:
                    continue
                out['time_predictions'].append({
                    'slot_id': idx,
                    'start_offset_hours': float(slot_time[0]),
                    'peak_offset_hours': float(slot_time[1]),
                    'end_offset_hours': float(slot_time[2])
                })
        if 'processed_event_probs' in predictions:
            probs = predictions['processed_event_probs']
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            out['slot_event_probabilities'] = [
                {'slot_id': idx, 'event_probability': float(prob)}
                for idx, prob in enumerate(np.asarray(probs).reshape(-1))
            ]
            out['event_probability'] = float(np.max(np.asarray(probs).reshape(-1))) if np.asarray(probs).size > 0 else 0.0
