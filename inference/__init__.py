"""推理模块。"""
import logging

from .predictor import SolarFlarePredictor
from .post_processing import PostProcessor
from .uncertainty_estimation import UncertaintyEstimator

logger = logging.getLogger(__name__)

try:
    from .visualization import PredictionVisualizer
except Exception as exc:
    # Keep predictor/post-processing importable even if matplotlib/Pillow is unavailable.
    logger.warning("导入 PredictionVisualizer 失败: %s", exc)
    PredictionVisualizer = None

__all__ = [
    "SolarFlarePredictor",
    "PostProcessor",
    "UncertaintyEstimator",
    "PredictionVisualizer",
]
