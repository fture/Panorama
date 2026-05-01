"""Shared dataclasses and type aliases for the panorama pipeline.

This file intentionally contains only simple, beginner-friendly structures.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class PairSample:
    """Container for one adjacent image pair.

    Attributes:
        pair_id: Unique sample id for reproducibility.
        image_a: First image (H, W, 3), uint8.
        image_b: Second image (H, W, 3), uint8.
        dynamic_mask_a: Optional moving-object mask for image_a (H, W), bool/uint8.
        dynamic_mask_b: Optional moving-object mask for image_b (H, W), bool/uint8.
    """

    pair_id: str
    image_a: np.ndarray
    image_b: np.ndarray
    dynamic_mask_a: Optional[np.ndarray] = None
    dynamic_mask_b: Optional[np.ndarray] = None


@dataclass
class AlignmentOutput:
    """Output of the alignment stage.

    Attributes:
        warped_b_to_a: image_b warped into image_a frame.
        overlap_mask: mask where both images are valid after warping.
        residual_flow_mag: simple residual-flow magnitude proxy.
        extras: dictionary for algorithm-specific intermediate values.
    """

    warped_b_to_a: np.ndarray
    overlap_mask: np.ndarray
    residual_flow_mag: np.ndarray
    extras: Dict[str, np.ndarray]
