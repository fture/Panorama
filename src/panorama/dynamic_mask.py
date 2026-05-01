"""Dynamic object masking utilities (Phase 4).

This educational module provides a simple motion-based dynamic mask.
Later you can replace it with YOLO-seg / Mask2Former style semantic masks.
"""

import cv2
import numpy as np


def estimate_dynamic_mask(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Estimate moving-object mask using dense optical flow magnitude.

    Returns a binary mask where high flow magnitude implies potential motion.
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    thr = float(np.percentile(mag, 85))
    return (mag > thr).astype(np.uint8)
