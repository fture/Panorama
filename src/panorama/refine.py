"""Seam-local refinement.

Educational implementation:
- detects seam band
- lightly smooths residual artifacts in seam neighborhood

This is NOT the full LPAM implementation, but a code scaffold that maps the
idea to an understandable local operation.
"""

import cv2
import numpy as np


def seam_local_refine(
    image_a: np.ndarray,
    warped_b: np.ndarray,
    seam_band: np.ndarray,
    blur_ksize: int = 5,
) -> np.ndarray:
    """Refine seam neighborhood by local residual-aware fusion.

    Implementation notes for students:
    1. Compute absolute difference in seam band.
    2. Use a soft confidence weight (lower residual -> higher confidence).
    3. Blend only inside seam band; outside keep original warped image.
    """
    diff = np.mean(np.abs(image_a.astype(np.float32) - warped_b.astype(np.float32)), axis=2)
    conf_b = np.exp(-diff / 25.0)
    conf_b = cv2.GaussianBlur(conf_b, (blur_ksize, blur_ksize), 0)
    conf_b = np.clip(conf_b, 0.0, 1.0)

    conf_b_3 = conf_b[..., None]
    fused = (1.0 - conf_b_3) * image_a.astype(np.float32) + conf_b_3 * warped_b.astype(np.float32)

    out = warped_b.astype(np.float32).copy()
    band = seam_band.astype(bool)
    out[band] = fused[band]
    return np.clip(out, 0, 255).astype(np.uint8)
