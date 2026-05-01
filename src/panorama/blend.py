"""Blending and composition (Phase 5 final composition stage)."""

import cv2
import numpy as np


def alpha_blend(image_a: np.ndarray, warped_b: np.ndarray, seam_x: np.ndarray, feather: int = 32) -> np.ndarray:
    """Feather blend around seam path.

    Left side favors A, right side favors warped B.
    """
    h, w = image_a.shape[:2]
    alpha = np.zeros((h, w), dtype=np.float32)
    for y, x in enumerate(seam_x):
        l = max(0, x - feather)
        r = min(w - 1, x + feather)
        if r > l:
            alpha[y, :l] = 1.0
            alpha[y, l:r + 1] = np.linspace(1.0, 0.0, r - l + 1)
            alpha[y, r + 1:] = 0.0
        else:
            alpha[y, :x] = 1.0
            alpha[y, x:] = 0.0

    alpha3 = alpha[..., None]
    out = alpha3 * image_a.astype(np.float32) + (1.0 - alpha3) * warped_b.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
