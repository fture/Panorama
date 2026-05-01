"""Seam optimization utilities.

Includes a simple dynamic-programming seam path for educational use.
Graph-cut can be added later as a drop-in replacement.
"""

import numpy as np


def min_vertical_seam(cost: np.ndarray) -> np.ndarray:
    """Find minimum-cost top-to-bottom seam using dynamic programming.

    Returns:
        seam_x: integer x-position for each y row.
    """
    h, w = cost.shape
    dp = np.full((h, w), np.inf, dtype=np.float32)
    parent = np.full((h, w), -1, dtype=np.int32)

    dp[0] = cost[0]

    for y in range(1, h):
        for x in range(w):
            left = max(0, x - 1)
            right = min(w - 1, x + 1)
            prev_xs = np.arange(left, right + 1)
            prev_vals = dp[y - 1, prev_xs]
            best_idx = int(np.argmin(prev_vals))
            best_prev_x = int(prev_xs[best_idx])
            dp[y, x] = cost[y, x] + dp[y - 1, best_prev_x]
            parent[y, x] = best_prev_x

    seam_x = np.zeros((h,), dtype=np.int32)
    seam_x[-1] = int(np.argmin(dp[-1]))

    for y in range(h - 2, -1, -1):
        seam_x[y] = parent[y + 1, seam_x[y + 1]]

    return seam_x


def seam_band_mask(seam_x: np.ndarray, width: int, img_w: int) -> np.ndarray:
    """Create a binary seam band mask around seam path for local refinement."""
    h = seam_x.shape[0]
    m = np.zeros((h, img_w), dtype=np.uint8)
    half = max(1, width // 2)
    for y, x in enumerate(seam_x):
        m[y, max(0, x - half):min(img_w, x + half + 1)] = 1
    return m
