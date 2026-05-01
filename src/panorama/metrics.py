"""Evaluation metrics for Phase 5 comparison experiments."""

import numpy as np


def overlap_l1(image_a: np.ndarray, image_b: np.ndarray, overlap_mask: np.ndarray) -> float:
    """Mean L1 error on overlap region."""
    m = overlap_mask.astype(bool)
    if m.sum() == 0:
        return float('nan')
    diff = np.mean(np.abs(image_a.astype(np.float32) - image_b.astype(np.float32)), axis=2)
    return float(diff[m].mean())


def seam_band_l1(image_a: np.ndarray, image_b: np.ndarray, seam_band: np.ndarray) -> float:
    """Mean L1 error in seam-local region."""
    m = seam_band.astype(bool)
    if m.sum() == 0:
        return float('nan')
    diff = np.mean(np.abs(image_a.astype(np.float32) - image_b.astype(np.float32)), axis=2)
    return float(diff[m].mean())
