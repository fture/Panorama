"""Alignment module.

This module exposes a beginner-readable baseline aligner interface and a
placeholder adapter for UDIS++ integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import AlignmentOutput, PairSample


@dataclass
class BaselineHomographyAligner:
    """Classic ORB+RANSAC homography aligner.

    Why this class exists:
    - Students can read and run a full alignment baseline end-to-end.
    - It creates a stable baseline before integrating UDIS++.
    """

    nfeatures: int = 3000
    ransac_thresh: float = 3.0

    def run(self, sample: PairSample) -> AlignmentOutput:
        """Align image_b into image_a coordinate system.

        Steps:
        1. Detect ORB keypoints and descriptors.
        2. Match descriptors with Hamming distance.
        3. Estimate homography using RANSAC.
        4. Warp image_b to image_a frame.
        5. Build overlap mask and a simple residual map.
        """

        a = sample.image_a
        b = sample.image_b
        gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=self.nfeatures)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)

        if des_a is None or des_b is None:
            raise RuntimeError("Descriptor extraction failed. Check image quality.")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_b, des_a)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) < 10:
            raise RuntimeError("Not enough matches for homography estimation.")

        src = np.float32([kp_b[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        h_mat, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if h_mat is None:
            raise RuntimeError("Homography estimation failed.")

        h, w = a.shape[:2]
        warped_b = cv2.warpPerspective(b, h_mat, (w, h))

        valid_b = cv2.warpPerspective(np.ones((b.shape[0], b.shape[1]), dtype=np.uint8), h_mat, (w, h))
        valid_a = np.ones((h, w), dtype=np.uint8)
        overlap = ((valid_a > 0) & (valid_b > 0)).astype(np.uint8)

        residual = np.mean(np.abs(a.astype(np.float32) - warped_b.astype(np.float32)), axis=2)

        return AlignmentOutput(
            warped_b_to_a=warped_b,
            overlap_mask=overlap,
            residual_flow_mag=residual,
            extras={
                "homography": h_mat,
                "inlier_mask": inlier_mask.astype(np.uint8),
            },
        )
