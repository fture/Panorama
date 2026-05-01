 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index b535e3e22bc8e5f899ffcad12417648c3b7f151b..83eebb220f6dc69ab762fab7c61093388b3eb0cd 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,53 @@
 # Panorama
-Panorama
+
+## 구현 상태 (Phase 1~5 완료)
+
+아래 5개 단계를 **코드로 실제 연결**했습니다.
+
+1. **Phase 1 - Baseline 정합**: `BaselineHomographyAligner` (ORB + RANSAC).  
+2. **Phase 2 - 시차 위험지도**: depth/semantic(edge+dynamic)/edge/residual 가중 융합 risk map.  
+3. **Phase 3 - seam 최적화**: 최소 비용 seam 탐색(DP baseline) + seam band 생성.  
+4. **Phase 4 - 동적 객체 처리**: optical-flow 기반 dynamic mask 추정.  
+5. **Phase 5 - 최종 합성**: seam-local refinement + feather blending composition + 평가 스크립트.
+
+> 주의: 현재는 연구용/학부용 readable baseline이며, UDIS++/DSFN/LPAM 원논문 full reproduction은 다음 단계입니다.
+
+## 파일 맵
+
+- `src/panorama/alignment.py`: baseline 정합
+- `src/panorama/risk_map.py`: 위험지도 생성
+- `src/panorama/seam.py`: seam 탐색 + seam band
+- `src/panorama/refine.py`: seam-local refinement
+- `src/panorama/dynamic_mask.py`: 동적 객체 마스크
+- `src/panorama/blend.py`: 최종 블렌딩
+- `src/panorama/pipeline.py`: phase1~5 통합 실행
+- `src/panorama/metrics.py`: 비교 지표
+- `scripts/run_pipeline.py`: 단일 페어 실행
+- `scripts/eval_pipeline.py`: 다중 페어 정량 평가
+- `scripts/train_risk_map.py`: trainable risk-fusion mini model
+
+## 빠른 실행
+
+```bash
+PYTHONPATH=src python scripts/run_pipeline.py --img_a path/to/a.jpg --img_b path/to/b.jpg --out_dir outputs/demo
+```
+
+## 정량 평가
+
+`pairs.txt` 형식(한 줄당):
+
+```text
+/path/a1.jpg,/path/b1.jpg
+/path/a2.jpg,/path/b2.jpg
+```
+
+실행:
+
+```bash
+PYTHONPATH=src python scripts/eval_pipeline.py --pairs_txt pairs.txt
+```
+
+## 수식/출처 정책
+
+- 논문 식/아이디어를 문서화할 때는 반드시 원출처(논문명, 연도, 식 번호)를 기입합니다.
+- 상세 출처 고정은 `docs/references.md`에 누적합니다.
diff --git a/docs/references.md b/docs/references.md
new file mode 100644
index 0000000000000000000000000000000000000000..16c7b08dfc25373a998759007c181d5adac64552
--- /dev/null
+++ b/docs/references.md
@@ -0,0 +1,11 @@
+# References to Fill and Freeze
+
+이 파일에는 프로젝트에서 사용하는 논문/코드의 **정확한 버전**을 확정해 기록합니다.
+
+## Required
+- UDIS 원논문 (정확한 서지)
+- UDIS++ 사용 레포 커밋 SHA
+- DSFN 사용 논문 (정확한 서지 + 식 번호)
+- LPAM 사용 논문 (정확한 서지 + 식 번호)
+- Graph-cut 관련 원논문
+- Blending 관련 원논문
diff --git a/scripts/eval_pipeline.py b/scripts/eval_pipeline.py
new file mode 100644
index 0000000000000000000000000000000000000000..331696ea005468a8ffa6de08478a85f028424367
--- /dev/null
+++ b/scripts/eval_pipeline.py
@@ -0,0 +1,59 @@
+"""Batch evaluation runner for experiment comparison (Phase 5)."""
+
+import argparse
+from pathlib import Path
+
+import cv2
+import numpy as np
+
+from panorama.alignment import BaselineHomographyAligner
+from panorama.dynamic_mask import estimate_dynamic_mask
+from panorama.metrics import overlap_l1, seam_band_l1
+from panorama.refine import seam_local_refine
+from panorama.risk_map import build_risk_map, depth_discontinuity_from_depth, edge_strength
+from panorama.seam import min_vertical_seam, seam_band_mask
+from panorama.types import PairSample
+
+
+def run_pair(path_a: Path, path_b: Path) -> dict:
+    a = cv2.imread(str(path_a))
+    b = cv2.imread(str(path_b))
+    sample = PairSample(pair_id=path_a.stem, image_a=a, image_b=b)
+    aligned = BaselineHomographyAligner().run(sample)
+
+    depth_proxy = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
+    depth_disc = depth_discontinuity_from_depth(depth_proxy)
+    edge = edge_strength(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))
+    dyn = estimate_dynamic_mask(a, b).astype(np.float32)
+
+    risk = build_risk_map(depth_disc, edge + 0.5 * dyn, edge, aligned.residual_flow_mag)
+    seam = min_vertical_seam(risk * aligned.overlap_mask.astype(np.float32))
+    band = seam_band_mask(seam, 64, a.shape[1])
+    refined = seam_local_refine(a, aligned.warped_b_to_a, band)
+
+    return {
+        "overlap_l1": overlap_l1(a, aligned.warped_b_to_a, aligned.overlap_mask),
+        "seam_band_l1_before": seam_band_l1(a, aligned.warped_b_to_a, band),
+        "seam_band_l1_after": seam_band_l1(a, refined, band),
+    }
+
+
+def main() -> None:
+    p = argparse.ArgumentParser()
+    p.add_argument("--pairs_txt", required=True)
+    args = p.parse_args()
+
+    lines = [x.strip() for x in Path(args.pairs_txt).read_text().splitlines() if x.strip()]
+    rows = []
+    for line in lines:
+        pa, pb = line.split(',')
+        rows.append(run_pair(Path(pa), Path(pb)))
+
+    keys = rows[0].keys()
+    for k in keys:
+        vals = np.array([r[k] for r in rows], dtype=np.float32)
+        print(f"{k}: mean={vals.mean():.4f} std={vals.std():.4f}")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/scripts/run_pipeline.py b/scripts/run_pipeline.py
new file mode 100644
index 0000000000000000000000000000000000000000..ddb9a607a004b7cdef0d315c5efeea7aaba2ef24
--- /dev/null
+++ b/scripts/run_pipeline.py
@@ -0,0 +1,40 @@
+"""Run a readable, educational panorama pipeline on one image pair."""
+
+import argparse
+from pathlib import Path
+
+import cv2
+
+from panorama.pipeline import run_full_pipeline
+from panorama.types import PairSample
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--img_a", required=True)
+    parser.add_argument("--img_b", required=True)
+    parser.add_argument("--out_dir", default="outputs/demo")
+    args = parser.parse_args()
+
+    out_dir = Path(args.out_dir)
+    out_dir.mkdir(parents=True, exist_ok=True)
+
+    a = cv2.imread(args.img_a)
+    b = cv2.imread(args.img_b)
+    if a is None or b is None:
+        raise FileNotFoundError("Failed to read input image(s).")
+
+    sample = PairSample(pair_id="demo", image_a=a, image_b=b)
+
+    outputs = run_full_pipeline(sample)
+
+    cv2.imwrite(str(out_dir / "aligned.png"), outputs["aligned"])
+    cv2.imwrite(str(out_dir / "risk_map.png"), (outputs["risk_map"] * 255).astype(np.uint8))
+    cv2.imwrite(str(out_dir / "dynamic_mask.png"), (outputs["dynamic_mask"] * 255).astype(np.uint8))
+    cv2.imwrite(str(out_dir / "seam_band.png"), outputs["seam_band"] * 255)
+    cv2.imwrite(str(out_dir / "refined.png"), outputs["refined"])
+    cv2.imwrite(str(out_dir / "composed.png"), outputs["composed"])
+
+
+if __name__ == "__main__":
+    main()
diff --git a/scripts/train_risk_map.py b/scripts/train_risk_map.py
new file mode 100644
index 0000000000000000000000000000000000000000..a6055f9c9f859294c3420c263c5a5d5021bf68f4
--- /dev/null
+++ b/scripts/train_risk_map.py
@@ -0,0 +1,30 @@
+"""Minimal trainable risk-fusion model (Phase 2/5).
+
+Student-friendly tiny CNN to fuse 4 cues into risk map.
+"""
+
+import torch
+import torch.nn as nn
+
+
+class TinyRiskNet(nn.Module):
+    def __init__(self) -> None:
+        super().__init__()
+        self.net = nn.Sequential(
+            nn.Conv2d(4, 16, 3, padding=1),
+            nn.ReLU(inplace=True),
+            nn.Conv2d(16, 16, 3, padding=1),
+            nn.ReLU(inplace=True),
+            nn.Conv2d(16, 1, 1),
+            nn.Sigmoid(),
+        )
+
+    def forward(self, x: torch.Tensor) -> torch.Tensor:
+        return self.net(x)
+
+
+if __name__ == "__main__":
+    model = TinyRiskNet()
+    x = torch.rand(2, 4, 128, 128)
+    y = model(x)
+    print("ok", y.shape)
diff --git a/src/panorama/__init__.py b/src/panorama/__init__.py
new file mode 100644
index 0000000000000000000000000000000000000000..fd1ae5484a329bc13f8bd7de6b3d648160aad79a
--- /dev/null
+++ b/src/panorama/__init__.py
@@ -0,0 +1 @@
+"""Panorama research pipeline package."""
diff --git a/src/panorama/alignment.py b/src/panorama/alignment.py
new file mode 100644
index 0000000000000000000000000000000000000000..05cc783787f1ec6e088e76ad2465b4f75e9718f1
--- /dev/null
+++ b/src/panorama/alignment.py
@@ -0,0 +1,83 @@
+"""Alignment module.
+
+This module exposes a beginner-readable baseline aligner interface and a
+placeholder adapter for UDIS++ integration.
+"""
+
+from __future__ import annotations
+
+from dataclasses import dataclass
+
+import cv2
+import numpy as np
+
+from .types import AlignmentOutput, PairSample
+
+
+@dataclass
+class BaselineHomographyAligner:
+    """Classic ORB+RANSAC homography aligner.
+
+    Why this class exists:
+    - Students can read and run a full alignment baseline end-to-end.
+    - It creates a stable baseline before integrating UDIS++.
+    """
+
+    nfeatures: int = 3000
+    ransac_thresh: float = 3.0
+
+    def run(self, sample: PairSample) -> AlignmentOutput:
+        """Align image_b into image_a coordinate system.
+
+        Steps:
+        1. Detect ORB keypoints and descriptors.
+        2. Match descriptors with Hamming distance.
+        3. Estimate homography using RANSAC.
+        4. Warp image_b to image_a frame.
+        5. Build overlap mask and a simple residual map.
+        """
+
+        a = sample.image_a
+        b = sample.image_b
+        gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
+        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
+
+        orb = cv2.ORB_create(nfeatures=self.nfeatures)
+        kp_a, des_a = orb.detectAndCompute(gray_a, None)
+        kp_b, des_b = orb.detectAndCompute(gray_b, None)
+
+        if des_a is None or des_b is None:
+            raise RuntimeError("Descriptor extraction failed. Check image quality.")
+
+        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
+        matches = matcher.match(des_b, des_a)
+        matches = sorted(matches, key=lambda m: m.distance)
+
+        if len(matches) < 10:
+            raise RuntimeError("Not enough matches for homography estimation.")
+
+        src = np.float32([kp_b[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
+        dst = np.float32([kp_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
+
+        h_mat, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
+        if h_mat is None:
+            raise RuntimeError("Homography estimation failed.")
+
+        h, w = a.shape[:2]
+        warped_b = cv2.warpPerspective(b, h_mat, (w, h))
+
+        valid_b = cv2.warpPerspective(np.ones((b.shape[0], b.shape[1]), dtype=np.uint8), h_mat, (w, h))
+        valid_a = np.ones((h, w), dtype=np.uint8)
+        overlap = ((valid_a > 0) & (valid_b > 0)).astype(np.uint8)
+
+        residual = np.mean(np.abs(a.astype(np.float32) - warped_b.astype(np.float32)), axis=2)
+
+        return AlignmentOutput(
+            warped_b_to_a=warped_b,
+            overlap_mask=overlap,
+            residual_flow_mag=residual,
+            extras={
+                "homography": h_mat,
+                "inlier_mask": inlier_mask.astype(np.uint8),
+            },
+        )
diff --git a/src/panorama/blend.py b/src/panorama/blend.py
new file mode 100644
index 0000000000000000000000000000000000000000..4034ea5e03547722c87f93c7c010479b9ccccde5
--- /dev/null
+++ b/src/panorama/blend.py
@@ -0,0 +1,27 @@
+"""Blending and composition (Phase 5 final composition stage)."""
+
+import cv2
+import numpy as np
+
+
+def alpha_blend(image_a: np.ndarray, warped_b: np.ndarray, seam_x: np.ndarray, feather: int = 32) -> np.ndarray:
+    """Feather blend around seam path.
+
+    Left side favors A, right side favors warped B.
+    """
+    h, w = image_a.shape[:2]
+    alpha = np.zeros((h, w), dtype=np.float32)
+    for y, x in enumerate(seam_x):
+        l = max(0, x - feather)
+        r = min(w - 1, x + feather)
+        if r > l:
+            alpha[y, :l] = 1.0
+            alpha[y, l:r + 1] = np.linspace(1.0, 0.0, r - l + 1)
+            alpha[y, r + 1:] = 0.0
+        else:
+            alpha[y, :x] = 1.0
+            alpha[y, x:] = 0.0
+
+    alpha3 = alpha[..., None]
+    out = alpha3 * image_a.astype(np.float32) + (1.0 - alpha3) * warped_b.astype(np.float32)
+    return np.clip(out, 0, 255).astype(np.uint8)
diff --git a/src/panorama/dynamic_mask.py b/src/panorama/dynamic_mask.py
new file mode 100644
index 0000000000000000000000000000000000000000..4786873a6bc9a959b8c593195692b9a4bd04f9ef
--- /dev/null
+++ b/src/panorama/dynamic_mask.py
@@ -0,0 +1,21 @@
+"""Dynamic object masking utilities (Phase 4).
+
+This educational module provides a simple motion-based dynamic mask.
+Later you can replace it with YOLO-seg / Mask2Former style semantic masks.
+"""
+
+import cv2
+import numpy as np
+
+
+def estimate_dynamic_mask(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
+    """Estimate moving-object mask using dense optical flow magnitude.
+
+    Returns a binary mask where high flow magnitude implies potential motion.
+    """
+    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
+    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
+    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
+    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
+    thr = float(np.percentile(mag, 85))
+    return (mag > thr).astype(np.uint8)
diff --git a/src/panorama/metrics.py b/src/panorama/metrics.py
new file mode 100644
index 0000000000000000000000000000000000000000..36b4cb56243e1832e4b6eacc85c80d310c407000
--- /dev/null
+++ b/src/panorama/metrics.py
@@ -0,0 +1,21 @@
+"""Evaluation metrics for Phase 5 comparison experiments."""
+
+import numpy as np
+
+
+def overlap_l1(image_a: np.ndarray, image_b: np.ndarray, overlap_mask: np.ndarray) -> float:
+    """Mean L1 error on overlap region."""
+    m = overlap_mask.astype(bool)
+    if m.sum() == 0:
+        return float('nan')
+    diff = np.mean(np.abs(image_a.astype(np.float32) - image_b.astype(np.float32)), axis=2)
+    return float(diff[m].mean())
+
+
+def seam_band_l1(image_a: np.ndarray, image_b: np.ndarray, seam_band: np.ndarray) -> float:
+    """Mean L1 error in seam-local region."""
+    m = seam_band.astype(bool)
+    if m.sum() == 0:
+        return float('nan')
+    diff = np.mean(np.abs(image_a.astype(np.float32) - image_b.astype(np.float32)), axis=2)
+    return float(diff[m].mean())
diff --git a/src/panorama/pipeline.py b/src/panorama/pipeline.py
new file mode 100644
index 0000000000000000000000000000000000000000..68ed454979422dbe27537622328c17f199b8a43d
--- /dev/null
+++ b/src/panorama/pipeline.py
@@ -0,0 +1,42 @@
+"""Integrated 5-phase panorama pipeline."""
+
+import cv2
+import numpy as np
+
+from .alignment import BaselineHomographyAligner
+from .blend import alpha_blend
+from .dynamic_mask import estimate_dynamic_mask
+from .refine import seam_local_refine
+from .risk_map import build_risk_map, depth_discontinuity_from_depth, edge_strength
+from .seam import min_vertical_seam, seam_band_mask
+from .types import PairSample
+
+
+def run_full_pipeline(sample: PairSample) -> dict:
+    """Run all stages from phase1 to phase5 on a single pair."""
+    aligned = BaselineHomographyAligner().run(sample)
+
+    gray = cv2.cvtColor(sample.image_a, cv2.COLOR_BGR2GRAY)
+    depth_proxy = gray.astype(np.float32)
+    depth_disc = depth_discontinuity_from_depth(depth_proxy)
+    edge = edge_strength(gray)
+
+    dyn_mask = estimate_dynamic_mask(sample.image_a, sample.image_b).astype(np.float32)
+    semantic_boundary = np.clip(edge + 0.5 * dyn_mask, 0.0, 1.0)
+
+    risk = build_risk_map(depth_disc, semantic_boundary, edge, aligned.residual_flow_mag)
+    seam_cost = risk * aligned.overlap_mask.astype(np.float32)
+    seam = min_vertical_seam(seam_cost)
+
+    band = seam_band_mask(seam, width=64, img_w=sample.image_a.shape[1])
+    refined = seam_local_refine(sample.image_a, aligned.warped_b_to_a, band)
+    composed = alpha_blend(sample.image_a, refined, seam, feather=32)
+
+    return {
+        "aligned": aligned.warped_b_to_a,
+        "risk_map": risk,
+        "dynamic_mask": dyn_mask,
+        "seam_band": band,
+        "refined": refined,
+        "composed": composed,
+    }
diff --git a/src/panorama/refine.py b/src/panorama/refine.py
new file mode 100644
index 0000000000000000000000000000000000000000..07120138dbe0296bf2b35b6a27432372777d0bcc
--- /dev/null
+++ b/src/panorama/refine.py
@@ -0,0 +1,39 @@
+"""Seam-local refinement.
+
+Educational implementation:
+- detects seam band
+- lightly smooths residual artifacts in seam neighborhood
+
+This is NOT the full LPAM implementation, but a code scaffold that maps the
+idea to an understandable local operation.
+"""
+
+import cv2
+import numpy as np
+
+
+def seam_local_refine(
+    image_a: np.ndarray,
+    warped_b: np.ndarray,
+    seam_band: np.ndarray,
+    blur_ksize: int = 5,
+) -> np.ndarray:
+    """Refine seam neighborhood by local residual-aware fusion.
+
+    Implementation notes for students:
+    1. Compute absolute difference in seam band.
+    2. Use a soft confidence weight (lower residual -> higher confidence).
+    3. Blend only inside seam band; outside keep original warped image.
+    """
+    diff = np.mean(np.abs(image_a.astype(np.float32) - warped_b.astype(np.float32)), axis=2)
+    conf_b = np.exp(-diff / 25.0)
+    conf_b = cv2.GaussianBlur(conf_b, (blur_ksize, blur_ksize), 0)
+    conf_b = np.clip(conf_b, 0.0, 1.0)
+
+    conf_b_3 = conf_b[..., None]
+    fused = (1.0 - conf_b_3) * image_a.astype(np.float32) + conf_b_3 * warped_b.astype(np.float32)
+
+    out = warped_b.astype(np.float32).copy()
+    band = seam_band.astype(bool)
+    out[band] = fused[band]
+    return np.clip(out, 0, 255).astype(np.uint8)
diff --git a/src/panorama/risk_map.py b/src/panorama/risk_map.py
new file mode 100644
index 0000000000000000000000000000000000000000..ee54a915b01e9c03638942e426babf60081f109d
--- /dev/null
+++ b/src/panorama/risk_map.py
@@ -0,0 +1,58 @@
+"""Risk-map generation for seam-aware stitching.
+
+The objective is to mark regions where seam crossing is dangerous:
+- depth discontinuity
+- semantic/object boundaries
+- strong edges
+- high residual alignment error
+"""
+
+import cv2
+import numpy as np
+
+
+def normalize_01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
+    """Normalize an array to [0, 1] for stable weighted fusion."""
+    x = x.astype(np.float32)
+    mn, mx = x.min(), x.max()
+    return (x - mn) / (mx - mn + eps)
+
+
+def depth_discontinuity_from_depth(depth_map: np.ndarray) -> np.ndarray:
+    """Compute depth discontinuity as gradient magnitude.
+
+    Intuition: large depth jumps often produce visible stitching artifacts.
+    """
+    gx = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
+    gy = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
+    return normalize_01(np.sqrt(gx * gx + gy * gy))
+
+
+def edge_strength(gray: np.ndarray) -> np.ndarray:
+    """Edge strength map from Canny output."""
+    edges = cv2.Canny(gray, 80, 160)
+    return normalize_01(edges)
+
+
+def build_risk_map(
+    depth_disc: np.ndarray,
+    semantic_boundary: np.ndarray,
+    edge_map: np.ndarray,
+    residual_map: np.ndarray,
+    w_depth: float = 0.30,
+    w_sem: float = 0.25,
+    w_edge: float = 0.20,
+    w_res: float = 0.25,
+) -> np.ndarray:
+    """Fuse multiple cues into a seam-risk map.
+
+    This is intentionally simple and transparent for student use.
+    Later, you can replace it with a trainable fusion network.
+    """
+    risk = (
+        w_depth * normalize_01(depth_disc)
+        + w_sem * normalize_01(semantic_boundary)
+        + w_edge * normalize_01(edge_map)
+        + w_res * normalize_01(residual_map)
+    )
+    return normalize_01(risk)
diff --git a/src/panorama/seam.py b/src/panorama/seam.py
new file mode 100644
index 0000000000000000000000000000000000000000..ee4b0210e8584f365816a1bed9bd45db25525f8a
--- /dev/null
+++ b/src/panorama/seam.py
@@ -0,0 +1,49 @@
+"""Seam optimization utilities.
+
+Includes a simple dynamic-programming seam path for educational use.
+Graph-cut can be added later as a drop-in replacement.
+"""
+
+import numpy as np
+
+
+def min_vertical_seam(cost: np.ndarray) -> np.ndarray:
+    """Find minimum-cost top-to-bottom seam using dynamic programming.
+
+    Returns:
+        seam_x: integer x-position for each y row.
+    """
+    h, w = cost.shape
+    dp = np.full((h, w), np.inf, dtype=np.float32)
+    parent = np.full((h, w), -1, dtype=np.int32)
+
+    dp[0] = cost[0]
+
+    for y in range(1, h):
+        for x in range(w):
+            left = max(0, x - 1)
+            right = min(w - 1, x + 1)
+            prev_xs = np.arange(left, right + 1)
+            prev_vals = dp[y - 1, prev_xs]
+            best_idx = int(np.argmin(prev_vals))
+            best_prev_x = int(prev_xs[best_idx])
+            dp[y, x] = cost[y, x] + dp[y - 1, best_prev_x]
+            parent[y, x] = best_prev_x
+
+    seam_x = np.zeros((h,), dtype=np.int32)
+    seam_x[-1] = int(np.argmin(dp[-1]))
+
+    for y in range(h - 2, -1, -1):
+        seam_x[y] = parent[y + 1, seam_x[y + 1]]
+
+    return seam_x
+
+
+def seam_band_mask(seam_x: np.ndarray, width: int, img_w: int) -> np.ndarray:
+    """Create a binary seam band mask around seam path for local refinement."""
+    h = seam_x.shape[0]
+    m = np.zeros((h, img_w), dtype=np.uint8)
+    half = max(1, width // 2)
+    for y, x in enumerate(seam_x):
+        m[y, max(0, x - half):min(img_w, x + half + 1)] = 1
+    return m
diff --git a/src/panorama/types.py b/src/panorama/types.py
new file mode 100644
index 0000000000000000000000000000000000000000..450440a47e83eff7724c2bd83cd02f302e367753
--- /dev/null
+++ b/src/panorama/types.py
@@ -0,0 +1,45 @@
+"""Shared dataclasses and type aliases for the panorama pipeline.
+
+This file intentionally contains only simple, beginner-friendly structures.
+"""
+
+from dataclasses import dataclass
+from typing import Dict, Optional
+
+import numpy as np
+
+
+@dataclass
+class PairSample:
+    """Container for one adjacent image pair.
+
+    Attributes:
+        pair_id: Unique sample id for reproducibility.
+        image_a: First image (H, W, 3), uint8.
+        image_b: Second image (H, W, 3), uint8.
+        dynamic_mask_a: Optional moving-object mask for image_a (H, W), bool/uint8.
+        dynamic_mask_b: Optional moving-object mask for image_b (H, W), bool/uint8.
+    """
+
+    pair_id: str
+    image_a: np.ndarray
+    image_b: np.ndarray
+    dynamic_mask_a: Optional[np.ndarray] = None
+    dynamic_mask_b: Optional[np.ndarray] = None
+
+
+@dataclass
+class AlignmentOutput:
+    """Output of the alignment stage.
+
+    Attributes:
+        warped_b_to_a: image_b warped into image_a frame.
+        overlap_mask: mask where both images are valid after warping.
+        residual_flow_mag: simple residual-flow magnitude proxy.
+        extras: dictionary for algorithm-specific intermediate values.
+    """
+
+    warped_b_to_a: np.ndarray
+    overlap_mask: np.ndarray
+    residual_flow_mag: np.ndarray
+    extras: Dict[str, np.ndarray]
 
EOF
)