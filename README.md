# Panorama

## 구현 상태 (Phase 1~5 완료)

아래 5개 단계를 **코드로 실제 연결**했습니다.

1. **Phase 1 - Baseline 정합**: `BaselineHomographyAligner` (ORB + RANSAC).  
2. **Phase 2 - 시차 위험지도**: depth/semantic(edge+dynamic)/edge/residual 가중 융합 risk map.  
3. **Phase 3 - seam 최적화**: 최소 비용 seam 탐색(DP baseline) + seam band 생성.  
4. **Phase 4 - 동적 객체 처리**: optical-flow 기반 dynamic mask 추정.  
5. **Phase 5 - 최종 합성**: seam-local refinement + feather blending composition + 평가 스크립트.

> 주의: 현재는 연구용/학부용 readable baseline이며, UDIS++/DSFN/LPAM 원논문 full reproduction은 다음 단계입니다.

## 파일 맵

- `src/panorama/alignment.py`: baseline 정합
- `src/panorama/risk_map.py`: 위험지도 생성
- `src/panorama/seam.py`: seam 탐색 + seam band
- `src/panorama/refine.py`: seam-local refinement
- `src/panorama/dynamic_mask.py`: 동적 객체 마스크
- `src/panorama/blend.py`: 최종 블렌딩
- `src/panorama/pipeline.py`: phase1~5 통합 실행
- `src/panorama/metrics.py`: 비교 지표
- `scripts/run_pipeline.py`: 단일 페어 실행
- `scripts/eval_pipeline.py`: 다중 페어 정량 평가
- `scripts/train_risk_map.py`: trainable risk-fusion mini model

## 빠른 실행

```bash
PYTHONPATH=src python scripts/run_pipeline.py --img_a path/to/a.jpg --img_b path/to/b.jpg --out_dir outputs/demo
```

## 정량 평가

`pairs.txt` 형식(한 줄당):

```text
/path/a1.jpg,/path/b1.jpg
/path/a2.jpg,/path/b2.jpg
```

실행:

```bash
PYTHONPATH=src python scripts/eval_pipeline.py --pairs_txt pairs.txt
```

## 수식/출처 정책

- 논문 식/아이디어를 문서화할 때는 반드시 원출처(논문명, 연도, 식 번호)를 기입합니다.
- 상세 출처 고정은 `docs/references.md`에 누적합니다.
