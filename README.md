# Panorama (Research-Oriented, Student-Friendly)

이 프로젝트는 **대시차 강건 파노라마 스티칭**을 목표로 하며, 아래 파이프라인을 코드로 분해해 실험 가능하게 구성합니다.

1. Alignment (UDIS++ baseline + classic baseline)
2. Parallax Risk Map (DSFN 아이디어 차용)
3. Seam Optimization (cost map + seam solver)
4. Seam-Local Refinement (LPAM soft seam 아이디어 차용)
5. Blending / Composition

---

## 왜 DSFN/LPAM을 명시적으로 반영하는가?

- DSFN 계열의 핵심 아이디어는 **시차/경계 위험 지역을 명시적으로 인식**해 스티칭 의사결정(특히 seam)을 돕는 것입니다.
- LPAM 계열의 핵심 아이디어는 **전역 재정렬보다 seam 주변 국소 문제를 집중 수정**해 artifact를 줄이는 것입니다.

이 저장소는 위 핵심 아이디어를 학부 수준에서 이해 가능한 구조로 먼저 구현한 뒤, 점진적으로 원 논문 구현에 가까워지도록 설계했습니다.

---

## 데이터셋 선택 상세 가이드

### A. UDIS 계열 데이터 (필수)
- **선택 이유**
  - UDIS++ 정합 성능 재현 및 파인튜닝에 직접 연결됨.
  - 동일 계열 데이터에서 baseline을 잡아야 개선량 해석이 명확함.
- **주의점**
  - 실제 사용자 장면(강한 동적 객체, 반사, 야간)과 domain gap 존재 가능.

### B. 자체 수집 연속 촬영 데이터 (필수)
- **선택 이유**
  - 최종 배포/사용 장면과 가장 유사한 데이터.
  - seam-local refinement 및 동적 객체 ghosting 억제 성능을 현실적으로 검증 가능.
- **권장 수집 규칙**
  - 실내/실외/근거리 전경/사람·차량 포함 장면을 균형 있게.
  - 각 장면마다 3~8장 인접 프레임 확보.
  - 촬영 메타데이터(초점거리, 노출, 시간대) 저장.

### C. 보조 사전학습 데이터 (선택)
- 예: COCO 기반 synthetic pair 생성
- **선택 이유**
  - 동적 객체 마스크 및 위험지도 보조학습 데이터 확장에 유리.

---

## 수식/아이디어 출처 (필수 명시)

아래는 본 프로젝트 설계 시 직접 참고하는 1차/원출처 중심 문헌입니다.

- UDIS / UDIS++ 계열: 
  - Nie et al., “Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images.” (UDIS)  
  - 관련 후속 구현/개선(UDIS++) 공개 코드 및 논문 설명.
- DSFN 계열(시차 인식·경계 인식 기반 스티칭):
  - (프로젝트에서 사용하는 정확한 DSFN 논문 버전을 확정 후, 저자/연도/링크를 `docs/references.md`에 고정)
- LPAM soft seam 계열:
  - (프로젝트에서 사용하는 정확한 LPAM 논문 버전을 확정 후, 저자/연도/링크를 `docs/references.md`에 고정)
- Graph-cut seam 최적화:
  - Boykov and Jolly, 2001 (interactive graph cuts) — seam energy 최소화의 고전적 기반.
- Multi-band blending:
  - Burt and Adelson, 1983.

> 중요: README/보고서/발표에서 식을 사용할 때는 **해당 식이 나온 원 논문 식 번호**를 함께 표기하세요.

---

## 코드 구조 (학부 수준, 주석 매우 상세)

- `src/panorama/alignment.py`
  - ORB+RANSAC baseline aligner (이해용)
  - UDIS++ adapter를 이 위치에 추가 예정
- `src/panorama/risk_map.py`
  - depth/semantic/edge/residual 기반 risk map 융합
- `src/panorama/seam.py`
  - seam path 계산 (현재 DP 버전, 추후 graph-cut 교체 가능)
- `src/panorama/refine.py`
  - seam-local refinement (LPAM 아이디어 매핑 버전)
- `scripts/run_pipeline.py`
  - 단일 쌍 테스트 실행 스크립트

---

## 로컬 실행

```bash
PYTHONPATH=src python scripts/run_pipeline.py --img_a path/to/a.jpg --img_b path/to/b.jpg --out_dir outputs/demo
```

출력:
- `warped_b.png`
- `risk_map.png`
- `seam_band.png`
- `refined.png`

---

## 다음 구현 TODO

1. UDIS++ 공식 코드 연동 adapter 작성 (`src/panorama/alignment.py`)
2. 실제 depth estimator(MiDaS/DPT) 연결
3. semantic boundary를 segmentation model 기반으로 교체
4. seam solver를 graph-cut으로 교체
5. LPAM 논문 수식에 맞춘 soft seam mask prediction 모듈 추가
6. 정량 평가 지표 스크립트(`src/panorama/metrics/`) 확장
