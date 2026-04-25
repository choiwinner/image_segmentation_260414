# 이미지 전처리 및 가드 존 마크 검출 기능 추가

## 배경

`main3.py`로 실행 시 체크포인트 기반 동작은 정상이나, 실제 이미지에서는 예제와 달리 **이미지가 선명하지 않거나 초점이 다른 경우** 마크 검출 성능이 떨어지는 문제가 있음.

## 제안하는 변경 사항

**기존 파일(`main3.py`, `processor.py`, `dual_roi_utils.py`)은 일체 수정하지 않음.**

---

### 1. [NEW] `image_preprocessor.py` — 이미지 전처리 모듈

CLAHE/equalizeHist 기반 전처리 기능을 독립 모듈로 구현.

| 함수 | 설명 |
|---|---|
| `apply_equalize_hist(image)` | `cv2.equalizeHist` 적용 (그레이스케일 → BGR 복원) |
| `apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8))` | `cv2.createCLAHE` 적용 |
| `show_preprocessing_comparison(original, processed_hist, processed_clahe)` | 원본/equalizeHist/CLAHE 3개 이미지를 나란히 표시 |
| `ask_user_preprocessing_choice()` | 사용자에게 전처리 방법 선택 또는 스킵을 묻는 함수 |

**동작 흐름:**
1. before/after 이미지 각각에 equalizeHist와 CLAHE를 적용
2. 원본 vs 전처리 결과를 matplotlib 창으로 표시
3. 사용자가 선택: `0` 원본 유지, `1` equalizeHist, `2` CLAHE
4. 선택된 이미지로 이후 분석 진행

---

### 2. [NEW] `guard_zone_detector.py` — 가드 존 내 마크 직접 검출 모듈

차분(difference) 비교 없이 **단일 이미지의 가드 존 영역 내에서 마크만 직접 찾는 로직**.

| 함수 | 설명 |
|---|---|
| `detect_marks_in_guard_zone(image, roi_rect, guard_percentage, detector)` | ROI에서 가드 존 계산 → 가드 존 영역 크롭 → SAM2로 마크 검출 |
| `filter_dark_marks(masks, image, darkness_threshold=100)` | 검출된 마스크 중 어두운(검은) 마크만 필터링 |

**동작 흐름:**
1. ROI 사각형에서 가드 존 (sub_rect) 계산
2. 가드 존 영역만 이미지에서 크롭
3. 크롭된 이미지에서 SAM2 자동 세그멘테이션 실행
4. 검출된 마스크 중 평균 밝기가 어두운 것만 마크로 판정
5. 크롭 좌표를 원본 좌표로 변환하여 반환

> [!NOTE]
> 이 기능은 **"차이점 검출 전에 간단하게 마크가 있는지 먼저 확인"**하는 용도입니다. 
> 차분 기반 분석과 병행하여 사용할 수 있습니다.

---

### 3. [NEW] `main4.py` — 통합 파이프라인

`main3.py`의 기존 파이프라인을 기반으로 아래 단계를 **앞부분에** 추가:

```
[기존 main3.py 흐름]
1. 이미지 로드
2. 이미지 정합
                        ← [NEW] 전처리 단계 삽입
3. 파라미터 설정
4. ROI 검출
                        ← [NEW] 가드 존 마크 빠른 검출 (선택 옵션)
5. 인터랙티브 프롬프트
6. SAM2 분석
7. 시각화
```

**추가되는 사용자 상호작용:**

**Step 2.5 (전처리):**
```
[정보] 이미지 전처리 옵션:
  전처리 전/후 이미지 비교 표시...
  > 전처리 방법을 선택하세요 (0:원본유지, 1:equalizeHist, 2:CLAHE): 
```

**Step 4.5 (가드 존 빠른 검출):**
```
  > 가드 존 내 마크 빠른 검출을 실행할까요? (y/n): 
  > ROI 1: 가드 존 내 검출된 마크 3개
  > ROI 2: 가드 존 내 검출된 마크 2개
```

---

## 파일 구조 요약

| 파일 | 상태 | 역할 |
|---|---|---|
| `main3.py` | 기존 유지 | 기존 인터랙티브 분석 파이프라인 |
| `processor.py` | 기존 유지 | SAM2 모델, 정합, 분석 핵심 로직 |
| `dual_roi_utils.py` | 기존 유지 | ROI 검출, 가드 존, 시각화 유틸 |
| `image_preprocessor.py` | **[NEW]** | CLAHE/equalizeHist 전처리 + 비교 표시 |
| `guard_zone_detector.py` | **[NEW]** | 가드 존 내 마크 직접 검출 |
| `main4.py` | **[NEW]** | 전처리 + 가드 존 검출 통합 파이프라인 |

## Verification Plan

### 수동 검증
1. `main4.py` 실행 → 전처리 비교 이미지가 정상 표시되는지 확인
2. 전처리 선택 (0/1/2) 후 정상적으로 다음 단계 진행 확인
3. 가드 존 마크 빠른 검출 옵션 동작 확인
4. 기존 `main3.py` 실행이 영향 없이 정상 동작하는지 확인

### 자동 검증
```bash
python -c "from image_preprocessor import apply_clahe, apply_equalize_hist; print('import OK')"
python -c "from guard_zone_detector import detect_marks_in_guard_zone; print('import OK')"
```
