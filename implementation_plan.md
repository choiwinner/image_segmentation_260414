# Probe Card Contact Mark 검출 구현 계획

전후 이미지 비교를 통해 새롭게 생성된 contact mark를 픽셀 단위로 검출하는 시스템을 구축합니다.

## User Review Required

> [!IMPORTANT]
> - **이미지 정합(Registration)**: 전후 사진 촬영 시 미세한 위치 차이가 발생할 수 있으므로, SAM2 적용 전 OpenCV의 ECC 알고리즘을 사용하여 이미지를 정밀하게 정렬합니다.
> - **SAM2 활용**: Meta의 최신 SOTA 모델인 SAM2를 사용하여 contact mark들을 개별 객체로 분리하여 인식합니다.
> - **비교 알고리즘**: 이전 이미지의 mark들과 현재 이미지의 mark들을 IoU(Intersection over Union) 기반으로 대조하여, 새롭게 추가된 mark만을 식별합니다.

## Proposed Changes

### [Component] Environment Setup & Data Prep
- `uv`를 사용한 가상환경 구축 및 필수 라이브러리 설치
- SAM2 체크포인트 다운로드 및 설정

### [Component] Image Processing Module
#### [NEW] [processor.py](file:///c:/python/image_segmentation/processor.py)
- `ImageAligner`: 전후 이미지 정합 (ECC 알고리즘 활용)
- `MarkDetector`: SAM2를 이용한 mark segmentation
- `ChangeAnalyzer`: 전후 mark 비교 및 신규 mark 필터링

### [Component] Visualization & CLI
#### [NEW] [main.py](file:///c:/python/image_segmentation/main.py)
- 전체 프로세스 실행 및 결과 시각화 (`koreanize-matplotlib` 활용)

## Verification Plan

### Automated Tests
- 합성 데이터(이미지에 인위적으로 mark 추가)를 사용하여 검출 정확도 테스트
- `pytest`를 통한 이미지 정합 오차 측정 테스트

### Manual Verification
- 제공된 샘플 이미지(전후 사진)를 입력하여 시각적으로 신규 mark가 정확히 빨간색으로 표시되는지 확인
- 픽셀 단위 마스크의 정교함 확인
