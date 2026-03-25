# 기술 스터디 가이드: Probe Card Contact Mark 검출 시스템

이 문서는 본 프로젝트에 적용된 핵심 기술의 원리와 구현 코드의 구조를 분석하고, 관련 분야의 심화 학습을 돕기 위해 작성되었습니다.

---

## 1. 적용 기술 스택 (Core Technologies)

### 🚀 Meta SAM2 (Segment Anything Model 2)
- **역할**: 이미지 내의 모든 Contact Mark를 정교하게 객체 단위로 분리(Segmentation).
- **특징**: 
    - **Zero-shot**: 학습 데이터에 없던 새로운 형태의 mark도 추가 학습 없이 즉시 인식 가능.
    - **Hiera**: Hierarchical Vision Transformer 구조를 사용하여 이전 SAM보다 훨씬 빠르고 정확한 성능을 제공.
    - **Promptable**: 점, 박스, 또는 마스크 자체를 힌트로 사용하여 특정 영역만 정교하게 추출 가능.

### 📐 Image Registration (ECC 알고리즘)
- **역할**: '전' 이미지와 '후' 이미지의 미세한 위치 차이(진동, 정렬 오차)를 보정.
- **원리**: **Enhanced Correlation Coefficient (ECC)** 알고리즘을 사용하여 두 이미지 간의 상관 계수를 수렴시키며 최적의 변환 행렬(Euclidean, Affine 등)을 찾아냄.
- **필요성**: 픽셀 단위 비교 시 위치가 단 1픽셀만 어긋나도 모든 마크가 '새로운 마크'로 오인될 수 있기 때문에 필수적인 전처리 과정임.

### 🧬 IoU (Intersection over Union) 분석
- **역할**: 이전 이미지에 존재하던 마크와 현재 이미지의 마스크를 비교하여 '변화'를 감지.
- **로직**: 두 마스크의 교집합(Intersection) 영역을 합집합(Union) 영역으로 나눈 값이 임계값(Threshold) 이하면 새롭게 생성된 마크로 판단.

---

## 2. 코드 구조 리뷰 (Code Review)

### [processor.py](file:///c:/python/image_segmentation/processor.py) - 핵심 엔진
- **[ImageAligner](file:///c:/python/image_segmentation/processor.py#9-43)**:
    - `cv2.findTransformECC`를 사용해 기하학적 보정 수행.
    - 하드웨어적 한계(진동 등)를 소프트웨어적으로 극복하는 핵심 로직.
- **[MarkDetector](file:///c:/python/image_segmentation/processor.py#44-86)**:
    - `cuda` 감지 로직을 포함하여 `TF32`, `bfloat16`, `autocast`를 통한 GPU 가속 최적화 구현.
    - `torch.inference_mode()`를 사용하여 불필요한 그래디언트 계산을 방지하고 메모리 효율 극대화.
- **[ChangeAnalyzer](file:///c:/python/image_segmentation/processor.py#87-119)**:
    - 리스트 내 모든 객체를 전수 비교하여 `IoU` 기반의 필터링 수행.

### [predict.py](file:///c:/python/image_segmentation/predict.py) - 활용 인터페이스
- `argparse`를 통한 CLI 환경 구축.
- 프로덕션 환경에서 실제 이미지 경로를 인자로 받아 처리할 수 있는 구조.

---

## 3. 심화 학습 로직 (Study Topics)

### Step 1: Vision Transformer (ViT) 이해
SAM2의 근간이 되는 **Transformer** 아키텍처가 어떻게 이미지를 패치 단위로 나누고, 전역적인 문맥(Global Context)을 파악하는지 학습하면 좋습니다.
- 추천 키워드: `Attention Mechanism`, `Patch Embedding`, `Position Encoding`.

### Step 2: 최적화된 추론 (Optimization)
본 프로젝트에 적용된 가속 기법 외에 더 고도화된 최적화 기법을 찾아보세요.
- 추천 키워드: `TensorRT` (NVIDIA 전용 엔진), `ONNX Runtime`, `Quantization` (양자화).

### Step 3: 고급 이미지 정합 기술
ECC 외에도 특징점(Feature points) 기반의 정합 방법을 공부하면 더 극단적인 회전이나 왜곡 상황에 대처할 수 있습니다.
- 추천 키워드: `SIFT`, `ORB`, `RANSAC`, `Homography Matrix`.

---

## 4. 라이브러리 관리 (Modern Python)
- **uv**: 기존의 `pip`보다 수십 배 빠른 패키지 설치 및 의존성 해결(Rust 기반).
- **pyproject.toml**: 단순 패키지 목록이 아닌, PyTorch CUDA 인덱스와 같이 복잡한 소스 설정을 한 곳에서 관리하는 현대적 표준.
