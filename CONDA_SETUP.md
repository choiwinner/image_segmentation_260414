# Conda 환경 설치 가이드

이 프로젝트는 원래 `uv`를 사용하여 관리되었으나, `conda` 환경에서 설치하고 실행하려는 경우 아래의 단계를 따르세요.

## 0. 시작 전: CUDA 및 드라이버 버전 확인

PyTorch 설치 전, 본인의 PC 환경을 먼저 확인해야 합니다. 터미널에서 아래 명령어들을 실행해 보세요.

### 1) NVIDIA 드라이버 및 지원 가능 버전 확인
```bash
nvidia-smi
```
출력 화면의 우측 상단 **CUDA Version**을 확인하세요. 이는 드라이버가 지원하는 **최대 버전**입니다.

### 2) CUDA Toolkit 설치 버전 확인
```bash
nvcc --version
```
`release` 뒤의 숫자(예: 12.1)가 실제 설치된 버전입니다.

> [!TIP]
> 이 프로젝트는 **CUDA 12.1**을 기준으로 가이드합니다. 본인의 버전이 이보다 낮다면 [NVIDIA 공식 홈페이지](https://developer.nvidia.com/cuda-downloads)에서 드라이버 및 툴킷을 업데이트하는 것을 권장합니다.

## 1. Conda 환경 생성 및 활성화

먼저 Python 3.12 버전을 사용하는 새로운 conda 환경을 생성합니다.

```bash
# 환경 생성
conda create -n image-seg python=3.12 -y

# 환경 활성화
conda activate image-seg
```

## 2. PyTorch 및 관련 패키지 설치 (CUDA 12.4 최적화)

사용자님의 시스템 환경(CUDA 12.4)에 최적화된 PyTorch 버전 설치 명령어입니다.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

> [!TIP]
> 상기 명령어는 확인된 시스템 CUDA 버전(12.4)에 맞춰 수정되었습니다. 만약 다른 버전의 CUDA를 원하신다면 [PyTorch 공식 홈페이지](https://pytorch.org/get-started/locally/)를 참조하세요.

## 3. 기타 종속성 설치

나머지 필요한 패키지들을 설치합니다.

```bash
pip install koreanize-matplotlib matplotlib numpy opencv-python pillow sam2
```

## 4. 모델 가중치(Checkpoint) 및 필수 파일 다운로드

Python 패키지(`pip`) 설치 외에도 모델 실행을 위해 대용량 가중치 파일(`.pt`)과 설정 파일이 필요합니다.

### 4.1. SAM2 모델 가중치 다운로드
이 프로젝트는 **SAM2 Hiera-Large** 모델을 사용합니다. 아래 명령어를 사용하여 프로젝트 루트 디렉토리에 직접 다운로드해야 합니다.

**Windows (PowerShell)**:
```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" -OutFile "sam2_hiera_large.pt"
```

**Linux / macOS (curl)**:
```bash
curl -L -O https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### 4.2. Hugging Face를 통한 다운로드 (권장)
가장 안정적인 방법은 `huggingface_hub` 라이브러리를 사용하는 것입니다.

**라이브러리 설치**:
```bash
pip install huggingface_hub
```

**Python 스크립트로 다운로드**:
```python
from huggingface_hub import hf_hub_download

# sam2_hiera_large.pt 다운로드
hf_hub_download(
    repo_id="facebook/sam2-hiera-large",
    filename="sam2_hiera_large.pt",
    local_dir="./",
    local_dir_use_symlinks=False
)
```

**CLI를 사용하여 다운로드**:
```bash
huggingface-cli download facebook/sam2-hiera-large sam2_hiera_large.pt --local-dir . --local-dir-use-symlinks False
```

### 4.3. 브라우저를 통한 직접 다운로드 (웹 페이지)
터미널 명령어 사용이 익숙하지 않다면, 브라우저에서 허깅페이스 페이지에 접속하여 직접 다운로드할 수 있습니다. 각 모델별로 **가중치 파일(`.pt`)**과 **설정 파일(`.yaml`)** 두 가지를 모두 다운로드해야 합니다.

1.  사용하고자 하는 모델의 허깅페이스 페이지에 접속합니다. (아래 4.4 섹션의 링크 참조)
2.  **Files and versions** 탭을 클릭합니다.
3.  목록에서 `.pt` 파일과 `.yaml` 파일 오른쪽의 다운로드 아이콘(↓)을 클릭하여 각각 다운로드합니다.
4.  다운로드된 파일들을 프로젝트의 **최상위 폴더(Root)**인 `image_segmentation_260325` 폴더 안으로 이동시킵니다.

> [!IMPORTANT]
> 반드시 프로젝트 루트(최상위 폴더)에 위치시켜야 합니다. 하위 폴더에 넣으면 모델 로드 시 파일을 찾을 수 없다는 에러가 발생할 수 있습니다.

### 4.4. 모델별 가중치 및 설정 파일 목록
사용자의 PC 환경(VRAM 용량 등)에 맞춰 적절한 모델을 선택하세요. 각 모델별로 필요한 파일과 다운로드 링크는 다음과 같습니다.

| 모델 크기 | 설정 파일 (.yaml) | 가중치 파일 (.pt) | 허깅페이스 레포지토리 링크 |
| :--- | :--- | :--- | :--- |
| **Tiny** | `sam2_hiera_t.yaml` | `sam2_hiera_tiny.pt` | [facebook/sam2-hiera-tiny](https://huggingface.co/facebook/sam2-hiera-tiny/tree/main) |
| **Small** | `sam2_hiera_s.yaml` | `sam2_hiera_small.pt` | [facebook/sam2-hiera-small](https://huggingface.co/facebook/sam2-hiera-small/tree/main) |
| **Base Plus** | `sam2_hiera_b+.yaml` | `sam2_hiera_base_plus.pt` | [facebook/sam2-hiera-base-plus](https://huggingface.co/facebook/sam2-hiera-base-plus/tree/main) |
| **Large** | `sam2_hiera_l.yaml` | `sam2_hiera_large.pt` | [facebook/sam2-hiera-large](https://huggingface.co/facebook/sam2-hiera-large/tree/main) |

**CLI(터미널) 다운로드 예시 (Tiny 모델 기준)**:
```bash
# Weights 다운로드
huggingface-cli download facebook/sam2-hiera-tiny sam2_hiera_tiny.pt --local-dir . --local-dir-use-symlinks False
# Config 다운로드
huggingface-cli download facebook/sam2-hiera-tiny sam2_hiera_t.yaml --local-dir . --local-dir-use-symlinks False
```

> [!IMPORTANT]
> - 가중치 파일은 용량이 크므로(약 1GB 내외) 설치 시 네트워크 환경을 확인하세요.
> - 파일명은 반드시 소스 코드에 지정된 이름(`sam2_hiera_large.pt` 등)과 일치해야 합니다.
> - 프로젝트 폴더 내에 정확히 위치해야 모델 로드 시 에러가 발생하지 않습니다.

설치가 완료되면 아래 스크립트를 실행하여 PyTorch와 CUDA가 정상적으로 인식되는지 확인합니다.

```bash
python check_cuda.py
```

## 6. 프로젝트 실행

모든 패키지와 모델 가중치가 준비되었다면 프로젝트를 실행합니다.

```bash
python main.py
```

## 7. 자주 묻는 질문 (FAQ)

### Q: PC에 이미 다른 버전의 PyTorch나 패키지가 설치되어 있는데 충돌하지 않나요?
**A: 아니요, 전혀 충돌하지 않습니다.** 

가상 환경(Conda)을 사용하는 주된 이유가 바로 이러한 충돌을 방지하기 위함입니다. `conda create`로 생성된 환경은 시스템의 다른 환경이나 전역(Global) 설정으로부터 완전히 **격리된 독립적인 공간**입니다. 

- **격리된 패키지 관리**: 각 가상 환경은 자신만의 패키지 저장소(`site-packages`)를 가집니다.
- **버전 자유도**: 다른 프로젝트에서는 구버전 Torch를 사용하더라도, 이 환경에서는 최신 버전인 2.5.1을 안전하게 사용할 수 있습니다.
- **전환 용이**: `conda activate` 명령어를 통해 필요할 때만 특정 환경의 설정으로 전환하여 사용할 수 있습니다.

## 8. 보급형 GPU (GTX 1650 등) 환경 가이드

비디오 메모리(VRAM)가 4GB 내외인 GTX 1650 환경에서도 본 프로젝트를 실행할 수 있습니다. 다만, 메모리 관리를 위해 아래 설정을 권장합니다.

### 8.1. CUDA 버전 추천
**CUDA 12.1** 또는 **12.4** 버전 설치를 권장합니다. (이미 해당 버전이 설치되어 있다면 드라이버만 최신으로 유지하세요.)

### 8.2. 가벼운 모델(Tiny/Small/Base Plus) 사용 가이드
GTX 1650 등 VRAM이 부족한 환경에서는 모델 크기를 줄여서 실행해야 합니다. 아래의 표를 참고하여 `processor.py` 또는 `main.py`의 `MarkDetector` 선언 부분을 수정하세요.

| 선택 모델 | `model_cfg` (설정 파일) | `checkpoint` (가중치 파일) | 비고 |
| :--- | :--- | :--- | :--- |
| **Tiny** | `"sam2_hiera_t.yaml"` | `"sam2_hiera_tiny.pt"` | 가장 빠름, 저사양 권장 |
| **Small** | `"sam2_hiera_s.yaml"` | `"sam2_hiera_small.pt"` | 속도와 정확도 균형 |
| **Base Plus** | `"sam2_hiera_b+.yaml"` | `"sam2_hiera_base_plus.pt"` | 표준 성능 |
| **Large** (기본값) | `"sam2_hiera_l.yaml"` | `"sam2_hiera_large.pt"` | 최고 정확도, 고사양 필요 |

**코드 수정 예시 (`processor.py`):**
```python
# 46번 라인 근처 MarkDetector 클래스의 __init__ 인자 수정
def __init__(self, model_cfg="sam2_hiera_t.yaml", checkpoint="sam2_hiera_tiny.pt"):
    # (나머지 코드는 동일하게 유지)
```

또는 **`main.py`**에서 명시적으로 호출할 때 변경할 수도 있습니다:
```python
# main.py에서 detector 초기화 시 변경
detector = MarkDetector(model_cfg="sam2_hiera_t.yaml", checkpoint="sam2_hiera_tiny.pt")
```

### 8.3. 성능 최적화 팁
코드 내에서 아래의 기술을 사용하면 메모리 점유율을 대폭 낮출 수 있습니다 (본 프로젝트의 `processor.py`에 이미 적용되어 있습니다).

- **Inference Mode**: `with torch.inference_mode():` 사용
- **Mixed Precision**: `with torch.autocast("cuda", dtype=torch.bfloat16):` 사용

> [!TIP]
> 만약 Tiny 모델을 사용해도 끊김이 심하다면, 이미지를 입력하기 전에 `cv2.resize`를 통해 이미지 해상도를 절반 정도로 낮추어 처리하는 것이 큰 도움이 됩니다.
