# Probe Card Contact Mark PMI Analysis System (main9.py) 기술 가이드

이 문서는 프로브 카드(Probe Card)의 접촉 마크(Scrub Mark)를 정밀 분석하고 계측하는 **PMI (Probe Mark Inspection) 분석 시스템 (`main9.py`)**의 기술 가이드입니다. 

기존 버전에서 지적된 패드(Pad) 경계면 오검출 문제를 극복하기 위해 컴퓨터 비전 알고리즘과 딥러닝 세그멘테이션 모델을 유기적으로 결합하였으며, 산업용 장비 수준의 정밀 계측 엔진 및 누적 편차 추적을 위한 SPC 드리프트 분석 기술을 구현했습니다.

---

## 1. 기술 스택 (Technical Stack)

시스템은 아래의 기술 및 라이브러리를 기반으로 구축되었습니다.

*   **Language**: Python 3.10+
*   **Computer Vision**: OpenCV (Gabor Filter, Bilateral Filter, CLAHE, Image Registration/Alignment, Contour Analysis)
*   **Deep Learning**: Segment Anything Model 2 (SAM2) - (인터랙티브 프롬프트 기반 세그멘테이션)
*   **Data Processing**: NumPy (고속 행렬 연산 및 기하 데이터 처리)
*   **Visualization**: Matplotlib, `koreanize-matplotlib` (한글 폰트 완벽 지원 및 6패널 대시보드 리포팅)
*   **Database**: JSON 파일 기반 데이터베이스 (`learning_db.json`)
*   **Configuration**: JSON 설정 관리 (`analysis_config.json`)
*   **Output Formats**: CSV 정량적 레포트 및 PNG 포맷의 시각적 종합 분석 보고서

---

## 2. 상세 구현 기술 및 이론

### 2.1. 이미지 정합 (Alignment) & 전처리 (Preprocessing)
1.  **이미지 정합 (ImageAligner)**: 접촉 전(Before) 이미지와 접촉 후(After) 이미지 사이의 미세한 픽셀 어긋남을 보정하기 위해 특징점 기반 호모그래피(Homography) 변환을 수행하여 완전한 좌표 동기화를 이룹니다.
2.  **주파수 공간 필터링 (Gabor Filter)**: 특정 방향(Theta)과 주파수(Lambda)를 갖는 노이즈(예: 프로브 카드 패드 표면의 특정 패턴 및 미세 스크래치 결)를 제거하거나 억제하기 위해, 고속 푸리에 변환(FFT) 영역에서 가보어 필터 커널을 밴드 리젝트(Band Reject) 필터 형태로 로컬 적용합니다.
3.  **에지 보존 스무딩 (Bilateral Filter)**: 이미지의 에지(경계선)는 뭉개지 않고 내부 잔여 노이즈만 효율적으로 스무딩합니다.
4.  **대비 강화 (CLAHE)**: 이미지 전반의 조명 불균일성을 억제하고 마크와 패드 표면 사이의 고대비 경계를 확보하기 위해 대비 제한 적응형 히스토그램 평활화(Contrast Limited Adaptive Histogram Equalization)를 적용합니다.

### 2.2. Active Learning & SAM2 세그멘테이션
*   **SAM2 (Segment Anything Model 2)**: 검출할 마크의 특징점을 사용자가 제공하면(Prompt), 픽셀 단위의 정밀한 세그멘테이션 마스크를 도출합니다.
*   **특징 추출 (FeatureExtractor)**: 마스크 내부의 시각적 및 형태학적 정보를 분석하여 고유 프로필로 변환합니다.
    *   *시각적 특징*: CIELAB 및 HSV 색상 영역에서의 평균값 및 표준편차
    *   *형태학적 특징*: 원형도(Circularity), 종횡비(Aspect Ratio), 조밀도(Solidity), 면적(Area)
*   **피처 매칭 (FeatureMatcher)**: 새로 탐색된 마크 후보군의 특징 벡터와 학습 DB(`learning_db.json`)에 등록된 마크 및 노이즈 프로필 사이의 가중 거리(Weighted Distance)를 측정합니다.
    $$\text{Total Distance} = 1.5 \times \text{Color Dist} + 1.0 \times \text{Circularity Diff} + 0.8 \times \text{Aspect Ratio Diff} + 0.5 \times \text{Solidity Diff}$$
    이 거리를 기반으로 후보군이 '마크'일 확률을 $0.0 \sim 1.0$ 사이의 스코어로 수치화하고 불필요한 노이즈를 자동 필터링합니다.

### 2.3. 산업 PMI 수준 정밀 계측 엔진 (MarkMetrology)
각 검출 마크에 대해 산업 규격 수준의 기하학적 수치 및 등급을 정량 판정합니다.

1.  **Tilted Rectangle Fitting (경사 바운딩 박스 피팅)**:
    *   마크의 윤곽선에 대해 `cv2.minAreaRect`를 적용하여 경사각을 고려한 최소 사각형을 계산합니다.
    *   이를 통해 정확한 장축(Major Axis)과 단축(Minor Axis)을 측정하고, 실제 스크럽 진행 방향의 각도(Scrub Angle, $-90^\circ \sim +90^\circ$ 정규화)를 산출합니다.
2.  **패드 경계 최소 거리 측정 (Min Edge Distance)**:
    *   마크의 윤곽선 외곽 좌표들로부터 패드 ROI 테두리 4변(좌, 우, 상, 하)까지의 거리를 개별 계산하여 최소 경계 거리를 확인합니다.
    *   경계선 이내로 지나치게 밀착되면 패시베이션(Passivation) 막 손상 위험(Defective/Marginal)으로 분류됩니다.
3.  **중심 오프셋 (Offset & Normalized Offset)**:
    *   패드 중심 좌표 대비 마크 무게중심의 X, Y 편차를 측정하며, 패드 전체 크기 대비 상대 비율(Normalized Offset)로 정규화합니다.
4.  **Overdrive 추정**:
    *   스크럽의 장축 길이를 패드의 기준 치수(가로/세로 중 최솟값)와 비교하여 과도한 눌림(Overdrive 과다) 또는 접촉 미달(Overdrive 미달) 여부를 판단합니다.
5.  **마크 깊이/강도 추정 (Delta L*)**:
    *   마크 영역 내의 CIELAB 색상계 $L^*$ (밝기) 성분을 접촉 전(Before)과 후(After)를 비교하여 변화량($\Delta L^*$)을 추출합니다. 변화량이 음수로 커질수록 표면이 깊게 눌려 어두운 섀도가 많이 형성되었음을 의미합니다.
6.  **등급 분류 (Grade Classification)**:
    *   **Good**: 정상적인 접촉. 경계 여유가 충분하고 적절한 Overdrive 비율 범위에 속함.
    *   **Marginal**: 경계선에 다소 근접하거나 접촉 상태가 권장 범위를 아슬아슬하게 통과하는 상태.
    *   **Defective**: 패시베이션 파괴 위험(경계선 매우 인접), No Contact(접촉 불량), 과접촉(Punch-through 현상 위험)에 해당할 시 판정됨.

### 2.4. SPC 드리프트 분석 (DriftAnalyzer)
*   **통계적 공정 제어 (Statistical Process Control)**:
    여러 차례의 정밀 검사 세션을 거치는 동안 특정 패드에서 마크 중심점이 한 방향으로 서서히 이동하는 드리프트(Drift) 현상을 조기에 파악합니다.
*   **제어 한계선 (UCL/LCL)**:
    누적 히스토리 데이터의 오프셋 표준편차($\sigma$)를 계산하고, $\pm 2\sigma$ 관리 제어 상한선/하한선을 차트에 실시간으로 표기합니다.
*   **경향성 판정**:
    최근 3세션 연속으로 기준 임계값(2px) 이상 누적 이동 시 추세(`drift_→우측`, `drift_↓하단` 등)를 판정하여 장비 헤드의 정렬 보정 지표로 제공합니다.

---

## 3. 실행 방법 (Execution Guide)

### 3.1. 환경 구성 및 가상환경 설정
프로젝트 폴더 내에서 `uv` 가상환경 도구를 사용하여 파이썬 환경을 생성하고 의존 패키지를 구성합니다.

```bash
# 1. uv를 통한 가상환경 (.venv) 생성
uv venv

# 2. 가상환경 활성화 (Windows PowerShell 기준)
.venv\Scripts\activate

# 3. 필요한 핵심 패키지 설치
uv pip install opencv-python numpy matplotlib koreanize-matplotlib
# (SAM2 및 추가 패키지가 요구되는 경우 함께 설치합니다)
```

### 3.2. 실행 단계
터미널에서 가상환경이 활성화된 상태로 `main9.py`를 실행합니다.

```bash
python main9.py
```

### 3.3. 대화형 실행 프로세스 (CLI/GUI 동작 설명)
프로그램이 구동되면 콘솔 터미널과 OpenCV GUI, Matplotlib 창을 번갈아 사용하며 분석이 진행됩니다.

1.  **기존 학습 DB 초기화 여부 결정**
    *   터미널에 `기존 학습 DB(learning_db.json)를 초기화하고 처음부터 새로 학습하시겠습니까? (y/n)` 문구가 표시됩니다.
    *   완전 청정 상태에서 특징 프로필을 다시 구축하려면 `y`를 입력하고, 기존 학습 데이터를 누적하여 사용하려면 `n` (또는 엔터)을 누릅니다.
2.  **분석 대상 이미지 조합 선택**
    *   제공되는 테스트 이미지 조합(1~3번) 중 분석을 원하는 번호를 입력합니다. (예: `1` 입력 시 `before.jpg` / `after.jpg` 로딩)
3.  **로드 이미지 쌍 유효성 확인 (Matplotlib 창)**
    *   Before/After 원본 쌍이 포함된 팝업 창이 뜹니다. 이미지를 눈으로 확인하고 **이 팝업 창을 완전히 닫아야** 다음 정합 및 분석 단계로 넘어갑니다.
4.  **ROI 검출 분할 방향 선택**
    *   패드가 위치한 레이아웃 방향을 선택합니다 (`1` : 좌우 패드, `2` : 상하 패드).
5.  **Active Learning 인터랙티브 피드백 제어 (OpenCV 창)**
    *   `Active Learning Feedback` 제목의 OpenCV 윈도우가 생성됩니다. 화면 우측의 'After' 이미지 영역을 보며 마우스를 조작합니다.
    *   **조작 및 단축키 안내**:
        *   **마우스 좌클릭 (Positive)**: 미검출된 마크 영역의 중심 부분을 클릭하면 초록색 표시와 함께 마크로 강제 추가 지정됩니다.
        *   **마우스 우클릭 (Negative)**: 패드 가장자리 경계면에 잘못 검출된 원형 노이즈(패시베이션 반사 등)를 클릭하면 파란색 X 표시가 되며 탐지 대상에서 즉각 제외됩니다.
        *   **`[Space]` 또는 `[Enter]`**: 클릭 피드백을 적용하여 SAM2 모델을 통해 마스크를 즉각 갱신합니다.
        *   **`[u]` (Undo)**: 직전에 실수로 클릭한 포인트를 한 단계 취소합니다.
        *   **`[r]` (Reset)**: 수동 피드백 포인트를 전부 리셋하고 초기 탐지 상태로 되돌립니다.
        *   **`[s]` (Save & Exit)**: 현재 검출 상태를 확정하고, 유효한 마크 및 노이즈 프로필을 특징 DB(`learning_db.json`)에 학습 및 세션 오프셋 기록을 저장한 후 분석을 마무리합니다.
        *   **`[q]` (Quit)**: 아무것도 저장하지 않고 즉시 종료합니다.
6.  **정량 보고서 확인 및 결과 출력**
    *   분석이 완료되면 `reports` 디렉토리 아래에 타임스탬프가 지정된 정량적 분석 CSV 파일(`pmi_report_YYYYMMDD_HHMMSS.csv`)이 생성됩니다.
    *   동시에 화면에 **6패널 종합 시각적 대시보드 리포트** 팝업(Matplotlib)이 나타나며 정밀 Fitting 결과, 패드 경계선 거리 화살표, 전치(transposed) 요약 테이블, SPC 누적 오프셋 추세 차트를 가시적으로 바로 확인할 수 있습니다.
