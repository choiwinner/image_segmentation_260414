"""
이미지 전처리 모듈
- cv2.equalizeHist: 히스토그램 균등화
- cv2.createCLAHE: 적응적 히스토그램 균등화 (Contrast Limited Adaptive Histogram Equalization)

전처리 전/후 이미지를 사용자에게 표시하고 적용 여부를 결정하도록 함.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib


def apply_equalize_hist(image):
    """cv2.equalizeHist를 적용하여 이미지 대비를 개선함.
    
    Args:
        image: BGR 또는 그레이스케일 이미지
        
    Returns:
        전처리된 BGR 이미지
    """
    if len(image.shape) == 3:
        # BGR → YCrCb 변환 후 Y채널(밝기)에만 균등화 적용
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = cv2.equalizeHist(image)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """cv2.createCLAHE를 적용하여 지역적 대비를 개선함.
    
    Args:
        image: BGR 또는 그레이스케일 이미지
        clip_limit: 대비 제한 값 (기본 2.0)
        tile_grid_size: 타일 크기 (기본 8x8)
        
    Returns:
        전처리된 BGR 이미지
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 3:
        # BGR → YCrCb 변환 후 Y채널(밝기)에만 CLAHE 적용
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = clahe.apply(image)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def show_preprocessing_comparison(original, processed_hist, processed_clahe, title_prefix=""):
    """원본, equalizeHist, CLAHE 결과를 나란히 표시함.
    
    Args:
        original: 원본 BGR 이미지
        processed_hist: equalizeHist 적용 이미지
        processed_clahe: CLAHE 적용 이미지
        title_prefix: 제목 앞에 붙일 접두어 (예: "Before -", "After -")
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    prefix = f"{title_prefix} " if title_prefix else ""
    
    axes[0].set_title(f"{prefix}원본", fontsize=14)
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].axis('off')
    
    axes[1].set_title(f"{prefix}equalizeHist", fontsize=14)
    axes[1].imshow(cv2.cvtColor(processed_hist, cv2.COLOR_BGR2RGB))
    axes[1].axis('off')
    
    axes[2].set_title(f"{prefix}CLAHE", fontsize=14)
    axes[2].imshow(cv2.cvtColor(processed_clahe, cv2.COLOR_BGR2RGB))
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)


def ask_user_preprocessing_choice():
    """사용자에게 전처리 방법 선택을 묻는 함수.
    
    Returns:
        int: 0(원본 유지), 1(equalizeHist), 2(CLAHE)
    """
    print("\n  전처리 방법을 선택하세요:")
    print("    0: 원본 유지 (전처리 없음)")
    print("    1: equalizeHist (전역 히스토그램 균등화)")
    print("    2: CLAHE (적응적 히스토그램 균등화, 권장)")
    
    while True:
        choice = input("  > 선택 (0/1/2, 기본값 0): ").strip()
        if choice == '':
            return 0
        if choice in ('0', '1', '2'):
            return int(choice)
        print("    [!] 0, 1, 2 중 하나를 입력해 주세요.")


def preprocess_images(img_before, img_after):
    """before/after 이미지에 대해 전처리 옵션을 표시하고 사용자 선택을 적용함.
    
    Args:
        img_before: 전처리 전 before 이미지 (BGR)
        img_after: 전처리 전 after 이미지 (BGR)
        
    Returns:
        tuple: (전처리된 img_before, 전처리된 img_after)
    """
    # 각 이미지에 대해 전처리 결과 생성
    before_hist = apply_equalize_hist(img_before)
    before_clahe = apply_clahe(img_before)
    after_hist = apply_equalize_hist(img_after)
    after_clahe = apply_clahe(img_after)
    
    # 비교 이미지 표시
    print("  [정보] 전처리 비교 이미지를 표시합니다...")
    show_preprocessing_comparison(img_before, before_hist, before_clahe, title_prefix="Before")
    show_preprocessing_comparison(img_after, after_hist, after_clahe, title_prefix="After")
    
    # 사용자 선택
    choice = ask_user_preprocessing_choice()
    
    # 열려있는 비교 창 닫기
    plt.close('all')
    
    if choice == 1:
        print("  [적용] equalizeHist 전처리를 적용합니다.")
        return before_hist, after_hist
    elif choice == 2:
        print("  [적용] CLAHE 전처리를 적용합니다.")
        return before_clahe, after_clahe
    else:
        print("  [적용] 원본 이미지를 그대로 사용합니다.")
        return img_before, img_after
