"""
가드 존 내 마크 직접 검출 모듈

차분(difference) 비교 없이 단일 이미지의 가드 존 영역 안에서
마크(어두운 점)를 직접 찾는 간단한 검출 로직.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from dual_roi_utils import calculate_sub_rectangle


def detect_marks_in_guard_zone(image, roi_rect, guard_percentage, detector, 
                                min_area=20, darkness_threshold=120):
    """가드 존 영역 내에서 마크를 직접 검출함.
    
    Args:
        image: BGR 이미지
        roi_rect: ROI 사각형 좌표 (np.array, shape (4,1,2))
        guard_percentage: 가드 존 백분율 (%)
        detector: MarkDetector 인스턴스
        min_area: 최소 마크 면적
        darkness_threshold: 마크로 판단할 평균 밝기 상한값 (이 값보다 어두우면 마크)
        
    Returns:
        list: 검출된 마크 딕셔너리 리스트 (좌표는 원본 이미지 기준)
    """
    # 1. 가드 존 계산
    sub_rect = calculate_sub_rectangle(roi_rect, guard_percentage)
    if sub_rect is None:
        return []
    
    # 2. 가드 존 바운딩 박스로 크롭 영역 결정
    x, y, w, h = cv2.boundingRect(sub_rect)
    
    # 이미지 범위 클리핑
    img_h, img_w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return []
    
    # 3. 가드 존 영역 크롭
    cropped = image[y1:y2, x1:x2].copy()
    
    # 4. SAM2 자동 세그멘테이션으로 마크 후보 검출
    masks = detector.get_masks(cropped)
    if not masks:
        return []
    
    # 5. 어두운 마크만 필터링 + 최소 면적 필터
    filtered = filter_dark_marks(masks, cropped, darkness_threshold, min_area)
    
    # 6. 크롭 좌표를 원본 좌표로 변환
    result_marks = []
    for mask_dict in filtered:
        # 원본 크기의 빈 마스크 생성
        full_mask = np.zeros((img_h, img_w), dtype=bool)
        seg = mask_dict['segmentation'].astype(bool)
        full_mask[y1:y2, x1:x2] = seg
        
        result_marks.append({
            'segmentation': full_mask,
            'area': int(seg.sum()),
            'predicted_iou': mask_dict.get('predicted_iou', 0),
            'source': 'guard_zone_direct'
        })
    
    return result_marks


def filter_dark_marks(masks, image, darkness_threshold=120, min_area=20):
    """검출된 마스크 중 어두운(검은) 마크만 필터링.
    
    Args:
        masks: SAM2에서 생성된 마스크 리스트
        image: 크롭된 BGR 이미지
        darkness_threshold: 평균 밝기가 이 값 미만이면 마크로 판정
        min_area: 최소 면적
        
    Returns:
        list: 필터링된 마스크 리스트
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    filtered = []
    for mask_dict in masks:
        seg = mask_dict['segmentation'].astype(bool)
        area = int(seg.sum())
        
        # 면적 필터
        if area < min_area:
            continue
        
        # 밝기 필터: 마스크 영역의 평균 밝기가 임계값보다 어두운 것만
        mean_brightness = gray[seg].mean() if seg.any() else 255
        if mean_brightness < darkness_threshold:
            filtered.append(mask_dict)
    
    return filtered


def visualize_guard_zone_marks(image, roi_results_with_gz, title="가드 존 내 마크 검출 결과"):
    """가드 존 내 검출된 마크를 시각화.
    
    Args:
        image: BGR 이미지
        roi_results_with_gz: ROI별 가드 존 검출 결과 리스트
            각 항목: {'rect': ..., 'sub_rect': ..., 'gz_marks': [...]}
        title: 그래프 제목
    """
    fig, axes = plt.subplots(1, len(roi_results_with_gz), figsize=(9 * len(roi_results_with_gz), 8))
    if len(roi_results_with_gz) == 1:
        axes = [axes]
    
    for i, res in enumerate(roi_results_with_gz):
        vis = image.copy()
        
        # 마크 오버레이 (빨간색)
        for mark in res['gz_marks']:
            m = mark['segmentation'].astype(bool)
            vis[m] = [0, 0, 255]
        
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        
        # ROI 및 가드 존 표시
        cv2.polylines(vis_rgb, [res['rect']], True, (0, 255, 0), 2)
        cv2.polylines(vis_rgb, [res['sub_rect']], True, (255, 255, 0), 2)
        
        axes[i].set_title(f"ROI {i+1} - 마크 {len(res['gz_marks'])}개", fontsize=14)
        axes[i].imshow(vis_rgb)
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show(block=False)
