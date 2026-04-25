"""
main4.py — 이미지 전처리 + 가드 존 마크 검출 통합 파이프라인

main3.py의 기존 기능을 유지하면서 아래 기능을 추가:
  - 이미지 전처리 (equalizeHist / CLAHE) + 전후 비교 표시
  - 가드 존 내 마크 직접 검출 (차분 없이 단일 이미지 기반)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from processor import ImageAligner, MarkDetector, ChangeAnalyzer
from dual_roi_utils import (
    find_top_rectangles, 
    calculate_sub_rectangle, 
    is_mask_in_roi, 
    check_guard_zone, 
    visualize_dual_results, 
    select_multiple_rectangles_manually,
    collect_interactive_prompts,
    save_prompts,
    load_prompts
)
from image_preprocessor import preprocess_images
from guard_zone_detector import detect_marks_in_guard_zone, visualize_guard_zone_marks
import os
import time


def create_dummy_data_dual():
    """테스트를 위한 듀얼 사각형 더미 데이터 생성 (ROI 2개 대응)"""
    base = np.random.randint(50, 70, (600, 900), dtype=np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    
    cv2.rectangle(base, (80, 150), (380, 450), (200, 200, 200), -1)
    cv2.rectangle(base, (90, 160), (370, 440), (100, 100, 100), -1)
    cv2.rectangle(base, (500, 150), (800, 450), (200, 200, 200), -1)
    cv2.rectangle(base, (510, 160), (790, 440), (100, 100, 100), -1)
    
    img_before = base.copy()
    cv2.circle(img_before, (200, 300), 12, (40, 40, 40), -1)
    cv2.circle(img_before, (650, 250), 10, (30, 30, 30), -1)
    
    img_after = img_before.copy()
    cv2.circle(img_after, (230, 305), 10, (220, 220, 220), -1)
    cv2.circle(img_after, (780, 430), 12, (230, 230, 230), -1)
    
    return img_before, img_after


def main():
    print("\n" + "="*60)
    print("Probe Card Contact Mark Analysis (main4.py)")
    print("  — 이미지 전처리 + 가드 존 마크 검출 통합")
    print("="*60)
    
    # ── 1. 데이터 로드 ──
    image_before_path = "before.jpg"
    image_after_path = "after.jpg"
    
    if os.path.exists(image_before_path) and os.path.exists(image_after_path):
        print(f"\n[정보] 실제 이미지 로드: {image_before_path}, {image_after_path}")
        img_before = cv2.imread(image_before_path)
        img_after = cv2.imread(image_after_path)
    else:
        print("\n[정보] 테스트용 듀얼 더미 데이터를 생성합니다.")
        img_before, img_after = create_dummy_data_dual()
        
    # ── 2. 이미지 정합 ──
    print("\n[1/7] 이미지 정합(Alignment) 중...")
    aligner = ImageAligner()
    img_after_aligned, _ = aligner.align(img_before, img_after)
    
    # ROI 검출을 위해 원본 정합 이미지 보관 (전처리가 ROI 검출을 방해할 수 있음)
    img_after_for_roi = img_after_aligned.copy()

    # ── 2.5 [NEW] 이미지 전처리 ──
    print("\n[2/7] 이미지 전처리 (선택 사항)...")
    print("  전처리 비교 이미지를 확인한 후 적용 여부를 선택하세요.")
    img_before, img_after_aligned = preprocess_images(img_before, img_after_aligned)
    
    # ── 3. 분석 파라미터 입력 ──
    print("\n[3/7] 분석 파라미터 설정 (Enter 입력 시 기본값 사용)")
    def get_input(prompt, default_val):
        user_val = input(f"  > {prompt} (기본값 {default_val}): ").strip()
        if not user_val:
            return default_val
        try:
            return type(default_val)(user_val)
        except ValueError:
            print(f"    [!] 입력 오류. 기본값 {default_val} 사용.")
            return default_val

    diff_th = get_input("차분 임계값 (0~255)", 30)
    min_a = get_input("최소 마크 면적", 20)
    overlap_th = get_input("겹침 허용 비율 (0~1.0)", 0.8)
    guard_percentage = get_input("가드 존 백분율 (%)", 80.0)
    
    # ── 4. ROI 검출 ──
    # ROI 검출은 전처리 전 이미지를 사용하여 안정성 확보
    print("\n[4/7] 사각형 영역(ROI) 검출...")
    rois = find_top_rectangles(img_after_for_roi, n=2)
    if not rois or len(rois) < 2:
        print("  [!] 2개의 사각형을 자동으로 찾지 못했습니다. 수동 선택을 실행합니다.")
        rois = select_multiple_rectangles_manually(img_after_for_roi, n=2)
    
    if not rois or len(rois) < 2:
        print("[에러] ROI가 정상적으로 설정되지 않았습니다.")
        return

    # ── 4.5 [NEW] 가드 존 내 마크 빠른 검출 ──
    print("\n[4.5/7] 가드 존 내 마크 빠른 검출 (선택 사항)...")
    gz_choice = input("  > 가드 존 내 마크 빠른 검출을 실행할까요? (y/n, 기본값 n): ").strip().lower()
    
    if gz_choice == 'y':
        print("  [정보] SAM2 모델 로드 중 (가드 존 검출용)...")
        gz_detector = MarkDetector()
        if gz_detector.predictor is None:
            print("  [경고] SAM2 로드 실패. 가드 존 빠른 검출을 건너뜁니다.")
        else:
            gz_results = []
            for i, rect in enumerate(rois):
                sub_rect = calculate_sub_rectangle(rect, guard_percentage)
                gz_marks = detect_marks_in_guard_zone(
                    img_after_aligned, rect, guard_percentage, gz_detector,
                    min_area=min_a
                )
                gz_results.append({
                    'rect': rect,
                    'sub_rect': sub_rect,
                    'gz_marks': gz_marks
                })
                print(f"    ROI {i+1}: 가드 존 내 검출된 마크 {len(gz_marks)}개")
            
            # 결과 시각화
            visualize_guard_zone_marks(img_after_aligned, gz_results)
            
            gz_continue = input("\n  > 차분 기반 상세 분석도 계속 진행할까요? (y/n, 기본값 y): ").strip().lower()
            plt.close('all')
            if gz_continue == 'n':
                print("\n[완료] 가드 존 빠른 검출 결과로 종료합니다.")
                return

    # ── 5. 인터랙티브 프롬프트 설정 ──
    print("\n[5/7] 인터랙티브 세그멘테이션 설정 (배경/마크 학습)...")
    prompt_file = "prompts.json"
    points, labels = None, None
    
    if os.path.exists(prompt_file):
        ans = input(f"  > 기존에 저장된 프롬프트({prompt_file})를 로드할까요? (y/n): ").strip().lower()
        if ans == 'y':
            points, labels = load_prompts(prompt_file)
            print(f"    [정보] {len(points)}개의 점을 불러왔습니다.")
            
    if points is None:
        print("  > 배경(금색)과 마크(검은색)를 클릭하여 구분해 주세요.")
        points, labels = collect_interactive_prompts(img_after_aligned)
        if points:
            ans = input("  > 이 프롬프트를 저장할까요? (y/n): ").strip().lower()
            if ans == 'y':
                save_prompts(points, labels, prompt_file)

    if not points:
        print("[경고] 프롬프트가 선택되지 않았습니다. 기본 Zero-shot 모드로 진행합니다.")
        points, labels = [], []

    # ── 6. SAM2 기반 마크 검출 및 분석 ──
    print("\n[6/7] SAM2 마크 검출 및 분석 중...")
    detector = MarkDetector()
    if detector.predictor is None:
        print("[에러] SAM2 체크포인트를 찾을 수 없습니다.")
        return

    start_time = time.time()
    
    masks_before = detector.get_masks(img_before)
    analyzer = ChangeAnalyzer(diff_threshold=diff_th, min_area=min_a, overlap_threshold=overlap_th)
    
    # 차분 후보지 추출
    candidates, thresh_map = analyzer.get_difference_candidates(img_before, img_after_aligned)
    
    refined_new_marks = []
    for cand_pt in candidates:
        combined_points = points + [cand_pt]
        combined_labels = labels + [1]
        
        masks = detector.get_masks_from_points(img_after_aligned, combined_points, combined_labels)
        if not masks:
            continue
            
        n_mask = masks[0]
        n_seg = n_mask['segmentation']
        
        # 차분 맵과 교집합
        n_seg_refined = np.logical_and(n_seg, thresh_map > 0)
        refined_area = int(n_seg_refined.sum())
        
        if refined_area < min_a:
            continue
            
        n_mask = dict(n_mask)
        n_mask['segmentation'] = n_seg_refined
        n_mask['area'] = refined_area
        refined_new_marks.append(n_mask)

    print(f"  > 분석 소요 시간: {time.time() - start_time:.2f}초")
    
    # ── ROI별 독립 판정 ──
    roi_results = []
    for i, rect in enumerate(rois):
        sub_rect = calculate_sub_rectangle(rect, guard_percentage)
        marks_in_roi = [m for m in refined_new_marks if is_mask_in_roi(m, rect)]
        is_pass = check_guard_zone(marks_in_roi, sub_rect)
        
        roi_results.append({
            'rect': rect,
            'sub_rect': sub_rect,
            'is_pass': is_pass,
            'marks': marks_in_roi
        })
        
        status = "PASS" if is_pass else "FAIL"
        print(f"  > ROI {i+1} 결과: {status} (내부 검출 마크: {len(marks_in_roi)}개)")

    # ── 7. 최종 시각화 ──
    print("\n[7/7] 결과 시각화 중...")
    visualize_dual_results(img_before, img_after_aligned, roi_results)


if __name__ == "__main__":
    main()
