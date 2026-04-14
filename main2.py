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
    select_multiple_rectangles_manually
)
import os
import time

def create_dummy_data_dual():
    """테스트를 위한 듀얼 사각형 더미 데이터 생성 (ROI 2개 대응)"""
    # 기본 배경
    base = np.random.randint(50, 70, (600, 900), dtype=np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    
    # 2개의 사각형 영역 정의 (거의 비슷한 크기)
    # ROI 1 (왼쪽)
    cv2.rectangle(base, (80, 150), (380, 450), (200, 200, 200), -1)
    cv2.rectangle(base, (90, 160), (370, 440), (100, 100, 100), -1)
    
    # ROI 2 (오른쪽)
    cv2.rectangle(base, (500, 150), (800, 450), (200, 200, 200), -1)
    cv2.rectangle(base, (510, 160), (790, 440), (100, 100, 100), -1)
    
    img_before = base.copy()
    # 기존 마크 시뮬레이션
    cv2.circle(img_before, (200, 300), 12, (40, 40, 40), -1)
    cv2.circle(img_before, (650, 250), 10, (30, 30, 30), -1)
    
    # Contact 후 (정렬 및 새 마크)
    img_after = img_before.copy()
    
    # Case 1: ROI 1 (가드 존 내부 - PASS 예상)
    cv2.circle(img_after, (230, 305), 10, (220, 220, 220), -1)
    
    # Case 2: ROI 2 (가드 존 이탈 - FAIL 예상)
    # 가드 존(80%) 기준 끝자락에 표시
    cv2.circle(img_after, (780, 430), 12, (230, 230, 230), -1)
    
    # Case 3: ROI 외부의 변화 (무시되어야 함)
    cv2.rectangle(img_after, (420, 50), (460, 90), (255, 255, 255), -1)
    
    return img_before, img_after

def main():
    print("\n" + "="*50)
    print("Probe Card Dual ROI Contact Mark Analysis (main2.py)")
    print("="*50)
    
    # 1. 데이터 로드
    image_before_path = "before.jpg"
    image_after_path = "after.jpg"
    
    if os.path.exists(image_before_path) and os.path.exists(image_after_path):
        print(f"[정보] 실제 이미지 로드: {image_before_path}, {image_after_path}")
        img_before = cv2.imread(image_before_path)
        img_after = cv2.imread(image_after_path)
    else:
        print("[정보] 테스트용 듀얼 더미 데이터를 생성합니다.")
        img_before, img_after = create_dummy_data_dual()
        
    # 2. 이미지 정합
    print("\n[1/5] 이미지 정합(Alignment) 중...")
    aligner = ImageAligner()
    img_after_aligned, _ = aligner.align(img_before, img_after)
    
    # 3. 분석 파라미터 입력
    print("\n[2/5] 분석 파라미터 설정 (Enter 입력 시 기본값 사용)")
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
    
    # 4. SAM2 기반 마크 검출
    print("\n[3/5] SAM2 마크 검출 및 분석 중...")
    detector = MarkDetector()
    if detector.mask_generator is None:
        print("[에러] SAM2 체크포인트를 찾을 수 없습니다.")
        return

    start_time = time.time()
    masks_before = detector.get_masks(img_before)
    analyzer = ChangeAnalyzer(diff_threshold=diff_th, min_area=min_a, overlap_threshold=overlap_th)
    
    # 전체 이미지에서 신규 마크 후보 추출
    global_new_marks = analyzer.find_new_marks_refined(img_before, img_after_aligned, masks_before, detector)
    print(f"  > 분석 소요 시간: {time.time() - start_time:.2f}초")
    
    # 5. 듀얼 ROI 검출 및 영역별 판정
    print("\n[4/5] 사각형 영역(ROI) 검출 및 독립 판정...")
    rois = find_top_rectangles(img_after_aligned, n=2)
    
    # ROI 확인 및 수동 선택 전환
    if not rois or len(rois) < 2:
        print("  [!] 2개의 사각형을 자동으로 찾지 못했습니다. 수동 선택을 실행합니다.")
        rois = select_multiple_rectangles_manually(img_after_aligned, n=2)
    else:
        # 자동 검출 결과 미리보기 (matplotlib으로 렌더링 보장)
        confirm_img = img_after_aligned.copy()
        for i, r in enumerate(rois):
            cv2.polylines(confirm_img, [r], True, (0, 255, 0), 3)
            sub_r = calculate_sub_rectangle(r, guard_percentage)
            if sub_r is not None:
                cv2.polylines(confirm_img, [sub_r], True, (255, 255, 0), 2)
            x, y, _, _ = cv2.boundingRect(r)
            cv2.putText(confirm_img, f"ROI {i+1}", (x, max(y-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(cv2.cvtColor(confirm_img, cv2.COLOR_BGR2RGB))
        ax.set_title("[자동 검출된 ROI 미리보기]  창을 닫으면 터미널에서 확인 입력", fontsize=13)
        ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)  # 창이 그려지도록 잠시 대기
        
        print("-" * 30)
        ans = input("검출된 ROI 2개가 올바른가요? (y/n): ").strip().lower()
        plt.close()
        
        if ans != 'y':
            rois = select_multiple_rectangles_manually(img_after_aligned, n=2)

    if not rois or len(rois) < 2:
        print("[에러] ROI가 정상적으로 설정되지 않았습니다.")
        return

    # ROI별 독립 판정 로직
    roi_results = []
    for i, rect in enumerate(rois):
        # ROI 내부 가드 존 계산
        sub_rect = calculate_sub_rectangle(rect, guard_percentage)
        
        # 1. 독립 필터링: 해당 ROI 영역 내부에 있는 마크만 추출 (외부 마크 철저 무시)
        marks_in_roi = [m for m in global_new_marks if is_mask_in_roi(m, rect)]
        
        # 2. 독립 판정: 가드 존 이탈 여부 확인
        is_pass = check_guard_zone(marks_in_roi, sub_rect)
        
        roi_results.append({
            'rect': rect,
            'sub_rect': sub_rect,
            'is_pass': is_pass,
            'marks': marks_in_roi
        })
        
        status = "PASS" if is_pass else "FAIL"
        print(f"  > ROI {i+1} 결과: {status} (내부 검출 마크: {len(marks_in_roi)}개)")

    # 6. 최종 시각화
    print("\n[5/5] 결과 시각화 중...")
    visualize_dual_results(img_before, img_after_aligned, roi_results)

if __name__ == "__main__":
    main()
