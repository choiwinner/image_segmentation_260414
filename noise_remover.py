import cv2
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import json
import os

def collect_color(image, title="Select Color", instruction="Click a point"):
    """사용자가 마우스로 클릭한 지점의 색상을 추출 (공통 함수)"""
    clone = image.copy()
    selected_color = [None] 

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            color = image[y, x]
            selected_color[0] = color.tolist() # BGR
            print(f"  [확인] 선택된 색상 (BGR): {selected_color[0]}")
            cv2.circle(clone, (x, y), 5, (0, 0, 255), 2)
            cv2.putText(clone, f"Selected: {selected_color[0]}", (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(title, clone)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_callback)
    cv2.imshow(title, clone)
    
    print(f"\n  [안내] {instruction}")
    print("  선택 완료 후, 이미지 창에서 Enter 또는 'q'를 눌러 종료하세요.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 13: 
            break
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(title)
    return selected_color[0]

def save_colors(bg_color, mark_color, filename="noise_colors.json"):
    """추출한 색상 정보 저장"""
    try:
        with open(filename, 'w') as f:
            json.dump({"bg_color": bg_color, "mark_color": mark_color}, f)
        print(f"  [정보] 색상 정보가 {filename}에 저장되었습니다.")
    except Exception as e:
        print(f"  [오류] 색상 저장 실패: {e}")

def load_colors(filename="noise_colors.json"):
    """저장된 색상 정보 로드"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def remove_small_noise(image, threshold_area=15, bg_color=None, mark_color=None, sensitivity=40):
    """
    배경색 및 마크색과의 유사도를 이용하여 노이즈를 탐지하고 제거함.
    """
    if image is None:
        return None, 0
    
    if bg_color is None:
        bg_color = [100, 150, 180] # 기본값 (금색 계열 추정)
    
    bg_color_np = np.array(bg_color, dtype=np.uint8)
    
    # 1. 배경색과의 밝기 차이 기반 (기존 방식: 배경보다 어두운 부분)
    diff_from_bg = cv2.subtract(bg_color_np, image)
    gray_from_bg = cv2.cvtColor(diff_from_bg, cv2.COLOR_BGR2GRAY)
    _, thresh_bg = cv2.threshold(gray_from_bg, sensitivity, 255, cv2.THRESH_BINARY)

    # 2. [추가] 마크 색상과의 유사도 기반 로직
    if mark_color is not None:
        mark_color_np = np.array(mark_color, dtype=np.uint8)
        # 각 픽셀과 마크 색상 간의 거리(유사도) 계산
        # L1 거리를 빠르게 계산하여 임계값 처리
        diff_mark = cv2.absdiff(image, mark_color_np)
        dist_mark = cv2.cvtColor(diff_mark, cv2.COLOR_BGR2GRAY)
        
        # 마크 색상과 충분히 가까운(유사한) 영역만 필터링 (임계값 50 정도 사용)
        _, thresh_mark = cv2.threshold(dist_mark, 50, 255, cv2.THRESH_BINARY_INV)
        
        # 배경색과 다르고 + 마크색과 유사한 영역만 최종 후보로 선정
        final_thresh = cv2.bitwise_and(thresh_bg, thresh_mark)
    else:
        final_thresh = thresh_bg
    
    # 3. 모폴로지 연산으로 작은 조각 병합
    kernel = np.ones((3,3), np.uint8)
    final_thresh = cv2.morphologyEx(final_thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. 레이블링 및 소형 객체 제거
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_thresh)
    
    result = image.copy()
    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= threshold_area:
            result[labels == i] = bg_color_np
            count += 1
            
    return result, count

def visualize_dual_noise_removal(before_orig, before_proc, after_orig, after_proc, count_b, count_a, rois=None):
    """Before/After 모드 전후 비교 시각화 (ROI 영역만 강조)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 시각화용 이미지 생성
    def get_roi_focused_image(orig, proc, rois):
        if rois is None:
            return proc
        mask = np.zeros(orig.shape[:2], dtype=np.uint8)
        for rect in rois:
            cv2.drawContours(mask, [rect], -1, 255, -1)
        
        # ROI 외부는 어둡게 처리 (원본의 30% 밝기)
        dimmed = cv2.addWeighted(proc, 0.3, np.zeros_like(proc), 0.7, 0)
        # ROI 내부는 결과 이미지 사용
        focused = np.where(mask[:,:,None] == 255, proc, dimmed)
        return focused

    vis_before_orig = before_orig.copy()
    vis_after_orig = after_orig.copy()
    
    if rois is not None:
        for rect in rois:
            cv2.polylines(vis_before_orig, [rect], True, (0, 255, 0), 2)
            cv2.polylines(vis_after_orig, [rect], True, (0, 255, 0), 2)
    
    focused_before = get_roi_focused_image(before_orig, before_proc, rois)
    focused_after = get_roi_focused_image(after_orig, after_proc, rois)
            
    # Before
    axes[0, 0].set_title("Before (원본 + ROI)", fontsize=12)
    axes[0, 1].set_title(f"Before (ROI 내부 마크 검출: {count_b}개)", fontsize=12)
    axes[0, 0].imshow(cv2.cvtColor(vis_before_orig, cv2.COLOR_BGR2RGB))
    axes[0, 1].imshow(cv2.cvtColor(focused_before, cv2.COLOR_BGR2RGB))
    
    # After
    axes[1, 0].set_title("After (원본 + ROI)", fontsize=12)
    axes[1, 1].set_title(f"After (ROI 내부 마크 검출: {count_a}개)", fontsize=12)
    axes[1, 0].imshow(cv2.cvtColor(vis_after_orig, cv2.COLOR_BGR2RGB))
    axes[1, 1].imshow(cv2.cvtColor(focused_after, cv2.COLOR_BGR2RGB))
    
    for ax in axes.ravel(): ax.axis('off')
    
    plt.tight_layout()
    plt.show(block=False)

def ask_user_noise_removal(img_before, img_after, rois=None):
    """사용자에게 노이즈 제거 적용 여부를 확인받으며 파라미터를 조정함"""
    print("\n" + "-"*40)
    print("[소형 노이즈 제거 및 마크 검출 피드백]")
    print("-"*40)
    
    colors = load_colors()
    bg_color = colors.get("bg_color") if colors else None
    mark_color = colors.get("mark_color") if colors else None
    
    if bg_color and mark_color:
        print(f"  [정보] 저장된 색상 로드됨 (배경: {bg_color}, 마크: {mark_color})")
        ans = input(f"  > 저장된 색상 정보를 사용할까요? (y/n, 기본값 y): ").strip().lower()
        if ans == 'n':
            bg_color = collect_color(img_after, "Background Color", "이미지에서 '금색 배경' 부분을 클릭해 주세요.")
            mark_color = collect_color(img_after, "Mark Color", "이미지에서 '검은색 마크' 부분을 클릭해 주세요.")
            if bg_color and mark_color: save_colors(bg_color, mark_color)
    else:
        bg_color = collect_color(img_after, "Background Color", "이미지에서 '금색 배경' 부분을 클릭해 주세요.")
        mark_color = collect_color(img_after, "Mark Color", "이미지에서 '검은색 마크' 부분을 클릭해 주세요.")
        if bg_color and mark_color: save_colors(bg_color, mark_color)

    threshold = 15
    sensitivity = 40
    
    while True:
        try:
            print(f"\n  [현재 설정] 면적: {threshold}, 민감도: {sensitivity}")
            u_th = input(f"  > 제거할 최대 면적 (현재 {threshold}, 변경 시 숫자 입력): ").strip()
            u_se = input(f"  > 탐지 민감도 (현재 {sensitivity}, 변경 시 숫자 입력): ").strip()
            
            if u_th: threshold = int(u_th)
            if u_se: sensitivity = int(u_se)
        except ValueError:
            print("  [!] 숫자를 입력해 주세요.")
            continue
            
        print(f"  [정보] ROI 기반 마크 검출 및 제거 처리 중...")
        
        # 실제 처리는 전체 이미지에 하되, 카운팅이나 시각화 피드백은 ROI 기반으로 가능
        # ROI 내의 마크들만 필터링하여 시각화에 반영
        processed_before, count_b = remove_small_noise(img_before, threshold, bg_color, mark_color, sensitivity)
        processed_after, count_a = remove_small_noise(img_after, threshold, bg_color, mark_color, sensitivity)
        
        # 시각화 (ROI가 있으면 표시)
        visualize_dual_noise_removal(img_before, processed_before, img_after, processed_after, count_b, count_a, rois)
        
        print("\n  > 결과가 만족스러우신가요?")
        print("    y: 적용하고 다음 단계로")
        print("    n: 적용하지 않고 건너뜀")
        print("    r: 파라미터 재설정 (다시 시도)")
        choice = input("  선택: ").strip().lower()
        
        plt.close('all')
        
        if choice == 'y':
            print(f"  [적용] 노이즈 제거 완료. (B: {count_b}, A: {count_a})")
            return processed_before, processed_after
        elif choice == 'n':
            print("  [알림] 노이즈 제거를 적용하지 않습니다.")
            return img_before, img_after
        elif choice == 'r':
            continue
        else:
            print("  [알림] y를 입력한 것으로 간주하여 적용합니다.")
            return processed_before, processed_after
