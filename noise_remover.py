import cv2
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import json
import os

def collect_background_color(image):
    """사용자가 마우스로 클릭한 지점의 색상을 배경색으로 추출"""
    clone = image.copy()
    bg_color = [None] 

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            color = image[y, x]
            bg_color[0] = color.tolist() # BGR
            print(f"  [확인] 선택된 배경색 (BGR): {bg_color[0]}")
            cv2.circle(clone, (x, y), 5, (0, 0, 255), 2)
            cv2.putText(clone, f"Selected: {bg_color[0]}", (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Click Background Color", clone)

    win_name = "Click Background Color"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)
    cv2.imshow(win_name, clone)
    
    print("\n  [안내] 이미지에서 배경(금색 부분)의 깨끗한 지점을 클릭해 주세요.")
    print("  선택 완료 후, 이미지 창에서 Enter 또는 'q'를 눌러 종료하세요.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 13: 
            break
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(win_name)
    return bg_color[0]

def save_bg_color(color, filename="bg_color.json"):
    """추출한 배경색 저장"""
    try:
        with open(filename, 'w') as f:
            json.dump({"bg_color": color}, f)
        print(f"  [정보] 배경색 정보가 {filename}에 저장되었습니다.")
    except Exception as e:
        print(f"  [오류] 배경색 저장 실패: {e}")

def load_bg_color(filename="bg_color.json"):
    """저장된 배경색 로드"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return data.get("bg_color")
        except:
            return None
    return None

def remove_small_noise(image, threshold_area=15, background_color=None, sensitivity=40):
    """
    배경색과의 차이를 이용하여 노이즈를 탐지하고 제거함.
    sensitivity: 배경색보다 얼마나 더 어두워야 노이즈로 간주할 것인가 (기본값 40)
    """
    if image is None:
        return None, 0
    
    # 배경색이 없으면 자동 계산
    if background_color is None:
        background_color = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8).tolist()
    
    bg_color_np = np.array(background_color, dtype=np.uint8)
    
    # 1. 배경색과의 밝기 차이 계산 (배경보다 어두운 부분 탐지)
    # 이미지의 각 픽셀과 배경색의 차이를 구함
    diff = cv2.subtract(bg_color_np, image) # bg - image (어두울수록 큰 값)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 2. 임계값(sensitivity) 이상으로 어두운 영역을 노이즈 후보로 선정
    _, thresh = cv2.threshold(diff_gray, sensitivity, 255, cv2.THRESH_BINARY)
    
    # 3. 모폴로지 연산으로 작은 조각 병합
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. 레이블링 및 소형 객체 제거
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    
    result = image.copy()
    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= threshold_area:
            result[labels == i] = bg_color_np
            count += 1
            
    return result, count

def visualize_dual_noise_removal(before_orig, before_proc, after_orig, after_proc, count_b, count_a):
    """Before/After 모드 전후 비교 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Before
    axes[0, 0].set_title("Before (원본)", fontsize=12)
    axes[0, 0].imshow(cv2.cvtColor(before_orig, cv2.COLOR_BGR2RGB))
    axes[0, 0].axis('off')
    
    axes[0, 1].set_title(f"Before (제거됨: {count_b}개)", fontsize=12)
    axes[0, 1].imshow(cv2.cvtColor(before_proc, cv2.COLOR_BGR2RGB))
    axes[0, 1].axis('off')
    
    # After
    axes[1, 0].set_title("After (원본)", fontsize=12)
    axes[1, 0].imshow(cv2.cvtColor(after_orig, cv2.COLOR_BGR2RGB))
    axes[1, 0].axis('off')
    
    axes[1, 1].set_title(f"After (제거됨: {count_a}개)", fontsize=12)
    axes[1, 1].imshow(cv2.cvtColor(after_proc, cv2.COLOR_BGR2RGB))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)

def ask_user_noise_removal(img_before, img_after):
    """사용자에게 노이즈 제거 적용 여부를 확인받음"""
    print("\n" + "-"*40)
    print("[소형 노이즈 제거 설정]")
    print("-"*40)
    
    bg_color = load_bg_color()
    if bg_color:
        ans = input(f"  > 저장된 배경색 {bg_color}을(를) 사용할까요? (y/n, 기본값 y): ").strip().lower()
        if ans == 'n':
            bg_color = collect_background_color(img_after)
            if bg_color: save_bg_color(bg_color)
    else:
        bg_color = collect_background_color(img_after)
        if bg_color: save_bg_color(bg_color)

    try:
        threshold = int(input("\n  > 제거할 최대 면적 (기본값 15): ") or "15")
        sensitivity = int(input("  > 탐지 민감도 (작을수록 더 많이 탐지, 기본값 40): ") or "40")
    except ValueError:
        threshold = 15
        sensitivity = 40
        
    print(f"  [정보] 노이즈 제거 처리 중...")
    
    processed_before, count_b = remove_small_noise(img_before, threshold, bg_color, sensitivity)
    processed_after, count_a = remove_small_noise(img_after, threshold, bg_color, sensitivity)
    
    # Before/After 동시 시각화
    visualize_dual_noise_removal(img_before, processed_before, img_after, processed_after, count_b, count_a)
    
    choice = input("\n  > 이 노이즈 제거 결과를 적용할까요? (y/n, 기본값 y): ").strip().lower()
    plt.close('all')
    
    if choice == 'n':
        print("  [알림] 노이즈 제거를 적용하지 않습니다.")
        return img_before, img_after
    else:
        print(f"  [적용] 노이즈 제거가 완료되었습니다. (B: {count_b}, A: {count_a})")
        return processed_before, processed_after
