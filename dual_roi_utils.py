import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_top_rectangles(image, n=2):
    """이미지를 좌우로 나누어 각각 가장 큰 사각형을 찾음 (사용자 제안 방식)"""
    h, w = image.shape[:2]
    mid_x = w // 2
    
    # 좌측 영역
    left_half = image[:, :mid_x]
    # 우측 영역
    right_half = image[:, mid_x:]
    
    def find_single_largest_rect(img_half):
        if len(img_half.shape) == 3:
            gray = cv2.cvtColor(img_half, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_half
        
        h_img, w_img = gray.shape[:2]
        half_area = h_img * w_img
        min_area = half_area * 0.03  # 최소: 반쪽 이미지의 3%

        def _is_full_boundary(box_pts, tol=0.08):
            """사각형이 반쪽 이미지 가장자리 3면 이상에 닿으면 전체 경계로 판단 (제외 대상)"""
            x, y, bw, bh = cv2.boundingRect(box_pts)
            touches = [
                x < w_img * tol,                    # 왼쪽 가장자리
                y < h_img * tol,                    # 위쪽 가장자리
                (x + bw) > w_img * (1 - tol),       # 오른쪽 가장자리
                (y + bh) > h_img * (1 - tol),       # 아래쪽 가장자리
            ]
            return sum(touches) >= 3  # 3면 이상 닿으면 전체 경계 사각형

        def _extract_candidates(binary_map):
            """이진 맵에서 유효한 사각형 후보를 면적 내림차순으로 수집"""
            kernel = np.ones((7, 7), np.uint8)
            closed = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                rect = cv2.minAreaRect(cnt)
                (_, (rw, rh), _) = rect
                rect_area = rw * rh
                rectangularity = area / rect_area if rect_area > 0 else 0
                if rectangularity > 0.55:
                    box = cv2.boxPoints(rect)
                    box_pts = np.int32(box).reshape((-1, 1, 2))
                    # 이미지 전체 가장자리에 닿는 사각형(배경 경계) 제외
                    if not _is_full_boundary(box_pts):
                        candidates.append((area, box_pts))
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates

        # --- 전략 0: 대형 블러로 노이즈 충분히 제거 후 Otsu (최우선) ---
        # 노이즈가 심한 이미지에서 균일한 gray box를 검출하는 데 가장 효과적
        large_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh_large = cv2.threshold(large_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cands = _extract_candidates(thresh_large)
        if cands:
            return cands[0][1]

        # --- 전략 1: 소형 블러 + Otsu ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cands = _extract_candidates(thresh_otsu)
        if cands:
            return cands[0][1]
        
        # --- 전략 2: Bilateral + Adaptive 이진화 ---
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh_adapt = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 51, 5)
        cands = _extract_candidates(thresh_adapt)
        if cands:
            return cands[0][1]
        
        # --- 전략 3: Canny 엣지 기반 ---
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        cands = _extract_candidates(dilated)
        if cands:
            return cands[0][1]
        
        return None

    left_rect = find_single_largest_rect(left_half)
    right_rect_half = find_single_largest_rect(right_half)
    
    results = []
    if left_rect is not None:
        results.append(left_rect)
    if right_rect_half is not None:
        # 우측 사각형은 x좌표에 mid_x를 더해줘야 함
        right_rect = right_rect_half.copy()
        for p in range(len(right_rect)):
            right_rect[p, 0, 0] += mid_x
        results.append(right_rect)
        
    return results

def calculate_sub_rectangle(rect, length_percentage):
    """주어진 사각형의 길이 대비 특정 백분율을 가진 중심 기반 사각형(가드 존) 계산"""
    if rect is None:
        return None
        
    x, y, w, h = cv2.boundingRect(rect)
    center_x, center_y = x + w/2, y + h/2
    
    scale = length_percentage / 100.0
    new_w = w * scale
    new_h = h * scale
    
    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2
    
    sub_rect = np.array([
        [[int(new_x), int(new_y)]],
        [[int(new_x + new_w), int(new_y)]],
        [[int(new_x + new_w), int(new_y + new_h)]],
        [[int(new_x), int(new_y + new_h)]]
    ], dtype=np.int32)
    
    return sub_rect

def is_mask_in_roi(mask_dict, roi_rect):
    """마스크의 중심점이 특정 ROI 사각형 내부에 있는지 확인"""
    mask = mask_dict['segmentation']
    coords = np.argwhere(mask) # [y, x]
    if coords.size == 0:
        return False
    
    # 사각 영역 바운딩 박스
    x, y, w, h = cv2.boundingRect(roi_rect)
    y_min, y_max = y, y + h
    x_min, x_max = x, x + w
    
    # 마스크 포인트들의 평균(중심점) 좌표
    center_y, center_x = np.mean(coords, axis=0)
    
    return (y_min <= center_y <= y_max) and (x_min <= center_x <= x_max)

def check_guard_zone(marks, sub_rect):
    """특정 사각형의 가드 존 이탈 여부 확인 (독립 판정용)"""
    if sub_rect is None or not marks:
        return True # 마크가 없으면 통과
        
    x, y, w, h = cv2.boundingRect(sub_rect)
    x1, y1, x2, y2 = x, y, x + w, y + h
    
    for mark in marks:
        mask = mark['segmentation']
        coords = np.argwhere(mask)
        if coords.size == 0: continue
            
        # 하나라도 가드 존을 벗어나면 False
        if np.any(coords[:, 0] < y1) or np.any(coords[:, 0] > y2) or \
           np.any(coords[:, 1] < x1) or np.any(coords[:, 1] > x2):
            return False
            
    return True

def select_multiple_rectangles_manually(image, n=2):
    """사용자가 마우스로 여러 개의 사각형을 수동으로 선택하도록 함"""
    all_rects = []
    
    for i in range(n):
        print(f"\n[{i+1}/{n}] 번째 사각형의 4개 꼭짓점을 클릭해 주세요.")
        clone = image.copy()
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append((x, y))
                    cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                    cv2.imshow(f"Select ROI {i+1}", clone)
                    if len(points) > 1:
                        cv2.line(clone, points[-2], points[-1], (255, 0, 0), 2)
                        cv2.imshow(f"Select ROI {i+1}", clone)
                    if len(points) == 4:
                        cv2.line(clone, points[3], points[0], (255, 0, 0), 2)
                        cv2.imshow(f"Select ROI {i+1}", clone)

        win_name = f"Select ROI {i+1}"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, mouse_callback)
        cv2.imshow(win_name, clone)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            # 4개 다 선택했거나, 'q' 또는 'Enter' 누르면 종료
            if len(points) == 4 or key == ord('q') or key == 13:
                break
            
            # 창이 닫혔는지 확인하는 간접적인 방법 (플랫폼별 상이할 수 있음)
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        try:
            cv2.destroyWindow(win_name)
            # 창이 완전히 닫히도록 짧게 대기
            cv2.waitKey(100)
        except:
            pass
        
        if len(points) == 4:
            all_rects.append(np.array(points).reshape((-1, 1, 2)).astype(np.int32))
        else:
            print(f"  [!] {i+1}번째 사각형이 정확히 선택되지 않았습니다.")
            cv2.destroyAllWindows()
            return None
            
    cv2.destroyAllWindows()
    return all_rects

import json
import os

def collect_interactive_prompts(image):
    """사용자가 마우스로 Positive(좌클릭) 및 Negative(우클릭) 점을 선택하도록 함"""
    clone = image.copy()
    points = []
    labels = [] # 1: Positive (Mark), 0: Negative (Background)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1) # Red for Positive
            cv2.putText(clone, "P", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Select Prompts (L:Mark, R:BG)", clone)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y))
            labels.append(0)
            cv2.circle(clone, (x, y), 5, (255, 0, 0), -1) # Blue for Negative
            cv2.putText(clone, "N", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow("Select Prompts (L:Mark, R:BG)", clone)

    win_name = "Select Prompts (L:Mark, R:BG)"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)
    cv2.imshow(win_name, clone)
    
    print("-" * 40)
    print("마크(검은색)는 **좌클릭(P)**, 배경(금색)은 **우클릭(N)** 해주세요.")
    print("선택 완료 후, **이미지 창에서** Enter 또는 'q'를 눌러 종료하세요.")
    print("-" * 40)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 13:
            break
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(win_name)
    return points, labels

def save_prompts(points, labels, filename="prompts.json"):
    """선택한 프롬프트 점과 라벨을 JSON 파일로 저장"""
    data = {
        "points": points,
        "labels": labels
    }
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"[정보] 프롬프트가 {filename}에 저장되었습니다.")

def load_prompts(filename="prompts.json"):
    """저장된 프롬프트 데이터를 로드"""
    if not os.path.exists(filename):
        return None, None
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["points"], data["labels"]

def visualize_dual_results(img_before, img_after, roi_results):
    """두 개의 영역에 대한 독립적인 판정 결과를 시각화"""
    plt.figure(figsize=(18, 6))
    
    # 1. 원본 이미지 (Before)
    plt.subplot(1, 3, 1)
    plt.title("Contact 전")
    plt.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 2. ROI 및 가드 존 표시
    plt.subplot(1, 3, 2)
    plt.title("검출된 ROI 및 가드 존")
    img_roi = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
    for i, res in enumerate(roi_results):
        cv2.polylines(img_roi, [res['rect']], True, (0, 255, 0), 2)
        cv2.polylines(img_roi, [res['sub_rect']], True, (255, 255, 0), 2)
        x, y, _, _ = cv2.boundingRect(res['rect'])
        cv2.putText(img_roi, f"ROI {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    plt.imshow(img_roi)
    plt.axis('off')
    
    # 3. 신규 마크 및 독립 판정 결과
    plt.subplot(1, 3, 3)
    plt.title("검출 결과 및 개별 판정")
    res_img = img_after.copy()
    for res in roi_results:
        for mark in res['marks']:
            m = mark['segmentation'].astype(bool)
            res_img[m] = [0, 0, 255] # Red
    
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    for res in roi_results:
        cv2.polylines(res_img_rgb, [res['rect']], True, (0, 255, 0), 2)
        cv2.polylines(res_img_rgb, [res['sub_rect']], True, (255, 255, 0), 2)
        
    plt.imshow(res_img_rgb)
    
    # 좌측 상단에 판정 결과 텍스트 표시
    for i, res in enumerate(roi_results):
        color = 'lime' if res['is_pass'] else 'red'
        status = 'PASS' if res['is_pass'] else 'FAIL'
        plt.text(10, 40 + (i*50), f"ROI {i+1}: {status}", 
                 color=color, fontsize=16, fontweight='bold', 
                 bbox={'facecolor': 'black', 'alpha': 0.6})
                 
    plt.axis('off')
    plt.tight_layout()
    plt.show()
