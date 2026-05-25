import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def find_top_rectangles(image, n=2, direction="horizontal"):
    """이미지를 좌우 또는 상하로 나누어 패드만 정확하게 검출 (8점 검증 및 수축 보정 추가)"""
    h, w = image.shape[:2]
    
    def refine_roi_by_pixels(img_half, rect_pts):
        x, y, bw, bh = cv2.boundingRect(rect_pts)
        if bw < 50 or bh < 50: return rect_pts
        roi_gray = cv2.cvtColor(img_half[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY) if len(img_half.shape)==3 else img_half[y:y+bh, x:x+bw]
        col_means = np.mean(roi_gray, axis=0); row_means = np.mean(roi_gray, axis=1)
        target_mean = np.mean(col_means[len(col_means)//4 : 3*len(col_means)//4])
        new_x, new_w_end, new_y, new_h_end = 0, len(col_means)-1, 0, len(row_means)-1
        for i in range(len(col_means)):
            if abs(col_means[i] - target_mean) < 25: new_x = i; break
        for i in range(len(col_means)-1, -1, -1):
            if abs(col_means[i] - target_mean) < 25: new_w_end = i; break
        for i in range(len(row_means)):
            if abs(row_means[i] - target_mean) < 25: new_y = i; break
        for i in range(len(row_means)-1, -1, -1):
            if abs(row_means[i] - target_mean) < 25: new_h_end = i; break
        return np.array([[[x+new_x, y+new_y]], [[x+new_w_end, y+new_y]], [[x+new_w_end, y+new_h_end]], [[x+new_x, y+new_h_end]]], dtype=np.int32)

    def find_single_pad(img_half):
        gray = cv2.cvtColor(img_half, cv2.COLOR_BGR2GRAY) if len(img_half.shape)==3 else img_half
        h_half, w_half = gray.shape[:2]
        
        # 1차 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu 이진화 적용
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 배경/패드 색상 반전 여부 자동 판단 개선
        total_mean = np.mean(thresh)
        center_mean = np.mean(thresh[h_half//4 : 3*h_half//4, w_half//4 : 3*w_half//4])
        if center_mean < total_mean:
            thresh = cv2.bitwise_not(thresh)
            
        # 노이즈를 닫고 연결하기 위한 모폴로지 닫기/열기 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 패드가 작게 생성되거나 짤려도 잡히도록 최소 면적 필터 완화 (4% -> 2%)
            if area < (h_half * w_half * 0.02): 
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 외곽 테두리를 패드로 잡지 않기 위한 한계치 완화 (96% -> 98%)
            if bw > w_half * 0.98 or bh > h_half * 0.98: 
                continue
                
            # 사각형도(Rectangularity) 필터 완화 (0.75 -> 0.55): 모서리가 둥글거나 다각형 형태도 잡음
            rectangularity = area / (bw * bh) if (bw > 0 and bh > 0) else 0.0
            if rectangularity > 0.55:
                # 중심 근처에 있을수록 가중 점수 부여
                score = area * (1.0 - abs((x + bw/2) - w_half/2) / w_half)
                candidates.append((score, cnt))
                
        # [백업 대비책] 만약 이진화 및 모폴로지로 패드가 완전히 분리가 안 되었을 때 Canny 에지 검출로 재시도
        if not candidates:
            edges = cv2.Canny(gray, 30, 100)
            edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
            contours_backup, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_backup:
                area = cv2.contourArea(cnt)
                if area < (h_half * w_half * 0.02): 
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw > w_half * 0.98 or bh > h_half * 0.98: 
                    continue
                rectangularity = area / (bw * bh) if (bw > 0 and bh > 0) else 0.0
                if rectangularity > 0.55:
                    score = area * (1.0 - abs((x + bw/2) - w_half/2) / w_half)
                    candidates.append((score, cnt))
                    
        if not candidates: 
            return None
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        x, y, bw, bh = cv2.boundingRect(candidates[0][1])
        return refine_roi_by_pixels(img_half, np.array([[[x, y]], [[x+bw, y]], [[x+bw, y+bh]], [[x, y+bh]]], dtype=np.int32))

    results = []
    if direction == "vertical":
        mid_y = h // 2
        # 위쪽 패드 검출
        t = find_single_pad(image[:mid_y, :])
        if t is not None: results.append(t)
        # 아래쪽 패드 검출 및 y좌표 보정
        b = find_single_pad(image[mid_y:, :])
        if b is not None:
            b_shifted = b.copy(); b_shifted[:, 0, 1] += mid_y; results.append(b_shifted)
    else:
        mid_x = w // 2
        # 왼쪽 패드 검출
        l = find_single_pad(image[:, :mid_x])
        if l is not None: results.append(l)
        # 오른쪽 패드 검출 및 x좌표 보정
        r = find_single_pad(image[:, mid_x:])
        if r is not None:
            r_shifted = r.copy(); r_shifted[:, 0, 0] += mid_x; results.append(r_shifted)
            
    return results

def calculate_sub_rectangle(rect, length_percentage):
    if rect is None: return None
    x, y, w, h = cv2.boundingRect(rect)
    cx, cy, s = x + w/2, y + h/2, length_percentage / 100.0
    nw, nh = w * s, h * s
    nx, ny = cx - nw/2, cy - nh/2
    return np.array([[[int(nx), int(ny)]], [[int(nx+nw), int(ny)]], [[int(nx+nw), int(ny+nh)]], [[int(nx), int(ny+nh)]]], dtype=np.int32)

def is_mask_in_roi(m_dict, roi_rect):
    mask = m_dict['segmentation']; coords = np.argwhere(mask)
    if coords.size == 0: return False
    x, y, w, h = cv2.boundingRect(roi_rect); cy, cx = np.mean(coords, axis=0)
    return (y <= cy <= y+h) and (x <= cx <= x+w)

def check_guard_zone(marks, sub_rect):
    if sub_rect is None or not marks: return True
    x, y, w, h = cv2.boundingRect(sub_rect); x1, y1, x2, y2 = x, y, x + w, y + h
    for m in marks:
        c = np.argwhere(m['segmentation'])
        if c.size == 0: continue
        # 마크의 중심점이 가드존 내부에 있는지 판단 (너무 엄격한 1픽셀 아웃을 방지)
        cy, cx = np.mean(c, axis=0)
        if not (y1 <= cy <= y2 and x1 <= cx <= x2):
            return False
    return True

def show_manual_roi_guide_plot(image):
    """수동 ROI 지정 방법을 사용자에게 한글 가이드 다이얼로그로 먼저 보여주는 예시 창"""
    import matplotlib.pyplot as plt
    import koreanize_matplotlib
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_title("[가이드] 수동 ROI 패드 꼭짓점 지정 예시\n(반드시 그림의 1 -> 2 -> 3 -> 4 순서대로 꼭짓점 4곳을 클릭해 주세요!)", 
                 fontsize=13, fontweight='bold', color='navy')
    
    # 원본 이미지 출력
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 예시 사각형 좌표 시각화 (이미지 중심의 15% 크기 수준 가상 예시)
    h, w = image.shape[:2]
    ex_x, ex_y, ex_w, ex_h = int(w * 0.15), int(h * 0.25), int(w * 0.20), int(h * 0.45)
    pts = [
        (ex_x, ex_y),          # 1: 좌상
        (ex_x + ex_w, ex_y),   # 2: 우상
        (ex_x + ex_w, ex_y + ex_h), # 3: 우하
        (ex_x, ex_y + ex_h)    # 4: 좌하
    ]
    
    # 예시 다각형 박스 덧그리기
    import matplotlib.patches as patches
    rect_patch = patches.Rectangle((ex_x, ex_y), ex_w, ex_h, linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
    ax.add_patch(rect_patch)
    
    # 순서 숫자 1, 2, 3, 4 마커 드로잉
    for idx, (px, py) in enumerate(pts):
        ax.plot(px, py, 'ro', markersize=12)
        ax.text(px, py - 18, f"클릭 {idx+1}", color='yellow', fontsize=10, fontweight='bold', 
                ha='center', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
        
    ax.text(w // 2, h - 25, "💡 예시를 확인하신 후, 이 창을 닫으면 실제 마우스 클릭 창이 활성화됩니다.", 
            color='darkred', fontsize=11, fontweight='bold', ha='center', bbox=dict(facecolor='yellow', alpha=0.9, pad=4))
            
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def select_multiple_rectangles_manually(image, n=2):
    # 수동 클릭 창을 구동하기 전에 가이드 예시 플롯 노출
    show_manual_roi_guide_plot(image)
    
    all_rects = []
    for i in range(n):
        clone = image.copy(); pts = []
        def cb(e, x, y, f, p):
            if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((x, y)); cv2.circle(clone, (x, y), 5, (0,0,255), -1)
                if len(pts) > 1: cv2.line(clone, pts[-2], pts[-1], (255,0,0), 2)
                if len(pts) == 4: cv2.line(clone, pts[3], pts[0], (255,0,0), 2); cv2.imshow("Manual ROI", clone)
        cv2.imshow("Manual ROI", clone); cv2.setMouseCallback("Manual ROI", cb)
        while True:
            k = cv2.waitKey(1) & 0xFF
            if len(pts) == 4 or k == ord('q') or k == 13: break
        cv2.destroyAllWindows()
        if len(pts) == 4: all_rects.append(np.array(pts).reshape((-1, 1, 2)).astype(np.int32))
    return all_rects

def collect_interactive_prompts(image):
    clone = image.copy(); p, l = [], []
    def cb(e, x, y, f, pa):
        if e == cv2.EVENT_LBUTTONDOWN: p.append((x,y)); l.append(1); cv2.circle(clone, (x,y), 5, (0,0,255),-1)
        elif e == cv2.EVENT_RBUTTONDOWN: p.append((x,y)); l.append(0); cv2.circle(clone, (x,y), 5, (255,0,0),-1)
        cv2.imshow("Prompts", clone)
    cv2.imshow("Prompts", clone); cv2.setMouseCallback("Prompts", cb)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 13: break
    cv2.destroyAllWindows(); return p, l

def save_prompts(p, l, f="prompts.json"):
    with open(f, 'w') as fo: json.dump({"points": p, "labels": l}, fo)
def load_prompts(f="prompts.json"):
    if not os.path.exists(f): return None, None
    with open(f, 'r') as fo: d = json.load(fo); return d["points"], d["labels"]

def visualize_dual_results(img_b, img_a, res_list, config_info=None):
    """판정 결과 텍스트를 ROI 박스 위로 더 높게 조정"""
    import koreanize_matplotlib
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 3, 1); plt.title("Contact Before"); plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("ROI & Guard Zone")
    img_roi = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    for r in res_list:
        cv2.polylines(img_roi, [r['rect']], True, (0,255,0), 2); cv2.polylines(img_roi, [r['sub_rect']], True, (255,255,0), 2)
    plt.imshow(img_roi); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Final Results")
    res_img = img_a.copy()
    for r in res_list:
        for m in r['marks']:
            mask = m['segmentation'].astype(np.uint8)
            res_img[mask > 0] = [0, 0, 255]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 1: continue
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(res_img, (int(cx), int(cy)), int(radius) + 5, (0, 0, 255), 1)
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    for r in res_list:
        cv2.polylines(res_img_rgb, [r['rect']], True, (0,255,0), 2); cv2.polylines(res_img_rgb, [r['sub_rect']], True, (255,255,0), 2)
    plt.imshow(res_img_rgb)
    
    # [핵심 수정] 박스를 침범하지 않도록 y좌표 오프셋을 -15에서 -50으로 대폭 올림
    for i, r in enumerate(res_list):
        x, y, w, h = cv2.boundingRect(r['rect'])
        c, s = ('lime', 'PASS') if r['is_pass'] else ('red', 'FAIL')
        plt.text(x + w//2, y - 50, f"ROI {i+1}: {s}", color=c, fontsize=15, fontweight='bold', ha='center', bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 3})
    
    if config_info:
        info_text = " | ".join([f"{k}: {v}" for k, v in config_info.items()])
        plt.figtext(0.5, 0.02, f"[설정 파라미터 현황] {info_text}", ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    plt.axis('off'); plt.tight_layout(); plt.show()
