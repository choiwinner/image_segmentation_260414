import cv2
import numpy as np
import random

def create_large_complex_mark(img, center, base_size):
    """흐리지 않고 선명하지만, 인위적인 테두리가 없는 프로브 마크 생성"""
    cx, cy = center
    bw, bh = base_size
    gray_val = random.randint(40, 90)
    color = (gray_val, gray_val + random.randint(-4, 4), gray_val + random.randint(-3, 3))
    
    num_pts = random.randint(10, 16)
    pts = []
    for i in range(num_pts):
        px = cx + random.randint(-bw//2, bw//2)
        py = cy + random.randint(-bh//2, bh//2)
        pts.append([px, py])
    
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    temp_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(temp_mask, [pts], 255)
    
    mark_patch = np.full((img.shape[0], img.shape[1], 3), color, dtype=np.uint8)
    inner_noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    mark_patch = np.clip(mark_patch.astype(np.int16) + inner_noise, 0, 255).astype(np.uint8)
    
    mask_inv = cv2.bitwise_not(temp_mask)
    img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    img_fg = cv2.bitwise_and(mark_patch, mark_patch, mask=temp_mask)
    img[:] = cv2.add(img_bg, img_fg)

def add_noise_in_roi(img, cx, cy, w, h, count=5):
    """특정 ROI 내부에만 미세 노이즈 추가"""
    for _ in range(count):
        nx = random.randint(cx - w//2 + 20, cx + w//2 - 20)
        ny = random.randint(cy - h//2 + 20, cy + h//2 - 20)
        sz = random.randint(1, 3) # 아주 작은 노이즈
        c = random.randint(40, 110)
        cv2.circle(img, (nx, ny), sz, (c, c, c), -1)

def generate():
    width, height = 1000, 800
    img = np.full((height, width, 3), [220, 230, 240], dtype=np.uint8)
    
    # 사각형 패드 2개
    roi_w, roi_h = 320, 450
    rois = [(250, 400), (750, 400)]
    
    for cx, cy in rois:
        # 대비를 높인 패드 색상
        cv2.rectangle(img, (cx-roi_w//2, cy-roi_h//2), (cx+roi_w//2, cy+roi_h//2), (140, 150, 160), -1)
        # 선명한 테두리 (두께 2)
        cv2.rectangle(img, (cx-roi_w//2, cy-roi_h//2), (cx+roi_w//2, cy+roi_h//2), (60, 60, 60), 2)

    img_before = img.copy()
    # ROI 내부에 노이즈 5개씩 추가 (Before)
    for cx, cy in rois:
        add_noise_in_roi(img_before, cx, cy, roi_w, roi_h, count=5)
    
    # Before 마크
    create_large_complex_mark(img_before, (250, 300), (80, 140))
    create_large_complex_mark(img_before, (750, 300), (70, 120))
    
    img_after = img_before.copy()
    # After 마크 추가
    create_large_complex_mark(img_after, (265, 330), (85, 110))
    create_large_complex_mark(img_after, (750-roi_w//2, 550), (100, 140))
    create_large_complex_mark(img_after, (780, 500), (90, 80))
    
    # After에 신규 노이즈 추가 (패드 내 2~3개 더 추가)
    for cx, cy in rois:
        add_noise_in_roi(img_after, cx, cy, roi_w, roi_h, count=random.randint(2, 4))
    
    cv2.imwrite("before.jpg", img_before)
    cv2.imwrite("after.jpg", img_after)
    print("[완료] ROI 내부에 4~5개의 노이즈가 포함된 테스트 이미지가 생성되었습니다.")

if __name__ == "__main__":
    generate()
