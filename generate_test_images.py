import cv2
import numpy as np
import os

def generate_test_images():
    # 1. 배경 생성
    h, w = 600, 900
    base = np.full((h, w, 3), 60, dtype=np.uint8)
    
    # 노이즈 추가
    noise = np.random.normal(0, 5, (h, w, 3)).astype(np.uint8)
    base = cv2.add(base, noise)
    
    # 2. 2개의 ROI 사각형 (Pads) 생성
    def draw_pad_with_pattern(img, rect_outer, rect_inner):
        cv2.rectangle(img, rect_outer[0], rect_outer[1], (220, 220, 220), -1)
        x1, y1 = rect_inner[0]
        x2, y2 = rect_inner[1]
        cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), -1)
        
        # Pad 내부에 얼룩 패턴 추가
        roi_h = y2 - y1
        roi_w = x2 - x1
        yy, xx = np.mgrid[0:roi_h, 0:roi_w]
        stripe_pattern = np.sin((xx + yy) / 5.0) * 10.0
        spotted_pattern = np.sin(xx / 12.0) * np.sin(yy / 12.0) * 12.0
        blotchy_pattern = np.sin(xx / 30.0) * np.cos(yy / 30.0) * 8.0
        total_pattern = stripe_pattern + spotted_pattern + blotchy_pattern
        
        roi_area = img[y1:y2, x1:x2].astype(np.int16)
        for c in range(3):
            roi_area[:, :, c] = np.clip(roi_area[:, :, c] + total_pattern, 0, 255)
        img[y1:y2, x1:x2] = roi_area.astype(np.uint8)

    draw_pad_with_pattern(base, ((100, 150), (350, 450)), ((110, 160), (340, 440)))
    draw_pad_with_pattern(base, ((550, 150), (800, 450)), ((560, 160), (790, 440)))

    
    # 3. before.jpg 생성 (기존 마크)
    img_before = base.copy()
    # ROI 1의 기존 마크
    cv2.circle(img_before, (200, 300), 12, (40, 40, 40), -1)
    # ROI 2의 기존 마크
    cv2.circle(img_before, (650, 200), 15, (30, 30, 30), -1)
    
    cv2.imwrite("before.jpg", img_before)
    print("Successfully created before.jpg")
    
    # 4. after.jpg 생성 (겹치는 새로운 마크 추가)
    img_after = img_before.copy()
    
    # [핵심] ROI 1: 기존 마크 (200, 300)와 겹치는 새로운 마크 추가
    # 기존 마크와 동일한 어두운 색상으로 설정하여 난이도 상향
    cv2.circle(img_after, (212, 308), 12, (40, 40, 40), -1)
    
    # ROI 2: 새로운 마크 추가 (가드 존 바깥쪽 - FAIL 유도)
    cv2.circle(img_after, (780, 430), 10, (40, 40, 40), -1)
    
    # ROI 외부: 무시되어야 할 노이즈/변화 (어두운 사각형으로 변경)
    cv2.rectangle(img_after, (450, 50), (470, 70), (40, 40, 40), -1)
    
    cv2.imwrite("after.jpg", img_after)
    print("Successfully created after.jpg")

if __name__ == "__main__":
    generate_test_images()
