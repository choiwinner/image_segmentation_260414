import cv2
import numpy as np
import os

def generate_test_images():
    print("[정보] 테스트용 이미지 생성 중 (main5.py 기능 검증용)...")
    
    # 1. 배경 생성 (약간 노이즈 있는 금색/노란색 계열)
    # BGR: (Blue, Green, Red) -> Gold is roughly (50, 180, 220)
    h, w = 600, 1000
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:] = (60, 170, 210) # 밝은 금색 느낌
    
    # 미세한 질감 노이즈 추가
    noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 2. 사각형 영역 (ROI) 2개 생성
    # ROI 1: 좌측
    roi1_rect = [(100, 150), (450, 450)] # (x1, y1), (x2, y2)
    # ROI 2: 우측
    roi2_rect = [(550, 150), (900, 450)]
    
    def draw_roi(img, rect):
        # 외곽선은 약간 밝은 색
        cv2.rectangle(img, rect[0], rect[1], (180, 220, 240), -1)
        # 내부는 조금 더 진한 색
        inner = [(rect[0][0]+10, rect[0][1]+10), (rect[1][0]-10, rect[1][1]-10)]
        cv2.rectangle(img, inner[0], inner[1], (100, 150, 180), -1)

    # before 이미지 생성
    img_before = base.copy()
    draw_roi(img_before, roi1_rect)
    draw_roi(img_before, roi2_rect)
    
    # ── before의 특징점(마크) ──
    # ROI 1 내부에 기존 마크 1개
    cv2.circle(img_before, (200, 300), 12, (30, 30, 30), -1)
    # ROI 2 내부에 기존 마크 1개
    cv2.circle(img_before, (700, 250), 10, (20, 20, 20), -1)
    
    # ── [요청] 본체(사각형) 밖의 마크와 비슷한 노이즈 (검출되지 않아야 함) ──
    cv2.circle(img_before, (50, 50), 8, (40, 40, 40), -1) # 좌상단 밖
    cv2.circle(img_before, (950, 550), 10, (30, 30, 30), -1) # 우하단 밖
    cv2.circle(img_before, (500, 50), 7, (20, 20, 20), -1) # 중앙 상단 밖
    
    # ── [요청] 본체(사각형) 내부의 아주 작은 노이즈 (전처리로 제거되어야 함) ──
    # ROI 1 내부
    cv2.circle(img_before, (150, 200), 2, (10, 10, 10), -1)
    cv2.circle(img_before, (400, 400), 3, (15, 15, 15), -1)
    # ROI 2 내부
    cv2.circle(img_before, (600, 200), 2, (10, 10, 10), -1)
    
    # after 이미지 생성 (before에서 변화 추가)
    img_after = img_before.copy()
    
    # ── 새로운 마크 추가 (ROI 내부 가드존 통과 예정) ──
    # ROI 1: 새로운 마크 (중앙 근처)
    cv2.circle(img_after, (300, 300), 11, (45, 45, 45), -1)
    
    # ── 새로운 가드존 위반 마크 추가 (ROI 2의 경계선 근처) ──
    # ROI 2: 새로운 마크 (경계선 근처 - FAIL 유도)
    cv2.circle(img_after, (885, 435), 9, (35, 35, 35), -1)
    
    # ── [요청] After에서도 사각형 밖 노이즈 추가/변화 ──
    cv2.circle(img_after, (100, 550), 12, (40, 40, 40), -1) # 좌하단 밖 변화 (검출 무시되어야 함)
    
    # ── [요청] After에서도 사각형 내부 아주 작은 노이즈 추가 ──
    cv2.circle(img_after, (800, 200), 2, (10, 10, 10), -1) # ROI 2 내부
    
    # 미세한 위치 차이(Shift) 추가 (정합 테스트용)
    rows, cols = img_after.shape[:2]
    M = np.float32([[1, 0, 3], [0, 1, 2]]) # x+3, y+2 이동
    img_after = cv2.warpAffine(img_after, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    
    # 파일 저장
    cv2.imwrite("before.jpg", img_before)
    cv2.imwrite("after.jpg", img_after)
    
    print(f"[완료] before.jpg, after.jpg가 생성되었습니다.")
    print("  - 사각형 밖 노이즈: (50,50), (950,550), (500,50), (100,550) - ROI 제한 기능으로 무시되어야 함")
    print("  - 사각형 안 작은 노이즈: (150,200), (400,400), (600,200), (800,200) - 노이즈 제거 전처리로 제거되어야 함")

if __name__ == "__main__":
    generate_test_images()
