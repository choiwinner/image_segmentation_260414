import cv2
import numpy as np
from processor import MarkDetector, ChangeAnalyzer, visualize_results
import torch
import os

def evaluate_high_overlap():
    print("높은 중첩(High Overlap) 마크 검출 평가를 시작합니다.")
    
    # 1. 테스트 이미지 생성 (거의 겹쳐 있는 상황)
    # 배경: 회색 패드 느낌
    base = np.full((300, 300, 3), 220, dtype=np.uint8)
    cv2.rectangle(base, (20, 20), (280, 280), (180, 180, 180), -1)
    
    # Before: 기존 마크 하나 (검은색 원)
    img_before = base.copy()
    existing_center = (150, 150)
    radius = 30
    cv2.circle(img_before, existing_center, radius, (50, 50, 50), -1)
    
    # After: 기존 마크 + 70% 정도 겹치는 새로운 마크
    # 중심을 반지름의 0.6배만큼만 이동시켜 심하게 겹침 유도
    img_after = img_before.copy()
    new_center = (150 + int(radius * 0.6), 150) 
    cv2.circle(img_after, new_center, radius, (60, 60, 60), -1)
    
    print(f"테스트 이미지 생성 완료: 중심점 이동 {int(radius * 0.6)}px (매우 높은 중첩)")
    
    # 2. 검출기 및 분석기 초기화
    detector = MarkDetector()
    if detector.predictor is None:
        print("SAM2 모델을 로드할 수 없어 평가를 진행할 수 없습니다.")
        return
        
    analyzer = ChangeAnalyzer(diff_threshold=20, min_area=10)
    
    # 3. 마스크 추출 및 분석
    print("마스크 분석 중...")
    masks_before = detector.get_masks(img_before)
    
    # 개선된 차분 기반 로직 실행
    new_marks = analyzer.find_new_marks_refined(img_before, img_after, masks_before, detector)
    
    print("-" * 40)
    print(f"검출된 신규 Mark 개수: {len(new_marks)}")
    
    if len(new_marks) > 0:
        print("SUCCESS: 심한 중첩 상황에서도 신규 마크를 성공적으로 검출했습니다!")
    else:
        print("FAILURE: 중첩된 신규 마크를 검출하지 못했습니다.")
    print("-" * 40)
    
    # 4. 결과 시각화 저장
    # visualize_results는 plt.show()를 호출하므로, 서버 환경 등에서는 확인이 어려울 수 있음
    # 여기서는 결과 이미지를 파일로 저장하는 로직을 추가로 수행
    res_img = img_after.copy()
    for mask in new_marks:
        m = mask['segmentation'].astype(bool)
        res_img[m] = [0, 0, 255] # Red for new marks
    
    cv2.imwrite("evaluation_result.jpg", res_img)
    print("결과 이미지가 evaluation_result.jpg로 저장되었습니다.")

if __name__ == "__main__":
    evaluate_high_overlap()
