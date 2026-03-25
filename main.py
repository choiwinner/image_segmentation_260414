import cv2
import numpy as np
from processor import ImageAligner, MarkDetector, ChangeAnalyzer, visualize_results
import os
import time

def create_dummy_data():
    """테스트를 위한 더미 데이터 생성 (사각형 포함)"""
    # 기본 이미지 (노이즈가 섞인 배경)
    base = np.random.randint(50, 70, (500, 500), dtype=np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    
    # 큰 직사각형 추가 (검출 대상)
    cv2.rectangle(base, (50, 50), (450, 450), (200, 200, 200), -1)
    # 배경과 구분되는 내부 영역
    cv2.rectangle(base, (60, 60), (440, 440), (100, 100, 100), -1)
    
    # 기존 Mark (Dark circles)
    img_before = base.copy()
    cv2.circle(img_before, (100, 100), 20, (30, 30, 30), -1) # 기존 마크 1
    cv2.circle(img_before, (300, 200), 25, (40, 40, 40), -1) # 기존 마크 2
    
    # Contact 후 이미지 (약간의 이동 + 새로운 Mark)
    # 1. 미세한 위치 이동 (Translation)
    M = np.float32([[1, 0, 2], [0, 1, -1]])
    img_after = cv2.warpAffine(img_before, M, (500, 500))
    
    # 2. 새로운 Mark 추가 (Bright spot)
    # Case 1: 가드 존 내부 (PASS 예상) - 중앙 근처
    cv2.circle(img_after, (250, 250), 15, (220, 220, 220), -1) 
    
    # Case 2: 가드 존 외부 가능성 (FAIL 유도용) - 가장자리 근처
    # (사용자가 낮은 백분율을 입력하면 Fail이 날 수 있음)
    cv2.circle(img_after, (80, 80), 10, (230, 230, 230), -1)
    
    return img_before, img_after

def main():
    print("Probe Card Contact Mark 검출 데모를 시작합니다.")
    start_total = time.perf_counter()
    
    # 1. 데이터 준비 (실제 이미지 파일 로드 시도)
    image_before_path = "before.jpg"
    image_after_path = "after.jpg"
    
    if os.path.exists(image_before_path) and os.path.exists(image_after_path):
        print(f"실제 이미지 파일을 로드합니다: {image_before_path}, {image_after_path}")
        img_before = cv2.imread(image_before_path)
        img_after = cv2.imread(image_after_path)
    else:
        print("이미지 파일을 찾을 수 없어 테스트용 더미 데이터를 생성합니다.")
        img_before, img_after = create_dummy_data()
    
    # 2. 이미지 정합 (Alignment)
    print("이미지 정합 중...")
    start_align = time.perf_counter()
    aligner = ImageAligner()
    img_after_aligned, _ = aligner.align(img_before, img_after)
    end_align = time.perf_counter()
    print(f"  > 정합 소요 시간: {end_align - start_align:.4f}초")
    
    # 3. Mark 검출 (SAM2)
    print("SAM2를 이용한 Mark 검출 중... (체크포인트 필요)")
    start_detect = time.perf_counter()
    detector = MarkDetector()
    
    if detector.mask_generator is None:
        print("\n[!] SAM2 체크포인트가 없어 실제 모델 실행은 생략하고 Mock 결과를 사용합니다.")
        mock_mask = np.zeros((500, 500), dtype=bool)
        cv2.circle(mock_mask.view(np.uint8), (200, 400), 16, 1, -1)
        new_marks = [{'segmentation': mock_mask}]
        end_detect = time.perf_counter()
    else:
        masks_before = detector.get_masks(img_before)
        masks_after = detector.get_masks(img_after_aligned)
        end_detect = time.perf_counter()
        
        # 4. 차이 분석
        print("신규 Mark 분석 중...")
        start_analysis = time.perf_counter()
        analyzer = ChangeAnalyzer()
        new_marks = analyzer.find_new_marks(masks_before, masks_after)
        end_analysis = time.perf_counter()
        print(f"  > 분석 소요 시간: {end_analysis - start_analysis:.4f}초")
    
    print(f"  > 검출/추론 소요 시간: {end_detect - start_detect:.4f}초")
    
    end_total = time.perf_counter()
    print("-" * 40)
    print(f"전체 소요 시간: {end_total - start_total:.4f}초")
    print(f"검출된 신규 Mark 개수: {len(new_marks)}")
    print("-" * 40)
    
    # 5. 사각형 검출 및 가드 존 검증 (선택적 실행)
    print("-" * 40)
    run_validation = input("사각형 검출 및 가드 존 검증 기능을 실행하시겠습니까? (y/n): ").strip().lower()
    
    rect = None
    sub_rect = None
    is_pass = None
    
    if run_validation == 'y':
        try:
            percentage = float(input("가드 존으로 사용할 길이 백분율을 입력하세요 (예: 80): "))
        except ValueError:
            print("잘못된 입력입니다. 기본값 80%를 사용합니다.")
            percentage = 80.0
            
        from processor import find_largest_rectangle, calculate_sub_rectangle, is_within_guard_zone, select_rectangle_manually
        
        # 1-1. 자동 사각형 검출 시도
        rect = find_largest_rectangle(img_after_aligned)
        
        while True:
            if rect is not None:
                # 임시 시각화로 사용자 확인
                temp_img = img_after_aligned.copy()
                cv2.polylines(temp_img, [rect], True, (0, 255, 0), 2)
                cv2.imshow("Confirm Rectangle", temp_img)
                cv2.waitKey(1) # OpenCV 창을 강제로 갱신하여 표시되게 함
                
                print("-" * 40)
                print("검출된 사각형이 올바른가요?")
                ans = input("  > 승인하시려면 'y', 수동으로 설정하시려면 'n'을 입력하세요: ").strip().lower()
                cv2.destroyWindow("Confirm Rectangle")
                
                if ans == 'y':
                    break
            else:
                print("사각형을 자동으로 찾을 수 없습니다.")

            # 1-2. 수동 선택
            print("수동으로 사각형 꼭짓점 4개를 선택합니다...")
            rect = select_rectangle_manually(img_after_aligned)
            if rect is None:
                print("수동 선택에 실패했습니다. 다시 시도해 주세요.")
                continue

        # 2. 가드 존 계산 및 판정
        sub_rect = calculate_sub_rectangle(rect, percentage)
        is_pass = is_within_guard_zone(new_marks, sub_rect)
        
        print(f"  > 판정 결과: {'PASS' if is_pass else 'FAIL'}")

    # 6. 전체 결과 시각화
    visualize_results(img_before, img_after_aligned, new_marks, rect, sub_rect, is_pass)

if __name__ == "__main__":
    main()
