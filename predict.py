import cv2
import argparse
import sys
import time
from processor import ImageAligner, MarkDetector, ChangeAnalyzer, visualize_results

def main():
    parser = argparse.ArgumentParser(description="Probe Card Contact Mark 검출 (실제 이미지 비교)")
    parser.add_argument("--before", type=str, required=True, help="Contact 전 이미지 경로")
    parser.add_argument("--after", type=str, required=True, help="Contact 후 이미지 경로")
    parser.add_argument("--checkpoint", type=str, default="sam2_hiera_large.pt", help="SAM2 체크포인트 경로")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml", help="SAM2 모델 설정 파일")
    
    args = parser.parse_args()

    print(f"이미지 로딩 중...\n - Before: {args.before}\n - After: {args.after}")
    
    img_before = cv2.imread(args.before)
    img_after = cv2.imread(args.after)

    if img_before is None or img_after is None:
        print("Error: 이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)

    start_total = time.perf_counter()

    # 1. 이미지 정합 (Alignment)
    print("1. 이미지 정합 중 (ECC 알고리즘)...")
    start_time = time.perf_counter()
    aligner = ImageAligner()
    img_after_aligned, _ = aligner.align(img_before, img_after)
    print(f"   > 완료 ({time.perf_counter() - start_time:.4f}초)")

    # 2. Mark 검출 (SAM2)
    print("2. SAM2를 이용한 Mark 검출 및 분석 중...")
    start_time = time.perf_counter()
    detector = MarkDetector(model_cfg=args.model_cfg, checkpoint=args.checkpoint)
    
    if detector.mask_generator is None:
        print("Error: SAM2 모델을 로드할 수 없습니다. --checkpoint 경로를 확인하세요.")
        sys.exit(1)

    masks_before = detector.get_masks(img_before)
    masks_after = detector.get_masks(img_after_aligned)
    print(f"   > 검출 완료 ({time.perf_counter() - start_time:.4f}초)")

    # 3. 차이 분석
    print("3. 신규 Mark 분석 중...")
    start_time = time.perf_counter()
    analyzer = ChangeAnalyzer()
    new_marks = analyzer.find_new_marks(masks_before, masks_after)
    print(f"   > 분석 완료 ({time.perf_counter() - start_time:.4f}초)")

    print("-" * 40)
    print(f"전체 소요 시간: {time.perf_counter() - start_total:.4f}초")
    print(f"검출된 신규 Mark 개수: {len(new_marks)}")
    print("-" * 40)

    # 4. 결과 시괄화
    print("결과 창을 띄웁니다...")
    visualize_results(img_before, img_after_aligned, new_marks)

if __name__ == "__main__":
    main()
