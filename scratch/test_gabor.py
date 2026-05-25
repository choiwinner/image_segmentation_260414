import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import koreanize_matplotlib  # 한글 폰트 설정을 위한 임포트

def apply_gabor_filter_to_rois(img, rois, theta=np.pi/4, lam=5.0, ksize=31, sigma=4.0, gamma=0.5):
    """
    이미지에서 검출된 ROI 영역에 대해서만 1-Gabor 필터링을 수행하여 
    ROI 내부의 반복적인 배경 패턴(줄무늬 등)을 억제합니다.
    """
    img_out = img.copy()
    h, w = img.shape[:2]
    
    for idx, rect in enumerate(rois):
        x, y, w_box, h_box = cv2.boundingRect(rect)
        
        # 바운딩 박스 영역 내에서 ROI 마스크 생성
        mask = np.zeros((h_box, w_box), dtype=np.uint8)
        local_rect = rect - [x, y]
        cv2.drawContours(mask, [local_rect], -1, 255, -1)
        
        # ROI 내부 서브 이미지 추출
        sub_img = img[y:y+h_box, x:x+w_box]
        sub_gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        
        # 가보어 필터링 적용 (대역 차단)
        kh = min(ksize, h_box)
        kw = min(ksize, w_box)
        if kh % 2 == 0: kh -= 1
        if kw % 2 == 0: kw -= 1
        
        gabor_kernel = cv2.getGaborKernel((kh, kw), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
        
        # FFT
        f_transform = np.fft.fft2(sub_gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Kernel padding
        padded_kernel = np.zeros((h_box, w_box), dtype=np.float32)
        cy, cx = h_box // 2, w_box // 2
        padded_kernel[cy-kh//2 : cy+kh//2+1, cx-kw//2 : cx+kw//2+1] = gabor_kernel
        
        kernel_fft = np.fft.fftshift(np.fft.fft2(padded_kernel))
        kernel_magnitude = np.abs(kernel_fft)
        max_val = np.max(kernel_magnitude)
        if max_val > 0:
            kernel_magnitude = kernel_magnitude / max_val
            
        band_reject_mask = 1.0 - kernel_magnitude
        filtered_shift = f_shift * band_reject_mask
        
        filtered_ishift = np.fft.ifftshift(filtered_shift)
        image_filtered = np.abs(np.fft.ifft2(filtered_ishift))
        
        # 정규화
        image_normalized = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        filtered_bgr = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
        
        # ROI 마스크 영역만 합성
        for c in range(3):
            img_out[y:y+h_box, x:x+w_box, c] = np.where(
                mask == 255,
                filtered_bgr[:, :, c],
                img_out[y:y+h_box, x:x+w_box, c]
            )
            
    return img_out

def main():
    img_before = cv2.imread("before.jpg")
    if img_before is None:
        print("before.jpg 파일을 찾을 수 없습니다. 테스트 이미지를 먼저 생성해 주세요.")
        return
        
    # main6.py의 ROI 검출 방식 간소화 모방
    # 임의로 두 ROI 영역(사각형) 지정 (v5 이미지의 ROI 1, ROI 2 위치 기반)
    # ROI 1: [(100, 150), (450, 450)] -> 사각형의 4개 꼭짓점으로 contour 구성
    roi1 = np.array([[100, 150], [450, 150], [450, 450], [100, 450]], dtype=np.int32)
    # ROI 2: [(550, 150), (900, 450)]
    roi2 = np.array([[550, 150], [900, 150], [900, 450], [550, 450]], dtype=np.int32)
    rois = [roi1, roi2]
    
    # 여러 theta, lam 파라미터 조합으로 테스트
    # 예제 이미지의 줄무늬 패턴은 대각선 45도 방향 (theta = -np.pi/4 또는 np.pi/4)
    # 빗살 무늬 간격(주기)은 약 5~10픽셀 사이이므로 lam = 5.0, 7.0, 10.0 테스트
    
    theta_list = [np.pi/4, -np.pi/4, 0, np.pi/2]
    lam_list = [5.0, 7.0, 10.0]
    
    fig, axes = plt.subplots(len(theta_list), len(lam_list) + 1, figsize=(18, 12))
    
    for r_idx, theta in enumerate(theta_list):
        # 첫 열은 원본 이미지 표시
        axes[r_idx, 0].imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
        axes[r_idx, 0].set_title("원본 이미지")
        axes[r_idx, 0].axis('off')
        
        for c_idx, lam in enumerate(lam_list):
            filtered_img = apply_gabor_filter_to_rois(img_before, rois, theta=theta, lam=lam)
            
            axes[r_idx, c_idx + 1].imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
            axes[r_idx, c_idx + 1].set_title(f"theta={theta:.2f}, lam={lam:.1f}")
            axes[r_idx, c_idx + 1].axis('off')
            
    plt.tight_layout()
    plt.savefig("scratch/gabor_test_results.png")
    print("가보어 필터 테스트 결과를 scratch/gabor_test_results.png 로 저장했습니다.")

if __name__ == "__main__":
    main()
