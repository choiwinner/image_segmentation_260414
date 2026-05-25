import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib  # 한글 폰트 지원
from processor import ImageAligner, MarkDetector, ChangeAnalyzer
from dual_roi_utils import (
    find_top_rectangles, calculate_sub_rectangle, is_mask_in_roi, 
    check_guard_zone, visualize_dual_results, select_multiple_rectangles_manually,
    collect_interactive_prompts, save_prompts, load_prompts
)
from image_preprocessor import apply_equalize_hist, apply_clahe, show_preprocessing_comparison, ask_user_preprocessing_choice
from noise_remover import ask_user_noise_removal
import os
import json
import time

def apply_gabor_filter_to_rois(img, rois, theta=np.pi/4, lam=5.0, ksize=31, sigma=4.0, gamma=0.5):
    """
    이미지에서 검출된 ROI 영역에 대해서만 1-Gabor 대역 차단 필터링을 수행하여 
    ROI 내부의 반복적인 배경 패턴(줄무늬, 격자 등)을 효과적으로 제거합니다.
    """
    img_out = img.copy()
    h, w = img.shape[:2]
    
    for rect in rois:
        # ROI 영역의 바운딩 박스 추출
        x, y, w_box, h_box = cv2.boundingRect(rect)
        
        # 바운딩 박스 내 다각형 ROI 마스크 생성
        mask = np.zeros((h_box, w_box), dtype=np.uint8)
        local_rect = rect - [x, y]
        cv2.drawContours(mask, [local_rect], -1, 255, -1)
        
        # ROI 내부 서브 이미지 추출
        sub_img = img[y:y+h_box, x:x+w_box]
        sub_gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        
        # 가보어 필터 크기가 ROI 바운딩 박스보다 클 경우 크기 축소 예외 처리
        kh = min(ksize, h_box)
        kw = min(ksize, w_box)
        if kh % 2 == 0: kh -= 1
        if kw % 2 == 0: kw -= 1
        if kh <= 0 or kw <= 0:
            continue
            
        # 가보어 커널(대역 통과) 생성
        gabor_kernel = cv2.getGaborKernel((kh, kw), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
        
        # FFT(고속 푸리에 변환)를 통해 주파수 도메인으로 매핑
        f_transform = np.fft.fft2(sub_gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # 커널을 서브 이미지 크기에 맞게 패딩
        padded_kernel = np.zeros((h_box, w_box), dtype=np.float32)
        cy, cx = h_box // 2, w_box // 2
        padded_kernel[cy-kh//2 : cy+kh//2+1, cx-kw//2 : cx+kw//2+1] = gabor_kernel
        
        # 패딩된 가보어 커널의 주파수 도메인 스펙트럼 도출
        kernel_fft = np.fft.fftshift(np.fft.fft2(padded_kernel))
        kernel_magnitude = np.abs(kernel_fft)
        max_val = np.max(kernel_magnitude)
        if max_val > 0:
            kernel_magnitude = kernel_magnitude / max_val
            
        # 1-Gabor 대역 차단 필터 생성 (통과 대역을 1에서 차감)
        band_reject_mask = 1.0 - kernel_magnitude
        
        # 주파수 필터링 수행
        filtered_shift = f_shift * band_reject_mask
        
        # IFFT(역 고속 푸리에 변환)를 통한 공간 도메인 복원
        filtered_ishift = np.fft.ifftshift(filtered_shift)
        image_filtered = np.abs(np.fft.ifft2(filtered_ishift))
        
        # 노멀라이즈 및 BGR 변환
        image_normalized = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        filtered_bgr = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
        
        # 다각형 마스크 영역 내부에만 필터링 결과 적용
        for c in range(3):
            img_out[y:y+h_box, x:x+w_box, c] = np.where(
                mask == 255,
                filtered_bgr[:, :, c],
                img_out[y:y+h_box, x:x+w_box, c]
            )
            
    return img_out

def main():
    print("\n" + "="*60)
    print("Probe Card Contact Mark Analysis with Gabor Filter (main7.py)")
    print("="*60)
    
    # 1. 데이터 로드
    image_before_path, image_after_path = "before.jpg", "after.jpg"
    if os.path.exists(image_before_path) and os.path.exists(image_after_path):
        img_before, img_after = cv2.imread(image_before_path), cv2.imread(image_after_path)
    else:
        print("[오류] before.jpg 또는 after.jpg가 존재하지 않습니다.")
        return
        
    # 2. 이미지 정합
    print("\n[1/9] 이미지 정합(Alignment) 중...")
    img_after_aligned, _ = ImageAligner().align(img_before, img_after)
    
    config_file = "analysis_config.json"
    
    def load_full_config():
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f: 
                    return json.load(f)
            except: 
                pass
        return {
            "diff_th": 25, 
            "min_a": 5, 
            "overlap_th": 0.3, 
            "guard_percentage": 80.0,
            "gabor_theta": np.pi/4,
            "gabor_lam": 5.0,
            "roi_direction": "horizontal",
            "gt_bbox": None,
            "noise_threshold": 15,
            "noise_sensitivity": 40
        }
    
    def save_full_config(c):
        with open(config_file, 'w') as f: 
            json.dump(c, f)

    config = load_full_config()
    
    auto_mode = False
    if os.path.exists(config_file):
        ans = input("  > 저장된 설정을 불러와 자동으로 최종 결과까지 실행할까요? (y/n, 기본값 y): ").strip().lower()
        if ans != 'n':
            auto_mode = True
            print("  [알림] 자동 모드로 진행합니다. 대화형 입력을 건너뜁니다.")
        else:
            print("  [알림] 수동 모드로 진행합니다.")

    # 3. ROI 검출
    print("\n[2/9] 사각형 영역(ROI) 검출...")
    if auto_mode:
        roi_dir = config.get("roi_direction", "horizontal")
        print(f"  [자동] ROI 검출 방향: {roi_dir}")
    else:
        while True:
            dir_choice = input("  > ROI 검출 분할 방향을 선택하세요 (1: 좌우, 2: 상하, 기본값 1): ").strip()
            if dir_choice == "" or dir_choice == "1":
                roi_dir = "horizontal"
                break
            elif dir_choice == "2":
                roi_dir = "vertical"
                break
            else:
                print("    [!] 1 또는 2를 입력해 주세요.")
        config["roi_direction"] = roi_dir
            
    rois = find_top_rectangles(img_after_aligned, n=2, direction=roi_dir)
    if not rois or len(rois) < 2: 
        rois = select_multiple_rectangles_manually(img_after_aligned, n=2)
    if not rois: 
        print("[오류] ROI 영역을 찾지 못했습니다.")
        return

    # 4. 인터랙티브 파라미터 최적화 (가보어 필터 파라미터 포함)
    print("\n[3/9] 분석 및 가보어 파라미터 최적화 중...")
    initial_cfg = config.copy()
    
    def interactive_optimization(img_b_orig, img_a_orig, rois, cfg, p_info="", gt_bbox=None):
        from matplotlib.widgets import Slider
        analyzer = ChangeAnalyzer(diff_threshold=cfg.get('diff_th', 25), min_area=cfg.get('min_a', 5))
        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)
        plt.subplots_adjust(bottom=0.38)
        
        # 최초 설정값 기준 이미지 준비
        g_theta_init = cfg.get('gabor_theta', np.pi/4)
        g_lam_init = cfg.get('gabor_lam', 5.0)
        
        img_plot_b = ax_b.imshow(np.zeros_like(img_b_orig))
        ax_b.set_title("Before (Gabor 전처리 적용)"); ax_b.axis('off')
        
        img_plot_a = ax_a.imshow(np.zeros_like(img_a_orig))
        overlay = np.zeros((*img_a_orig.shape[:2], 4), dtype=np.uint8)
        mark_plot = ax_a.imshow(overlay)
        ax_a.set_title(f"After (Gabor 전처리 적용) | 프롬프트: {p_info}"); ax_a.axis('off')
        
        # 타겟 BBox가 지정된 경우, GT 마스크 생성 및 시각화용 patches.Rectangle 추가
        gt_mask = None
        if gt_bbox is not None:
            gx, gy, gw, gh = gt_bbox
            gt_mask = np.zeros(img_a_orig.shape[:2], dtype=np.uint8)
            gt_mask[gy:gy+gh, gx:gx+gw] = 255
            
            import matplotlib.patches as patches
            rect_patch = patches.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', label='Target BBox (GT)')
            ax_a.add_patch(rect_patch)
            ax_a.legend(loc='upper right')
        
        # 슬라이더 컨트롤들 정의
        s_diff = Slider(plt.axes([0.15, 0.28, 0.7, 0.03]), '차이 임계값 (Diff Th)', 0, 255, valinit=cfg.get('diff_th', 25), valstep=1)
        s_min_a = Slider(plt.axes([0.15, 0.23, 0.7, 0.03]), '최소 면적 (Min Area)', 1, 500, valinit=cfg.get('min_a', 5), valstep=1)
        s_overlap = Slider(plt.axes([0.15, 0.18, 0.7, 0.03]), '중첩 임계값 (Overlap Th)', 0.0, 1.0, valinit=cfg.get('overlap_th', 0.3))
        s_guard = Slider(plt.axes([0.15, 0.13, 0.7, 0.03]), '가드 영역 비율 (Guard %)', 10.0, 100.0, valinit=cfg.get('guard_percentage', 80.0))
        s_g_theta = Slider(plt.axes([0.15, 0.08, 0.7, 0.03]), '가보어 각도 (Gabor Theta)', 0.0, np.pi, valinit=g_theta_init)
        s_g_lam = Slider(plt.axes([0.15, 0.03, 0.7, 0.03]), '가보어 파장 (Gabor Lambda)', 1.0, 30.0, valinit=g_lam_init)
        
        res_cfg = cfg.copy()
        
        def update(val):
            # 슬라이더 값 획득
            diff_th = int(s_diff.val)
            min_a = int(s_min_a.val)
            overlap_th = s_overlap.val
            guard_pct = s_guard.val
            g_theta = s_g_theta.val
            g_lam = s_g_lam.val
            
            # 1. 가보어 필터링 적용
            t_img_b_gabor = apply_gabor_filter_to_rois(img_b_orig, rois, theta=g_theta, lam=g_lam)
            t_img_a_gabor = apply_gabor_filter_to_rois(img_a_orig, rois, theta=g_theta, lam=g_lam)
            
            # 1.5. 양방향 필터(Bilateral Filter)를 통해 패드의 고주파 그레인 노이즈 제거 (엣지는 보존하고 평활화)
            t_img_b_bilateral = cv2.bilateralFilter(t_img_b_gabor, 9, 75, 75)
            t_img_a_bilateral = cv2.bilateralFilter(t_img_a_gabor, 9, 75, 75)
            
            # 2. 소형 노이즈 제거 동기화 (튜닝 시에도 노이즈 제거 처리 적용)
            from noise_remover import remove_small_noise, load_colors
            colors = load_colors()
            bg_color = colors.get("bg_color") if colors else [100, 150, 180]
            mark_color = colors.get("mark_color") if colors else None
            noise_th = config.get("noise_threshold", 15)
            noise_sens = config.get("noise_sensitivity", 40)
            
            t_img_b_clean, _ = remove_small_noise(t_img_b_bilateral, noise_th, bg_color, mark_color, noise_sens)
            t_img_a_clean, _ = remove_small_noise(t_img_a_bilateral, noise_th, bg_color, mark_color, noise_sens)
            
            # 3. 대비 전처리 적용 (튜닝 시에는 대화형 창 없이 CLAHE를 기본으로 적용)
            t_img_b_proc = apply_clahe(t_img_b_clean)
            t_img_a_proc = apply_clahe(t_img_a_clean)
            
            # 이미지 뷰 데이터 업데이트
            img_plot_b.set_data(cv2.cvtColor(t_img_b_proc, cv2.COLOR_BGR2RGB))
            img_plot_a.set_data(cv2.cvtColor(t_img_a_proc, cv2.COLOR_BGR2RGB))
            
            # 4. 차영상 분석 수행
            analyzer.diff_threshold = diff_th
            _, thresh = analyzer.get_difference_candidates(t_img_b_proc, t_img_a_proc)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
            
            mask_valid = np.zeros_like(thresh)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_a:
                    cx, cy = centroids[i]
                    
                    # 사용자가 지정한 타겟 BBox가 있다면 BBox 외부 영역 후보 제외 (노이즈 강제 억제)
                    if gt_bbox is not None:
                        gx, gy, gw, gh = gt_bbox
                        if not (gx <= cx <= gx + gw and gy <= cy <= gy + gh):
                            continue
                            
                    for rect in rois:
                        if cv2.pointPolygonTest(rect, (float(cx), float(cy)), False) >= 0:
                            mask_valid[labels == i] = 255
                            break
                            
            new_overlay = np.zeros((*t_img_a_proc.shape[:2], 4), dtype=np.uint8)
            new_overlay[mask_valid > 0] = [255, 0, 0, 160] # 검출된 영역을 반투명 빨간색으로 오버레이
            mark_plot.set_data(new_overlay)
            
            # 타겟 마스크가 있는 경우 IoU 계산 및 타이틀 표시
            iou_str = ""
            if gt_mask is not None:
                intersection = np.logical_and(gt_mask == 255, mask_valid == 255).sum()
                union = np.logical_or(gt_mask == 255, mask_valid == 255).sum()
                iou = intersection / union if union > 0 else 0.0
                iou_str = f" | Target IoU: {iou:.3f}"
            
            ax_a.set_title(f"After (Gabor 전처리 적용) | 프롬프트: {p_info}{iou_str}"); ax_a.axis('off')
            
            res_cfg.update({
                "diff_th": diff_th, 
                "min_a": min_a, 
                "overlap_th": overlap_th, 
                "guard_percentage": guard_pct,
                "gabor_theta": g_theta,
                "gabor_lam": g_lam
            })
            fig.canvas.draw_idle()
            
        for s in [s_diff, s_min_a, s_overlap, s_guard, s_g_theta, s_g_lam]:
            s.on_changed(update)
            
        update(None)
        plt.show()
        return res_cfg

    # 프롬프트 로드
    p, l = load_prompts()
    p_count = f"Positive:{l.count(1)} / Negative:{l.count(0)}" if l else "없음"
    
    # [3.5 추가] 파라미터 최적화(IoU 계산)를 위한 타겟 마크 영역(BBox) 지정
    if auto_mode:
        gt_bbox = config.get("gt_bbox", None)
        if gt_bbox:
            print(f"  [자동] 지정된 타겟 바운딩 박스: x={gt_bbox[0]}, y={gt_bbox[1]}, w={gt_bbox[2]}, h={gt_bbox[3]}")
        else:
            print("  [자동] 저장된 타겟 바운딩 박스가 없습니다. 전체 영역 최적화를 진행합니다.")
    else:
        print("\n[3/9-추가] 파라미터 최적화(IoU 계산)를 위한 타겟 마크 영역(BBox) 지정...")
        print("  > 이미지 창이 뜨면 마우스 드래그로 실제 마크 영역을 지정한 후 엔터(Enter)나 스페이스(Space)를 누르세요.")
        print("  > 지정을 건너뛰려면 바로 엔터나 스페이스를 누르시면 됩니다.")
        
        cv2.namedWindow("Select Target Mark BBox", cv2.WINDOW_NORMAL)
        r_bbox = cv2.selectROI("Select Target Mark BBox", img_after_aligned, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target Mark BBox")
        
        # w, h가 0보다 클 때만 유효한 BBox로 판단
        gt_bbox = r_bbox if (r_bbox[2] > 0 and r_bbox[3] > 0) else None
        if gt_bbox:
            config["gt_bbox"] = list(gt_bbox)
            print(f"  [확인] 지정된 타겟 바운딩 박스: x={gt_bbox[0]}, y={gt_bbox[1]}, w={gt_bbox[2]}, h={gt_bbox[3]}")
        else:
            config["gt_bbox"] = None
            print("  [알림] 타겟 영역 지정을 건너뛰었습니다. 기존 전체 영역 최적화를 진행합니다.")
    
    # 인터랙티브 파라미터 튜닝 실행
    if auto_mode:
        print("\n[3/9] 자동 모드: 기존 파라미터를 그대로 사용합니다. (파라미터 튜닝 창 스킵)")
        opt_cfg = {
            "diff_th": config.get("diff_th", 25),
            "min_a": config.get("min_a", 5),
            "overlap_th": config.get("overlap_th", 0.3),
            "guard_percentage": config.get("guard_percentage", 80.0),
            "gabor_theta": config.get("gabor_theta", np.pi/4),
            "gabor_lam": config.get("gabor_lam", 5.0)
        }
    else:
        opt_cfg = interactive_optimization(img_before, img_after_aligned, rois, initial_cfg, p_count, gt_bbox)
        config.update(opt_cfg)
        save_full_config(config)
    
    # 5. 소형 노이즈 제거 전처리 (대화형 실행)
    # 가보어 필터의 최적값을 원본 이미지에 최종 적용한 상태에서 노이즈 제거 검토
    print("\n[4/9] 소형 노이즈 제거 전처리...")
    img_before_gabor = apply_gabor_filter_to_rois(img_before, rois, theta=opt_cfg['gabor_theta'], lam=opt_cfg['gabor_lam'])
    img_after_gabor = apply_gabor_filter_to_rois(img_after_aligned, rois, theta=opt_cfg['gabor_theta'], lam=opt_cfg['gabor_lam'])
    
    # 양방향 필터(Bilateral Filter)를 통해 고주파 그레인 노이즈 평활화
    img_before_bilateral = cv2.bilateralFilter(img_before_gabor, 9, 75, 75)
    img_after_bilateral = cv2.bilateralFilter(img_after_gabor, 9, 75, 75)
    
    if auto_mode:
        print("  [자동] 기존 설정값으로 노이즈 제거를 수행합니다.")
        from noise_remover import remove_small_noise, load_colors
        colors = load_colors()
        bg_color = colors.get("bg_color") if colors else None
        mark_color = colors.get("mark_color") if colors else None
        noise_th = config.get("noise_threshold", 15)
        noise_sens = config.get("noise_sensitivity", 40)
        img_before_clean, count_b = remove_small_noise(img_before_bilateral, noise_th, bg_color, mark_color, noise_sens)
        img_after_clean, count_a = remove_small_noise(img_after_bilateral, noise_th, bg_color, mark_color, noise_sens)
        print(f"  [확인] 노이즈 제거 완료 (Before: {count_b}개 제거, After: {count_a}개 제거)")
    else:
        img_before_clean, img_after_clean, noise_th, noise_sens = ask_user_noise_removal(img_before_bilateral, img_after_bilateral, rois)
        config["noise_threshold"] = noise_th
        config["noise_sensitivity"] = noise_sens
        save_full_config(config)
    
    # 6. 이미지 대비 및 필터링 완료 전처리
    print("\n[5/9] 이미지 대비 전처리...")
    if auto_mode:
        contrast_method = config.get("contrast_method", 2)
        print(f"  [자동] 대비 전처리 방식 적용 (방법: {contrast_method})")
        if contrast_method == 1:
            img_before_proc = apply_equalize_hist(img_before_clean)
            img_after_proc = apply_equalize_hist(img_after_clean)
        elif contrast_method == 2:
            img_before_proc = apply_clahe(img_before_clean)
            img_after_proc = apply_clahe(img_after_clean)
        else:
            img_before_proc, img_after_proc = img_before_clean, img_after_clean
    else:
        before_hist = apply_equalize_hist(img_before_clean)
        before_clahe = apply_clahe(img_before_clean)
        after_hist = apply_equalize_hist(img_after_clean)
        after_clahe = apply_clahe(img_after_clean)
        
        print("  [정보] 전처리 비교 이미지를 표시합니다...")
        show_preprocessing_comparison(img_before_clean, before_hist, before_clahe, title_prefix="Before")
        show_preprocessing_comparison(img_after_clean, after_hist, after_clahe, title_prefix="After")
        
        choice = ask_user_preprocessing_choice()
        plt.close('all')
        
        config["contrast_method"] = choice
        save_full_config(config)
        
        if choice == 1:
            print("  [적용] equalizeHist 전처리를 적용합니다.")
            img_before_proc, img_after_proc = before_hist, after_hist
        elif choice == 2:
            print("  [적용] CLAHE 전처리를 적용합니다.")
            img_before_proc, img_after_proc = before_clahe, after_clahe
        else:
            print("  [적용] 원본 이미지를 그대로 사용합니다.")
            img_before_proc, img_after_proc = img_before_clean, img_after_clean
    
    # 7. 세그멘테이션 프롬프트 기설정 확인
    if not l:
        if auto_mode:
            print("\n[6/9] 자동 모드: 세그멘테이션 프롬프트 설정을 건너뜁니다. (기존 로드된 데이터 사용)")
        else:
            print("\n[6/9] 세그멘테이션 프롬프트 설정..."); 
            p, l = collect_interactive_prompts(img_after_clean)
            if p: 
                save_prompts(p, l)

    # 8. 최종 마크 분석
    print("\n[7/9] 최종 마크 분석 중...")
    detector = MarkDetector()
    analyzer = ChangeAnalyzer(diff_threshold=opt_cfg['diff_th'], min_area=opt_cfg['min_a'])
    _, thresh_map = analyzer.get_difference_candidates(img_before_proc, img_after_proc)
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(thresh_map)
    
    refined_new_marks = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < opt_cfg['min_a']: 
            continue
        cand_pt = centroids[i]
        
        # 사용자가 지정한 타겟 BBox가 있다면 자동화 모드 여부와 관계없이 BBox 외부 후보를 완전히 제거합니다. (정합 완료되어 절대 좌표 유효함)
        if gt_bbox is not None:
            gx, gy, gw, gh = gt_bbox
            cx, cy = cand_pt
            if not (gx <= cx <= gx + gw and gy <= cy <= gy + gh):
                continue
                
        is_inside = False
        target_mask = np.zeros_like(thresh_map, dtype=np.uint8)
        
        for rect in rois:
            if cv2.pointPolygonTest(rect, (float(cand_pt[0]), float(cand_pt[1])), False) >= 0:
                cv2.drawContours(target_mask, [rect], -1, 255, -1)
                is_inside = True
                break
        if not is_inside: 
            continue
        
        raw_mask = (labels_map == i).astype(np.uint8) * 255
        
        # 수동 모드일 때는 이전에 수집한 포인트 프롬프트(p, l)를 사용하고,
        # 자동 모드(다른 이미지)일 때는 위치 편향을 방지하기 위해 Zero-shot 모드(새로 검출된 cand_pt 좌표 하나만 입력)로 SAM을 구동합니다.
        use_prompts = p if (not auto_mode and p) else None
        use_labels = l if (not auto_mode and l) else None
        
        if detector.predictor:
            if use_prompts:
                masks = detector.get_masks_from_points(img_after_clean, use_prompts + [cand_pt], use_labels + [1])
            else:
                masks = detector.get_masks_from_points(img_after_clean, [cand_pt], [1])
                
            if masks:
                ai_seg = masks[0]['segmentation'].astype(np.uint8) * 255
                final_seg = cv2.bitwise_or(raw_mask, cv2.bitwise_and(ai_seg, thresh_map))
            else: 
                final_seg = raw_mask
        else: 
            final_seg = raw_mask
        
        final_seg = cv2.bitwise_and(final_seg, target_mask)
        if int(final_seg.sum()) > 0:
            refined_new_marks.append({
                'segmentation': final_seg > 0, 
                'area': int((final_seg > 0).sum())
            })

    # 9. ROI 결과 판정
    print("\n[8/9] ROI 결과 및 가드존 위반 여부 분석...")
    roi_results = []
    for i, rect in enumerate(rois):
        sub = calculate_sub_rectangle(rect, opt_cfg['guard_percentage'])
        marks = [m for m in refined_new_marks if is_mask_in_roi(m, rect)]
        roi_results.append({
            'rect': rect, 
            'sub_rect': sub, 
            'is_pass': check_guard_zone(marks, sub), 
            'marks': marks
        })

    # 10. 최종 시각화
    print("\n[9/9] 분석 결과 시각화 출력...")
    cfg_disp = {
        **opt_cfg, 
        "Prompts": f"Positive:{l.count(1)} / Negative:{l.count(0)}" if l else "Zero-shot"
    }
    visualize_dual_results(img_before, img_after_aligned, roi_results, cfg_disp)

if __name__ == "__main__":
    main()
