import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from processor import ImageAligner, MarkDetector, ChangeAnalyzer
from dual_roi_utils import (
    find_top_rectangles, calculate_sub_rectangle, is_mask_in_roi, 
    check_guard_zone, visualize_dual_results, select_multiple_rectangles_manually,
    collect_interactive_prompts, save_prompts, load_prompts
)
from image_preprocessor import preprocess_images
from noise_remover import ask_user_noise_removal
import os
import json
import time

def main():
    print("\n" + "="*60)
    print("Probe Card Contact Mark Analysis (main6.py)")
    print("="*60)
    
    # 1. 데이터 로드
    image_before_path, image_after_path = "before.jpg", "after.jpg"
    if os.path.exists(image_before_path) and os.path.exists(image_after_path):
        img_before, img_after = cv2.imread(image_before_path), cv2.imread(image_after_path)
    else: return
        
    # 2. 이미지 정합
    print("\n[1/8] 이미지 정합(Alignment) 중...")
    img_after_aligned, _ = ImageAligner().align(img_before, img_after)
    
    # 3. ROI 검출
    print("\n[2/8] 사각형 영역(ROI) 검출...")
    rois = find_top_rectangles(img_after_aligned, n=2)
    if not rois or len(rois) < 2: rois = select_multiple_rectangles_manually(img_after_aligned, n=2)
    if not rois: return

    # 4. 소형 노이즈 제거 전처리
    print("\n[3/8] 소형 노이즈 제거 전처리...")
    img_before, img_after_aligned = ask_user_noise_removal(img_before, img_after_aligned, rois)
    
    # 5. 이미지 추가 전처리
    print("\n[4/8] 이미지 대비 전처리...")
    img_before_proc, img_after_proc = preprocess_images(img_before, img_after_aligned)
    
    # 6. 인터랙티브 파라미터 최적화
    print("\n[5/8] 분석 파라미터 최적화 중...")
    config_file = "analysis_config.json"
    def load_cfg():
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f: return json.load(f)
            except: pass
        return {"diff_th": 25, "min_a": 5, "overlap_th": 0.3, "guard_percentage": 80.0}
    
    def save_cfg(c):
        with open(config_file, 'w') as f: json.dump(c, f)

    initial_cfg = load_cfg()
    
    def interactive_optimization(img_b, img_a, rois, cfg, p_info=""):
        from matplotlib.widgets import Slider
        analyzer = ChangeAnalyzer(diff_threshold=cfg['diff_th'], min_area=cfg['min_a'])
        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)
        plt.subplots_adjust(bottom=0.3)
        
        ax_b.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)); ax_b.set_title("Before (Reference)"); ax_b.axis('off')
        img_plot = ax_a.imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
        overlay = np.zeros((*img_a.shape[:2], 4), dtype=np.uint8); mark_plot = ax_a.imshow(overlay)
        ax_a.set_title(f"After (Optimization) | Prompts: {p_info}"); ax_a.axis('off')
        
        s_diff = Slider(plt.axes([0.15, 0.22, 0.7, 0.03]), 'Diff Thresh', 0, 255, valinit=cfg['diff_th'], valstep=1)
        s_min_a = Slider(plt.axes([0.15, 0.18, 0.7, 0.03]), 'Min Area', 1, 500, valinit=cfg['min_a'], valstep=1)
        s_overlap = Slider(plt.axes([0.15, 0.14, 0.7, 0.03]), 'Overlap Th', 0.0, 1.0, valinit=cfg['overlap_th'])
        s_guard = Slider(plt.axes([0.15, 0.10, 0.7, 0.03]), 'Guard %', 10.0, 100.0, valinit=cfg['guard_percentage'])
        
        res_cfg = cfg.copy()
        def update(val):
            analyzer.diff_threshold = int(s_diff.val)
            _, thresh = analyzer.get_difference_candidates(img_b, img_a)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
            mask_valid = np.zeros_like(thresh)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= int(s_min_a.val):
                    cx, cy = centroids[i]
                    for rect in rois:
                        if cv2.pointPolygonTest(rect, (float(cx), float(cy)), False) >= 0:
                            mask_valid[labels == i] = 255; break
            new_overlay = np.zeros((*img_a.shape[:2], 4), dtype=np.uint8)
            new_overlay[mask_valid > 0] = [255, 0, 0, 160]; mark_plot.set_data(new_overlay)
            res_cfg.update({"diff_th": int(s_diff.val), "min_a": int(s_min_a.val), "overlap_th": s_overlap.val, "guard_percentage": s_guard.val})
            fig.canvas.draw_idle()
        for s in [s_diff, s_min_a, s_overlap, s_guard]: s.on_changed(update)
        update(None); plt.show(); return res_cfg

    # 프롬프트 로드
    p, l = load_prompts(); p_count = f"P:{l.count(1)} / N:{l.count(0)}" if l else "None"
    opt_cfg = interactive_optimization(img_before_proc, img_after_proc, rois, initial_cfg, p_count)
    save_cfg(opt_cfg)

    # 7. 세그멘테이션 프롬프트 기설정 확인
    if not l:
        print("\n[6/8] 세그멘테이션 프롬프트 설정..."); p, l = collect_interactive_prompts(img_after_aligned)
        if p: save_prompts(p, l)

    # 8. 최종 마크 분석
    print("\n[7/8] 최종 마크 분석 중...")
    detector = MarkDetector(); analyzer = ChangeAnalyzer(diff_threshold=opt_cfg['diff_th'], min_area=opt_cfg['min_a'])
    _, thresh_map = analyzer.get_difference_candidates(img_before_proc, img_after_proc)
    num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(thresh_map)
    
    refined_new_marks = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < opt_cfg['min_a']: continue
        cand_pt = centroids[i]; is_inside, target_mask = False, np.zeros_like(thresh_map, dtype=np.uint8)
        for rect in rois:
            if cv2.pointPolygonTest(rect, (float(cand_pt[0]), float(cand_pt[1])), False) >= 0:
                cv2.drawContours(target_mask, [rect], -1, 255, -1); is_inside = True; break
        if not is_inside: continue
        
        raw_mask = (labels_map == i).astype(np.uint8) * 255
        if detector.predictor and p:
            masks = detector.get_masks_from_points(img_after_aligned, p + [cand_pt], l + [1])
            if masks:
                ai_seg = masks[0]['segmentation'].astype(np.uint8) * 255
                final_seg = cv2.bitwise_or(raw_mask, cv2.bitwise_and(ai_seg, thresh_map))
            else: final_seg = raw_mask
        else: final_seg = raw_mask
        
        final_seg = cv2.bitwise_and(final_seg, target_mask)
        if int(final_seg.sum()) > 0:
            refined_new_marks.append({'segmentation': final_seg > 0, 'area': int((final_seg > 0).sum())})

    # ROI 결과 판정
    roi_results = []
    for i, rect in enumerate(rois):
        sub = calculate_sub_rectangle(rect, opt_cfg['guard_percentage'])
        marks = [m for m in refined_new_marks if is_mask_in_roi(m, rect)]
        roi_results.append({'rect': rect, 'sub_rect': sub, 'is_pass': check_guard_zone(marks, sub), 'marks': marks})

    # 9. 최종 시각화
    cfg_disp = {**opt_cfg, "Prompts": f"P:{l.count(1)} / N:{l.count(0)}" if l else "Zero-shot"}
    visualize_dual_results(img_before, img_after_aligned, roi_results, cfg_disp)

if __name__ == "__main__":
    main()
