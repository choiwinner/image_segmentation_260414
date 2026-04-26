import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import koreanize_matplotlib

class ImageAligner:
    """전후 이미지의 미세한 위치 차이를 보정하는 클래스"""
    def __init__(self):
        # ECC 알고리즘을 위한 파라미터 설정
        self.number_of_iterations = 5000
        self.termination_eps = 1e-10
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                         self.number_of_iterations, self.termination_eps)

    def align(self, ref_img, target_img):
        """ref_img 기준으로 target_img를 정렬함"""
        # 그레이스케일 변환
        if len(ref_img.shape) == 3:
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_img
            target_gray = target_img

        # 변환 행렬 초기화 (유사 변환 방식 사용)
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            # ECC 변환 찾기
            (cc, warp_matrix) = cv2.findTransformECC(ref_gray, target_gray, warp_matrix, 
                                                    cv2.MOTION_EUCLIDEAN, self.criteria)
            # 이미지 워핑
            sz = ref_img.shape
            aligned_img = cv2.warpAffine(target_img, warp_matrix, (sz[1], sz[0]), 
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return aligned_img, warp_matrix
        except cv2.error as e:
            print(f"Alignment 실패: {e}. 원본 이미지를 그대로 반환합니다.")
            return target_img, np.eye(2, 3, dtype=np.float32)

class MarkDetector:
    """SAM2를 사용하여 Contact Mark를 검출하는 클래스"""
    def __init__(self, model_cfg="sam2.1_hiera_b+.yaml", checkpoint="sam2.1_hiera_base_plus.pt"):
        # 장치 선택: CUDA가 가능하면 사용, 아니면 CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # TF32 가속 활성화 (Ampere 이상의 GPU에서 효과적)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device("cpu")
            
        print(f"Using device: {self.device}")
        
        # 모델 로드
        try:
            # 설정 파일과 체크포인트의 절대 경로 확보
            import os
            base_path = os.path.dirname(os.path.abspath(__file__))
            full_model_cfg = os.path.join(base_path, model_cfg) if not os.path.isabs(model_cfg) else model_cfg
            full_checkpoint = os.path.join(base_path, checkpoint) if not os.path.isabs(checkpoint) else checkpoint
            
            # SAM2 모델 빌드 및 장치 할당
            self.sam2 = build_sam2(full_model_cfg, full_checkpoint, device=self.device, apply_postprocessing=True)
            self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)
            self.predictor = SAM2ImagePredictor(self.sam2)
        except Exception as e:
            print(f"SAM2 모델 로드 실패: {e}")
            self.mask_generator = None
            self.predictor = None

    def get_masks(self, image):
        if self.mask_generator is None:
            return []
        
        # SAM2는 RGB 이미지를 기대함
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # 추론 가속: GPU 환경에서는 autocast와 inference_mode 사용
        if self.device.type == "cuda":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks = self.mask_generator.generate(image_rgb)
        else:
            with torch.inference_mode():
                masks = self.mask_generator.generate(image_rgb)
                
        return masks

    def get_masks_from_points(self, image, points, labels=None):
        """특정 점(Point Prompt) 및 라벨(Positive/Negative)을 기반으로 마스크를 생성함"""
        if self.predictor is None:
            return []
            
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        self.predictor.set_image(image_rgb)
        
        input_points = np.array(points)
        if labels is None:
            input_labels = np.ones(len(points)) # 모두 Positive prompt
        else:
            input_labels = np.array(labels)
        
        if self.device.type == "cuda":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=False
                )
        else:
            with torch.inference_mode():
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=False
                )
        
        # 결과를 SAM2AutomaticMaskGenerator 형식과 유사하게 변환
        formatted_masks = []
        for i, mask in enumerate(masks):
            formatted_masks.append({
                'segmentation': mask,
                'area': np.sum(mask),
                'predicted_iou': scores[i],
                'point_coords': [points[i]]
            })
        return formatted_masks

class ChangeAnalyzer:
    """전후 마스크를 비교하여 새로운 마크를 찾는 클래스"""
    def __init__(self, iou_threshold=0.3, diff_threshold=30, min_area=20, 
                 overlap_threshold=0.8, size_ratio_threshold=0.2, 
                 strict_iou_threshold=0.7, duplicate_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.overlap_threshold = overlap_threshold
        self.size_ratio_threshold = size_ratio_threshold
        self.strict_iou_threshold = strict_iou_threshold
        self.duplicate_threshold = duplicate_threshold

    def compute_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0
        return intersection / union

    def find_new_marks(self, old_masks, new_masks):
        """
        old_masks: 이전 이미지에서 검출된 마스크 리스트
        new_masks: 이후 이미지에서 검출된 마스크 리스트
        반환: 새롭게 생성된 것으로 판단되는 마스크 리스트
        """
        new_mark_indices = []
        
        for i, n_mask in enumerate(new_masks):
            is_existing = False
            for o_mask in old_masks:
                iou = self.compute_iou(o_mask['segmentation'], n_mask['segmentation'])
                if iou > self.iou_threshold:
                    is_existing = True
                    break
            
            if not is_existing:
                new_mark_indices.append(i)
        
        return [new_masks[idx] for idx in new_mark_indices]

    def get_difference_candidates(self, img_before, img_after):
        """이미지 차분을 통해 변화가 발생한 후보 영역의 중심점들을 추출함 (CIELAB 색상 분석 적용)"""
        # 1. 색상 공간 변환 (CIELAB는 인간의 시각적 차이를 가장 잘 반영함)
        before_lab = cv2.cvtColor(img_before, cv2.COLOR_BGR2Lab)
        after_lab = cv2.cvtColor(img_after, cv2.COLOR_BGR2Lab)
        
        # 2. Lab 각 채널별 차이 계산
        diff_lab = cv2.absdiff(before_lab, after_lab)
        
        # 3. 유클리드 거리 근사 (L, a, b 차이의 가중 합산)
        # 밝기(L) 차이뿐만 아니라 색상(a, b) 차이를 충분히 반영
        l, a, b = cv2.split(diff_lab)
        
        # 각 채널의 기여도를 고려하여 통합된 델타 맵 생성
        # a, b 채널은 미세한 색상 변화를 잡기 위해 가중치를 높임
        diff_total = cv2.addWeighted(l, 0.5, a, 0.25, 0)
        diff_total = cv2.addWeighted(diff_total, 1.0, b, 0.25, 0)
        
        # 4. 적응형 임계값 또는 사용자 설정 임계값 적용
        _, thresh = cv2.threshold(diff_total, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # 5. 모폴로지 연산으로 작은 노이즈 제거 및 인접 영역 병합
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 6. 레이블링을 통해 후보 영역 추출
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        
        candidates = []
        for i in range(1, num_labels): # 0번은 배경
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                candidates.append(centroids[i])
                
        return candidates, thresh

    def find_new_marks_refined(self, img_before, img_after, old_masks, detector):
        """개선된 차분 기반 신규 마크 검출 로직"""
        # 1. 차분을 통한 후보지 추출 (차분 맵도 함께 확보)
        candidates, thresh_map = self.get_difference_candidates(img_before, img_after)
        if not candidates:
            return []
            
        # 2. 각 후보지에 대해 정밀 마스크 추출 (Prompt 기반)
        refined_new_marks = []
        
        for point in candidates:
            # 해당 점을 중심으로 SAM2 마스크 생성
            masks = detector.get_masks_from_points(img_after, [point])
            if not masks:
                continue
                
            n_mask = masks[0]
            n_seg = n_mask['segmentation']
            
            # 2.5 마스크를 '순수 변화 영역'으로 정제 (SAM2 마스크 ∩ 차분 맵)
            # 이렇게 하면 기존 마크와 겹치더라도 '새로 추가된 픽셀'만 빨간색으로 표시됨
            n_seg_refined = np.logical_and(n_seg, thresh_map > 0)
            refined_area = int(n_seg_refined.sum())
            
            # 정제 후 면적이 너무 작으면 (노이즈) 무시
            if refined_area < self.min_area:
                continue
            
            # 변화 비율이 10% 미만이면 기존 마크로 판정
            change_ratio = refined_area / n_mask['area'] if n_mask['area'] > 0 else 0
            if change_ratio < 0.10:
                continue
            
            # 마스크 데이터를 정제된 버전으로 교체 (이 단계가 핵심)
            n_mask = dict(n_mask)  # 원본 참조 보호를 위해 복사
            n_mask['segmentation'] = n_seg_refined
            n_mask['area'] = refined_area
            n_seg = n_seg_refined  # 이후 IoU 계산에도 정제 버전 사용
            
            # 3. 기존 마스크들과 비교 (기존 마스크와 겹치더라도 '새로운 영역'의 비율이 높으면 신규로 간주)
            is_new = True
            for o_mask in old_masks:
                o_seg = o_mask['segmentation']
                
                # IoU 계산
                intersection = np.logical_and(n_seg, o_seg).sum()
                union = np.logical_or(n_seg, o_seg).sum()
                iou = intersection / union if union > 0 else 0
                
                # 만약 기존 마스크와 거의 완벽하게 겹친다면(IoU가 매우 높음) 기존 마스크로 판정
                if iou > self.strict_iou_threshold: 
                    is_new = False
                    break
                
                # 단순히 겹치는 정도가 아니라, 새 마스크 면적의 대부분이 기존 마스크에 포함되는지 확인
                overlap_ratio = intersection / n_mask['area'] if n_mask['area'] > 0 else 0
                
                # 면적 비율 확인: 새 마스크와 기존 마스크의 크기가 비슷할 때만 동일 마크로 간주
                # (큰 배경 마스크 안에 작은 신규 마크가 들어온 경우를 허용하기 위함)
                size_ratio = n_mask['area'] / o_mask['area'] if o_mask['area'] > 0 else 0
                
                # 임계값 이상 겹치고, 크기 차이가 크지 않다면 기존 마크로 판단
                if overlap_ratio > self.overlap_threshold and size_ratio > self.size_ratio_threshold: 
                    is_new = False
                    break

            if is_new:
                # 중복 추가 방지 (이미 추가된 마스크와 많이 겹치는지 확인)
                duplicate = False
                for r_mask in refined_new_marks:
                    if self.compute_iou(n_seg, r_mask['segmentation']) > self.duplicate_threshold:
                        duplicate = True
                        break
                if not duplicate:
                    refined_new_marks.append(n_mask)
        
        return refined_new_marks

def visualize_results(img_before, img_after, new_marks, rect=None, sub_rect=None, is_pass=None):
    """결과 시각화"""
    plt.figure(figsize=(18, 6))
    
    # 1. Contact 전
    plt.subplot(1, 3, 1)
    plt.title("Contact 전")
    plt.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 2. Contact 후 (정합 완료)
    plt.subplot(1, 3, 2)
    plt.title("Contact 후 (정합 완료)")
    img_after_rgb = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
    
    # 사각형 표시 (실선 및 점선)
    if rect is not None:
        cv2.polylines(img_after_rgb, [rect], True, (0, 255, 0), 2) # Green Solid
    if sub_rect is not None:
        # OpenCV는 점선을 직접 지원하지 않으므로, matplotlib에서 그리거나 커스텀 루프로 구현
        # 여기서는 시각적 편의를 위해 다른 색상으로 표시하거나 matplotlib line style 사용
        cv2.polylines(img_after_rgb, [sub_rect], True, (255, 255, 0), 2) # Yellow (Dotted 대체 색상)
        
    plt.imshow(img_after_rgb)
    plt.axis('off')
    
    # 3. 검출된 새로운 Mark 및 판정
    plt.subplot(1, 3, 3)
    status_str = ""
    if is_pass is not None:
        status_str = f" [{ 'PASS' if is_pass else 'FAIL' }]"
    plt.title(f"검출된 새로운 Mark (Red){status_str}")
    
    res_img = img_after.copy()
    # 새로운 마크를 빨간색으로 오버레이
    for mask in new_marks:
        m = mask['segmentation'].astype(bool)
        res_img[m] = [0, 0, 255] # BGR format for Red
        
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    
    # 사각형 다시 표시 (결과창에도)
    if rect is not None:
        cv2.polylines(res_img_rgb, [rect], True, (0, 255, 0), 2)
    if sub_rect is not None:
        cv2.polylines(res_img_rgb, [sub_rect], True, (255, 255, 0), 2)
        
    plt.imshow(res_img_rgb)
    
    # PASS/FAIL 텍스트 강조
    if is_pass is not None:
        color = 'lime' if is_pass else 'red'
        plt.text(10, 30, 'PASS' if is_pass else 'FAIL', 
                 color=color, fontsize=20, fontweight='bold', 
                 bbox={'facecolor': 'black', 'alpha': 0.5})
                 
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def find_largest_rectangle(image):
    """이미지 내에서 가장 큰 직사각형(또는 4각형)을 찾음"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 노이즈 제거 및 이진화
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    best_rect = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: # 너무 작은 영역 제외
            continue
            
        # 윤곽선 근사화
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # 꼭짓점이 4개인 경우
        if len(approx) == 4:
            if area > max_area:
                max_area = area
                best_rect = approx
                
    return best_rect

def calculate_sub_rectangle(rect, length_percentage):
    """주어진 사각형의 길이 대비 특정 백분율을 가진 중심 기반 사각형 계산"""
    if rect is None:
        return None
        
    # rect coordinates: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    # 단순화를 위해 bounding box 기준으로 계산 후 비율 조정
    x, y, w, h = cv2.boundingRect(rect)
    center_x, center_y = x + w/2, y + h/2
    
    # 길이 비율에 따른 가로세로 스케일 팩터
    scale = length_percentage / 100.0
    
    new_w = w * scale
    new_h = h * scale
    
    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2
    
    sub_rect = np.array([
        [[int(new_x), int(new_y)]],
        [[int(new_x + new_w), int(new_y)]],
        [[int(new_x + new_w), int(new_y + new_h)]],
        [[int(new_x), int(new_y + new_h)]]
    ], dtype=np.int32)
    
    return sub_rect

def is_within_guard_zone(new_marks, sub_rect):
    """검출된 모든 마크가 가드 존(sub_rect) 내부에 있는지 확인"""
    if sub_rect is None or not new_marks:
        return True
        
    # sub_rect를 ROI 마스크로 변환하여 확인
    # bounding box 방식으로 체크 (성능 및 정확도 고려)
    x, y, w, h = cv2.boundingRect(sub_rect)
    x1, y1, x2, y2 = x, y, x + w, y + h
    
    for mark in new_marks:
        mask = mark['segmentation']
        # mask가 True인 픽셀들의 좌표 추출
        coords = np.argwhere(mask) # [y, x] 형식
        if coords.size == 0:
            continue
            
        # 모든 y좌표가 [y1, y2], x좌표가 [x1, x2] 사이에 있는지 확인
        if np.any(coords[:, 0] < y1) or np.any(coords[:, 0] > y2) or \
           np.any(coords[:, 1] < x1) or np.any(coords[:, 1] > x2):
            return False
            
    return True

def select_rectangle_manually(image):
    """사용자가 마우스로 4개의 꼭짓점을 클릭하여 사각형을 선택하도록 함"""
    clone = image.copy()
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select 4 Corners", clone)
                if len(points) > 1:
                    cv2.line(clone, points[-2], points[-1], (255, 0, 0), 2)
                    cv2.imshow("Select 4 Corners", clone)
                if len(points) == 4:
                    cv2.line(clone, points[3], points[0], (255, 0, 0), 2)
                    cv2.imshow("Select 4 Corners", clone)

    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", mouse_callback)
    cv2.imshow("Select 4 Corners", clone)
    
    print("-" * 40)
    print("이미지 창에서 사각형의 4개 꼭짓점을 클릭해 주세요 (순서대로).")
    print("4개 클릭 완료 후, **이미지 창을 선택한 상태에서** 아무 키나 누르면 설정이 완료됩니다.")
    print("-" * 40)
    cv2.waitKey(0)
    cv2.destroyWindow("Select 4 Corners")
    
    if len(points) == 4:
        # np.array 형식으로 변환 ([[x1,y1]], [[x2,y2]], ...)
        return np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    else:
        print("4개의 점이 정확히 선택되지 않았습니다.")
        return None
