import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib  # 한글 폰트 지원
from processor import ImageAligner, MarkDetector, ChangeAnalyzer
from dual_roi_utils import (
    find_top_rectangles, calculate_sub_rectangle, is_mask_in_roi, 
    check_guard_zone, visualize_dual_results, select_multiple_rectangles_manually
)
from image_preprocessor import apply_equalize_hist, apply_clahe
import os
import json
import time

# =========================================================================
# 1. Active Learning 데이터베이스 관리 클래스
# =========================================================================
class ActiveLearningDB:
    def __init__(self, db_path="learning_db.json"):
        self.db_path = db_path
        self.data = self.load_db()

    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"  [경고] DB 로드 실패: {e}. 새로운 학습 데이터베이스를 시작합니다.")
        return {"mark_profiles": [], "noise_profiles": [], "roi_prompts": []}

    def save_db(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
            print(f"  [성공] 학습 데이터베이스 저장 완료: {self.db_path}")
            print(f"         (학습된 마크 프로필: {len(self.data['mark_profiles'])}개, 노이즈 프로필: {len(self.data['noise_profiles'])}개)")
        except Exception as e:
            print(f"  [오류] DB 저장 실패: {e}")

    def add_mark_profile(self, profile):
        # 중복 방지를 위한 간단한 필터링: 기존 프로필과 너무 똑같은 것은 제외
        for existing in self.data["mark_profiles"]:
            diff = np.linalg.norm(np.array(existing["lab_mean"]) - np.array(profile["lab_mean"]))
            if diff < 1.0 and abs(existing["circularity"] - profile["circularity"]) < 0.05:
                return  # 거의 동일한 특징은 추가하지 않음
        self.data["mark_profiles"].append(profile)

    def add_noise_profile(self, profile):
        for existing in self.data["noise_profiles"]:
            diff = np.linalg.norm(np.array(existing["lab_mean"]) - np.array(profile["lab_mean"]))
            if diff < 1.0 and abs(existing["circularity"] - profile["circularity"]) < 0.05:
                return
        self.data["noise_profiles"].append(profile)

    def save_roi_prompts(self, roi_index, normalized_points, labels):
        # 정합된 좌표계 대비 일반화된 프롬프트를 누적 저장 (동작 템플릿용)
        # 이전 기록이 있다면 덮어쓰고, 없으면 새로 추가
        updated = False
        for entry in self.data["roi_prompts"]:
            if entry.get("roi_index") == roi_index:
                entry["normalized_points"] = normalized_points
                entry["labels"] = labels
                updated = True
                break
        if not updated:
            self.data["roi_prompts"].append({
                "roi_index": roi_index,
                "normalized_points": normalized_points,
                "labels": labels
            })

# =========================================================================
# 2. 특징(Feature) 추출 클래스
# =========================================================================
class FeatureExtractor:
    @staticmethod
    def extract_features(img, mask):
        """
        마스크 영역 내부의 이미지에서 시각적(Lab/HSV 색상) 특징 및 형태학적(Circularity 등) 특징 추출
        """
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        coords = np.argwhere(mask_u8 > 0)
        if coords.size == 0:
            return None

        features = {}

        # 1. 색상 공간 변환
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 마스크 내 픽셀 추출
        pixels_lab = img_lab[mask_u8 > 0]
        pixels_hsv = img_hsv[mask_u8 > 0]

        # Lab 평균 및 표준편차
        lab_mean = np.mean(pixels_lab, axis=0)
        lab_std = np.std(pixels_lab, axis=0)
        features["lab_mean"] = lab_mean.tolist()
        features["lab_std"] = lab_std.tolist()

        # HSV 평균 및 표준편차
        hsv_mean = np.mean(pixels_hsv, axis=0)
        hsv_std = np.std(pixels_hsv, axis=0)
        features["hsv_mean"] = hsv_mean.tolist()
        features["hsv_std"] = hsv_std.tolist()

        # 2. 형태학적 분석
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(cnt))
            perimeter = float(cv2.arcLength(cnt, True))

            # Circularity (원형도)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
            features["circularity"] = min(circularity, 1.0)

            # Bounding Box 비율 (Aspect Ratio)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h) if h > 0 else 1.0
            features["aspect_ratio"] = aspect_ratio

            # Solidity (조밀도)
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 0 else 1.0
            features["solidity"] = solidity
            features["area"] = area
        else:
            features["circularity"] = 0.0
            features["aspect_ratio"] = 1.0
            features["solidity"] = 0.0
            features["area"] = 0.0

        return features

# =========================================================================
# 3. 특징 매칭 및 판별 엔진 클래스
# =========================================================================
class FeatureMatcher:
    def __init__(self, db):
        self.db = db

    def compute_distance(self, feat1, feat2):
        """두 피처 간의 가중 거리를 측정 (낮을수록 유사함)"""
        # 1. 색상 차이 (Lab 평균 공간에서의 유클리드 거리)
        lab1 = np.array(feat1["lab_mean"])
        lab2 = np.array(feat2["lab_mean"])
        color_dist = np.linalg.norm(lab1 - lab2)

        # 색상 거리를 적절하게 노멀라이즈 (보통 Lab 거리는 0~100 이상이 됨)
        norm_color_dist = color_dist / 35.0

        # 2. 형태학적 차이
        circ_diff = abs(feat1["circularity"] - feat2["circularity"])
        
        ar1 = feat1["aspect_ratio"]
        ar2 = feat2["aspect_ratio"]
        ar_diff = min(abs(np.log(ar1) - np.log(ar2)), 1.0) if (ar1 > 0 and ar2 > 0) else abs(ar1 - ar2)
        
        solid_diff = abs(feat1["solidity"] - feat2["solidity"])

        # 가중합 거리 도출 (색상 가중치: 1.5, 형태 가중치: 1.0/0.5)
        total_dist = (
            norm_color_dist * 1.5 +
            circ_diff * 1.0 +
            ar_diff * 0.8 +
            solid_diff * 0.5
        )
        return total_dist

    def score_candidate(self, cand_features):
        """
        후보 영역이 노이즈인지 마크인지 스코어링 (0.0: 완전 노이즈 ~ 1.0: 완전 마크)
        """
        mark_profiles = self.db.data["mark_profiles"]
        noise_profiles = self.db.data["noise_profiles"]

        if not mark_profiles:
            # 학습 데이터가 전혀 없는 초기 상태: 중립값 0.5 반환
            return 0.5, 999.0, 999.0

        # 마크 DB 중 가장 유사한 특징과의 최소 거리
        min_d_mark = min([self.compute_distance(cand_features, mp) for mp in mark_profiles])

        # 노이즈 DB 중 가장 유사한 특징과의 최소 거리
        if noise_profiles:
            min_d_noise = min([self.compute_distance(cand_features, np_prof) for np_prof in noise_profiles])
        else:
            min_d_noise = 2.0  # 노이즈 정보가 없을 때의 디폴트 안전마진

        # 마크에 가까우며 노이즈에서 멀수록 1.0에 접근
        score = min_d_noise / (min_d_mark + min_d_noise + 1e-6)
        return score, min_d_mark, min_d_noise

# =========================================================================
# 4. 이미지 전반 유사성 비교기
# =========================================================================
def find_most_similar_saved_config(img, rois, db):
    """
    현재 이미지의 패드 영역 히스토그램을 DB에 저장된 예전 히스토그램과 비교하여
    가장 높은 유사도를 가진 설정값을 찾아 반환합니다.
    """
    # 본 예제에서는 최적의 매칭을 탐색하는 구조적 뼈대를 제공합니다.
    # 만약 DB가 비어있다면 디폴트 설정을 리턴합니다.
    return None

# =========================================================================
# 5. 핵심 대화형 Active Learning 피드백 루프 (OpenCV GUI)
# =========================================================================
def run_active_learning_gui(img_before, img_after, detector, rois, diff_thresh_map, db, initial_cfg):
    """
    사용자로부터 실시간 마우스 클릭 피드백을 받아 마스크를 즉시 갱신하고
    수정이 완료되면 학습 DB에 특징 데이터와 프롬프트 좌표를 저장하는 GUI 제어 루프.
    """
    print("\n" + "-"*50)
    print("[지속적 학습] Active Learning 대화형 피드백 세션 시작")
    print("  * 마크 검출 결과를 보며 클릭 피드백을 줄 수 있습니다.")
    print("  * 마우스 좌클릭 : 미검출된 마크 추가 (Positive / 초록색 표시)")
    print("  * 마우스 우클릭 : 오검출된 노이즈 제거 (Negative / 파란색 표시)")
    print("  * [Space / Enter] : 피드백 반영하여 마스크 재추론")
    print("  * [u] : 직전 클릭 취소 (Undo)")
    print("  * [r] : 모든 클릭 초기화")
    print("  * [s] : 현재 상태를 확정하고 특징 및 프롬프트 학습 DB에 저장")
    print("  * [q] : 저장하지 않고 종료")
    print("-"*50)

    h, w = img_after.shape[:2]
    
    # ROI 인덱스별 클릭 포인트 및 라벨 관리
    # roi_clicks[roi_idx] = {"points": [(x,y), ...], "labels": [1, 0, ...], "is_manual": [True, False, ...]}
    roi_clicks = {i: {"points": [], "labels": [], "is_manual": []} for i in range(len(rois))}

    # DB에서 과거에 학습되어 전이된 포인트가 있다면 로딩
    transfer_points_loaded = False
    for entry in db.data.get("roi_prompts", []):
        roi_idx = entry.get("roi_index")
        if roi_idx in roi_clicks:
            norm_pts = entry.get("normalized_points", [])
            lbls = entry.get("labels", [])
            
            # 현재 ROI의 바운딩 박스를 기준으로 실제 픽셀 좌표로 변환
            if len(rois) > roi_idx:
                rect = rois[roi_idx]
                rx, ry, rw, rh = cv2.boundingRect(rect)
                for pt, l in zip(norm_pts, lbls):
                    px = int(pt[0] * rw + rx)
                    py = int(pt[1] * rh + ry)
                    roi_clicks[roi_idx]["points"].append((px, py))
                    roi_clicks[roi_idx]["labels"].append(l)
                    roi_clicks[roi_idx]["is_manual"].append(False) # DB 전이 포인트는 수동 클릭 아님 (화면 표시 배제)
                transfer_points_loaded = True

    if transfer_points_loaded:
        print("  [알림] 이전 학습 DB로부터 영역 기반 프롬프트 포인트가 자동으로 전이되었습니다.")

    # 마우스 콜백 이벤트 함수
    current_roi_idx = [0] # 가변 참조를 위한 리스트 구조
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            # 듀얼 화면 중 오른쪽 (After 결과) 영역을 클릭했을 때만 좌표 맵핑하여 접수
            if x < w:
                print("    [알림] 마크/노이즈 피드백은 오른쪽 'After' 이미지 영역을 클릭해 주세요.")
                return
                
            x_after = x - w
            y_after = y
            
            # 클릭 좌표가 어느 ROI에 해당하는지 파악
            target_roi = -1
            for idx, rect in enumerate(rois):
                if cv2.pointPolygonTest(rect, (float(x_after), float(y_after)), False) >= 0:
                    target_roi = idx
                    break
            
            if target_roi == -1:
                print("    [경고] 클릭한 위치가 검출된 패드 ROI 내부가 아닙니다.")
                return

            label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
            roi_clicks[target_roi]["points"].append((x_after, y_after))
            roi_clicks[target_roi]["labels"].append(label)
            roi_clicks[target_roi]["is_manual"].append(True) # 사용자가 직접 클릭한 포인트 마킹
            
            # 화면 즉시 임시 갱신
            draw_overlay()

    cv2.namedWindow("Active Learning Feedback", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Active Learning Feedback", on_mouse)

    current_masks = [] # 현재 세그멘테이션 결과 리스트

    def update_segmentation():
        nonlocal current_masks
        current_masks = []
        
        # 1. 차영상 기반 레이블링 후보군 중심점 수집
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(diff_thresh_map)
        anchors = [] # [(x, y, is_from_diff, label_idx)]
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= initial_cfg.get('min_a', 5):
                anchors.append((centroids[i][0], centroids[i][1], True, i))
                
        # 2. 사용자가 추가한 수동 Positive 포인트들도 앵커 리스트에 병합 (미검출 마크 완전 복원용)
        for roi_idx, clicks in roi_clicks.items():
            for pt, lbl, is_man in zip(clicks["points"], clicks["labels"], clicks["is_manual"]):
                if lbl == 1: # Positive 마크 추가점
                    # 기존 차영상 앵커와의 중복 확인 (반경 10픽셀 이내)
                    duplicated = False
                    for ax, ay, _, _ in anchors:
                        if np.linalg.norm(np.array([ax, ay]) - np.array(pt)) < 10.0:
                            duplicated = True
                            break
                    if not duplicated:
                        anchors.append((pt[0], pt[1], False, -1))

        # 임시 후보 마스크 리스트
        temp_masks = []
        
        for ax, ay, is_from_diff, label_idx in anchors:
            cand_pt = (ax, ay)
            
            # 해당 앵커가 위치한 ROI 탐색
            target_roi_idx = -1
            target_poly = None
            for r_idx, rect in enumerate(rois):
                if cv2.pointPolygonTest(rect, (float(ax), float(ay)), False) >= 0:
                    target_roi_idx = r_idx
                    target_poly = rect
                    break
            
            if target_roi_idx == -1:
                continue
                
            # 기본 마스크 형태 설정
            if is_from_diff and label_idx > 0:
                raw_mask = (labels_map == label_idx).astype(np.uint8) * 255
            else:
                raw_mask = np.zeros_like(diff_thresh_map, dtype=np.uint8)
                
            # 해당 ROI의 피드백 포인트 수집
            feedback_pts = roi_clicks[target_roi_idx]["points"]
            feedback_lbls = roi_clicks[target_roi_idx]["labels"]

            # SAM2를 이용한 정밀 세그멘테이션 재예측
            if detector.predictor:
                # 앵커 자체(1: Positive) 및 누적 피드백 포인트를 결합하여 프롬프트 구성
                input_pts = [cand_pt] + feedback_pts
                input_lbls = [1] + feedback_lbls
                
                # 좌표 중복 제거
                unique_pts = []
                unique_lbls = []
                seen = set()
                for pt, l in zip(input_pts, input_lbls):
                    pt_rounded = (int(pt[0]), int(pt[1]))
                    if pt_rounded not in seen:
                        seen.add(pt_rounded)
                        unique_pts.append(pt)
                        unique_lbls.append(l)
                
                masks = detector.get_masks_from_points(img_after, unique_pts, unique_lbls)
                if masks:
                    ai_seg = masks[0]['segmentation'].astype(np.uint8) * 255
                    if is_from_diff:
                        # 차영상 후보였던 경우: 차영상 영역과 비트와이즈 AND 연산 후 병합
                        final_seg = cv2.bitwise_or(raw_mask, cv2.bitwise_and(ai_seg, diff_thresh_map))
                    else:
                        # 사용자가 직접 추가한 마크인 경우: AI 모델의 예측을 그대로 활용
                        final_seg = ai_seg
                else:
                    final_seg = raw_mask
            else:
                final_seg = raw_mask

            # 검출 결과를 소속 ROI 영역 내부로 엄격히 제한
            roi_mask = np.zeros_like(diff_thresh_map, dtype=np.uint8)
            cv2.drawContours(roi_mask, [target_poly], -1, 255, -1)
            
            # [경계 노이즈 차단] 패드 테두리선 부근의 정합 오차 에지 노이즈를 배제하기 위해
            # ROI 마스크 영역을 안쪽으로 8픽셀 수축(Erosion)하여 차단합니다.
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            roi_mask_eroded = cv2.erode(roi_mask, kernel_erode, iterations=1)
            final_seg = cv2.bitwise_and(final_seg, roi_mask_eroded)

            if int(final_seg.sum()) > 0:
                mask_bool = final_seg > 0
                
                # 특징 추출 수행
                feat = FeatureExtractor.extract_features(img_after, mask_bool)
                if feat is not None:
                    temp_masks.append({
                        'segmentation': mask_bool,
                        'area': int(mask_bool.sum()),
                        'roi_idx': target_roi_idx,
                        'features': feat,
                        'is_from_diff': is_from_diff
                    })

        # 3. 실시간 DB 학습 수행 (Real-time DB Update)
        # 마크 프로필 업데이트
        for m in temp_masks:
            has_positive_feedback = False
            for pt, lbl, is_man in zip(roi_clicks[m['roi_idx']]["points"], roi_clicks[m['roi_idx']]["labels"], roi_clicks[m['roi_idx']]["is_manual"]):
                if lbl == 1 and is_man:
                    if m['segmentation'][int(pt[1]), int(pt[0])]:
                        has_positive_feedback = True
                        break
            
            # 수동으로 마크를 직접 추가했거나, 아직 학습 DB에 마크 프로필이 전혀 없다면 차영상 마크 자동 학습
            if has_positive_feedback or (m['is_from_diff'] and len(db.data["mark_profiles"]) == 0):
                db.add_mark_profile(m['features'])

        # 노이즈 프로필 업데이트
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < initial_cfg.get('min_a', 5):
                continue
            cand_pt = centroids[i]
            
            # 수동으로 노이즈라고 마크(우클릭, label=0, is_manual=True)한 곳 근처 후보 학습
            for roi_idx, clicks in roi_clicks.items():
                for pt, lbl, is_man in zip(clicks["points"], clicks["labels"], clicks["is_manual"]):
                    if lbl == 0 and is_man:
                        dist = np.linalg.norm(np.array(cand_pt) - np.array(pt))
                        if dist < 20.0:
                            raw_mask = (labels_map == i).astype(np.uint8) * 255
                            feat_noise = FeatureExtractor.extract_features(img_after, raw_mask > 0)
                            if feat_noise is not None:
                                db.add_noise_profile(feat_noise)

        # 각 ROI별 프롬프트 좌표 업데이트
        for roi_idx, clicks in roi_clicks.items():
            if clicks["points"]:
                rect = rois[roi_idx]
                rx, ry, rw, rh = cv2.boundingRect(rect)
                normalized_pts = []
                for pt in clicks["points"]:
                    nx = (pt[0] - rx) / rw if rw > 0 else 0.5
                    ny = (pt[1] - ry) / rh if rh > 0 else 0.5
                    normalized_pts.append((nx, ny))
                db.save_roi_prompts(roi_idx, normalized_pts, clicks["labels"])

        # 파일 즉시 저장 (실시간 반영)
        db.save_db()

        # 4. 업데이트 완료된 DB 기준으로 순수 특징 매칭만 평가하여 최종 출력 마스크 구성 (수동 우회 필터 제거!)
        matcher = FeatureMatcher(db)
        for m in temp_masks:
            score, d_mark, d_noise = matcher.score_candidate(m['features'])
            
            # [기하학적 형태 필터] 극단적인 가로/세로 비율을 가진 선형 에지 형태의 가짜 노이즈는 강제 제외
            aspect_ratio = m['features'].get("aspect_ratio", 1.0)
            circularity = m['features'].get("circularity", 1.0)
            if (aspect_ratio > 3.5 or aspect_ratio < 0.28) and circularity < 0.30:
                # 패드 경계선의 전형적인 얇고 긴 에지 노이즈는 스킵
                continue
                
            # 오직 업데이트된 DB의 매칭 스코어 조건(0.35 이상)만 통과해야 최종 마크로 채택!
            if score < 0.35 and len(db.data["mark_profiles"]) > 0:
                continue
                
            current_masks.append({
                'segmentation': m['segmentation'],
                'area': m['area'],
                'roi_idx': m['roi_idx'],
                'features': m['features'],
                'score': score
            })

        # 5. 수동 피드백 포인트 동기화 및 클리어
        # 이번 세션에서 새로 클릭한 수동 포인트들을 DB 파일로 영구 저장 완료했으므로,
        # 메모리 상의 수동 포인트(is_manual=True)는 모두 비워주고,
        # DB에 저장된 최신 포인트 목록을 전부 is_manual=False(DB 전이 포인트) 상태로 roi_clicks에 동기화 로드합니다.
        # 이렇게 하면 수동 클릭된 X 마킹이 화면에서는 지워지되, 백그라운드 추론용 프롬프트로는 계속 기여합니다.
        for r_idx in range(len(rois)):
            roi_clicks[r_idx] = {"points": [], "labels": [], "is_manual": []}
            
        for entry in db.data.get("roi_prompts", []):
            roi_idx = entry.get("roi_index")
            if roi_idx in roi_clicks:
                norm_pts = entry.get("normalized_points", [])
                lbls = entry.get("labels", [])
                
                if len(rois) > roi_idx:
                    rect = rois[roi_idx]
                    rx, ry, rw, rh = cv2.boundingRect(rect)
                    for pt, l in zip(norm_pts, lbls):
                        px = int(pt[0] * rw + rx)
                        py = int(pt[1] * rh + ry)
                        roi_clicks[roi_idx]["points"].append((px, py))
                        roi_clicks[roi_idx]["labels"].append(l)
                        roi_clicks[roi_idx]["is_manual"].append(False)

    def draw_overlay():
        # 원본 이미지 복사
        display_img = img_after.copy()
        
        # 1. 검출된 최종 마스크 영역 오버레이 (빨간색 반투명) 및 마크 스코어/원형 하이라이트 표시
        mask_overlay = np.zeros_like(display_img)
        for m in current_masks:
            mask_overlay[m['segmentation']] = [0, 0, 255] # Red BGR
            
            m_u8 = m['segmentation'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                
                # [추가] 최종 결과물처럼 마크를 감싸는 빨간색 외곽 원 그리기
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(display_img, (int(cx), int(cy)), int(radius) + 5, (0, 0, 255), 2) # 빨간색 두께 2 원형 라인
                
                x_m, y_m, w_m, h_m = cv2.boundingRect(cnt)
                
                # 스코어 백분율 텍스트
                score_pct = int(m['score'] * 100)
                is_db_active = len(db.data["mark_profiles"]) > 0
                if is_db_active:
                    lbl_text = f"DB:{score_pct}%"
                    color_txt = (0, 255, 0) # 초록색
                else:
                    lbl_text = f"New:{score_pct}%"
                    color_txt = (0, 255, 255) # 노란색
                    
                # 텍스트 가독성을 높이기 위한 블랙 배경 박스 추가
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                text_size, _ = cv2.getTextSize(lbl_text, font, font_scale, thickness)
                
                # 텍스트가 빨간색 외곽 원과 겹치지 않게 y 좌표 조정
                tx = x_m + (w_m - text_size[0]) // 2
                ty = max(int(cy - radius - 10), 15)
                
                cv2.rectangle(display_img, (tx - 2, ty - text_size[1] - 2), 
                              (tx + text_size[0] + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(display_img, lbl_text, (tx, ty), font, font_scale, color_txt, thickness, cv2.LINE_AA)
        
        display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, 0.5, 0)
        
        # 2. ROI 박스 및 가드존 라인 표시
        for i, rect in enumerate(rois):
            # ROI 경계 (녹색)
            cv2.polylines(display_img, [rect], True, (0, 255, 0), 2)
            # 가드존 경계 (노란색)
            sub = calculate_sub_rectangle(rect, initial_cfg.get('guard_percentage', 80.0))
            cv2.polylines(display_img, [sub], True, (0, 255, 255), 2)
            
            # ROI 인덱스 번호 텍스트
            x, y, w_box, h_box = cv2.boundingRect(rect)
            cv2.putText(display_img, f"Pad {i+1}", (x + 10, y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 3. 사용자 피드백 마우스 포인트 맵핑
        for roi_idx, clicks in roi_clicks.items():
            for pt, lbl, is_man in zip(clicks["points"], clicks["labels"], clicks["is_manual"]):
                # 사용자가 이번 세션에 직접 우클릭한 마크 지우기(Negative) 점만 파란색 X자로 화면에 표시
                if is_man and lbl == 0:
                    cv2.drawMarker(display_img, pt, (255, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
                    cv2.circle(display_img, pt, 8, (255, 255, 255), 1)

        # Before 이미지와 After 오버레이 이미지를 가로로 결합하여 띄우기
        # Before 영역에 텍스트 표시
        before_display = img_before.copy()
        cv2.putText(before_display, "Before (Reference)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(display_img, "After (Feedback Here)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        combined_img = np.hstack((before_display, display_img))
        cv2.imshow("Active Learning Feedback", combined_img)

    # 최초 세그멘테이션 계산 및 드로잉
    update_segmentation()
    draw_overlay()

    save_and_exit = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        
        # Space / Enter 키 누를 시 세그멘테이션 모델 재실행 및 화면 업데이트
        if key == 32 or key == 13:
            print("  > 피드백을 적용하여 SAM2 및 특징 필터링 모델을 재추론합니다...")
            update_segmentation()
            draw_overlay()
            
        # 'u' 키: 마지막 마우스 포인트 실행 취소 (Undo) - 수동 포인트만 제거
        elif key == ord('u'):
            undone = False
            for r_idx in reversed(range(len(rois))):
                clicks = roi_clicks[r_idx]
                manual_indices = [idx for idx, is_man in enumerate(clicks["is_manual"]) if is_man]
                if manual_indices:
                    last_idx = manual_indices[-1]
                    clicks["points"].pop(last_idx)
                    clicks["labels"].pop(last_idx)
                    clicks["is_manual"].pop(last_idx)
                    undone = True
                    break
            if undone:
                print("  > 마지막 피드백 입력을 취소했습니다.")
                draw_overlay()
                
        # 'r' 키: 수동 피드백 포인트만 리셋
        elif key == ord('r'):
            for r_idx in range(len(rois)):
                clicks = roi_clicks[r_idx]
                new_pts = [p for p, is_man in zip(clicks["points"], clicks["is_manual"]) if not is_man]
                new_lbls = [l for l, is_man in zip(clicks["labels"], clicks["is_manual"]) if not is_man]
                new_is_man = [False] * len(new_pts)
                clicks["points"] = new_pts
                clicks["labels"] = new_lbls
                clicks["is_manual"] = new_is_man
            print("  > 수집된 수동 피드백 포인트를 리셋했습니다.")
            update_segmentation()
            draw_overlay()
            
        # 's' 키: 저장 및 종료
        elif key == ord('s'):
            save_and_exit = True
            break
            
        # 'q' 키: 저장하지 않고 종료
        elif key == ord('q'):
            print("  [알림] 저장하지 않고 세션을 종료합니다.")
            break

    cv2.destroyWindow("Active Learning Feedback")

    if save_and_exit:
        print("\n[학습] 현재 세션의 마크/노이즈 특징 프로필 및 전이용 프롬프트 저장 중...")
        
        # 1. 최종 검출된 마크의 특징 데이터 누적
        for m in current_masks:
            if m['features'] is not None:
                db.add_mark_profile(m['features'])
                
        # 2. 사용자가 노이즈로 마킹했거나, 마크 스코어가 너무 낮아 배제된 영역 중 오검출 노이즈의 특징 데이터 누적
        # (이번 세션에서 우클릭 근처 영역에서 픽셀 차분으로 검출되었던 부근의 노이즈 특징 수집)
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(diff_thresh_map)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < initial_cfg.get('min_a', 5):
                continue
            cand_pt = centroids[i]
            
            # 만약 이 후보 영역 근처에 사용자가 Negative(우클릭, label=0) 마킹을 했다면 노이즈 프로필로 학습
            for roi_idx, clicks in roi_clicks.items():
                for pt, lbl in zip(clicks["points"], clicks["labels"]):
                    if lbl == 0:
                        dist = np.linalg.norm(np.array(cand_pt) - np.array(pt))
                        if dist < 20.0:  # 반경 20픽셀 이내 매칭
                            raw_mask = (labels_map == i).astype(np.uint8) * 255
                            # 해당 노이즈 영역의 특징 분석
                            feat_noise = FeatureExtractor.extract_features(img_after, raw_mask > 0)
                            if feat_noise is not None:
                                db.add_noise_profile(feat_noise)
                                print(f"    - 노이즈 특징 프로필 1개 추출 완료 (Area: {feat_noise['area']})")

        # 3. 각 ROI 기준 프롬프트 좌표를 정규화하여 다음 이미지 전이용 템플릿으로 저장
        for roi_idx, clicks in roi_clicks.items():
            if clicks["points"]:
                rect = rois[roi_idx]
                rx, ry, rw, rh = cv2.boundingRect(rect)
                normalized_pts = []
                for pt in clicks["points"]:
                    nx = (pt[0] - rx) / rw if rw > 0 else 0.5
                    ny = (pt[1] - ry) / rh if rh > 0 else 0.5
                    normalized_pts.append((nx, ny))
                db.save_roi_prompts(roi_idx, normalized_pts, clicks["labels"])
                
        db.save_db()
        print("  [학습 완료] 데이터베이스 업데이트가 성공적으로 마무리되었습니다.")

    return current_masks

# =========================================================================
# 6. Gabor 필터 로컬 적용 함수
# =========================================================================
def apply_gabor_filter_to_rois(img, rois, theta=np.pi/4, lam=5.0, ksize=31, sigma=4.0, gamma=0.5):
    img_out = img.copy()
    h, w = img.shape[:2]
    
    for rect in rois:
        x, y, w_box, h_box = cv2.boundingRect(rect)
        mask = np.zeros((h_box, w_box), dtype=np.uint8)
        local_rect = rect - [x, y]
        cv2.drawContours(mask, [local_rect], -1, 255, -1)
        
        sub_img = img[y:y+h_box, x:x+w_box]
        sub_gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        
        kh = min(ksize, h_box)
        kw = min(ksize, w_box)
        if kh % 2 == 0: kh -= 1
        if kw % 2 == 0: kw -= 1
        if kh <= 0 or kw <= 0:
            continue
            
        gabor_kernel = cv2.getGaborKernel((kh, kw), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
        
        f_transform = np.fft.fft2(sub_gray)
        f_shift = np.fft.fftshift(f_transform)
        
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
        
        image_normalized = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        filtered_bgr = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
        
        for c in range(3):
            img_out[y:y+h_box, x:x+w_box, c] = np.where(
                mask == 255,
                filtered_bgr[:, :, c],
                img_out[y:y+h_box, x:x+w_box, c]
            )
            
    return img_out

def visualize_dual_results_with_scores(img_b, img_a, res_list, config_info=None):
    """
    Matplotlib 결과 창에서 어떤 후보가 DB 학습(유사도 매칭) 기반으로 
    검출된 마크인지 명확하게 스코어 및 텍스트와 함께 시각화합니다.
    """
    import koreanize_matplotlib
    plt.figure(figsize=(18, 8))
    
    # 1. Contact Before
    plt.subplot(1, 3, 1)
    plt.title("Contact Before (전)")
    plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 2. ROI & Guard Zone
    plt.subplot(1, 3, 2)
    plt.title("ROI & Guard Zone (가드존 영역)")
    img_roi = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    for r in res_list:
        cv2.polylines(img_roi, [r['rect']], True, (0, 255, 0), 2)
        cv2.polylines(img_roi, [r['sub_rect']], True, (255, 255, 0), 2)
    plt.imshow(img_roi)
    plt.axis('off')
    
    # 3. Final Results
    plt.subplot(1, 3, 3)
    plt.title("Final Results (인식된 마크 & 매칭 유사도)")
    res_img = img_a.copy()
    
    scores_to_plot = []  # [(cx, cy, txt, color), ...]
    
    for r in res_list:
        for m in r['marks']:
            mask = m['segmentation'].astype(np.uint8)
            # 빨간색 채색
            res_img[mask > 0] = [0, 0, 255]
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 1: 
                    continue
                # 검출 마크에 원형 테두리 하이라이트
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(res_img, (int(cx), int(cy)), int(radius) + 5, (0, 0, 255), 1)
                
                # 스코어링 정보를 토대로 표시 텍스트 생성
                score = m.get('score', 0.5)
                score_pct = int(score * 100)
                
                # DB 학습 데이터 유무에 따라 텍스트 표시
                txt = f"DB:{score_pct}%"
                color = 'lime' if score > 0.5 else 'yellow'
                scores_to_plot.append((cx, cy - radius - 8, txt, color))
                
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    for r in res_list:
        cv2.polylines(res_img_rgb, [r['rect']], True, (0, 255, 0), 2)
        cv2.polylines(res_img_rgb, [r['sub_rect']], True, (255, 255, 0), 2)
    plt.imshow(res_img_rgb)
    
    # 유사도 스코어 플로팅
    for cx, cy, txt, color in scores_to_plot:
        plt.text(cx, cy, txt, color=color, fontsize=10, fontweight='bold', ha='center',
                 bbox={'facecolor': 'black', 'alpha': 0.6, 'pad': 2})
                 
    # PASS/FAIL 텍스트 강조
    for i, r in enumerate(res_list):
        x, y, w, h = cv2.boundingRect(r['rect'])
        c, s = ('lime', 'PASS') if r['is_pass'] else ('red', 'FAIL')
        plt.text(x + w//2, y - 50, f"ROI {i+1}: {s}", color=c, fontsize=15, 
                 fontweight='bold', ha='center', bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 3})
                 
    if config_info:
        info_text = " | ".join([f"{k}: {v}" for k, v in config_info.items()])
        plt.figtext(0.5, 0.02, f"[설정 파라미터 현황] {info_text}", ha="center", fontsize=12, 
                    bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
                    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# =========================================================================
# 7. 메인 실행 함수
# =========================================================================
def main():
    print("\n" + "="*60)
    print("Probe Card Contact Mark Analysis with Active Learning (main8.py)")
    print("="*60)
    
    # 학습 데이터베이스(learning_db.json) 초기화 선택 옵션 추가
    db_file = "learning_db.json"
    if os.path.exists(db_file):
        init_choice = input("  > 기존 학습 DB(learning_db.json)를 초기화하고 처음부터 새로 학습하시겠습니까? (y/n, 기본값 n): ").strip().lower()
        if init_choice == 'y':
            try:
                os.remove(db_file)
                print("  [성공] 기존 학습 DB 파일이 성공적으로 초기화되었습니다.")
            except Exception as e:
                print(f"  [경고] 학습 DB 파일 삭제에 실패했습니다: {e}")
    
    # 1. 사용자로부터 테스트 이미지 파일 세트 번호 입력받음 (검증 다양화)
    print("  [선택 가능한 이미지 조합]")
    print("    1 : before.jpg / after.jpg (기본 테스트 세트)")
    print("    2 : before_new1.jpg / after_new1.jpg (두 번째 테스트 세트)")
    print("    3 : before_new2.jpg / after_new2.jpg (세 번째 테스트 세트)")
    
    choice = input("  > 분석할 이미지 조합을 선택하세요 (1~3, 기본값 1): ").strip()
    if choice == "2":
        image_before_path, image_after_path = "before_new1.jpg", "after_new1.jpg"
    elif choice == "3":
        image_before_path, image_after_path = "before_new2.jpg", "after_new2.jpg"
    else:
        image_before_path, image_after_path = "before.jpg", "after.jpg"

    if os.path.exists(image_before_path) and os.path.exists(image_after_path):
        img_before = cv2.imread(image_before_path)
        img_after = cv2.imread(image_after_path)
        print(f"  [로딩 완료] {image_before_path} 및 {image_after_path}")
        
        # 로딩 확인을 위한 Matplotlib 가시화
        import matplotlib.pyplot as plt
        import koreanize_matplotlib
        
        print("\n[확인] 로드된 이미지 쌍을 확인해 주세요. 창을 닫으면 이미지 정합을 시작합니다.")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
        ax1.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Before (이전) 이미지: {image_before_path}", fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"After (이후) 이미지: {image_after_path}", fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        plt.suptitle("💡 이미지가 정상적으로 로드되었는지 확인하고, 이 창을 닫으십시오. (창을 닫으면 정합 및 분석 시작)", 
                     fontsize=12, fontweight='bold', color='darkred')
        plt.tight_layout()
        plt.show()
    else:
        print(f"  [오류] 지정된 파일({image_before_path} 또는 {image_after_path})이 존재하지 않습니다.")
        return

    # 2. 이미지 정합
    print("\n[1/8] 이미지 정합(Alignment) 처리 중...")
    img_after_aligned, _ = ImageAligner().align(img_before, img_after)

    # 3. Active Learning DB 초기화 및 로드
    db = ActiveLearningDB("learning_db.json")

    # 4. 설정 파라미터 로드
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
            "noise_threshold": 15,
            "noise_sensitivity": 40
        }
    
    config = load_full_config()

    # 5. ROI 검출 (분할 방향 선택)
    print("\n[2/8] 사각형 영역(ROI) 검출 및 분할...")
    while True:
        dir_choice = input("  > ROI 검출 분할 방향을 선택하세요 (1: 좌우(좌우 패드), 2: 상하(상하 패드), 기본값 1): ").strip()
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
        print("  [오류] ROI 영역을 검출하거나 선택하지 못했습니다.")
        return

    # 6. 최적 가보어 및 필터 전처리 파라미터 적용
    print("\n[3/8] 노이즈 제거 및 주파수 필터링(Gabor) 적용...")
    # 가보어 대역 차단 필터
    img_before_gabor = apply_gabor_filter_to_rois(img_before, rois, theta=config['gabor_theta'], lam=config['gabor_lam'])
    img_after_gabor = apply_gabor_filter_to_rois(img_after_aligned, rois, theta=config['gabor_theta'], lam=config['gabor_lam'])
    
    # 양방향 필터로 표면 그레인 완화
    img_before_bilateral = cv2.bilateralFilter(img_before_gabor, 9, 75, 75)
    img_after_bilateral = cv2.bilateralFilter(img_after_gabor, 9, 75, 75)

    # 7. 노이즈 제거 및 CLAHE 고대비 전처리
    print("\n[4/8] 소형 노이즈 제거 및 CLAHE 대비 개선...")
    from noise_remover import remove_small_noise, load_colors
    colors = load_colors()
    bg_color = colors.get("bg_color") if colors else None
    mark_color = colors.get("mark_color") if colors else None
    noise_th = config.get("noise_threshold", 15)
    noise_sens = config.get("noise_sensitivity", 40)
    
    img_before_clean, count_b = remove_small_noise(img_before_bilateral, noise_th, bg_color, mark_color, noise_sens)
    img_after_clean, count_a = remove_small_noise(img_after_bilateral, noise_th, bg_color, mark_color, noise_sens)
    print(f"  - 노이즈 제거 완료 (Before: {count_b}개 제거, After: {count_a}개 제거)")

    # CLAHE
    img_before_proc = apply_clahe(img_before_clean)
    img_after_proc = apply_clahe(img_after_clean)

    # 8. 차영상 계산 및 레이블 후보 도출
    print("\n[5/8] 차영상 맵 도출 및 후보 영역 계산...")
    analyzer = ChangeAnalyzer(diff_threshold=config['diff_th'], min_area=config['min_a'])
    _, diff_thresh_map = analyzer.get_difference_candidates(img_before_proc, img_after_proc)

    # 9. SAM2 모델 로드
    print("\n[6/8] SAM2 모델 로딩...")
    detector = MarkDetector()

    # 10. Active Learning 인터랙티브 세션 기동
    print("\n[7/8] 대화형 Active Learning 피드백 세션 구동...")
    # OpenCV 창을 통해 실시간 마우스 피드백을 수집 및 학습 (Before와 After 비교 시각화용 인수 추가)
    final_masks = run_active_learning_gui(img_before_clean, img_after_clean, detector, rois, diff_thresh_map, db, config)

    # 11. 최종 가드존 및 ROI 결과 판정
    print("\n[8/8] 가드존 위반 여부 최종 판정...")
    roi_results = []
    for i, rect in enumerate(rois):
        sub_rect = calculate_sub_rectangle(rect, config['guard_percentage'])
        # 해당 ROI에 포함되는 최종 검출 마스크 분류
        marks_in_roi = []
        for m in final_masks:
            if m['roi_idx'] == i:
                marks_in_roi.append(m)
                
        is_pass = check_guard_zone(marks_in_roi, sub_rect)
        roi_results.append({
            'rect': rect,
            'sub_rect': sub_rect,
            'is_pass': is_pass,
            'marks': marks_in_roi
        })
        print(f"  - ROI {i+1} 결과: {'PASS' if is_pass else 'FAIL'} (검출된 마크 수: {len(marks_in_roi)}개)")

    # 12. 결과 가시화 및 한글 폰트 적용
    print("\n[완료] 최종 시각화 출력을 생성합니다.")
    config_disp = {
        "차이임계값": config['diff_th'],
        "최소면적": config['min_a'],
        "가드비율": f"{config['guard_percentage']}%",
        "가보어각도": round(config['gabor_theta'], 3),
        "학습데이터": f"Mark {len(db.data['mark_profiles'])}개 / Noise {len(db.data['noise_profiles'])}개"
    }
    visualize_dual_results_with_scores(img_before, img_after_aligned, roi_results, config_disp)

if __name__ == "__main__":
    main()
