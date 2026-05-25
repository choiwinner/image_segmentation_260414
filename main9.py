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
import csv
from datetime import datetime

# =========================================================================
# 1. Active Learning 데이터베이스 관리 클래스 (드리프트 히스토리 확장)
# =========================================================================
class ActiveLearningDB:
    def __init__(self, db_path="learning_db.json"):
        self.db_path = db_path
        self.data = self.load_db()

    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 드리프트 히스토리 키가 없으면 추가
                    if "drift_history" not in data:
                        data["drift_history"] = []
                    return data
            except Exception as e:
                print(f"  [경고] DB 로드 실패: {e}. 새로운 학습 데이터베이스를 시작합니다.")
        return {"mark_profiles": [], "noise_profiles": [], "roi_prompts": [], "drift_history": []}

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

    def add_drift_record(self, record):
        """드리프트 분석용 세션 기록 추가"""
        self.data["drift_history"].append(record)

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
# 4. 산업 PMI 수준 마크 정밀 계측 엔진 (MarkMetrology) [NEW in main9]
# =========================================================================
class MarkMetrology:
    """
    산업 PMI(Probe Mark Inspection) 수준의 마크 정밀 계측 엔진.
    각 마크에 대해 Tilted Rectangle Fitting, 패드 경계 거리, Overdrive 추정,
    마크 강도(Intensity), 등급 분류(Good/Marginal/Defective)를 수행합니다.
    """

    # 등급 판정 임계값 (산업 일반 기준, 프로브 카드 규격에 따라 조정 가능)
    EDGE_DIST_GOOD_RATIO = 0.10       # 패드 폭 대비 10% 이상이면 Good
    EDGE_DIST_MARGINAL_RATIO = 0.03   # 패드 폭 대비 3% 이상이면 Marginal
    OVERDRIVE_GOOD_MIN = 0.05         # 스크럽 장축/패드 폭 비율 최소 5%
    OVERDRIVE_GOOD_MAX = 0.35         # 스크럽 장축/패드 폭 비율 최대 35%
    OVERDRIVE_DEFECT_MAX = 0.50       # 50% 초과 시 과접촉 Defective
    OVERDRIVE_DEFECT_MIN = 0.02       # 2% 미만 시 미접촉 Defective
    CIRCULARITY_MIN_GOOD = 0.25       # 원형도 최소 기준

    @staticmethod
    def measure_mark(mask, pad_rect, img_before, img_after):
        """
        단일 마크에 대한 정밀 계측 수행.
        
        Args:
            mask: 마크의 세그멘테이션 마스크 (bool 또는 uint8)
            pad_rect: 소속 패드의 ROI 사각형 (np.array, shape=(4,1,2))
            img_before: Before 이미지 (BGR)
            img_after: After 이미지 (BGR)
            
        Returns:
            dict: 정밀 계측 결과 딕셔너리
        """
        result = {}
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        
        # 패드 바운딩 박스 기본 정보
        px, py, pw, ph = cv2.boundingRect(pad_rect)
        pad_cx = px + pw / 2.0
        pad_cy = py + ph / 2.0
        pad_ref_dim = min(pw, ph)  # 경계 거리 비율 계산을 위한 기준 치수
        
        # 윤곽선 추출
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1:
            return None

        # ---------------------------------------------------------------
        # (1) Tilted Rectangle Fitting (스크럽 마크 정밀 계측)
        # ---------------------------------------------------------------
        min_rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
        (mcx, mcy), (rect_w, rect_h), angle = min_rect
        
        # 장축/단축 결정 (OpenCV minAreaRect의 w/h는 방향에 따라 바뀔 수 있음)
        major_axis = max(rect_w, rect_h)
        minor_axis = min(rect_w, rect_h)
        
        # 스크럽 방향 각도 보정: 장축 방향이 실제 스크럽 방향
        if rect_w < rect_h:
            scrub_angle = angle + 90.0
        else:
            scrub_angle = angle
        # 각도를 -90 ~ +90 범위로 정규화
        while scrub_angle > 90:
            scrub_angle -= 180
        while scrub_angle < -90:
            scrub_angle += 180
        
        result['center'] = (float(mcx), float(mcy))
        result['major_axis'] = float(major_axis)
        result['minor_axis'] = float(minor_axis)
        result['scrub_angle'] = float(scrub_angle)
        result['area'] = float(cv2.contourArea(cnt))
        result['tilted_rect'] = min_rect  # 시각화용 원본 데이터
        
        # ---------------------------------------------------------------
        # (2) 패드 경계 거리 측정 (4변 최소 거리)
        # ---------------------------------------------------------------
        # 마크 윤곽선의 모든 점에서 패드 4변까지의 최소 거리 계산
        cnt_points = cnt.reshape(-1, 2)  # (N, 2) 형태
        
        # 패드 4변의 직선 방정식 계산 (좌/우/상/하)
        dist_left = float(np.min(cnt_points[:, 0] - px))
        dist_right = float(np.min((px + pw) - cnt_points[:, 0]))
        dist_top = float(np.min(cnt_points[:, 1] - py))
        dist_bottom = float(np.min((py + ph) - cnt_points[:, 1]))
        
        result['dist_left'] = max(dist_left, 0.0)
        result['dist_right'] = max(dist_right, 0.0)
        result['dist_top'] = max(dist_top, 0.0)
        result['dist_bottom'] = max(dist_bottom, 0.0)
        result['min_edge_dist'] = min(result['dist_left'], result['dist_right'],
                                       result['dist_top'], result['dist_bottom'])
        
        # ---------------------------------------------------------------
        # (3) 중심 오프셋 (패드 중심 대비 마크 중심의 편차)
        # ---------------------------------------------------------------
        offset_x = mcx - pad_cx
        offset_y = mcy - pad_cy
        result['offset_x'] = float(offset_x)
        result['offset_y'] = float(offset_y)
        
        # 정규화된 오프셋 (패드 폭/높이 대비 비율, -0.5 ~ +0.5)
        result['norm_offset_x'] = float(offset_x / pw) if pw > 0 else 0.0
        result['norm_offset_y'] = float(offset_y / ph) if ph > 0 else 0.0

        # ---------------------------------------------------------------
        # (4) Overdrive 추정 (스크럽 장축 길이 / 패드 폭 비율)
        # ---------------------------------------------------------------
        overdrive_ratio = major_axis / pad_ref_dim if pad_ref_dim > 0 else 0.0
        result['overdrive_ratio'] = float(overdrive_ratio)

        # ---------------------------------------------------------------
        # (5) 마크 깊이/강도 추정 (Before/After 간 CIELAB L* 채널 차이)
        # ---------------------------------------------------------------
        before_lab = cv2.cvtColor(img_before, cv2.COLOR_BGR2Lab)
        after_lab = cv2.cvtColor(img_after, cv2.COLOR_BGR2Lab)
        
        # 마크 영역 내의 L* 채널 평균값 추출
        l_before = before_lab[:, :, 0].astype(np.float32)
        l_after = after_lab[:, :, 0].astype(np.float32)
        
        mask_bool = mask_u8 > 0
        if np.any(mask_bool):
            mean_l_before = float(np.mean(l_before[mask_bool]))
            mean_l_after = float(np.mean(l_after[mask_bool]))
            delta_l = mean_l_after - mean_l_before  # 음수면 어두워진 것 (마크가 눌린 것)
        else:
            mean_l_before = 0.0
            mean_l_after = 0.0
            delta_l = 0.0
        
        result['intensity_L_before'] = mean_l_before
        result['intensity_L_after'] = mean_l_after
        result['intensity_delta_L'] = float(delta_l)

        # ---------------------------------------------------------------
        # (6) 등급 분류 (Good / Marginal / Defective)
        # ---------------------------------------------------------------
        grade, reasons = MarkMetrology._classify_grade(result, pad_ref_dim)
        result['grade'] = grade
        result['grade_reasons'] = reasons
        
        return result

    @staticmethod
    def _classify_grade(metrology, pad_ref_dim):
        """
        계측 데이터를 기반으로 마크를 3등급으로 분류합니다.
        
        Returns:
            tuple: (grade: str, reasons: list[str])
        """
        reasons = []
        
        min_dist = metrology['min_edge_dist']
        od_ratio = metrology['overdrive_ratio']
        major = metrology['major_axis']
        minor = metrology['minor_axis']
        
        # Defective 조건 체크 (하나라도 해당하면 Defective)
        is_defective = False
        
        # 패드 경계 근접 (패시베이션 손상 위험)
        edge_ratio = min_dist / pad_ref_dim if pad_ref_dim > 0 else 0
        if edge_ratio <= MarkMetrology.EDGE_DIST_MARGINAL_RATIO:
            is_defective = True
            reasons.append(f"경계 침범 위험: 최소 경계거리 {min_dist:.1f}px (패드 대비 {edge_ratio*100:.1f}%)")
        
        # 과접촉 (Punch-through 위험)
        if od_ratio > MarkMetrology.OVERDRIVE_DEFECT_MAX:
            is_defective = True
            reasons.append(f"과접촉(Punch-through): Overdrive 비율 {od_ratio*100:.1f}%")
        
        # 미접촉 (접촉 불량)
        if od_ratio < MarkMetrology.OVERDRIVE_DEFECT_MIN:
            is_defective = True
            reasons.append(f"미접촉(No Contact): Overdrive 비율 {od_ratio*100:.1f}%")
        
        if is_defective:
            return "Defective", reasons
        
        # Marginal 조건 체크
        is_marginal = False
        
        if edge_ratio <= MarkMetrology.EDGE_DIST_GOOD_RATIO:
            is_marginal = True
            reasons.append(f"경계 근접 주의: 최소 경계거리 {min_dist:.1f}px (패드 대비 {edge_ratio*100:.1f}%)")
        
        if od_ratio < MarkMetrology.OVERDRIVE_GOOD_MIN:
            is_marginal = True
            reasons.append(f"접촉 약함: Overdrive 비율 {od_ratio*100:.1f}% (권장 {MarkMetrology.OVERDRIVE_GOOD_MIN*100:.0f}%+)")
        
        if od_ratio > MarkMetrology.OVERDRIVE_GOOD_MAX:
            is_marginal = True
            reasons.append(f"접촉 과다 주의: Overdrive 비율 {od_ratio*100:.1f}% (권장 {MarkMetrology.OVERDRIVE_GOOD_MAX*100:.0f}% 이하)")
        
        if is_marginal:
            return "Marginal", reasons
        
        # Good 등급
        reasons.append(f"정상 접촉: 경계여유 {edge_ratio*100:.1f}%, Overdrive {od_ratio*100:.1f}%")
        return "Good", reasons

# =========================================================================
# 5. 마크 위치 드리프트 분석 클래스 [NEW in main9]
# =========================================================================
class DriftAnalyzer:
    """
    다수 세션에 걸쳐 마크 중심 위치의 드리프트(편차)를 
    통계적으로 추적 및 분석하는 SPC(Statistical Process Control) 분석기.
    """
    
    def __init__(self, db):
        self.db = db
    
    def record_session(self, roi_idx, metrology_list, image_name=""):
        """
        현재 세션의 마크 계측 결과를 드리프트 히스토리에 기록합니다.
        
        Args:
            roi_idx: ROI 인덱스
            metrology_list: 해당 ROI의 마크별 계측 결과 리스트
            image_name: 분석한 이미지 파일명
        """
        if not metrology_list:
            return
        
        # 마크들의 평균 중심 오프셋 계산
        offsets_x = [m['offset_x'] for m in metrology_list if m is not None]
        offsets_y = [m['offset_y'] for m in metrology_list if m is not None]
        
        if not offsets_x:
            return
            
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": image_name,
            "roi_index": roi_idx,
            "mark_count": len(metrology_list),
            "mean_offset_x": float(np.mean(offsets_x)),
            "mean_offset_y": float(np.mean(offsets_y)),
            "std_offset_x": float(np.std(offsets_x)) if len(offsets_x) > 1 else 0.0,
            "std_offset_y": float(np.std(offsets_y)) if len(offsets_y) > 1 else 0.0,
            "mean_overdrive": float(np.mean([m['overdrive_ratio'] for m in metrology_list if m])),
            "mean_scrub_angle": float(np.mean([m['scrub_angle'] for m in metrology_list if m])),
        }
        
        self.db.add_drift_record(record)
    
    def get_drift_stats(self, roi_idx):
        """
        특정 ROI의 누적 드리프트 통계를 반환합니다.
        
        Returns:
            dict: 통계 요약 딕셔너리
        """
        history = [r for r in self.db.data.get("drift_history", []) 
                    if r.get("roi_index") == roi_idx]
        
        if not history:
            return {
                'session_count': 0,
                'mean_offset_x': 0.0, 'mean_offset_y': 0.0,
                'std_offset_x': 0.0, 'std_offset_y': 0.0,
                'trend_direction': 'insufficient_data'
            }
        
        all_ox = [r['mean_offset_x'] for r in history]
        all_oy = [r['mean_offset_y'] for r in history]
        
        stats = {
            'session_count': len(history),
            'mean_offset_x': float(np.mean(all_ox)),
            'mean_offset_y': float(np.mean(all_oy)),
            'std_offset_x': float(np.std(all_ox)) if len(all_ox) > 1 else 0.0,
            'std_offset_y': float(np.std(all_oy)) if len(all_oy) > 1 else 0.0,
        }
        
        # 드리프트 방향 판정
        if len(history) >= 3:
            # 최근 3세션의 X/Y 이동 경향 분석
            recent_ox = [r['mean_offset_x'] for r in history[-3:]]
            recent_oy = [r['mean_offset_y'] for r in history[-3:]]
            
            dx = recent_ox[-1] - recent_ox[0]
            dy = recent_oy[-1] - recent_oy[0]
            
            threshold = 2.0  # 2px 이상 이동 시 드리프트로 판정
            directions = []
            if abs(dx) > threshold:
                directions.append('→우측' if dx > 0 else '←좌측')
            if abs(dy) > threshold:
                directions.append('↓하단' if dy > 0 else '↑상단')
            
            if directions:
                stats['trend_direction'] = 'drift_' + '+'.join(directions)
            else:
                stats['trend_direction'] = 'stable'
        else:
            stats['trend_direction'] = 'monitoring' if len(history) >= 2 else 'insufficient_data'
        
        return stats
    
    def plot_drift_chart(self, roi_idx, ax=None):
        """X/Y 오프셋 산점도 및 트렌드 차트를 시각화합니다."""
        history = [r for r in self.db.data.get("drift_history", []) 
                    if r.get("roi_index") == roi_idx]
        
        if not history:
            if ax:
                ax.text(0.5, 0.5, '드리프트 데이터 없음\n(2회 이상 분석 필요)', 
                        transform=ax.transAxes, ha='center', va='center', fontsize=11)
                ax.set_title(f'ROI {roi_idx+1} 드리프트 이력')
            return
        
        sessions = list(range(1, len(history) + 1))
        ox = [r['mean_offset_x'] for r in history]
        oy = [r['mean_offset_y'] for r in history]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(sessions, ox, 'b-o', label='X 오프셋', markersize=6)
        ax.plot(sessions, oy, 'r-s', label='Y 오프셋', markersize=6)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        
        # ±2σ 제어 한계선 표시 (SPC UCL/LCL)
        if len(sessions) >= 3:
            mean_ox, std_ox = np.mean(ox), np.std(ox)
            mean_oy, std_oy = np.mean(oy), np.std(oy)
            combined_std = max(std_ox, std_oy, 1.0)
            
            ax.axhline(y=2*combined_std, color='orange', linestyle=':', alpha=0.7, label=f'UCL (+2σ = {2*combined_std:.1f}px)')
            ax.axhline(y=-2*combined_std, color='orange', linestyle=':', alpha=0.7, label=f'LCL (-2σ = {-2*combined_std:.1f}px)')
        
        ax.set_xlabel('세션 번호')
        ax.set_ylabel('오프셋 (px)')
        ax.set_title(f'ROI {roi_idx+1} 마크 위치 드리프트 이력')
        ax.legend(fontsize=8, loc='center left')
        ax.grid(True, alpha=0.3)

# =========================================================================
# 6. 정량 보고서 자동 생성 클래스 [NEW in main9]
# =========================================================================
class ReportGenerator:
    """PMI 정량 보고서 자동 생성기 (CSV 파일 + 시각적 요약 보고서)"""
    
    @staticmethod
    def generate_csv(roi_results_with_metrology, output_dir="reports"):
        """
        마크별 상세 계측치를 CSV로 출력합니다.
        
        Args:
            roi_results_with_metrology: ROI별 결과 + 마크별 계측 데이터
            output_dir: 출력 디렉토리
            
        Returns:
            str: 생성된 CSV 파일 경로
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"pmi_report_{timestamp}.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 헤더 행
            writer.writerow([
                'ROI', '마크번호', '등급', '등급사유',
                '중심X', '중심Y', '장축(px)', '단축(px)', '스크럽각도(°)', '면적(px²)',
                '경계거리_좌(px)', '경계거리_우(px)', '경계거리_상(px)', '경계거리_하(px)', '최소경계거리(px)',
                '오프셋X(px)', '오프셋Y(px)', '정규오프셋X', '정규오프셋Y',
                'Overdrive비율(%)', '강도ΔL*',
                'ROI판정'
            ])
            
            for r in roi_results_with_metrology:
                roi_label = f"ROI {r['roi_index'] + 1}"
                roi_pass = 'PASS' if r['is_pass'] else 'FAIL'
                
                for j, met in enumerate(r.get('metrology', [])):
                    if met is None:
                        continue
                    writer.writerow([
                        roi_label, j + 1, met['grade'], '; '.join(met['grade_reasons']),
                        f"{met['center'][0]:.1f}", f"{met['center'][1]:.1f}",
                        f"{met['major_axis']:.1f}", f"{met['minor_axis']:.1f}",
                        f"{met['scrub_angle']:.1f}", f"{met['area']:.0f}",
                        f"{met['dist_left']:.1f}", f"{met['dist_right']:.1f}",
                        f"{met['dist_top']:.1f}", f"{met['dist_bottom']:.1f}",
                        f"{met['min_edge_dist']:.1f}",
                        f"{met['offset_x']:.1f}", f"{met['offset_y']:.1f}",
                        f"{met['norm_offset_x']:.3f}", f"{met['norm_offset_y']:.3f}",
                        f"{met['overdrive_ratio']*100:.1f}", f"{met['intensity_delta_L']:.1f}",
                        roi_pass
                    ])
        
        print(f"  [보고서] CSV 보고서 저장 완료: {csv_path}")
        return csv_path
    
    @staticmethod
    def generate_visual_report(img_before, img_after, roi_results_with_metrology, 
                               drift_analyzer, config_info=None):
        """
        6패널 종합 시각 보고서를 Matplotlib로 생성합니다.
        
        패널 구성:
          [1] Before 원본           [2] After 정합 + ROI 표시
          [3] 마크 오버레이 + Tilted Rect + 등급  [4] 패드 경계거리 시각화
          [5] 마크별 계측 요약 테이블       [6] 드리프트 차트
        """
        import koreanize_matplotlib
        from matplotlib.gridspec import GridSpec
        
        # 피규어 크기와 배경색 설정 (현대적인 연한 회색 대시보드 배경)
        fig = plt.figure(figsize=(24, 15), facecolor='#F8F9FA')
        fig.suptitle("🔬 PMI (Probe Mark Inspection) 정량 분석 보고서", 
                     fontsize=20, fontweight='bold', color='#1A252C', y=0.985)
        
        # 2행 3열 그리드스펙 정의 (여백 및 서브플롯 간 간격을 충분히 확보)
        gs = GridSpec(2, 3, figure=fig, left=0.03, right=0.97, bottom=0.06, top=0.90, wspace=0.20, hspace=0.25)
        
        # 이미지의 종횡비 계산 (세로형 이미지로 인한 레이아웃 찌그러짐 및 겹침 방지)
        h_img, w_img = img_after.shape[:2]
        img_aspect = h_img / w_img if w_img > 0 else 1.5
        
        # 등급별 색상 정의
        grade_colors = {
            'Good': (0, 220, 0),        # 녹색 (BGR)
            'Marginal': (0, 200, 255),   # 주황색/황색 (BGR)
            'Defective': (0, 0, 255)     # 빨간색 (BGR)
        }
        grade_colors_plt = {
            'Good': '#D4EDDA',       # 파스텔 연녹색
            'Marginal': '#FFF3CD',   # 파스텔 연황색
            'Defective': '#F8D7DA'   # 파스텔 연적색
        }
        
        def apply_card_style(ax, title, is_image=True):
            """서브플롯을 감싸는 현대적인 카드 스타일 디자인 적용"""
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color='#2C3E50')
            ax.patch.set_facecolor('#FFFFFF')
            
            # 부드러운 회색 테두리 설정
            for name, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('#E2E8F0')
                spine.set_linewidth(1.5)
                
            if is_image:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # 종횡비를 이미지와 매칭하여 세로로 긴 뷰포트 통일
                ax.set_box_aspect(img_aspect)
            else:
                # 차트 등 축 정보가 있는 경우 전체 가로 공간 활용
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#CBD5E1')
                ax.spines['bottom'].set_color('#CBD5E1')
                ax.tick_params(colors='#475569', labelsize=10)
                ax.grid(True, linestyle='--', alpha=0.5, color='#F1F5F9')

        # --- [패널 1] Before 원본 ---
        ax1 = fig.add_subplot(gs[0, 0])
        apply_card_style(ax1, "① Before (접촉 전)", is_image=True)
        ax1.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
        
        # --- [패널 2] After + ROI ---
        ax2 = fig.add_subplot(gs[0, 1])
        apply_card_style(ax2, "② After (접촉 후) + ROI", is_image=True)
        img_roi = cv2.cvtColor(img_after.copy(), cv2.COLOR_BGR2RGB)
        for r in roi_results_with_metrology:
            cv2.polylines(img_roi, [r['rect']], True, (0, 255, 0), 2)
            cv2.polylines(img_roi, [r['sub_rect']], True, (255, 255, 0), 2)
        ax2.imshow(img_roi)
        
        # --- [패널 3] 마크 검출 + 정밀 계측 ---
        ax3 = fig.add_subplot(gs[0, 2])
        apply_card_style(ax3, "③ 마크 검출 + 정밀 계측 (Tilted Rectangle)", is_image=True)
        res_img = img_after.copy()
        
        for r in roi_results_with_metrology:
            # ROI 및 가드존 라인
            cv2.polylines(res_img, [r['rect']], True, (0, 255, 0), 2)
            cv2.polylines(res_img, [r['sub_rect']], True, (0, 255, 255), 1)
            
            for j, (m, met) in enumerate(zip(r['marks'], r.get('metrology', []))):
                if met is None:
                    continue
                
                # 마크 영역 등급별 색상 오버레이
                grade_color = grade_colors.get(met['grade'], (255, 255, 255))
                mask_bool = m['segmentation'].astype(bool)
                overlay = res_img.copy()
                overlay[mask_bool] = grade_color
                res_img = cv2.addWeighted(res_img, 0.6, overlay, 0.4, 0)
                
                # Tilted Rectangle 그리기
                box = cv2.boxPoints(met['tilted_rect'])
                box = np.intp(box)
                cv2.drawContours(res_img, [box], 0, grade_color, 2)
                
                # 장축/단축 라인 그리기
                center = met['center']
                angle_rad = np.radians(met['scrub_angle'])
                half_major = met['major_axis'] / 2.0
                dx = half_major * np.cos(angle_rad)
                dy = half_major * np.sin(angle_rad)
                pt1 = (int(center[0] - dx), int(center[1] - dy))
                pt2 = (int(center[0] + dx), int(center[1] + dy))
                cv2.line(res_img, pt1, pt2, (255, 255, 0), 1, cv2.LINE_AA)
                
                # 마크 번호 + 등급 텍스트
                tx = int(center[0])
                ty = max(int(center[1] - met['major_axis']/2 - 15), 15)
                label = f"#{j+1} {met['grade']}"
                cv2.putText(res_img, label, (tx - 30, ty), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, grade_color, 1, cv2.LINE_AA)
            
            # ROI 판정 표시
            x_r, y_r, w_r, h_r = cv2.boundingRect(r['rect'])
            pass_txt = 'PASS' if r['is_pass'] else 'FAIL'
            pass_col = (0, 255, 0) if r['is_pass'] else (0, 0, 255)
            cv2.putText(res_img, f"ROI {r['roi_index']+1}: {pass_txt}", 
                       (x_r + 5, y_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pass_col, 2, cv2.LINE_AA)
        
        ax3.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        
        # --- [패널 4] 경계거리 히트맵 시각화 ---
        ax4 = fig.add_subplot(gs[1, 0])
        apply_card_style(ax4, "④ 패드 경계 거리 분석", is_image=True)
        
        dist_img = img_after.copy()
        for r in roi_results_with_metrology:
            cv2.polylines(dist_img, [r['rect']], True, (0, 255, 0), 2)
            
            for j, met in enumerate(r.get('metrology', [])):
                if met is None:
                    continue
                cx, cy = int(met['center'][0]), int(met['center'][1])
                px_r, py_r, pw_r, ph_r = cv2.boundingRect(r['rect'])
                
                # 4방향 화살표로 최소 경계거리 표시 (텍스트를 화살표 중간 지점으로 분산 배치하여 겹침 방지)
                # 좌측
                cv2.arrowedLine(dist_img, (cx, cy), (px_r, cy), (0, 0, 255), 1, tipLength=0.05)
                mid_x = (cx + px_r) // 2
                cv2.putText(dist_img, f"{met['dist_left']:.0f}", 
                           (mid_x - 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
                # 우측
                cv2.arrowedLine(dist_img, (cx, cy), (px_r + pw_r, cy), (0, 0, 255), 1, tipLength=0.05)
                mid_x = (cx + px_r + pw_r) // 2
                cv2.putText(dist_img, f"{met['dist_right']:.0f}", 
                           (mid_x - 10, cy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
                # 상단
                cv2.arrowedLine(dist_img, (cx, cy), (cx, py_r), (0, 255, 0), 1, tipLength=0.05)
                mid_y = (cy + py_r) // 2
                cv2.putText(dist_img, f"{met['dist_top']:.0f}", 
                           (cx + 6, mid_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
                # 하단
                cv2.arrowedLine(dist_img, (cx, cy), (cx, py_r + ph_r), (0, 255, 0), 1, tipLength=0.05)
                mid_y = (cy + py_r + ph_r) // 2
                cv2.putText(dist_img, f"{met['dist_bottom']:.0f}", 
                           (cx - 24, mid_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 최소 경계거리 강조 텍스트
                min_dist = met['min_edge_dist']
                min_color = (0, 255, 0) if met['grade'] == 'Good' else ((0, 165, 255) if met['grade'] == 'Marginal' else (0, 0, 255))
                cv2.putText(dist_img, f"Min:{min_dist:.0f}px", 
                           (cx - 25, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, min_color, 1, cv2.LINE_AA)
        
        ax4.imshow(cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB))
        
        # --- [패널 5] 마크별 계측 요약 테이블 ---
        ax5 = fig.add_subplot(gs[1, 1])
        apply_card_style(ax5, "⑤ 마크 계측 요약 테이블", is_image=True)
        ax5.set_box_aspect(None) # 종횡비 제한 해제
        ax5.axis('off') # 불필요한 테두리 및 눈금 숨김
        
        # 데이터를 전치(transpose)하여 세로 공간을 알차게 활용
        row_labels = ['마크', '등급', '장축 (px)', '단축 (px)', '각도 (°)', '최소경계 (px)', 'OD비율 (%)', 'ΔL*']
        # 속성 레이블을 첫 열로 포함하여 표 전체 구조를 2차원 리스트로 구성
        transposed_data = [[label] for label in row_labels]
        table_colors = [['#1E293B'] for _ in range(8)]
        
        has_data = False
        for r in roi_results_with_metrology:
            for j, met in enumerate(r.get('metrology', [])):
                if met is None:
                    continue
                has_data = True
                transposed_data[0].append(f"ROI{r['roi_index']+1}-#{j+1}")
                transposed_data[1].append(met['grade'])
                transposed_data[2].append(f"{met['major_axis']:.1f}")
                transposed_data[3].append(f"{met['minor_axis']:.1f}")
                transposed_data[4].append(f"{met['scrub_angle']:.1f}")
                transposed_data[5].append(f"{met['min_edge_dist']:.1f}")
                transposed_data[6].append(f"{met['overdrive_ratio']*100:.1f}")
                transposed_data[7].append(f"{met['intensity_delta_L']:.1f}")
                
                # 등급 셀 하이라이트
                g_color = grade_colors_plt.get(met['grade'], '#FFFFFF')
                table_colors[0].append('#F8FAFC')
                table_colors[1].append(g_color)
                for i in range(2, 8):
                    table_colors[i].append('#FFFFFF')
        
        if has_data:
            # bbox를 이용해 5번 영역(100%)을 빈틈없이 채우도록 배치
            table = ax5.table(cellText=transposed_data, cellColours=table_colors, 
                             loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            
            # 셀 테두리 및 헤더 서식 설정
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('#E2E8F0')
                if col == 0: # 첫 번째 열 (속성명 레이블)
                    cell.set_text_props(color='white', fontweight='bold', fontsize=12)
                elif row == 0 and col > 0: # 최상단 마크명 레이블
                    cell.set_text_props(fontweight='bold', color='#1A252C')
        else:
            ax5.text(0.5, 0.5, '검출된 마크 없음', transform=ax5.transAxes, 
                    ha='center', va='center', fontsize=13, color='#64748B', fontweight='bold')
        
        # --- [패널 6] 드리프트 차트 ---
        ax6 = fig.add_subplot(gs[1, 2])
        if drift_analyzer and roi_results_with_metrology:
            first_roi_idx = roi_results_with_metrology[0]['roi_index']
            drift_analyzer.plot_drift_chart(first_roi_idx, ax=ax6)
            apply_card_style(ax6, f"⑥ ROI {first_roi_idx+1} 마크 위치 드리프트 이력", is_image=False)
        else:
            apply_card_style(ax6, "⑥ 드리프트 분석 이력", is_image=False)
            ax6.text(0.5, 0.5, '드리프트 데이터 없음\n(2회 이상 분석 필요)', 
                    transform=ax6.transAxes, ha='center', va='center', fontsize=11, color='#64748B', fontweight='bold')
        
        # 하단 설정 정보 표시
        if config_info:
            info_text = " | ".join([f"{k}: {v}" for k, v in config_info.items()])
            plt.figtext(0.04, 0.02, f"[설정 파라미터] {info_text}", ha="left", fontsize=9.5, color='#475569',
                       bbox={"facecolor":"#FFFFFF", "edgecolor":"#E2E8F0", "alpha":0.95, "pad":6, "boxstyle":"round,pad=0.6"})
        
        plt.show()

# =========================================================================
# 7. 이미지 전반 유사성 비교기
# =========================================================================
def find_most_similar_saved_config(img, rois, db):
    return None

# =========================================================================
# 8. 핵심 대화형 Active Learning 피드백 루프 (OpenCV GUI)
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
    roi_clicks = {i: {"points": [], "labels": [], "is_manual": []} for i in range(len(rois))}

    # DB에서 과거에 학습되어 전이된 포인트가 있다면 로딩
    transfer_points_loaded = False
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
                transfer_points_loaded = True

    if transfer_points_loaded:
        print("  [알림] 이전 학습 DB로부터 영역 기반 프롬프트 포인트가 자동으로 전이되었습니다.")

    # 마우스 콜백 이벤트 함수
    current_roi_idx = [0]
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            if x < w:
                print("    [알림] 마크/노이즈 피드백은 오른쪽 'After' 이미지 영역을 클릭해 주세요.")
                return
                
            x_after = x - w
            y_after = y
            
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
            roi_clicks[target_roi]["is_manual"].append(True)
            
            draw_overlay()

    cv2.namedWindow("Active Learning Feedback", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Active Learning Feedback", on_mouse)

    current_masks = []

    def update_segmentation():
        nonlocal current_masks
        current_masks = []
        
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(diff_thresh_map)
        anchors = []
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= initial_cfg.get('min_a', 5):
                anchors.append((centroids[i][0], centroids[i][1], True, i))
                
        for roi_idx, clicks in roi_clicks.items():
            for pt, lbl, is_man in zip(clicks["points"], clicks["labels"], clicks["is_manual"]):
                if lbl == 1:
                    duplicated = False
                    for ax, ay, _, _ in anchors:
                        if np.linalg.norm(np.array([ax, ay]) - np.array(pt)) < 10.0:
                            duplicated = True
                            break
                    if not duplicated:
                        anchors.append((pt[0], pt[1], False, -1))

        temp_masks = []
        
        for ax, ay, is_from_diff, label_idx in anchors:
            cand_pt = (ax, ay)
            
            target_roi_idx = -1
            target_poly = None
            for r_idx, rect in enumerate(rois):
                if cv2.pointPolygonTest(rect, (float(ax), float(ay)), False) >= 0:
                    target_roi_idx = r_idx
                    target_poly = rect
                    break
            
            if target_roi_idx == -1:
                continue
                
            if is_from_diff and label_idx > 0:
                raw_mask = (labels_map == label_idx).astype(np.uint8) * 255
            else:
                raw_mask = np.zeros_like(diff_thresh_map, dtype=np.uint8)
                
            feedback_pts = roi_clicks[target_roi_idx]["points"]
            feedback_lbls = roi_clicks[target_roi_idx]["labels"]

            if detector.predictor:
                input_pts = [cand_pt] + feedback_pts
                input_lbls = [1] + feedback_lbls
                
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
                        final_seg = cv2.bitwise_or(raw_mask, cv2.bitwise_and(ai_seg, diff_thresh_map))
                    else:
                        final_seg = ai_seg
                else:
                    final_seg = raw_mask
            else:
                final_seg = raw_mask

            roi_mask = np.zeros_like(diff_thresh_map, dtype=np.uint8)
            cv2.drawContours(roi_mask, [target_poly], -1, 255, -1)
            
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            roi_mask_eroded = cv2.erode(roi_mask, kernel_erode, iterations=1)
            final_seg = cv2.bitwise_and(final_seg, roi_mask_eroded)

            if int(final_seg.sum()) > 0:
                mask_bool = final_seg > 0
                
                feat = FeatureExtractor.extract_features(img_after, mask_bool)
                if feat is not None:
                    temp_masks.append({
                        'segmentation': mask_bool,
                        'area': int(mask_bool.sum()),
                        'roi_idx': target_roi_idx,
                        'features': feat,
                        'is_from_diff': is_from_diff
                    })

        # 실시간 DB 학습 수행
        for m in temp_masks:
            has_positive_feedback = False
            for pt, lbl, is_man in zip(roi_clicks[m['roi_idx']]["points"], roi_clicks[m['roi_idx']]["labels"], roi_clicks[m['roi_idx']]["is_manual"]):
                if lbl == 1 and is_man:
                    if m['segmentation'][int(pt[1]), int(pt[0])]:
                        has_positive_feedback = True
                        break
            
            if has_positive_feedback or (m['is_from_diff'] and len(db.data["mark_profiles"]) == 0):
                db.add_mark_profile(m['features'])

        # 노이즈 프로필 업데이트
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < initial_cfg.get('min_a', 5):
                continue
            cand_pt = centroids[i]
            
            for roi_idx, clicks in roi_clicks.items():
                for pt, lbl, is_man in zip(clicks["points"], clicks["labels"], clicks["is_manual"]):
                    if lbl == 0 and is_man:
                        dist = np.linalg.norm(np.array(cand_pt) - np.array(pt))
                        if dist < 20.0:
                            raw_mask = (labels_map == i).astype(np.uint8) * 255
                            feat_noise = FeatureExtractor.extract_features(img_after, raw_mask > 0)
                            if feat_noise is not None:
                                db.add_noise_profile(feat_noise)

        # ROI별 프롬프트 좌표 업데이트
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

        # 업데이트된 DB 기준으로 순수 특징 매칭만 평가
        matcher = FeatureMatcher(db)
        for m in temp_masks:
            score, d_mark, d_noise = matcher.score_candidate(m['features'])
            
            aspect_ratio = m['features'].get("aspect_ratio", 1.0)
            circularity = m['features'].get("circularity", 1.0)
            if (aspect_ratio > 3.5 or aspect_ratio < 0.28) and circularity < 0.30:
                continue
                
            if score < 0.35 and len(db.data["mark_profiles"]) > 0:
                continue
                
            current_masks.append({
                'segmentation': m['segmentation'],
                'area': m['area'],
                'roi_idx': m['roi_idx'],
                'features': m['features'],
                'score': score
            })

        # 수동 피드백 포인트 동기화 및 클리어
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
                        px_val = int(pt[0] * rw + rx)
                        py_val = int(pt[1] * rh + ry)
                        roi_clicks[roi_idx]["points"].append((px_val, py_val))
                        roi_clicks[roi_idx]["labels"].append(l)
                        roi_clicks[roi_idx]["is_manual"].append(False)

    def draw_overlay():
        display_img = img_after.copy()
        
        mask_overlay = np.zeros_like(display_img)
        for m in current_masks:
            mask_overlay[m['segmentation']] = [0, 0, 255]
            
            m_u8 = m['segmentation'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(display_img, (int(cx), int(cy)), int(radius) + 5, (0, 0, 255), 2)
                
                x_m, y_m, w_m, h_m = cv2.boundingRect(cnt)
                
                score_pct = int(m['score'] * 100)
                is_db_active = len(db.data["mark_profiles"]) > 0
                if is_db_active:
                    lbl_text = f"DB:{score_pct}%"
                    color_txt = (0, 255, 0)
                else:
                    lbl_text = f"New:{score_pct}%"
                    color_txt = (0, 255, 255)
                    
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                text_size, _ = cv2.getTextSize(lbl_text, font, font_scale, thickness)
                
                tx = x_m + (w_m - text_size[0]) // 2
                ty = max(int(cy - radius - 10), 15)
                
                cv2.rectangle(display_img, (tx - 2, ty - text_size[1] - 2), 
                              (tx + text_size[0] + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(display_img, lbl_text, (tx, ty), font, font_scale, color_txt, thickness, cv2.LINE_AA)
        
        display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, 0.5, 0)
        
        for i, rect in enumerate(rois):
            cv2.polylines(display_img, [rect], True, (0, 255, 0), 2)
            sub = calculate_sub_rectangle(rect, initial_cfg.get('guard_percentage', 80.0))
            cv2.polylines(display_img, [sub], True, (0, 255, 255), 2)
            
            x, y, w_box, h_box = cv2.boundingRect(rect)
            cv2.putText(display_img, f"Pad {i+1}", (x + 10, y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        for roi_idx, clicks in roi_clicks.items():
            for pt, lbl, is_man in zip(clicks["points"], clicks["labels"], clicks["is_manual"]):
                if is_man and lbl == 0:
                    cv2.drawMarker(display_img, pt, (255, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
                    cv2.circle(display_img, pt, 8, (255, 255, 255), 1)

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
        
        if key == 32 or key == 13:
            print("  > 피드백을 적용하여 SAM2 및 특징 필터링 모델을 재추론합니다...")
            update_segmentation()
            draw_overlay()
            
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
            
        elif key == ord('s'):
            save_and_exit = True
            break
            
        elif key == ord('q'):
            print("  [알림] 저장하지 않고 세션을 종료합니다.")
            break

    cv2.destroyWindow("Active Learning Feedback")

    if save_and_exit:
        print("\n[학습] 현재 세션의 마크/노이즈 특징 프로필 및 전이용 프롬프트 저장 중...")
        
        for m in current_masks:
            if m['features'] is not None:
                db.add_mark_profile(m['features'])
                
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(diff_thresh_map)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < initial_cfg.get('min_a', 5):
                continue
            cand_pt = centroids[i]
            
            for roi_idx, clicks in roi_clicks.items():
                for pt, lbl in zip(clicks["points"], clicks["labels"]):
                    if lbl == 0:
                        dist = np.linalg.norm(np.array(cand_pt) - np.array(pt))
                        if dist < 20.0:
                            raw_mask = (labels_map == i).astype(np.uint8) * 255
                            feat_noise = FeatureExtractor.extract_features(img_after, raw_mask > 0)
                            if feat_noise is not None:
                                db.add_noise_profile(feat_noise)
                                print(f"    - 노이즈 특징 프로필 1개 추출 완료 (Area: {feat_noise['area']})")

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
# 9. Gabor 필터 로컬 적용 함수
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
        cy_k, cx_k = h_box // 2, w_box // 2
        padded_kernel[cy_k-kh//2 : cy_k+kh//2+1, cx_k-kw//2 : cx_k+kw//2+1] = gabor_kernel
        
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

# =========================================================================
# 10. 메인 실행 함수
# =========================================================================
def main():
    print("\n" + "="*60)
    print("Probe Card Contact Mark PMI Analysis System (main9.py)")
    print("  — 산업 PMI 수준 정밀 계측 + Active Learning 통합 시스템")
    print("="*60)
    
    # 학습 데이터베이스(learning_db.json) 초기화 선택 옵션
    db_file = "learning_db.json"
    if os.path.exists(db_file):
        init_choice = input("  > 기존 학습 DB(learning_db.json)를 초기화하고 처음부터 새로 학습하시겠습니까? (y/n, 기본값 n): ").strip().lower()
        if init_choice == 'y':
            try:
                os.remove(db_file)
                print("  [성공] 기존 학습 DB 파일이 성공적으로 초기화되었습니다.")
            except Exception as e:
                print(f"  [경고] 학습 DB 파일 삭제에 실패했습니다: {e}")
    
    # 1. 사용자로부터 테스트 이미지 파일 세트 번호 입력
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
    print("\n[1/10] 이미지 정합(Alignment) 처리 중...")
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
    print("\n[2/10] 사각형 영역(ROI) 검출 및 분할...")
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
    print("\n[3/10] 노이즈 제거 및 주파수 필터링(Gabor) 적용...")
    img_before_gabor = apply_gabor_filter_to_rois(img_before, rois, theta=config['gabor_theta'], lam=config['gabor_lam'])
    img_after_gabor = apply_gabor_filter_to_rois(img_after_aligned, rois, theta=config['gabor_theta'], lam=config['gabor_lam'])
    
    img_before_bilateral = cv2.bilateralFilter(img_before_gabor, 9, 75, 75)
    img_after_bilateral = cv2.bilateralFilter(img_after_gabor, 9, 75, 75)

    # 7. 노이즈 제거 및 CLAHE 고대비 전처리
    print("\n[4/10] 소형 노이즈 제거 및 CLAHE 대비 개선...")
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
    print("\n[5/10] 차영상 맵 도출 및 후보 영역 계산...")
    analyzer = ChangeAnalyzer(diff_threshold=config['diff_th'], min_area=config['min_a'])
    _, diff_thresh_map = analyzer.get_difference_candidates(img_before_proc, img_after_proc)

    # 9. SAM2 모델 로드
    print("\n[6/10] SAM2 모델 로딩...")
    detector = MarkDetector()

    # 10. Active Learning 인터랙티브 세션 기동
    print("\n[7/10] 대화형 Active Learning 피드백 세션 구동...")
    final_masks = run_active_learning_gui(img_before_clean, img_after_clean, detector, rois, diff_thresh_map, db, config)

    # =====================================================================
    # [NEW in main9] 11. 정밀 계측 (MarkMetrology) 수행
    # =====================================================================
    print("\n[8/10] 산업 PMI 수준 정밀 계측 수행 중...")
    
    roi_results = []
    all_metrology = {}  # roi_idx -> [metrology_dict, ...]
    
    for i, rect in enumerate(rois):
        sub_rect = calculate_sub_rectangle(rect, config['guard_percentage'])
        marks_in_roi = [m for m in final_masks if m['roi_idx'] == i]
        
        # 각 마크에 대한 정밀 계측 수행
        metrology_list = []
        for m in marks_in_roi:
            met = MarkMetrology.measure_mark(
                m['segmentation'], rect, img_before, img_after_aligned
            )
            metrology_list.append(met)
            
            if met is not None:
                grade_emoji = {'Good': '✅', 'Marginal': '⚠️', 'Defective': '❌'}.get(met['grade'], '?')
                print(f"    ROI {i+1} 마크: {grade_emoji} {met['grade']} | "
                      f"스크럽={met['major_axis']:.1f}x{met['minor_axis']:.1f}px "
                      f"θ={met['scrub_angle']:.1f}° | "
                      f"최소경계={met['min_edge_dist']:.1f}px | "
                      f"OD={met['overdrive_ratio']*100:.1f}% | "
                      f"ΔL*={met['intensity_delta_L']:.1f}")
        
        all_metrology[i] = metrology_list
        
        is_pass = check_guard_zone(marks_in_roi, sub_rect)
        
        roi_results.append({
            'roi_index': i,
            'rect': rect,
            'sub_rect': sub_rect,
            'is_pass': is_pass,
            'marks': marks_in_roi,
            'metrology': metrology_list
        })
        
        print(f"  - ROI {i+1} 결과: {'PASS' if is_pass else 'FAIL'} "
              f"(검출된 마크 수: {len(marks_in_roi)}개)")

    # =====================================================================
    # [NEW in main9] 12. 드리프트 분석
    # =====================================================================
    print("\n[9/10] 마크 위치 드리프트 분석...")
    drift_analyzer = DriftAnalyzer(db)
    
    for i, met_list in all_metrology.items():
        valid_mets = [m for m in met_list if m is not None]
        if valid_mets:
            drift_analyzer.record_session(i, valid_mets, image_after_path)
            
            stats = drift_analyzer.get_drift_stats(i)
            print(f"  - ROI {i+1} 드리프트: "
                  f"평균 오프셋=({stats['mean_offset_x']:.1f}, {stats['mean_offset_y']:.1f})px | "
                  f"표준편차=({stats['std_offset_x']:.1f}, {stats['std_offset_y']:.1f})px | "
                  f"추세={stats['trend_direction']} | "
                  f"누적 세션={stats['session_count']}회")
    
    # 드리프트 기록 포함하여 DB 저장
    db.save_db()

    # =====================================================================
    # [NEW in main9] 13. 정량 보고서 생성 (CSV + 시각 보고서)
    # =====================================================================
    print("\n[10/10] PMI 정량 보고서 생성 중...")
    
    # CSV 보고서 생성
    csv_path = ReportGenerator.generate_csv(roi_results, output_dir="reports")
    
    # 설정 정보 구성
    config_disp = {
        "차이임계값": config['diff_th'],
        "최소면적": config['min_a'],
        "가드비율": f"{config['guard_percentage']}%",
        "가보어각도": round(config['gabor_theta'], 3),
        "학습데이터": f"Mark {len(db.data['mark_profiles'])}개 / Noise {len(db.data['noise_profiles'])}개",
        "드리프트세션": f"{len(db.data.get('drift_history', []))}회"
    }
    
    # 6패널 시각 보고서 생성
    ReportGenerator.generate_visual_report(
        img_before, img_after_aligned, roi_results, drift_analyzer, config_disp
    )
    
    print("\n" + "="*60)
    print("[완료] PMI 분석이 성공적으로 완료되었습니다.")
    print(f"  - CSV 보고서: {csv_path}")
    print("="*60)

if __name__ == "__main__":
    main()
