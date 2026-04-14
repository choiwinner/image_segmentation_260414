import cv2
import numpy as np
from processor import ImageAligner, MarkDetector, ChangeAnalyzer
import torch

def reproduce_issue():
    # 1. Create dummy data with overlapping marks (Same color to force merging in SAM2)
    # Base background
    base = np.full((300, 300, 3), 200, dtype=np.uint8)
    
    # Existing mark in 'before'
    img_before = base.copy()
    cv2.circle(img_before, (150, 150), 20, (50, 50, 50), -1) 
    
    # New mark in 'after' overlapping the old one (Identical color)
    img_after = img_before.copy()
    cv2.circle(img_after, (170, 150), 20, (50, 50, 50), -1) 
    
    print("Reproduction images created. Overlapping marks with identical color.")
    
    # 2. Setup Detector and Analyzer
    detector = MarkDetector()
    if detector.predictor is None:
        print("SAM2 not available. Skipping.")
        return

    analyzer = ChangeAnalyzer(iou_threshold=0.3)
    
    # 3. Test Old Logic (for comparison)
    print("\n[Testing Old Logic]")
    masks_before = detector.get_masks(img_before)
    masks_after = detector.get_masks(img_after)
    old_new_marks = analyzer.find_new_marks(masks_before, masks_after)
    print(f"  > Old logic detected {len(old_new_marks)} new marks.")
    
    # 4. Test New Logic
    print("\n[Testing Refined Logic]")
    refined_new_marks = analyzer.find_new_marks_refined(img_before, img_after, masks_before, detector)
    print(f"  > Refined logic detected {len(refined_new_marks)} new marks.")
    
    # Validation
    if len(refined_new_marks) > 0:
        print("\nSUCCESS: Refined logic found the overlapping mark!")
    else:
        print("\nFAILURE: Refined logic also missed the overlapping mark.")

if __name__ == "__main__":
    reproduce_issue()
