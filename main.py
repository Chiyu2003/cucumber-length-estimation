import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Add root folder to python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from utils.image_processing import polygon_to_mask, draw_path, FX, FY, CX, CY
from utils.geometry import CameraIntrinsics, measure_m5

# Configurations
WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights", "cucumber_seg.pt")
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
DEPTH_SUFFIX = "_depth.npy"

def filter_duplicate_boxes(boxes, masks, confs=None, iou_thresh=0.40):
    if len(boxes) == 0:
        return [], []
    if confs is None:
        order = np.argsort([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])[::-1]
    else:
        order = np.argsort(confs)[::-1]
        
    keep_indices = []
    while len(order) > 0:
        i = order[0]
        keep_indices.append(i)
        if len(order) == 1:
            break
        box_i = boxes[i]
        rem_boxes = boxes[order[1:]]
        
        xi1 = np.maximum(box_i[0], rem_boxes[:, 0])
        yi1 = np.maximum(box_i[1], rem_boxes[:, 1])
        xi2 = np.minimum(box_i[2], rem_boxes[:, 2])
        yi2 = np.minimum(box_i[3], rem_boxes[:, 3])
        
        inter_area = np.maximum(0.0, xi2 - xi1) * np.maximum(0.0, yi2 - yi1)
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        rem_areas = (rem_boxes[:, 2] - rem_boxes[:, 0]) * (rem_boxes[:, 3] - rem_boxes[:, 1])
        iou = inter_area / (area_i + rem_areas - inter_area + 1e-8)
        
        mask_keep = np.where(iou <= iou_thresh)[0]
        order = order[mask_keep + 1]
        
    return [boxes[k] for k in keep_indices], [masks[k] for k in keep_indices]

def process_image(img_name, yolo, intrinsics):
    img_path = os.path.join(DATA_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        return
        
    # Load associated depth map (.npy)
    depth_name = img_name.replace(".png", DEPTH_SUFFIX).replace(".jpg", DEPTH_SUFFIX).replace(".jpeg", DEPTH_SUFFIX)
    depth_path = os.path.join(DATA_DIR, depth_name)
    if not os.path.exists(depth_path):
        # Fallback to search depth file containing '_depth' keyword
        base_name = os.path.splitext(img_name)[0]
        depth_path = os.path.join(DATA_DIR, f"{base_name}{DEPTH_SUFFIX}")
        
    if not os.path.exists(depth_path):
        print(f"[Warning] Depth file not found for {img_name}, skipping.")
        return
        
    depth_map = np.load(depth_path).astype(np.float32)
    if depth_map.max() > 50.0:  # Convert mm to meters if raw depth is in millimeters
        depth_map /= 1000.0
        
    # Segment instances with NMS iou threshold
    results = yolo.predict(source=img, conf=0.35, iou=0.40, verbose=False)[0]
    annotated = img.copy()
    
    if results.masks is not None:
        raw_boxes = results.boxes.xyxy.cpu().numpy()
        raw_confs = results.boxes.conf.cpu().numpy()
        raw_masks = results.masks.xy
        
        # Filter duplicate overlapping bounding boxes on the same cucumber
        boxes, masks = filter_duplicate_boxes(raw_boxes, raw_masks, confs=raw_confs, iou_thresh=0.40)
        
        # 1. 先合併所有小黃瓜的二值遮罩
        merged_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for poly in masks:
            mask = polygon_to_mask(poly, img.shape)
            merged_mask = cv2.bitwise_or(merged_mask, mask)
            
        # 2. 一次性與原圖融合，保證所有小黃瓜的綠色遮罩亮度完全一致
        mask_color = np.zeros_like(annotated)
        mask_color[merged_mask > 0] = (0, 220, 80)
        annotated = cv2.addWeighted(mask_color, 0.3, annotated, 0.7, 0)
        
        # 3. 繪製個別的骨架與資訊
        for idx, poly in enumerate(masks):
            mask = polygon_to_mask(poly, img.shape)
            res = measure_m5(mask, depth_map, intrinsics)
            if not res.success or res.length_cm is None:
                continue
                
            length_cm = res.length_cm
            
            # Project 3D Spline to 2D
            spline_3d = res.spline_points
            z_pts = spline_3d[:, 2]
            u_proj = (spline_3d[:, 0] * intrinsics.fx / z_pts) + intrinsics.cx
            v_proj = (spline_3d[:, 1] * intrinsics.fy / z_pts) + intrinsics.cy
            pts_2d = np.stack([u_proj, v_proj], axis=1).astype(np.int32)
            
            # 繪製中線與包圍盒
            cv2.polylines(annotated, [pts_2d], False, (0, 255, 255), 2, cv2.LINE_AA)
            x1, y1, x2, y2 = map(int, boxes[idx])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 2)
            
            label = f"{length_cm:.2f} cm"
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"-> Detected cucumber (index {idx}): {length_cm:.2f} cm")
            
    out_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
    cv2.imwrite(out_path, annotated)
    print(f"[Done] Annotated result saved to: {out_path}\n")

def main():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Missing weights file at {WEIGHTS_PATH}")
        sys.exit(1)
        
    yolo = YOLO(WEIGHTS_PATH)
    intrinsics = CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process all image files in data directory
    img_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print(f"No image files found in {DATA_DIR}")
        return
        
    print(f"Starting execution on {len(img_files)} images...")
    for img_file in sorted(img_files):
        process_image(img_file, yolo, intrinsics)

if __name__ == "__main__":
    main()
