#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m5_realsense_only/inference/predict_m5.py
=========================================
極簡推論程式：僅使用 YOLO11-seg 分割與 RealSense 3D M5 幾何運算進行長度量測（獨立 Repo 版本）。
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# 將當前與上層目錄加入 Python 搜尋路徑，確保可以載入同資料夾內的 utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(REPO_ROOT)

from utils.seg_utils import polygon_to_mask, draw_path, FX, FY, CX, CY
from utils.m5_geometry import CameraIntrinsics, measure_m5

# 模型與路徑設定
YOLO_MODEL_PATH = os.path.join(REPO_ROOT, "model", "best.pt")
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

def main():
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"❌ 找不到 YOLO 模型，請確認放置於: {YOLO_MODEL_PATH}")
        sys.exit(1)

    print("🔄 正在載入 YOLO11-seg 模型...")
    yolo = YOLO(YOLO_MODEL_PATH)
    intrinsics = CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 搜尋 dataset 資料夾底下的所有 color 照片
    images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"⚠️ 在 {DATASET_DIR} 中沒有找到任何圖片。")
        sys.exit(0)
        
    print(f"🚀 開始量測 (共 {len(images)} 張照片)...")
    for img_name in images:
        img_path = os.path.join(DATASET_DIR, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        h, w = img_bgr.shape[:2]
        
        # 尋找匹配的 RealSense 深度圖 (.npy)
        depth_map = None
        base_name = os.path.splitext(img_name)[0]
        depth_candidates = [
            os.path.join(DATASET_DIR, f"{base_name}_depth_raw.npy"),
            os.path.join(DATASET_DIR, img_name.replace("_color_clean.png", "_depth_raw.npy")),
            os.path.join(DATASET_DIR, img_name.replace(".png", "_depth_raw.npy")),
        ]
        
        for candidate in depth_candidates:
            if os.path.exists(candidate):
                depth_map = np.load(candidate).astype(np.float32)
                if depth_map.max() > 50.0:  # 如果是 mm 單位，轉換為米 (meters)
                    depth_map /= 1000.0
                print(f"🎯 讀取深度圖: {os.path.basename(candidate)}")
                break
                
        if depth_map is None:
            print(f"⚠️ 找不到 {img_name} 對應的 depth_raw.npy 深度檔，跳過此張量測。")
            continue
            
        # YOLO 預測分割遮罩
        results = yolo.predict(source=img_bgr, conf=0.25, verbose=False)[0]
        annotated_img = img_bgr.copy()
        
        if results.masks is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            masks_xy = results.masks.xy
            
            for idx, poly in enumerate(masks_xy):
                mask = polygon_to_mask(poly, img_bgr.shape)
                
                # 直接使用 Method M5 3D Spline 計算實際物理長度
                m5_res = measure_m5(mask, depth_map, intrinsics)
                if not m5_res.success or m5_res.length_cm is None:
                    print(f"   ⚠️ 小黃瓜 {idx+1} M5 重建失敗: {m5_res.failure_reason}")
                    continue
                
                length_cm = m5_res.length_cm
                
                # 取得 2D 投影點繪製骨架中線
                spline_3d = m5_res.spline_points
                z_pts = spline_3d[:, 2]
                u_proj = (spline_3d[:, 0] * intrinsics.fx / z_pts) + intrinsics.cx
                v_proj = (spline_3d[:, 1] * intrinsics.fy / z_pts) + intrinsics.cy
                path_pts = np.stack([u_proj, v_proj], axis=1).astype(np.int32)
                
                # 視覺化疊加
                # 1. 綠色半透明遮罩
                color_mask = np.zeros_like(annotated_img)
                color_mask[mask > 0] = (0, 220, 80)
                annotated_img = cv2.addWeighted(color_mask, 0.3, annotated_img, 0.7, 0)
                
                # 2. 黃色骨架中線
                cv2.polylines(annotated_img, [path_pts], False, (0, 255, 255), 2, cv2.LINE_AA)
                
                # 3. 紅色 Bounding Box 與長度文字
                x1, y1, x2, y2 = map(int, boxes[idx])
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 220), 2)
                cv2.putText(annotated_img, f"{length_cm:.2f} cm", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                print(f"   🥒 小黃瓜 {idx+1}: {length_cm:.2f} cm")
                
        out_path = os.path.join(OUTPUT_DIR, f"m5_{img_name}")
        cv2.imwrite(out_path, annotated_img)
        print(f"🎉 視覺化結果已儲存: {out_path}\n")

if __name__ == "__main__":
    main()
