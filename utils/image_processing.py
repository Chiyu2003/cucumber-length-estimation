import cv2
import numpy as np

FX, FY = 912.34, 911.85
CX, CY = 638.12, 357.45
DEPTH_SCALE = 1000.0

def deproject_pixel(u, v, depth_mm, fx=FX, fy=FY, cx=CX, cy=CY, depth_scale=DEPTH_SCALE):
    if depth_mm <= 0:
        return None
    z = float(depth_mm) / depth_scale
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)

def median_depth_at(depth_map, u, v, radius=3):
    h, w = depth_map.shape[:2]
    x0, x1 = max(0, int(round(u)) - radius), min(w, int(round(u)) + radius + 1)
    y0, y1 = max(0, int(round(v)) - radius), min(h, int(round(v)) + radius + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))

def polygon_to_mask(points, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if points is None or len(points) < 3:
        return mask
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def draw_path(image, path, color=(0, 255, 0), thickness=2):
    if len(path) < 2:
        return image
    pts = np.asarray(path, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA)
    return image
