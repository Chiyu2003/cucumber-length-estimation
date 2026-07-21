# -*- coding: utf-8 -*-
"""
utils/m5_medial_arc_spline.py - Re-implementation of Method M5 (Medial Arc Spline)
==================================================================================

This module implements Method M5 (Medial Arc Spline) from the CucumberVision paper:
1. Deprojection of SAM-masked pixels into a 3D Camera Coordinate Point Cloud.
2. Global median-depth filtering to remove leaf/branch background outliers.
3. PCA via SVD to find the principal cucumber growth/axis vector.
4. Binned cross-section slicing (default 25 bins) along the growth axis.
5. Vertebral centroid extraction.
6. Start/End tip anchors (none, single, or multiple) to refine boundary length.
7. Chord-length parameterization mapping vertebrae to u in [0, 1].
8. Natural Cubic Spline fitting (bc_type='natural') for X, Y, Z.
9. Speed integration of spline first-derivatives via composite trapezoidal rule.
10. Strict range validation and quality/occlusion metrics.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
from typing import Dict, Any, Tuple, Optional, List

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass(frozen=True)
class M5Config:
    depth_scale: float = 1.0
    depth_threshold_m: float = 0.035
    min_valid_points: int = 100
    num_bins: int = 25
    min_points_per_bin: int = 3
    min_spine_points: int = 4
    integration_samples: int = 500
    min_length_cm: float = 5.0
    max_length_cm: float = 80.0
    use_tip_anchors: bool = True
    tip_quantile: float = 0.002
    tip_anchor_strategy: str = "single"  # "none", "single", "multiple"
    max_missing_bin_ratio: float = 0.4
    max_consecutive_missing_bins: int = 5
    max_spline_polyline_ratio: float = 1.5

@dataclass
class M5Result:
    success: bool
    length_cm: Optional[float]
    failure_reason: Optional[str]
    raw_point_count: int
    filtered_point_count: int
    valid_bin_count: int
    skipped_bin_count: int
    principal_axis: Optional[np.ndarray]
    spine_points: Optional[np.ndarray]
    spline_points: Optional[np.ndarray]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Converts mask to a 2D boolean array."""
    if mask.ndim != 2:
        raise ValueError(f"Mask must be a 2D array, got shape {mask.shape}")
    return mask > 0

def normalize_depth_to_meters(depth: np.ndarray, scale: float) -> np.ndarray:
    """Converts depth to float64 in meters. Replaces <=0 and non-finite values with NaN."""
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be a 2D array, got shape {depth.shape}")
    depth_m = depth.astype(np.float64) * scale
    invalid = (depth_m <= 0) | (~np.isfinite(depth_m))
    depth_m[invalid] = np.nan
    return depth_m

def validate_intrinsics(intrinsics: CameraIntrinsics) -> None:
    """Validates that intrinsic focal lengths are positive."""
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError(f"Focal lengths fx, fy must be positive, got fx={intrinsics.fx}, fy={intrinsics.fy}")

def _is_edge_pixel_valid(
    ey: int, ex: int, ez: float,
    depth_m: np.ndarray,
    inner_depths_only: np.ndarray,
    mask_uint8: np.ndarray,
    window_radius: int
) -> bool:
    if ez <= 0 or not np.isfinite(ez):
        return False
        
    h, w = depth_m.shape
    y1, y2 = max(0, ey - window_radius), min(h, ey + window_radius + 1)
    x1, x2 = max(0, ex - window_radius), min(w, ex + window_radius + 1)
    
    local_inner = inner_depths_only[y1:y2, x1:x2]
    valid_local_inner = local_inner[(local_inner > 0) & np.isfinite(local_inner)]
    
    if len(valid_local_inner) > 0:
        local_median = np.median(valid_local_inner)
        return abs(ez - local_median) <= 0.015
        
    local_all = depth_m[y1:y2, x1:x2]
    valid_local_all = local_all[(local_all > 0) & np.isfinite(local_all) & (mask_uint8[y1:y2, x1:x2] > 0)]
    if len(valid_local_all) > 0:
        local_median = np.median(valid_local_all)
        return abs(ez - local_median) <= 0.015
        
    return False

def deproject_mask_to_point_cloud(
    mask_bool: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Deprojects masked pixels using pinhole camera equations with Boundary Depth Consistency Filtering."""
    if mask_bool.shape != depth_m.shape:
        raise ValueError(f"Mask shape {mask_bool.shape} does not match depth map shape {depth_m.shape}")
    
    # 1. Morphological erosion to separate stable inner region and sensitive edge boundary
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
    edge_mask = (mask_uint8 > 0) & (eroded_mask == 0)
    
    # 2. Get depths of stable inner region
    inner_depths_only = np.zeros_like(depth_m)
    inner_depths_only[eroded_mask > 0] = depth_m[eroded_mask > 0]
    
    v, u = np.nonzero(mask_bool)
    z = depth_m[v, u]
    
    valid_mask = (z > 0) & np.isfinite(z)
    
    # 3. Local Window Depth Consistency Filtering for Edge Pixels (radius=2 -> 5x5 window)
    h, w = mask_bool.shape
    window_radius = 2
    edge_y, edge_x = np.nonzero(edge_mask)
    
    if len(edge_y) > 0:
        # Create a lookup mapping (y, x) to validity for fast array-level masking
        edge_valid_map = np.ones_like(mask_bool, dtype=bool)
        
        for idx in range(len(edge_y)):
            ey, ex = edge_y[idx], edge_x[idx]
            ez = depth_m[ey, ex]
            edge_valid_map[ey, ex] = _is_edge_pixel_valid(
                ey, ex, ez, depth_m, inner_depths_only, mask_uint8, window_radius
            )
                    
        # Apply the edge validity map to the flat coordinate array
        is_edge_pixel = edge_mask[v, u]
        edge_pixel_validity = edge_valid_map[v, u]
        # Keep if not an edge pixel OR if the edge pixel is valid
        valid_mask = valid_mask & (~is_edge_pixel | edge_pixel_validity)
    
    u_valid = u[valid_mask].astype(np.float64)
    v_valid = v[valid_mask].astype(np.float64)
    z_valid = z[valid_mask]
    
    raw_pixel_count = len(v)
    valid_pixel_count = len(z_valid)
    invalid_ratio = (raw_pixel_count - valid_pixel_count) / max(1, raw_pixel_count)
    
    if valid_pixel_count == 0:
        points = np.empty((0, 3), dtype=np.float64)
    else:
        x = (u_valid - intrinsics.cx) * z_valid / intrinsics.fx
        y = (v_valid - intrinsics.cy) * z_valid / intrinsics.fy
        points = np.column_stack((x, y, z_valid))
        
    stats = {
        "raw_pixel_count": raw_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "invalid_ratio": invalid_ratio
    }
    return points, stats

def filter_points_by_median_depth(
    points: np.ndarray,
    threshold_m: float,
    fx: float = 912.34,
    fy: float = 911.85,
    cx: float = 638.12,
    cy: float = 357.45
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Filters 3D points using 3D plane trend depth fitting to support tilted cucumbers."""
    if len(points) == 0:
        return points, {"median_depth": 0.0, "removed_ratio": 0.0}
    
    median_z = float(np.median(points[:, 2]))
    u_pts = (points[:, 0] * fx / points[:, 2]) + cx
    v_pts = (points[:, 1] * fy / points[:, 2]) + cy
    
    try:
        A = np.column_stack([u_pts, v_pts, np.ones_like(u_pts)])
        res_plane, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
        z_pred = A @ res_plane
        inliers = np.abs(points[:, 2] - z_pred) <= threshold_m
    except Exception:
        inliers = np.abs(points[:, 2] - median_z) <= threshold_m
        
    filtered_points = points[inliers]
    removed_ratio = (len(points) - len(filtered_points)) / len(points)
    
    stats = {
        "median_depth": median_z,
        "removed_ratio": float(removed_ratio)
    }
    return filtered_points, stats

def compute_principal_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Finds the principal axis (v1) of the point cloud via SVD."""
    if len(points) == 0:
        raise ValueError("Cannot compute PCA of an empty point cloud.")
        
    center = points.mean(axis=0)
    centered = points - center
    
    # Degeneracy check
    if np.allclose(centered, 0.0):
        # Degenerate case where all points are identical
        principal_axis = np.array([0.0, 0.0, 1.0])
        singular_values = np.zeros(3)
        explained_variance = np.zeros(3)
    else:
        # SVD decomposition
        try:
            _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
            principal_axis = vt[0]
            # Normalize principal axis
            norm = np.linalg.norm(principal_axis)
            if norm > 0:
                principal_axis /= norm
            else:
                principal_axis = np.array([0.0, 0.0, 1.0])
            
            variance = singular_values ** 2
            total_var = np.sum(variance)
            explained_variance = variance / total_var if total_var > 0 else np.zeros(3)
        except np.linalg.LinAlgError:
            principal_axis = np.array([0.0, 0.0, 1.0])
            singular_values = np.zeros(3)
            explained_variance = np.zeros(3)
            
    # Deterministic orientation: ensure the axis points towards positive coordinates in its dominant component
    dom_idx = np.argmax(np.abs(principal_axis))
    if principal_axis[dom_idx] < 0:
        principal_axis = -principal_axis
        
    stats = {
        "singular_values": singular_values.tolist(),
        "explained_variance_ratio": explained_variance.tolist(),
        "center": center.tolist()
    }
    return center, principal_axis, stats

def assign_cross_section_bins(
    points: np.ndarray,
    center: np.ndarray,
    principal_axis: np.ndarray,
    num_bins: int,
    min_points_per_bin: int
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Slices the 3D points along the principal axis into cross-section bins."""
    projections = (points - center) @ principal_axis
    p_min, p_max = projections.min(), projections.max()
    
    edges = np.linspace(p_min, p_max, num_bins + 1)
    
    # Avoid out-of-bounds index for the absolute maximum value
    bin_ids = np.digitize(projections, edges[1:-1], right=False)
    
    bins_data = []
    for b in range(num_bins):
        in_bin = (bin_ids == b)
        bin_pts = points[in_bin]
        
        is_valid = len(bin_pts) >= min_points_per_bin
        bin_centroid = bin_pts.mean(axis=0) if is_valid else np.zeros(3)
        
        # Calculate point dispersion within the cross-section (distance from centroid)
        if is_valid:
            dispersion = float(np.mean(np.linalg.norm(bin_pts - bin_centroid, axis=1)))
            depth_std = float(np.std(bin_pts[:, 2]))
        else:
            dispersion = 0.0
            depth_std = 0.0
            
        bins_data.append({
            "bin_idx": b,
            "start": float(edges[b]),
            "end": float(edges[b+1]),
            "count": int(len(bin_pts)),
            "valid": is_valid,
            "centroid": bin_centroid if is_valid else None,
            "dispersion": dispersion,
            "depth_std": depth_std
        })
        
    return bins_data, projections

def _compute_multiple_anchors(
    points: np.ndarray,
    projections: np.ndarray,
    quantile: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    quantiles_start = np.linspace(quantile / 5.0, quantile, 5)
    quantiles_end = np.linspace(1.0 - quantile, 1.0 - (quantile / 5.0), 5)
    
    start_anchors = []
    for q in quantiles_start:
        q_val = np.quantile(projections, q)
        pts = points[projections <= q_val]
        if len(pts) > 0:
            start_anchors.append(pts.mean(axis=0))
            
    end_anchors = []
    for q in quantiles_end:
        q_val = np.quantile(projections, q)
        pts = points[projections >= q_val]
        if len(pts) > 0:
            end_anchors.append(pts.mean(axis=0))
            
    start_anchor = np.median(start_anchors, axis=0) if start_anchors else None
    end_anchor = np.median(end_anchors, axis=0) if end_anchors else None
    
    return start_anchor, end_anchor, start_anchors, end_anchors

def compute_tip_anchors(
    points: np.ndarray,
    projections: np.ndarray,
    strategy: str,
    quantile: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """Computes start and end tip anchors based on projection quantiles."""
    if strategy == "none" or len(points) == 0:
        return None, None, None, None
        
    if strategy == "single":
        q_low = np.quantile(projections, quantile)
        q_high = np.quantile(projections, 1.0 - quantile)
        
        start_pts = points[projections <= q_low]
        end_pts = points[projections >= q_high]
        
        start_anchor = start_pts.mean(axis=0) if len(start_pts) > 0 else None
        end_anchor = end_pts.mean(axis=0) if len(end_pts) > 0 else None
        
        return start_anchor, end_anchor, None, None
        
    if strategy == "multiple":
        return _compute_multiple_anchors(points, projections, quantile)
        
    return None, None, None, None

def clean_and_sort_spine(
    centroids: np.ndarray,
    center: np.ndarray,
    principal_axis: np.ndarray
) -> np.ndarray:
    """Sorts spine points along the principal axis and removes duplicates."""
    if len(centroids) < 2:
        return centroids
        
    # Project centroids onto the principal axis
    proj = (centroids - center) @ principal_axis
    sort_idx = np.argsort(proj)
    sorted_spine = centroids[sort_idx]
    
    # Remove overlapping points (distance < 1e-6)
    cleaned_spine = [sorted_spine[0]]
    for p in sorted_spine[1:]:
        if np.linalg.norm(p - cleaned_spine[-1]) >= 1e-6:
            cleaned_spine.append(p)
            
    return np.array(cleaned_spine)

def chord_length_parameterize(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the cumulative chord distance parameter u mapped to [0, 1]."""
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    
    total_length = cumulative[-1]
    if total_length <= 0:
        raise ValueError("Spine has degenerate total chord length of 0.")
        
    u = cumulative / total_length
    
    # Verify strict monotonicity
    if not np.all(np.diff(u) > 0):
        raise ValueError("Parameter space u is not strictly increasing.")
        
    return u, cumulative

def fit_natural_cubic_spline(u: np.ndarray, spine: np.ndarray) -> Tuple[CubicSpline, CubicSpline, CubicSpline]:
    """Fits independent natural cubic splines to X, Y, and Z coordinate sequences."""
    sx = CubicSpline(u, spine[:, 0], bc_type="natural")
    sy = CubicSpline(u, spine[:, 1], bc_type="natural")
    sz = CubicSpline(u, spine[:, 2], bc_type="natural")
    return sx, sy, sz

def integrate_spline_arc_length(
    sx: CubicSpline,
    sy: CubicSpline,
    sz: CubicSpline,
    samples: int
) -> Tuple[float, np.ndarray]:
    """Integrates the 3D speed along the cubic spline using composite trapezoidal rule."""
    u_dense = np.linspace(0.0, 1.0, samples)
    
    # Evaluate derivatives analytially
    dx = sx(u_dense, 1)
    dy = sy(u_dense, 1)
    dz = sz(u_dense, 1)
    
    speed = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if np.any(~np.isfinite(speed)):
        raise ValueError("Fitted spline contains non-finite speed derivatives.")
        
    # Trapezoidal rule integration
    if hasattr(np, "trapezoid"):
        length_m = np.trapezoid(speed, u_dense)
    else:
        length_m = np.trapz(speed, u_dense)
        
    spline_points = np.column_stack((sx(u_dense), sy(u_dense), sz(u_dense)))
    length_cm = length_m * 100.0
    return length_cm, spline_points

def _extract_valid_centroids(bins_data: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], int, int]:
    valid_centroids = []
    skipped_bin_count = 0
    consecutive_missing = 0
    max_consecutive_missing = 0
    
    for b in bins_data:
        if b["valid"]:
            valid_centroids.append(b["centroid"])
            consecutive_missing = 0
        else:
            skipped_bin_count += 1
            consecutive_missing += 1
            max_consecutive_missing = max(max_consecutive_missing, consecutive_missing)
            
    return valid_centroids, skipped_bin_count, max_consecutive_missing

def _build_spine(
    valid_centroids: List[np.ndarray],
    start_anchor: Optional[np.ndarray],
    end_anchor: Optional[np.ndarray],
    start_group: Optional[List[np.ndarray]],
    end_group: Optional[List[np.ndarray]],
    config: M5Config,
    center: np.ndarray,
    principal_axis: np.ndarray
) -> np.ndarray:
    spine_list = []
    if config.use_tip_anchors and start_anchor is not None:
        if config.tip_anchor_strategy == "multiple" and start_group is not None:
            spine_list.extend(start_group)
        else:
            spine_list.append(start_anchor)
            
    spine_list.extend(valid_centroids)
    
    if config.use_tip_anchors and end_anchor is not None:
        if config.tip_anchor_strategy == "multiple" and end_group is not None:
            spine_list.extend(end_group)
        else:
            spine_list.append(end_anchor)
            
    return clean_and_sort_spine(np.array(spine_list), center, principal_axis)

def _fail_m5_result(reason: str, diagnostics: Dict[str, Any], **kwargs) -> M5Result:
    defaults = {
        "raw_point_count": 0, "filtered_point_count": 0,
        "valid_bin_count": 0, "skipped_bin_count": 0,
        "principal_axis": None, "spine_points": None, "spline_points": None
    }
    defaults.update(kwargs)
    return M5Result(
        success=False, length_cm=None, failure_reason=reason,
        raw_point_count=defaults["raw_point_count"],
        filtered_point_count=defaults["filtered_point_count"],
        valid_bin_count=defaults["valid_bin_count"],
        skipped_bin_count=defaults["skipped_bin_count"],
        principal_axis=defaults["principal_axis"],
        spine_points=defaults["spine_points"],
        spline_points=defaults["spline_points"],
        diagnostics=diagnostics
    )

def measure_m5(
    mask: np.ndarray,
    aligned_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: Optional[M5Config] = None
) -> M5Result:
    """
    Executes the complete Method M5 (Medial Arc Spline) 3D length estimation.
    """
    if config is None:
        config = M5Config()
        
    diagnostics = {}
    
    # 1. Validation & Input Normalization
    try:
        mask_bool = normalize_mask(mask)
        depth_m = normalize_depth_to_meters(aligned_depth, config.depth_scale)
        validate_intrinsics(intrinsics)
    except ValueError as e:
        return _fail_m5_result("invalid_input", {"error": str(e)})
        
    if mask_bool.shape != depth_m.shape:
        return _fail_m5_result("shape_mismatch", {"mask_shape": mask_bool.shape, "depth_shape": depth_m.shape})
        
    # 2. Point Cloud Generation
    points, deproj_stats = deproject_mask_to_point_cloud(mask_bool, depth_m, intrinsics)
    diagnostics.update(deproj_stats)
    raw_point_count = len(points)
    
    if raw_point_count < config.min_valid_points:
        return _fail_m5_result("insufficient_valid_depth", diagnostics, raw_point_count=raw_point_count)
        
    # 3. Median-depth filtering
    filtered_points, filter_stats = filter_points_by_median_depth(points, config.depth_threshold_m)
    diagnostics.update(filter_stats)
    filtered_point_count = len(filtered_points)
    
    if filtered_point_count < config.min_valid_points:
        return _fail_m5_result("insufficient_points_after_filter", diagnostics, raw_point_count=raw_point_count, filtered_point_count=filtered_point_count)
        
    # 4. PCA / SVD Growth Axis Vector
    try:
        center, principal_axis, pca_stats = compute_principal_axis(filtered_points)
        diagnostics.update(pca_stats)
    except Exception as e:
        return _fail_m5_result("degenerate_point_cloud", {"error": str(e), **diagnostics}, raw_point_count=raw_point_count, filtered_point_count=filtered_point_count)
        
    # 5. Cross-section Binning
    bins_data, projections = assign_cross_section_bins(
        filtered_points, center, principal_axis, config.num_bins, config.min_points_per_bin
    )
    diagnostics.update({
        "bins_data": bins_data,
        "projection_min": float(projections.min()),
        "projection_max": float(projections.max()),
        "projection_range": float(np.ptp(projections))
    })
    
    if np.ptp(projections) < 1e-4:
        return _fail_m5_result("degenerate_projection", diagnostics, raw_point_count=raw_point_count, filtered_point_count=filtered_point_count, principal_axis=principal_axis)
        
    valid_centroids, skipped_bin_count, max_consecutive_missing = _extract_valid_centroids(bins_data)
    valid_bin_count = len(valid_centroids)
    missing_bin_ratio = skipped_bin_count / config.num_bins
    diagnostics.update({
        "missing_bin_ratio": missing_bin_ratio,
        "max_consecutive_missing_bins": max_consecutive_missing
    })
    
    # Occlusion limits validation
    if missing_bin_ratio > config.max_missing_bin_ratio or max_consecutive_missing > config.max_consecutive_missing_bins:
        spine_pts = np.array(valid_centroids) if valid_centroids else None
        return _fail_m5_result("insufficient_valid_bins", diagnostics, raw_point_count=raw_point_count, filtered_point_count=filtered_point_count, valid_bin_count=valid_bin_count, skipped_bin_count=skipped_bin_count, principal_axis=principal_axis, spine_points=spine_pts)
        
    # 6. Tip Anchors
    start_anchor, end_anchor, start_group, end_group = compute_tip_anchors(
        filtered_points, projections, config.tip_anchor_strategy, config.tip_quantile
    )
    
    diagnostics.update({
        "start_anchor": start_anchor.tolist() if start_anchor is not None else None,
        "end_anchor": end_anchor.tolist() if end_anchor is not None else None
    })
    
    spine_points = _build_spine(valid_centroids, start_anchor, end_anchor, start_group, end_group, config, center, principal_axis)
    
    if len(spine_points) < config.min_spine_points:
        return _fail_m5_result("insufficient_spine_points", diagnostics, raw_point_count=raw_point_count, filtered_point_count=filtered_point_count, valid_bin_count=valid_bin_count, skipped_bin_count=skipped_bin_count, principal_axis=principal_axis, spine_points=spine_points)
        
    # 7. Chord-length parameterization
    try:
        u, cumulative = chord_length_parameterize(spine_points)
        diagnostics["total_chord_length_cm"] = float(cumulative[-1] * 100.0)
    except Exception as e:
        return _fail_m5_result("invalid_chord_parameterization", {"error": str(e), **diagnostics}, raw_point_count=raw_point_count, filtered_point_count=filtered_point_count, valid_bin_count=valid_bin_count, skipped_bin_count=skipped_bin_count, principal_axis=principal_axis, spine_points=spine_points)
        
    # 8 & 9. Spline fitting & Integration
    try:
        sx, sy, sz = fit_natural_cubic_spline(u, spine_points)
        length_cm, spline_points = integrate_spline_arc_length(sx, sy, sz, config.integration_samples)
    except Exception as e:
        return _fail_m5_result("spline_fitting_failed", {"error": str(e), **diagnostics}, raw_point_count=raw_point_count, filtered_point_count=filtered_point_count, valid_bin_count=valid_bin_count, skipped_bin_count=skipped_bin_count, principal_axis=principal_axis, spine_points=spine_points)
        
    # 10. Spline/Polyline Ratio check (overshoot check)
    polyline_length_cm = float(cumulative[-1] * 100.0)
    ratio = length_cm / polyline_length_cm
    diagnostics["spline_polyline_ratio"] = ratio
    
    if ratio > config.max_spline_polyline_ratio:
        return _fail_m5_result("spline_fitting_failed", diagnostics, raw_point_count=raw_point_count, filtered_point_count=filtered_point_count, valid_bin_count=valid_bin_count, skipped_bin_count=skipped_bin_count, principal_axis=principal_axis, spine_points=spine_points, spline_points=spline_points)
        
    # 11. Range check
    if not (config.min_length_cm <= length_cm <= config.max_length_cm):
        diagnostics["unvalidated_length_cm"] = length_cm
        return _fail_m5_result("length_out_of_range", diagnostics, raw_point_count=raw_point_count, filtered_point_count=filtered_point_count, valid_bin_count=valid_bin_count, skipped_bin_count=skipped_bin_count, principal_axis=principal_axis, spine_points=spine_points, spline_points=spline_points)
        
    return M5Result(
        success=True, length_cm=length_cm, failure_reason=None,
        raw_point_count=raw_point_count, filtered_point_count=filtered_point_count,
        valid_bin_count=valid_bin_count, skipped_bin_count=skipped_bin_count,
        principal_axis=principal_axis, spine_points=spine_points, spline_points=spline_points,
        diagnostics=diagnostics
    )

def save_m5_debug_visualization(
    result: M5Result,
    rgb: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    output_path: str,
    intrinsics: CameraIntrinsics
) -> None:
    """Generates a comprehensive diagnostic visualization containing both a 2D overlay and a 3D view."""
    # Build a combined figure
    fig = plt.figure(figsize=(15, 7))
    
    # Pane 1: 2D Overlay
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_2d.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    
    # Draw Mask contour
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # Draw green contour
        pts = c.reshape(-1, 2)
        ax_2d.plot(pts[:, 0], pts[:, 1], 'g-', lw=2)
        
    # Project Spine/Spline points to 2D
    if result.success and result.spline_points is not None:
        spline = result.spline_points
        u = spline[:, 0] * intrinsics.fx / spline[:, 2] + intrinsics.cx
        v = spline[:, 1] * intrinsics.fy / spline[:, 2] + intrinsics.cy
        ax_2d.plot(u, v, 'c-', lw=3, label="Fitted Spline")
        
        # Project spine points
        spine = result.spine_points
        if spine is not None:
            su = spine[:, 0] * intrinsics.fx / spine[:, 2] + intrinsics.cx
            sv = spine[:, 1] * intrinsics.fy / spine[:, 2] + intrinsics.cy
            ax_2d.scatter(su, sv, color='red', s=40, zorder=5, label="Centroids")
            
        ax_2d.legend()
        ax_2d.set_title(f"2D Projection (Length: {result.length_cm:.2f} cm)")
    else:
        ax_2d.set_title(f"Inference Failed: {result.failure_reason}")
        
    # Pane 2: 3D Visualization
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Extract point cloud for plotting (downsampled for performance)
    mask_y, mask_x = np.nonzero(mask > 0)
    if len(mask_y) > 0:
        depth_m = depth.astype(np.float64) / 1000.0  # assume raw depth for visualization, scale as needed
        z = depth_m[mask_y, mask_x]
        valid = (z > 0) & np.isfinite(z)
        if np.any(valid):
            z_val = z[valid]
            x_val = (mask_x[valid] - intrinsics.cx) * z_val / intrinsics.fx
            y_val = (mask_y[valid] - intrinsics.cy) * z_val / intrinsics.fy
            
            # Downsample to max 1000 points
            stride = max(1, len(x_val) // 1000)
            ax_3d.scatter(x_val[::stride], y_val[::stride], z_val[::stride], color='gray', alpha=0.1, s=2)
            
    if result.spine_points is not None and len(result.spine_points) > 0:
        spine = result.spine_points
        ax_3d.plot(spine[:, 0], spine[:, 1], spine[:, 2], 'ro-', label="Spine path")
        
    if result.spline_points is not None and len(result.spline_points) > 0:
        spline = result.spline_points
        ax_3d.plot(spline[:, 0], spline[:, 1], spline[:, 2], 'c-', lw=3, label="3D Spline")
        
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.legend()
    ax_3d.set_title("3D Point Cloud and Spline")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
