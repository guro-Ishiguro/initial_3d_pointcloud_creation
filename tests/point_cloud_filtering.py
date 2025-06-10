# tests/point_cloud_filtering.py

import os
import numpy as np
import open3d as o3d
import logging
from numba import njit, prange

from utils import quaternion_to_rotation_matrix

# --- Numba高速化ヘルパー関数 ---

@njit
def get_patch(image, u, v, patch_radius):
    """画像から指定された座標中心のパッチを取得する"""
    h, w = image.shape[:2]
    u_min, u_max = u - patch_radius, u + patch_radius + 1
    v_min, v_max = v - patch_radius, v + patch_radius + 1
    if u_min < 0 or u_max > w or v_min < 0 or v_max > h:
        return None
    return image[v_min:v_max, u_min:u_max]

@njit
def compute_ncc(patch1, patch2):
    """2つのパッチ間の正規化相互相関 (NCC) を計算する"""
    if patch1 is None or patch2 is None: return -1.0
    
    p1_gray = (patch1[:, :, 0] * 0.299 + patch1[:, :, 1] * 0.587 + patch1[:, :, 2] * 0.114).flatten().astype(np.float32)
    p2_gray = (patch2[:, :, 0] * 0.299 + patch2[:, :, 1] * 0.587 + patch2[:, :, 2] * 0.114).flatten().astype(np.float32)

    mean1, mean2 = np.mean(p1_gray), np.mean(p2_gray)
    norm1, norm2 = p1_gray - mean1, p2_gray - mean2
    
    numerator = np.sum(norm1 * norm2)
    denominator = np.sqrt(np.sum(norm1**2) * np.sum(norm2**2))
    
    if denominator <= 1e-9: return -1.0

    return numerator / denominator

@njit
def _compute_photometric_cost(u, v, depth, ref_patch, ref_pose, K, neighbors_data, patch_radius):
    """
    指定された深度値における光度測定コストを計算する。
    コストは (1 - NCC) / 2 で定義され、0に近いほど良い。
    """
    if ref_patch is None or np.isnan(depth) or depth <= 0.1:
        return 1.0  # 最大コスト

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    p_cam = np.array([x_cam, y_cam, depth], dtype=np.float32)

    R_ref, T_ref = ref_pose
    p_world = (R_ref @ p_cam) + T_ref

    total_cost = 0.0
    valid_views = 0
    
    # neighbors_dataは3つのNumPy配列からなるタプル
    neighbor_Rs, neighbor_Ts, neighbor_images_arr = neighbors_data

    # 4次元の画像配列をループ
    for i in range(neighbor_images_arr.shape[0]):
        R_neighbor, T_neighbor = neighbor_Rs[i], neighbor_Ts[i]
        
        p_neighbor_cam = R_neighbor.T @ (p_world - T_neighbor)

        if p_neighbor_cam[2] <= 0.1: continue

        uv_hom = K @ p_neighbor_cam
        if abs(uv_hom[2]) < 1e-6: continue
        
        u_proj, v_proj = uv_hom[0] / uv_hom[2], uv_hom[1] / uv_hom[2]
        
        h_neighbor = neighbor_images_arr.shape[1] 
        #v_proj = h_neighbor - 1 - v_proj

        current_neighbor_image = neighbor_images_arr[i]
        neighbor_patch = get_patch(current_neighbor_image, int(round(u_proj)), int(round(v_proj)), patch_radius)
        if neighbor_patch is None: continue

        ncc = compute_ncc(ref_patch, neighbor_patch)
        total_cost += (1.0 - ncc) / 2.0
        valid_views += 1

    return total_cost / valid_views if valid_views > 0 else 1.0


@njit(parallel=True)
def _patchmatch_iteration(depth_map, depth_costs, ref_image, ref_pose, K, neighbors_data, patch_radius, iteration, is_odd_iteration):
    """PatchMatchの1イテレーション (伝播とランダム探索)"""
    h, w = depth_map.shape
    new_depth_map = np.copy(depth_map)

    for y in prange(1, h - 1):
        for x_ in range(1, w - 1):
            x = x_ if is_odd_iteration else w - 1 - x_
            
            current_depth = new_depth_map[y, x]
            ref_patch = get_patch(ref_image, x, y, patch_radius)
            if ref_patch is None: continue

            best_cost = _compute_photometric_cost(x, y, current_depth, ref_patch, ref_pose, K, neighbors_data, patch_radius)

            neighbor_depth = new_depth_map[y, x - 1] if is_odd_iteration else new_depth_map[y, x + 1]
            if not np.isnan(neighbor_depth):
                cost = _compute_photometric_cost(x, y, neighbor_depth, ref_patch, ref_pose, K, neighbors_data, patch_radius)
                if cost < best_cost:
                    best_cost = cost
                    current_depth = neighbor_depth

            neighbor_depth = new_depth_map[y - 1, x] if is_odd_iteration else new_depth_map[y + 1, x]
            if not np.isnan(neighbor_depth):
                cost = _compute_photometric_cost(x, y, neighbor_depth, ref_patch, ref_pose, K, neighbors_data, patch_radius)
                if cost < best_cost:
                    best_cost = cost
                    current_depth = neighbor_depth
            
            search_range = depth_costs[y, x]
            if not np.isnan(search_range):
                search_range /= (iteration + 1)
                random_depth = current_depth + (np.random.random() - 0.5) * search_range
                
                if random_depth > 0:
                    cost = _compute_photometric_cost(x, y, random_depth, ref_patch, ref_pose, K, neighbors_data, patch_radius)
                    if cost < best_cost:
                        current_depth = random_depth

            new_depth_map[y, x] = current_depth

    return new_depth_map


class PointCloudFilter:
    def __init__(self, config):
        self.config = config

    def refine_depth_with_patchmatch(self, cur_idx, initial_depth, depth_costs, images, camera_data, K,
                                     iterations=3, patch_radius=3):
        """
        PatchMatchアルゴリズムを用いて深度マップを最適化する。
        """
        logging.info(f"Starting PatchMatch refinement for image {cur_idx}...")
        
        num_points = np.count_nonzero(~np.isnan(initial_depth))
        if num_points == 0:
            logging.warning(f"Image {cur_idx} has no valid initial depth points. Skipping refinement.")
            return initial_depth

        ref_image = images[cur_idx]
        _, pos_ref, quat_ref = camera_data[cur_idx]
        R_ref = quaternion_to_rotation_matrix(*quat_ref).astype(np.float32)
        T_ref = np.array(pos_ref, dtype=np.float32)
        ref_pose = (R_ref, T_ref)

        neighbor_Rs, neighbor_Ts, neighbor_images_list = [], [], []
        for off in (-2, -1, 1, 2):
            idx = cur_idx + off
            if 0 <= idx < len(camera_data) and idx in images:
                _, pos, quat = camera_data[idx]
                neighbor_Rs.append(quaternion_to_rotation_matrix(*quat).astype(np.float32))
                neighbor_Ts.append(np.array(pos, dtype=np.float32))
                neighbor_images_list.append(images[idx])
        
        if not neighbor_images_list:
            logging.warning(f"No neighbor views found for image {cur_idx}. Skipping refinement.")
            return initial_depth

        try:
            neighbor_images_arr = np.array(neighbor_images_list, dtype=np.uint8)
        except ValueError as e:
            logging.error(f"Could not stack neighbor images into a single array for image {cur_idx}. They might have different dimensions. Error: {e}")
            logging.warning("Skipping refinement for this image.")
            return initial_depth
        
        neighbors_data = (
            np.array(neighbor_Rs, dtype=np.float32), 
            np.array(neighbor_Ts, dtype=np.float32), 
            neighbor_images_arr
        )
        
        optimized_depth = np.copy(initial_depth)

        for it in range(iterations):
            logging.info(f" Iteration {it + 1}/{iterations}")
            is_odd = (it % 2 == 0)
            optimized_depth = _patchmatch_iteration(
                optimized_depth, depth_costs, ref_image, ref_pose, K.astype(np.float32), 
                neighbors_data, patch_radius, it, is_odd
            )

        logging.info(f"Finished PatchMatch refinement for image {cur_idx}.")
        return optimized_depth