import os
import numpy as np
import cv2
import logging
from numba import njit, prange

from utils import quaternion_to_rotation_matrix, clear_folder

def visualize_reprojection(
    point_3d,
    ref_image,
    ref_uv,
    K,
    neighbors_info,
    patch_radius,
    window_title_prefix="",
):
    """
    特定の3D点の再投影結果を可視化する関数。
    """
    # 参照画像の準備
    vis_ref_image = ref_image.copy()
    ref_u, ref_v = int(round(ref_uv[0])), int(round(ref_uv[1]))
    cv2.circle(vis_ref_image, (ref_u, ref_v), 7, (0, 255, 0), 2)
    ref_patch = get_patch(ref_image, ref_u, ref_v, patch_radius)
    if ref_patch is not None:
        cv2.imshow(f"{window_title_prefix} Reference Patch", cv2.cvtColor(ref_patch, cv2.COLOR_RGB2BGR))

    cv2.imshow(f"{window_title_prefix} Reference Image", cv2.cvtColor(vis_ref_image, cv2.COLOR_RGB2BGR))

    # 近傍画像への再投影
    neighbor_Rs, neighbor_Ts, neighbor_images = neighbors_info
    for i, (R, T, image) in enumerate(zip(neighbor_Rs, neighbor_Ts, neighbor_images)):
        p_cam = R.T @ (point_3d - T)
        if p_cam[2] < 0.1: continue

        uv_hom = K @ p_cam
        if abs(uv_hom[2]) < 1e-5: continue

        u, v = (uv_hom[0] / uv_hom[2]), (uv_hom[1] / uv_hom[2])
        v_disp = image.shape[0] - 1 - v

        vis_neighbor_image = image.copy()
        if 0 <= u < vis_neighbor_image.shape[1] and 0 <= v_disp < vis_neighbor_image.shape[0]:
            cv2.circle(vis_neighbor_image, (int(round(u)), int(round(v_disp))), 7, (0, 0, 255), 2)
            neighbor_patch = get_patch(image, int(round(u)), int(round(v)), patch_radius)
            if neighbor_patch is not None and ref_patch is not None:
                ncc = compute_ncc(ref_patch, neighbor_patch)
                cv2.putText(vis_neighbor_image, f"NCC: {ncc:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.imshow(f"{window_title_prefix} Neighbor Patch {i+1}", cv2.cvtColor(neighbor_patch, cv2.COLOR_RGB2BGR))
        
        cv2.imshow(f"{window_title_prefix} Neighbor Image {i+1}", cv2.cvtColor(vis_neighbor_image, cv2.COLOR_RGB2BGR))

    print(f"[{window_title_prefix}] Showing reprojection. Press any key in a window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@njit
def get_patch(image, u, v, patch_radius):
    h, w = image.shape[:2]
    u_min, u_max = u - patch_radius, u + patch_radius + 1
    v_min, v_max = v - patch_radius, v + patch_radius + 1
    if u_min < 0 or u_max > w or v_min < 0 or v_max > h:
        return None
    return image[v_min:v_max, u_min:u_max]


@njit
def compute_ncc(patch1, patch2):
    if patch1 is None or patch2 is None: return -1.0
    p1_gray = (patch1[:, :, 0] * 0.299 + patch1[:, :, 1] * 0.587 + patch1[:, :, 2] * 0.114).flatten().astype(np.float32)
    p2_gray = (patch2[:, :, 0] * 0.299 + patch2[:, :, 1] * 0.587 + patch2[:, :, 2] * 0.114).flatten().astype(np.float32)
    mean1, mean2 = np.mean(p1_gray), np.mean(p2_gray)
    norm1, norm2 = p1_gray - mean1, p2_gray - mean2
    numerator = np.sum(norm1 * norm2)
    denominator = np.sqrt(np.sum(norm1**2) * np.sum(norm2**2))
    if denominator <= 1e-6: return -1.0

    return numerator / denominator

@njit
def _compute_photometric_cost_njit(point_3d, ref_patch, K, neighbor_Rs, neighbor_Ts, neighbor_images, patch_radius):
    total_cost = 0.0
    valid_views = 0
    for i in range(len(neighbor_images)):
        R, T, image = neighbor_Rs[i], neighbor_Ts[i], neighbor_images[i]
        
        p_cam = R.T @ (point_3d - T)

        # Z座標がカメラより後ろ、または近すぎる場合はスキップ
        if p_cam[2] < 0.1:
            continue

        uv_hom = K @ p_cam
        
        if abs(uv_hom[2]) < 1e-5:
            continue

        u, v = uv_hom[0] / uv_hom[2], uv_hom[1] / uv_hom[2]
        v = image.shape[0] - 1 - v

        neighbor_patch = get_patch(image, int(round(u)), int(round(v)), patch_radius)
        if neighbor_patch is None: continue

        ncc = compute_ncc(ref_patch, neighbor_patch)
        total_cost += (1.0 - ncc) / 2.0
        valid_views += 1

    return total_cost / valid_views if valid_views > 0 else 1.0

MIN_SEARCH_RANGE = 0.05      # 5cm
MAX_SEARCH_RANGE = 1.0       # 1m
SAMPLING_RESOLUTION = 0.01   # 探索の刻み幅 (1cm)
MIN_NUM_STEPS = 5            # 最小の探索ステップ数
MAX_NUM_STEPS = 30           # 最大の探索ステップ数

@njit
def _patchmatch_refinement_step(current_point, ref_patch, R_ref, T_ref, K, neighbors_info_njit,
                                patch_radius, search_range):
    best_point = current_point
    neighbor_Rs, neighbor_Ts, neighbor_images = neighbors_info_njit
    best_cost = _compute_photometric_cost_njit(current_point, ref_patch, K, neighbor_Rs, neighbor_Ts, neighbor_images, patch_radius)
    
    p_cam_ref = R_ref.T @ (best_point - T_ref)

    if p_cam_ref[2] <= 0.1: return best_point

    initial_cam_z = p_cam_ref[2]
    
    z_start = max(0.1, initial_cam_z - search_range / 2.0)
    z_end = initial_cam_z + search_range / 2.0

    num_steps = int(search_range / 0.01) # SAMPLING_RESOLUTION
    if num_steps < 5: # MIN_NUM_STEPS
        num_steps = 5
    elif num_steps > 30: # MAX_NUM_STEPS
        num_steps = 30

    for _ in range(num_steps):
        z_cand = np.random.uniform(z_start, z_end)
        
        candidate_point_cam = np.array([p_cam_ref[0], p_cam_ref[1], z_cand], dtype=np.float32)
        candidate_point = (R_ref @ candidate_point_cam) + T_ref

        cost = _compute_photometric_cost_njit(candidate_point, ref_patch, K, neighbor_Rs, neighbor_Ts, neighbor_images, patch_radius)
        if cost < best_cost:
            best_cost = cost
            best_point = candidate_point
    
    return best_point


class PointCloudFilter:
    def __init__(self, config):
        self.config = config

    def refine_points_with_patchmatch(self, cur_idx, points, colors, uvs, depth_costs, images, camera_data, K,
                                      iterations=3, patch_radius=3):
        logging.info(f"Starting PatchMatch refinement for image {cur_idx} with {len(points)} points...")
        
        optimized_points = np.copy(points)
        num_points = points.shape[0]
        if num_points == 0: return np.array([]), np.array([])

        _, pos_ref, quat_ref = camera_data[cur_idx]
        R_ref = quaternion_to_rotation_matrix(*quat_ref).astype(np.float32)
        T_ref = np.array(pos_ref, dtype=np.float32)

        neighbor_Rs, neighbor_Ts, neighbor_images = [], [], []
        for off in (-2, -1, 1, 2):
            idx = cur_idx + off
            if 0 <= idx < len(camera_data) and idx in images:
                _, pos, quat = camera_data[idx]
                neighbor_Rs.append(quaternion_to_rotation_matrix(*quat).astype(np.float32))
                neighbor_Ts.append(np.array(pos, dtype=np.float32))
                neighbor_images.append(images[idx])
        
        if not neighbor_images:
            return points, colors
        
        neighbors_info_njit = (np.array(neighbor_Rs), np.array(neighbor_Ts), tuple(neighbor_images))
            
        ref_image = images[cur_idx]

        for it in range(iterations):
            logging.info(f" Iteration {it + 1}/{iterations}")
            
            for p_idx in range(num_points):
                point_search_range = depth_costs[p_idx] / (it + 1)
                point_search_range = np.clip(point_search_range, 0.05, 1.0) # MIN/MAX_SEARCH_RANGE

                ref_u, ref_v = uvs[p_idx]
                ref_patch = get_patch(ref_image, ref_u, ref_v, patch_radius)
                if ref_patch is None: continue
                
                optimized_points[p_idx] = _patchmatch_refinement_step(
                    optimized_points[p_idx], ref_patch, R_ref, T_ref, K, neighbors_info_njit,
                    patch_radius, point_search_range)

        logging.info(f"Finished PatchMatch refinement for image {cur_idx}.")
        return optimized_points, colors