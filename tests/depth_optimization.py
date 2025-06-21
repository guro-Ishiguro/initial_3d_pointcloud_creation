# tests/depth_optimization.py

import numpy as np
import cv2
from numba import njit, prange
import logging
import config
import os
from utils import save_depth_map_as_image


@njit(fastmath=True)
def _bilinear_interpolate_jit(image, y, x):
    h, w = image.shape
    x = np.float32(x)
    y = np.float32(y)
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1
    if x1 < 0 or x2 >= w or y1 < 0 or y2 >= h:
        return 0.0
    q11, q12, q21, q22 = image[y1, x1], image[y2, x1], image[y1, x2], image[y2, x2]
    w1, w2, w3, w4 = (
        (x2 - x) * (y2 - y),
        (x - x1) * (y2 - y),
        (x2 - x) * (y - y1),
        (x - x1) * (y - y1),
    )
    return w1 * q11 + w2 * q21 + w3 * q12 + w4 * q22


@njit(fastmath=True)
def _compute_homography_jit(
    K_ref, R_ref, T_ref, K_src, R_src, T_src, plane_point_3d, plane_normal
):
    K_src = np.ascontiguousarray(K_src)
    R_src = np.ascontiguousarray(R_src)
    R_ref_inv = R_ref.T
    T_ref_inv = -R_ref_inv @ T_ref
    p_ref = (R_ref @ plane_point_3d.T + T_ref).T
    n_ref = R_ref @ plane_normal
    R_rel = R_src @ R_ref_inv
    T_rel = (R_src @ T_ref_inv.reshape(3, 1) + T_src.reshape(3, 1)).flatten()
    d = np.dot(n_ref, p_ref)
    if abs(d) < 1e-8:
        return np.eye(3, dtype=np.float32)
    H = R_rel + (T_rel.reshape(3, 1) @ n_ref.reshape(1, 3)) / d
    return np.ascontiguousarray(K_src) @ np.ascontiguousarray(H) @ np.linalg.inv(K_ref)


@njit(fastmath=True)
def _compute_weighted_zncc_cost_jit(patch_ref, warped_patch_src, sigma_color):
    """
    適応的支持領域重み付けを用いてZNCCコストを計算する。
    """
    patch_size = patch_ref.shape[0]
    half = patch_size // 2
    center_val = patch_ref[half, half]

    # 1. 重みカーネルを計算
    weights = np.zeros_like(patch_ref, dtype=np.float32)
    for r in range(patch_size):
        for c in range(patch_size):
            color_diff_sq = (patch_ref[r, c] - center_val) ** 2
            weights[r, c] = np.exp(-color_diff_sq / (2 * sigma_color**2))

    # 2. 重み付き統計量を計算
    sum_w = np.sum(weights)
    if sum_w < 1e-6:
        return 1.0

    mean_ref = np.sum(patch_ref * weights) / sum_w
    mean_src = np.sum(warped_patch_src * weights) / sum_w

    var_ref = np.sum(weights * (patch_ref - mean_ref) ** 2) / sum_w
    var_src = np.sum(weights * (warped_patch_src - mean_src) ** 2) / sum_w

    std_ref = np.sqrt(var_ref)
    std_src = np.sqrt(var_src)

    if std_ref < config.ZNCC_EPSILON or std_src < config.ZNCC_EPSILON:
        return 1.0

    # 3. 重み付きZNCCを計算
    numerator = (
        np.sum(weights * (patch_ref - mean_ref) * (warped_patch_src - mean_src)) / sum_w
    )
    denominator = std_ref * std_src

    # ZNCC値は-1から1なので、コストを0から1の範囲に変換
    return (1.0 - (numerator / denominator)) / 2.0


@njit(fastmath=True)
def _evaluate_cost_jit(
    r,
    c,
    depth,
    normal,
    patch_size,
    ref_image_gray,
    ref_pose_K,
    ref_pose_R,
    ref_pose_T,
    src_images_gray,
    src_K,
    src_R,
    src_T,
    top_k_costs,
    adaptive_weight_sigma_color,
):
    h, w = ref_image_gray.shape
    half = patch_size // 2

    if r - half < 0 or r + half + 1 > h or c - half < 0 or c + half + 1 > w:
        return 1.0

    x_cam = (c - ref_pose_K[0, 2]) * depth / ref_pose_K[0, 0]
    y_cam = (r - ref_pose_K[1, 2]) * depth / ref_pose_K[1, 1]
    point_3d_cam = np.array([x_cam, y_cam, depth], dtype=np.float32)
    R_ref_inv = ref_pose_R.T
    point_3d_world = R_ref_inv @ (point_3d_cam - ref_pose_T)
    normal_world = R_ref_inv @ normal
    patch_ref = ref_image_gray[r - half : r + half + 1, c - half : c + half + 1]
    num_neighbors = src_images_gray.shape[0]
    costs = np.zeros(num_neighbors, dtype=np.float32)
    for i in range(num_neighbors):
        H = _compute_homography_jit(
            ref_pose_K,
            ref_pose_R,
            ref_pose_T,
            src_K[i],
            src_R[i],
            src_T[i],
            point_3d_world,
            normal_world,
        )
        if np.isnan(H).any() or np.isinf(H).any():
            costs[i] = 1.0
            continue
        warped_patch = np.zeros_like(patch_ref, dtype=np.float32)
        for pr in range(patch_size):
            for pc in range(patch_size):
                u_ref, v_ref = c - half + pc, r - half + pr
                p_ref_h = np.array([u_ref, v_ref, 1.0], dtype=np.float32)
                p_src_h = H @ p_ref_h
                if abs(p_src_h[2]) < 1e-8:
                    warped_patch[pr, pc] = 0.0
                    continue
                u_src, v_src = p_src_h[0] / p_src_h[2], p_src_h[1] / p_src_h[2]
                warped_patch[pr, pc] = _bilinear_interpolate_jit(
                    src_images_gray[i], v_src, u_src
                )
        costs[i] = _compute_weighted_zncc_cost_jit(
            patch_ref, warped_patch, adaptive_weight_sigma_color
        )

    costs = np.sort(costs)
    top_k = min(top_k_costs, len(costs))
    return np.mean(costs[:top_k])


@njit(parallel=True, fastmath=True)
def _propagate_spatial_one_color_jit(
    depth_map,
    normal_map,
    cost_map,  # 更新対象のマップ
    propagation_mask,  # ★追加: 処理領域を制限するマスク
    neighbors_dr,
    neighbors_dc,  # 伝播方向 (NumPy配列に修正)
    color,  # 対象の色
    r_min,
    r_max,
    c_min,
    c_max,  # 処理範囲
    # パラメータ
    patch_size,
    top_k_costs,
    adaptive_weight_sigma_color,
    # 参照ビューのデータ
    ref_image_gray,
    ref_pose_K,
    ref_pose_R,
    ref_pose_T,
    # ソースビューのデータ
    src_images_gray,
    src_K,
    src_R,
    src_T,
):
    """
    指定された範囲のチェッカーボードの一つの色（赤または黒）のピクセルに対して空間伝播を実行する。
    """
    h, w = depth_map.shape
    # 指定された範囲のピクセルを並列で処理
    for r in prange(max(1, r_min), min(h - 1, r_max)):
        for c in range(max(1, c_min), min(w - 1, c_max)):
            # ★追加: マスク外のピクセルはスキップ
            if not propagation_mask[r, c]:
                continue

            # 対象の色（赤 or 黒）のピクセルのみを処理
            if (r + c) % 2 != color:
                continue

            if not np.isfinite(depth_map[r, c]):
                continue

            # 隣接ピクセルからより良い平面（深度と法線）を伝播させる
            for i in range(len(neighbors_dr)):
                dr = neighbors_dr[i]
                dc = neighbors_dc[i]
                nr, nc = r + dr, c + dc

                # 隣接ピクセルもマスク内である必要がある
                if not (0 <= nr < h and 0 <= nc < w and propagation_mask[nr, nc]):
                    continue

                neighbor_depth = depth_map[nr, nc]
                if not np.isfinite(neighbor_depth):
                    continue

                # 現在のピクセル(r, c)の位置で、隣接ピクセル(nr, nc)の平面を評価
                neighbor_normal = normal_map[nr, nc]
                new_cost = _evaluate_cost_jit(
                    r,
                    c,
                    neighbor_depth,
                    neighbor_normal,
                    patch_size,
                    ref_image_gray,
                    ref_pose_K,
                    ref_pose_R,
                    ref_pose_T,
                    src_images_gray,
                    src_K,
                    src_R,
                    src_T,
                    top_k_costs,
                    adaptive_weight_sigma_color,
                )

                # コストが改善されれば、現在のピクセルの平面を更新
                if new_cost < cost_map[r, c]:
                    depth_map[r, c] = neighbor_depth
                    normal_map[r, c] = neighbor_normal
                    cost_map[r, c] = new_cost


@njit(parallel=True, fastmath=True)
def _propagate_wavefront_jit(
    depth_map,
    normal_map,
    cost_map,
    visited_map,
    propagation_mask,  # ★追加: 処理領域を制限するマスク
    initial_wavefront_np,
    patch_size,
    top_k_costs,
    adaptive_weight_sigma_color,
    ref_image_gray,
    ref_pose_K,
    ref_pose_R,
    ref_pose_T,
    src_images_gray,
    src_K,
    src_R,
    src_T,
):
    """
    シードグリッドから波面を広げるように空間伝播を実行する。
    """
    h, w = depth_map.shape

    wavefront_A = np.zeros((h * w, 2), dtype=np.int32)
    wavefront_B = np.zeros((h * w, 2), dtype=np.int32)

    wave_A_count = len(initial_wavefront_np)
    if wave_A_count == 0:
        return
    wavefront_A[:wave_A_count] = initial_wavefront_np

    for i in range(wave_A_count):
        r, c = initial_wavefront_np[i]
        visited_map[r, c] = True

    current_wave = wavefront_A
    next_wave = wavefront_B
    current_count = wave_A_count

    neighbors_dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int8)
    neighbors_dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int8)

    loop_count = 0
    max_loops = h + w

    while current_count > 0 and loop_count < max_loops:
        for i in prange(current_count):
            r, c = current_wave[i]

            best_new_cost = cost_map[r, c]
            best_depth = depth_map[r, c]
            best_normal = normal_map[r, c]

            for j in range(len(neighbors_dr)):
                nr, nc = r + neighbors_dr[j], c + neighbors_dc[j]

                if 0 <= nr < h and 0 <= nc < w and visited_map[nr, nc]:
                    neighbor_depth = depth_map[nr, nc]
                    if not np.isfinite(neighbor_depth):
                        continue

                    neighbor_normal = normal_map[nr, nc]

                    new_cost = _evaluate_cost_jit(
                        r,
                        c,
                        neighbor_depth,
                        neighbor_normal,
                        patch_size,
                        ref_image_gray,
                        ref_pose_K,
                        ref_pose_R,
                        ref_pose_T,
                        src_images_gray,
                        src_K,
                        src_R,
                        src_T,
                        top_k_costs,
                        adaptive_weight_sigma_color,
                    )

                    if new_cost < best_new_cost:
                        best_new_cost = new_cost
                        best_depth = neighbor_depth
                        best_normal = neighbor_normal

            if best_new_cost < cost_map[r, c]:
                cost_map[r, c] = best_new_cost
                depth_map[r, c] = best_depth
                normal_map[r, c] = best_normal

        next_wave_count = 0
        for i in range(current_count):
            r, c = current_wave[i]

            for j in range(len(neighbors_dr)):
                nr, nc = r + neighbors_dr[j], c + neighbors_dc[j]

                # ★修正: マスク内かつ未訪問のピクセルを次の波面に追加
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and propagation_mask[nr, nc]
                    and not visited_map[nr, nc]
                ):
                    visited_map[nr, nc] = True
                    if next_wave_count < len(next_wave):
                        next_wave[next_wave_count, 0] = nr
                        next_wave[next_wave_count, 1] = nc
                        next_wave_count += 1

        current_wave, next_wave = next_wave, current_wave
        current_count = next_wave_count
        loop_count += 1


@njit(parallel=True, fastmath=True)
def _random_search_jit(
    depth_map,
    normal_map,
    cost_map,
    search_mask,  # ★追加: 処理領域を制限するマスク
    iteration,
    patch_size,
    top_k_costs,
    decay_rate,
    normal_search_angle,
    ref_image_gray,
    ref_pose_K,
    ref_pose_R,
    ref_pose_T,
    src_images_gray,
    src_K,
    src_R,
    src_T,
    adaptive_weight_sigma_color,
    depth_range_map,
):
    """画像全体に対してランダム探索を実行する"""
    h, w = depth_map.shape
    for r in prange(h):
        for c in range(w):
            # ★追加: マスク外のピクセルはスキップ
            if not search_mask[r, c]:
                continue

            if not np.isfinite(depth_map[r, c]):
                continue
            d_current = depth_map[r, c]
            d_range = depth_range_map[r, c]
            d_new = d_current + (np.random.rand() * 2 - 1) * d_range
            if d_new <= 0:
                continue

            n_current = normal_map[r, c]
            angle_rad = np.radians(
                (np.random.rand() * 2 - 1)
                * normal_search_angle
                * (decay_rate**iteration)
            )
            rand_axis = np.random.randn(3).astype(np.float32)
            rand_axis /= np.linalg.norm(rand_axis)
            cos_a, sin_a = np.float32(np.cos(angle_rad)), np.float32(np.sin(angle_rad))
            one_minus_cos_a = np.float32(1.0) - cos_a
            n_new = (
                n_current * cos_a
                + np.cross(rand_axis, n_current) * sin_a
                + rand_axis * np.dot(rand_axis, n_current) * one_minus_cos_a
            )
            n_new /= np.linalg.norm(n_new)

            new_cost = _evaluate_cost_jit(
                r,
                c,
                d_new,
                n_new,
                patch_size,
                ref_image_gray,
                ref_pose_K,
                ref_pose_R,
                ref_pose_T,
                src_images_gray,
                src_K,
                src_R,
                src_T,
                top_k_costs,
                adaptive_weight_sigma_color,
            )
            if new_cost < cost_map[r, c]:
                depth_map[r, c], normal_map[r, c], cost_map[r, c] = (
                    d_new,
                    n_new,
                    new_cost,
                )


@njit(fastmath=True)
def _check_geometric_consistency_jit(
    r_ref, c_ref, d_ref, K_ref, R_ref, T_ref, neighbor_poses, neighbor_depth_maps
):
    """
    参照ピクセルの深度値が、近傍ビューの深度マップと幾何学的に一貫しているかチェックする
    """
    consistent_views = 0
    h, w = neighbor_depth_maps[0].shape

    # 参照ビューのカメラ座標系での3D点を計算
    x_cam_ref = (c_ref - K_ref[0, 2]) * d_ref / K_ref[0, 0]
    y_cam_ref = (r_ref - K_ref[1, 2]) * d_ref / K_ref[1, 1]
    point_3d_cam_ref = np.array([x_cam_ref, y_cam_ref, d_ref], dtype=np.float32)

    # ワールド座標に変換
    point_3d_world = R_ref.T @ (point_3d_cam_ref - T_ref)

    # 各近傍ビューでチェック
    for i in range(len(neighbor_poses)):
        K_src, R_src, T_src = neighbor_poses[i]
        depth_map_src = neighbor_depth_maps[i]

        # 近傍ビューのカメラ座標に変換
        p_src_cam = R_src @ point_3d_world + T_src

        # カメラの後ろにある点は無視
        if p_src_cam[2] <= 0:
            continue

        # 近傍ビューの画像座標に投影
        p_src_img_h = K_src @ p_src_cam
        d_proj_src = p_src_img_h[2]  # 逆投影された深度
        u_src, v_src = p_src_img_h[0] / d_proj_src, p_src_img_h[1] / d_proj_src

        # 画像範囲外かチェック (整数座標に丸める前にチェック)
        if not (0 <= u_src < w and 0 <= v_src < h):
            continue

        # 最も近いピクセルの深度値を取得 (補間は複雑なので最近傍で)
        r_src, c_src = int(round(v_src)), int(round(u_src))

        if not (0 <= c_src < w and 0 <= r_src < h):
            continue

        d_actual_src = depth_map_src[r_src, c_src]

        # 近傍ビューの深度が有効かチェック
        if not np.isfinite(d_actual_src) or d_actual_src <= 0:
            continue

        # 幾何学的なエラーを計算 (相対深度差)
        relative_error = np.abs(d_proj_src - d_actual_src) / d_actual_src

        if relative_error < config.GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD:
            consistent_views += 1

    return consistent_views


@njit(fastmath=True)
def _check_photometric_consistency_jit(
    r,
    c,
    depth,
    ref_color_pixel,
    K,
    R_ref,
    T_ref,
    neighbor_images,
    neighbor_R,
    neighbor_T,
):
    """
    指定されたピクセルの深度値が、近傍ビューと光度的に一貫しているかチェックする
    """
    consistent_views = 0
    h, w, _ = neighbor_images[0].shape

    # 参照ビューのカメラ座標系での3D点を計算
    x_cam = (c - K[0, 2]) * depth / K[0, 0]
    y_cam = (r - K[1, 2]) * depth / K[1, 1]
    point_3d_cam = np.array([x_cam, y_cam, depth], dtype=np.float32)

    # ワールド座標に変換
    point_3d_world = R_ref.T @ (point_3d_cam - T_ref)

    # 各近傍ビューでチェック
    for i in range(len(neighbor_images)):
        R_src, T_src = neighbor_R[i], neighbor_T[i]

        # 近傍ビューのカメラ座標に変換
        p_src_cam = R_src @ point_3d_world + T_src

        # カメラの後ろにある点は無視
        if p_src_cam[2] <= 0:
            continue

        # 画像座標に投影
        p_src_img_h = K @ p_src_cam
        u_src, v_src = p_src_img_h[0] / p_src_img_h[2], p_src_img_h[1] / p_src_img_h[2]

        # 画像範囲内かチェック
        if not (0 <= u_src < w and 0 <= v_src < h):
            continue

        # 双線形補間で色を取得
        y, x = v_src, u_src
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1
        if not (x1 >= 0 and x2 < w and y1 >= 0 and y2 < h):
            continue

        q11, q12 = neighbor_images[i][y1, x1], neighbor_images[i][y1, x2]
        q21, q22 = neighbor_images[i][y2, x1], neighbor_images[i][y2, x2]
        w1, w2, w3, w4 = (
            (x2 - x) * (y2 - y),
            (x - x1) * (y2 - y),
            (x2 - x) * (y - y1),
            (x - x1) * (y - y1),
        )
        neighbor_color = w1 * q11 + w2 * q12 + w3 * q21 + w4 * q22

        # 色の差を計算 (L2ノルム)
        color_diff = np.sqrt(
            np.sum(
                (ref_color_pixel.astype(np.float32) - neighbor_color.astype(np.float32))
                ** 2
            )
        )

        if color_diff < config.FILTERING_COLOR_DIFFERENCE_THRESHOLD:
            consistent_views += 1

    return consistent_views


@njit(parallel=True)
def _initialize_normals_from_depth_jit(depth_map, K):
    h, w = depth_map.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    for r in prange(1, h - 1):
        for c in range(1, w - 1):
            if not np.isfinite(depth_map[r, c]):
                continue
            p_center = np.array(
                [
                    (c - cx) * depth_map[r, c] / fx,
                    (r - cy) * depth_map[r, c] / fy,
                    depth_map[r, c],
                ],
                dtype=np.float32,
            )
            p_right = np.array(
                [
                    (c + 1 - cx) * depth_map[r, c + 1] / fx,
                    (r - cy) * depth_map[r, c + 1] / fy,
                    depth_map[r, c + 1],
                ],
                dtype=np.float32,
            )
            p_down = np.array(
                [
                    (c - cx) * depth_map[r + 1, c] / fx,
                    (r + 1 - cy) * depth_map[r + 1, c] / fy,
                    depth_map[r + 1, c],
                ],
                dtype=np.float32,
            )
            if not (np.isfinite(p_right).all() and np.isfinite(p_down).all()):
                continue
            v_c, v_r = p_right - p_center, p_down - p_center
            normal = np.cross(v_r, v_c)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normals[r, c] = normal / norm
            else:
                normals[r, c] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


class DepthOptimization:
    def __init__(self, config):
        self.config = config
        if not hasattr(self.config, "ADAPTIVE_WEIGHT_SIGMA_COLOR"):
            logging.warning(
                "ADAPTIVE_WEIGHT_SIGMA_COLOR not found in config. Using default value 10.0."
            )
            self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR = 10.0

    def _debug_patch_visualization(
        self,
        r,
        c,
        depth,
        normal,
        ref_image,
        ref_pose,
        neighbor_views_data,
        title_prefix="",
    ):
        h, w, _ = ref_image.shape
        patch_size = self.config.PATCHMATCH_PATCH_SIZE
        half = patch_size // 2

        patch_display_size = (250, 250)

        orig_h, orig_w = ref_image.shape[:2]
        target_h = patch_display_size[1]
        target_w = int(orig_w * (target_h / orig_h))
        full_image_display_size = (target_w, target_h)

        K_ref, R_ref, T_ref = ref_pose["K"], ref_pose["R"], ref_pose["T"]
        x_cam = (c - K_ref[0, 2]) * depth / K_ref[0, 0]
        y_cam = (r - K_ref[1, 2]) * depth / K_ref[1, 1]
        point_3d_cam = np.array([x_cam, y_cam, depth])
        print(
            f"The coordinates of the pixel of interest in the camera coordinate system are {point_3d_cam}"
        )
        point_3d_world = R_ref.T @ (point_3d_cam - T_ref)
        print(
            f"The coordinates of the pixel of interest in the world coordinate system are {point_3d_world}"
        )
        normal_world = R_ref.T @ normal

        ref_image_with_point = ref_image.copy()
        cv2.rectangle(
            ref_image_with_point,
            (c - half, r - half),
            (c + half, r + half),
            (0, 0, 255),
            2,
        )
        cv2.circle(ref_image_with_point, (c, r), 5, (0, 255, 0), -1)
        ref_image_with_point_display = cv2.resize(
            ref_image_with_point,
            full_image_display_size,
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.putText(
            ref_image_with_point_display,
            "Full Reference",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        ref_patch = ref_image[r - half : r + half + 1, c - half : c + half + 1]
        ref_patch_display = cv2.resize(
            ref_patch, patch_display_size, interpolation=cv2.INTER_NEAREST
        )
        cv2.putText(
            ref_patch_display,
            "Reference Patch",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        for i, neighbor_view in enumerate(neighbor_views_data):
            src_image = neighbor_view["image"]
            H = _compute_homography_jit(
                K_ref.astype(np.float32),
                R_ref.astype(np.float32),
                T_ref.astype(np.float32),
                neighbor_view["K"].astype(np.float32),
                neighbor_view["R"].astype(np.float32),
                neighbor_view["T"].astype(np.float32),
                point_3d_world.astype(np.float32),
                normal_world.astype(np.float32),
            )

            warped_image = cv2.warpPerspective(ref_image, H, (w, h))
            warped_patch = warped_image[
                r - half : r + half + 1, c - half : c + half + 1
            ]
            warped_patch_display = cv2.resize(
                warped_patch, patch_display_size, interpolation=cv2.INTER_NEAREST
            )
            cv2.putText(
                warped_patch_display,
                f"Warped (View {i})",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            ref_corners = np.array(
                [
                    [c - half, r - half, 1],
                    [c + half, r - half, 1],
                    [c + half, r + half, 1],
                    [c - half, r + half, 1],
                ],
                dtype=np.float32,
            )
            transformed_corners_h = (H @ ref_corners.T).T
            transformed_corners_2d = (
                transformed_corners_h[:, :2] / (transformed_corners_h[:, 2:] + 1e-8)
            ).astype(np.int32)

            source_with_box = src_image.copy()
            cv2.polylines(
                source_with_box,
                [transformed_corners_2d],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )
            source_with_box_display = cv2.resize(
                source_with_box, full_image_display_size
            )
            cv2.putText(
                source_with_box_display,
                f"Location (View {i})",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            combined_view = np.hstack(
                [
                    ref_image_with_point_display,
                    ref_patch_display,
                    warped_patch_display,
                    source_with_box_display,
                ]
            )

            cv2.imshow(
                f"{title_prefix} - Debug View {i}",
                cv2.cvtColor(combined_view, cv2.COLOR_RGB2BGR),
            )

        print(
            f"{title_prefix} | Depth: {depth:.3f}, Normal: [{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}]"
        )

    def refine_depth_with_patchmatch(
        self,
        initial_depth,
        initial_depth_error,
        ref_image,
        ref_pose,
        neighbor_views_data,
        ref_idx=0,
    ):
        logging.info(
            f"Starting PatchMatch MVS depth refinement using '{self.config.CHOICED_PROPAGATION_METHOD}' method..."
        )

        h, w = initial_depth.shape
        depth_map = initial_depth.astype(np.float32)
        normal_map = _initialize_normals_from_depth_jit(
            depth_map, ref_pose["K"].astype(np.float32)
        )

        valid_initial_mask = np.isfinite(initial_depth)
        kernel = np.ones((7, 7), np.uint8)  # 膨張させるカーネルサイズ
        propagation_mask = cv2.dilate(
            valid_initial_mask.astype(np.uint8), kernel, iterations=1
        ).astype(np.bool_)

        ref_image_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ref_pose_K, ref_pose_R, ref_pose_T = (
            ref_pose["K"].astype(np.float32),
            ref_pose["R"].astype(np.float32),
            ref_pose["T"].astype(np.float32),
        )
        src_images_gray = np.stack(
            [
                cv2.cvtColor(view["image"], cv2.COLOR_RGB2GRAY).astype(np.float32)
                for view in neighbor_views_data
            ],
            axis=0,
        )
        src_K = np.stack(
            [view["K"].astype(np.float32) for view in neighbor_views_data], axis=0
        )
        src_R = np.stack(
            [view["R"].astype(np.float32) for view in neighbor_views_data], axis=0
        )
        src_T = np.stack(
            [view["T"].astype(np.float32) for view in neighbor_views_data], axis=0
        )
        cost_map = np.full((h, w), np.inf, dtype=np.float32)
        depth_map_prev = np.zeros_like(depth_map)
        convergence_threshold = 0.001

        if self.config.CHOICED_PROPAGATION_METHOD == "priority":
            logging.info("Calculating grid costs for priority propagation...")
            grid_rows = self.config.PROPAGATION_GRID_ROWS
            grid_cols = self.config.PROPAGATION_GRID_COLS
            grid_h = h // grid_rows
            grid_w = w // grid_cols
            grid_costs = np.full((grid_rows, grid_cols), np.inf, dtype=np.float32)

            for r_idx in range(grid_rows):
                for c_idx in range(grid_cols):
                    r_start = r_idx * grid_h
                    r_end = (r_idx + 1) * grid_h
                    c_start = c_idx * grid_w
                    c_end = (c_idx + 1) * grid_w
                    grid_patch = initial_depth_error[r_start:r_end, c_start:c_end]
                    valid_costs = grid_patch[np.isfinite(grid_patch)]
                    if len(valid_costs) > 0:
                        grid_costs[r_idx, c_idx] = np.median(valid_costs)

        # --- PatchMatch反復ループ ---
        for i in range(self.config.PATCHMATCH_ITERATIONS):
            np.copyto(depth_map_prev, depth_map)
            logging.info(
                f"PatchMatch Iteration {i+1}/{self.config.PATCHMATCH_ITERATIONS}"
            )

            # --- 1. 空間伝播 ---
            if self.config.CHOICED_PROPAGATION_METHOD == "checkerboard":
                if i % 2 == 0:
                    neighbors_dr = np.array([-1, 0], dtype=np.int8)
                    neighbors_dc = np.array([0, -1], dtype=np.int8)
                else:
                    neighbors_dr = np.array([1, 0], dtype=np.int8)
                    neighbors_dc = np.array([0, 1], dtype=np.int8)
                _propagate_spatial_one_color_jit(
                    depth_map,
                    normal_map,
                    cost_map,
                    propagation_mask,
                    neighbors_dr,
                    neighbors_dc,
                    0,
                    0,
                    h,
                    0,
                    w,
                    self.config.PATCHMATCH_PATCH_SIZE,
                    self.config.TOP_K_COSTS,
                    self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                    ref_image_gray,
                    ref_pose_K,
                    ref_pose_R,
                    ref_pose_T,
                    src_images_gray,
                    src_K,
                    src_R,
                    src_T,
                )
                _propagate_spatial_one_color_jit(
                    depth_map,
                    normal_map,
                    cost_map,
                    propagation_mask,
                    neighbors_dr,
                    neighbors_dc,
                    1,
                    0,
                    h,
                    0,
                    w,
                    self.config.PATCHMATCH_PATCH_SIZE,
                    self.config.TOP_K_COSTS,
                    self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                    ref_image_gray,
                    ref_pose_K,
                    ref_pose_R,
                    ref_pose_T,
                    src_images_gray,
                    src_K,
                    src_R,
                    src_T,
                )
            elif self.config.CHOICED_PROPAGATION_METHOD == "priority":
                if i == 0:
                    logging.info(
                        "Starting wavefront propagation from the best seed grid..."
                    )
                    visited_map = np.zeros_like(initial_depth, dtype=np.bool_)
                    grid_rows, grid_cols = (
                        self.config.PROPAGATION_GRID_ROWS,
                        self.config.PROPAGATION_GRID_COLS,
                    )
                    grid_h, grid_w = h // grid_rows, w // grid_cols
                    valid_grid_indices = np.where(np.isfinite(grid_costs))
                    if len(valid_grid_indices[0]) > 0:
                        min_cost_flat_idx = np.nanargmin(grid_costs)
                        r_idx, c_idx = np.unravel_index(
                            min_cost_flat_idx, grid_costs.shape
                        )
                        min_cost = grid_costs[r_idx, c_idx]
                        r_start, r_end = r_idx * grid_h, (r_idx + 1) * grid_h
                        c_start, c_end = c_idx * grid_w, (c_idx + 1) * grid_w
                        logging.info(
                            f"Seed grid found at index ({r_idx}, {c_idx}) with cost {min_cost:.4f}."
                        )

                        # シードグリッドはマスク内のみを訪問済みとする
                        seed_mask = np.zeros_like(visited_map)
                        seed_mask[r_start:r_end, c_start:c_end] = True
                        visited_map[seed_mask & propagation_mask] = True

                        initial_wavefront = []
                        for r_ in range(r_start, r_end):
                            for c_ in range(c_start, c_end):
                                if not visited_map[r_, c_]:
                                    continue
                                for dr in [-1, 0, 1]:
                                    for dc in [-1, 0, 1]:
                                        if dr == 0 and dc == 0:
                                            continue
                                        nr, nc = r_ + dr, c_ + dc
                                        if (
                                            0 <= nr < h
                                            and 0 <= nc < w
                                            and propagation_mask[nr, nc]
                                            and not visited_map[nr, nc]
                                        ):
                                            initial_wavefront.append((nr, nc))

                        initial_wavefront_np = np.array(
                            list(set(initial_wavefront)), dtype=np.int32
                        )

                        if len(initial_wavefront_np) > 0:
                            _propagate_wavefront_jit(
                                depth_map,
                                normal_map,
                                cost_map,
                                visited_map,
                                propagation_mask,
                                initial_wavefront_np,
                                self.config.PATCHMATCH_PATCH_SIZE,
                                self.config.TOP_K_COSTS,
                                self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                                ref_image_gray,
                                ref_pose_K,
                                ref_pose_R,
                                ref_pose_T,
                                src_images_gray,
                                src_K,
                                src_R,
                                src_T,
                            )
                        else:
                            logging.warning("Initial wavefront is empty.")
                    else:
                        logging.warning("No valid grids to start propagation.")
                else:
                    logging.info(
                        "Skipping spatial propagation for subsequent iterations."
                    )

            # --- 2. ランダム探索 ---
            depth_range_map = (
                initial_depth_error * (self.config.PATCHMATCH_DECAY_RATE**i)
            ).astype(np.float32)
            _random_search_jit(
                depth_map,
                normal_map,
                cost_map,
                propagation_mask,
                i,
                self.config.PATCHMATCH_PATCH_SIZE,
                self.config.TOP_K_COSTS,
                self.config.PATCHMATCH_DECAY_RATE,
                self.config.PATCHMATCH_NORMAL_SEARCH_ANGLE,
                ref_image_gray,
                ref_pose_K,
                ref_pose_R,
                ref_pose_T,
                src_images_gray,
                src_K,
                src_R,
                src_T,
                self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                depth_range_map,
            )
            if self.config.DEBUG_SAVE_DEPTH_MAPS:
                save_each_depth_dir = os.path.join(
                    config.DEPTH_IMAGE_DIR, f"depth_{ref_idx:04d}"
                )
                save_path = os.path.join(
                    save_each_depth_dir,
                    f"depth_iter_{i+1:02d}.png",
                )
                logging.info(f"Saving intermediate depth map to {save_path}")
                save_depth_map_as_image(depth_map.copy(), save_path)

            diff = np.abs(depth_map_prev - depth_map)
            valid_mask = (
                np.isfinite(depth_map_prev)
                & (depth_map_prev != 0)
                & np.isfinite(depth_map)
            )
            if np.sum(valid_mask) > 0:
                mean_change = np.mean(diff[valid_mask] / depth_map_prev[valid_mask])
                logging.info(f"Average depth change: {mean_change:.5f}")
                if mean_change < convergence_threshold:
                    logging.info(
                        f"Convergence reached at iteration {i+1}. Stopping early."
                    )
                    break
            else:
                logging.warning("No valid pixels for convergence check.")

        logging.info("PatchMatch MVS refinement finished.")
        final_depth_map = depth_map.copy()
        final_depth_map[~propagation_mask] = np.nan
        return final_depth_map

    def refine_depth_with_patchmatch_vanilla(
        self, ref_image, ref_pose, neighbor_views_data, ref_idx=0
    ):
        """
        通常のPatchMatch MVSを実行。深度は一様乱数で初期化し、探索範囲は固定値から減衰させる。
        """
        logging.info(
            "Starting VANILLA PatchMatch MVS depth refinement (random initialization)..."
        )
        h, w, _ = ref_image.shape

        # 1. 深度マップを一様乱数で初期化
        min_depth = self.config.PATCHMATCH_VANILLA_MIN_DEPTH
        max_depth = self.config.PATCHMATCH_VANILLA_MAX_DEPTH
        depth_map = np.random.uniform(min_depth, max_depth, (h, w)).astype(np.float32)

        # 2. 法線マップを初期化
        normal_map = _initialize_normals_from_depth_jit(
            depth_map, ref_pose["K"].astype(np.float32)
        )

        # 3. JITコンパイル用にデータを準備
        ref_image_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ref_pose_K, ref_pose_R, ref_pose_T = (
            ref_pose["K"].astype(np.float32),
            ref_pose["R"].astype(np.float32),
            ref_pose["T"].astype(np.float32),
        )
        src_images_gray = np.stack(
            [
                cv2.cvtColor(view["image"], cv2.COLOR_RGB2GRAY).astype(np.float32)
                for view in neighbor_views_data
            ],
            axis=0,
        )
        src_K = np.stack(
            [view["K"].astype(np.float32) for view in neighbor_views_data], axis=0
        )
        src_R = np.stack(
            [view["R"].astype(np.float32) for view in neighbor_views_data], axis=0
        )
        src_T = np.stack(
            [view["T"].astype(np.float32) for view in neighbor_views_data], axis=0
        )
        cost_map = np.full((h, w), np.inf, dtype=np.float32)

        # 4. PatchMatch反復ループ
        for i in range(self.config.PATCHMATCH_ITERATIONS):
            logging.info(
                f"Vanilla PatchMatch Iteration {i+1}/{self.config.PATCHMATCH_ITERATIONS}"
            )

            # --- 空間伝播 ---
            if i % 2 == 0:
                neighbors_dr = np.array([-1, 0], dtype=np.int8)
                neighbors_dc = np.array([0, -1], dtype=np.int8)
            else:
                neighbors_dr = np.array([1, 0], dtype=np.int8)
                neighbors_dc = np.array([0, 1], dtype=np.int8)

            _propagate_spatial_one_color_jit(
                depth_map,
                normal_map,
                cost_map,
                neighbors_dr,
                neighbors_dc,
                0,
                0,
                h,
                0,
                w,
                self.config.PATCHMATCH_PATCH_SIZE,
                self.config.TOP_K_COSTS,
                self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                ref_image_gray,
                ref_pose_K,
                ref_pose_R,
                ref_pose_T,
                src_images_gray,
                src_K,
                src_R,
                src_T,
            )
            _propagate_spatial_one_color_jit(
                depth_map,
                normal_map,
                cost_map,
                neighbors_dr,
                neighbors_dc,
                1,
                0,
                h,
                0,
                w,
                self.config.PATCHMATCH_PATCH_SIZE,
                self.config.TOP_K_COSTS,
                self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                ref_image_gray,
                ref_pose_K,
                ref_pose_R,
                ref_pose_T,
                src_images_gray,
                src_K,
                src_R,
                src_T,
            )

            # --- ランダム探索 ---
            depth_range_map = np.full(
                (h, w),
                self.config.PATCHMATCH_VANILLA_INITIAL_SEARCH_RANGE
                * (self.config.PATCHMATCH_DECAY_RATE**i),
                dtype=np.float32,
            )
            _random_search_jit(
                depth_map,
                normal_map,
                cost_map,
                i,
                self.config.PATCHMATCH_PATCH_SIZE,
                self.config.TOP_K_COSTS,
                self.config.PATCHMATCH_DECAY_RATE,
                self.config.PATCHMATCH_NORMAL_SEARCH_ANGLE,
                ref_image_gray,
                ref_pose_K,
                ref_pose_R,
                ref_pose_T,
                src_images_gray,
                src_K,
                src_R,
                src_T,
                self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                depth_range_map,
            )

            # イテレーションごとのデプスマップ保存
            if self.config.DEBUG_SAVE_DEPTH_MAPS:
                save_each_depth_dir = os.path.join(
                    config.DEPTH_IMAGE_DIR, f"depth_{ref_idx:04d}"
                )
                save_path = os.path.join(
                    save_each_depth_dir, f"depth_iter_{i+1:02d}.png"
                )
                logging.info(f"Saving intermediate depth map to {save_path}")
                save_depth_map_as_image(depth_map.copy(), save_path)

        logging.info("Vanilla PatchMatch MVS refinement finished.")
        return depth_map

    def filter_depth_map_by_geometric_consistency(
        self, ref_depth_map, ref_pose, neighbor_views_data, all_optimized_depths
    ):
        """
        複数ビュー間の幾何学的一貫性に基づいて深度マップをフィルタリングする
        """
        logging.info("Filtering depth map by geometric consistency...")
        h, w = ref_depth_map.shape
        filtered_depth_map = ref_depth_map.copy()

        # JIT用にデータを準備
        K_ref = ref_pose["K"].astype(np.float32)
        R_ref = ref_pose["R"].astype(np.float32)
        T_ref = ref_pose["T"].astype(np.float32)

        neighbor_poses = []
        neighbor_depth_maps = []
        for view in neighbor_views_data:
            view_idx = view["image_idx"]
            if view_idx in all_optimized_depths:
                neighbor_poses.append(
                    (
                        view["K"].astype(np.float32),
                        view["R"].astype(np.float32),
                        view["T"].astype(np.float32),
                    )
                )
                neighbor_depth_maps.append(
                    all_optimized_depths[view_idx].astype(np.float32)
                )

        if not neighbor_depth_maps:
            logging.warning(
                "No neighbor depth maps available for geometric consistency check."
            )
            return filtered_depth_map

        neighbor_depth_maps_np = np.stack(neighbor_depth_maps, axis=0)

        failures = 0
        for r in prange(h):
            for c in range(w):
                d_ref = filtered_depth_map[r, c]
                if not np.isfinite(d_ref) or d_ref <= 0:
                    continue

                consistent_views = _check_geometric_consistency_jit(
                    r,
                    c,
                    d_ref,
                    K_ref,
                    R_ref,
                    T_ref,
                    neighbor_poses,
                    neighbor_depth_maps_np,
                )

                if consistent_views < self.config.GEOMETRIC_MIN_CONSISTENT_VIEWS:
                    filtered_depth_map[r, c] = np.nan
                    failures += 1

        logging.info(
            f"{failures} points ({failures/(h*w)*100:.2f}%) invalidated by geometric consistency check."
        )
        return filtered_depth_map

    def filter_depth_map(self, depth_map, ref_image, ref_pose, neighbor_views_data):
        """
        コストと光度一貫性に基づいて深度マップをフィルタリングする
        """
        logging.info(
            "Filtering optimized depth map based on cost and photometric consistency..."
        )
        h, w = depth_map.shape
        filtered_depth_map = depth_map.copy()

        # データをJIT用に準備
        K = ref_pose["K"].astype(np.float32)
        R_ref = ref_pose["R"].astype(np.float32)
        T_ref = ref_pose["T"].astype(np.float32)

        neighbor_images = np.stack(
            [view["image"] for view in neighbor_views_data], axis=0
        )
        neighbor_R = np.stack(
            [view["R"].astype(np.float32) for view in neighbor_views_data], axis=0
        )
        neighbor_T = np.stack(
            [view["T"].astype(np.float32) for view in neighbor_views_data], axis=0
        )

        # 1. コストが高いピクセルを無効化
        # invalid_mask = cost_map > self.config.FILTERING_COST_THRESHOLD
        # filtered_depth_map[invalid_mask] = np.nan
        # logging.info(f"{np.sum(invalid_mask)} points invalidated by cost threshold.")

        # 2. 光度一貫性チェック
        consistency_failures = 0
        for r in range(h):
            for c in range(w):
                if not np.isfinite(filtered_depth_map[r, c]):
                    continue

                consistent_views = _check_photometric_consistency_jit(
                    r,
                    c,
                    filtered_depth_map[r, c],
                    ref_image[r, c],
                    K,
                    R_ref,
                    T_ref,
                    neighbor_images,
                    neighbor_R,
                    neighbor_T,
                )

                if consistent_views < self.config.FILTERING_MIN_CONSISTENT_VIEWS:
                    filtered_depth_map[r, c] = np.nan
                    consistency_failures += 1

        logging.info(
            f"{consistency_failures} points invalidated by photometric consistency check."
        )

        return filtered_depth_map
