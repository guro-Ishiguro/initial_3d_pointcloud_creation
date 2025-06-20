# tests/point_cloud_filtering.py

import numpy as np
import cv2
from numba import njit, prange
import logging
import config
import os
from utils import save_depth_map_as_image


# --- JITコンパイルされるコア関数群 ---
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
    パッチ中心との色の近さに応じて重みを付け、境界領域での頑健性を高める。
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


@njit(parallel=True)
def _patchmatch_iteration_jit(
    depth_map,
    normal_map,
    cost_map,
    depth_error_map,
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
):
    h, w = depth_map.shape
    for color in (0, 1):  # 空間伝播
        for r in prange(1, h - 1):
            for c in range(1, w - 1):
                if (r + c) % 2 != color:
                    continue
                if not np.isfinite(depth_map[r, c]):
                    continue
                for dr, dc in [(-1, 0), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if not np.isfinite(depth_map[nr, nc]):
                        continue

                    new_cost = _evaluate_cost_jit(
                        r,
                        c,
                        depth_map[nr, nc],
                        normal_map[nr, nc],
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
                        depth_map[r, c] = depth_map[nr, nc]
                        normal_map[r, c] = normal_map[nr, nc]
                        cost_map[r, c] = new_cost

    # --- ランダムサンプル探索 ---
    current_decay = decay_rate**iteration
    for r in prange(h):
        for c in range(w):
            if not np.isfinite(depth_map[r, c]):
                continue
            d_current = depth_map[r, c]
            d_range = (depth_error_map[r, c]) * current_decay
            d_new = d_current + (np.random.rand() * 2 - 1) * d_range
            if d_new <= 0:
                continue
            n_current = normal_map[r, c]
            angle_rad = np.radians(
                (np.random.rand() * 2 - 1) * normal_search_angle * current_decay
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


@njit(parallel=True)
def _patchmatch_iteration_vanilla_jit(
    depth_map,
    normal_map,
    cost_map,
    iteration,
    patch_size,
    top_k_costs,
    initial_search_range,
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
):
    """
    通常のPatchMatchの反復処理。
    探索範囲は深度誤差に依存せず、固定の初期値から減衰する。
    """
    h, w = depth_map.shape
    # --- 空間伝播 (変更なし) ---
    for color in (0, 1):
        for r in prange(1, h - 1):
            for c in range(1, w - 1):
                if (r + c) % 2 != color:
                    continue
                if not np.isfinite(depth_map[r, c]):
                    continue
                for dr, dc in [(-1, 0), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if not np.isfinite(depth_map[nr, nc]):
                        continue

                    new_cost = _evaluate_cost_jit(
                        r,
                        c,
                        depth_map[nr, nc],
                        normal_map[nr, nc],
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
                        depth_map[r, c] = depth_map[nr, nc]
                        normal_map[r, c] = normal_map[nr, nc]
                        cost_map[r, c] = new_cost

    # --- ランダムサンプル探索 (探索範囲の計算方法を変更) ---
    current_decay = decay_rate**iteration
    for r in prange(h):
        for c in range(w):
            if not np.isfinite(depth_map[r, c]):
                continue

            d_current = depth_map[r, c]
            # 深度誤差マップを使わず、固定の探索範囲を使用
            d_range = initial_search_range * current_decay
            d_new = d_current + (np.random.rand() * 2 - 1) * d_range
            if d_new <= 0:
                continue

            n_current = normal_map[r, c]
            angle_rad = np.radians(
                (np.random.rand() * 2 - 1) * normal_search_angle * current_decay
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
            "Starting PatchMatch MVS depth refinement with Adaptive Weighting..."
        )

        h, w = initial_depth.shape
        depth_map = initial_depth.astype(np.float32)
        normal_map = _initialize_normals_from_depth_jit(
            depth_map, ref_pose["K"].astype(np.float32)
        )

        if self.config.DEBUG_PATCH_MATCH_VISUALIZATION:
            debug_r, debug_c = self.config.DEBUG_PIXEL_COORDS
            if not (
                0 <= debug_r < h
                and 0 <= debug_c < w
                and np.isfinite(depth_map[debug_r, debug_c])
            ):
                logging.error(
                    f"Debug pixel ({debug_r}, {debug_c}) is out of bounds or has invalid depth. Aborting debug."
                )
                return initial_depth

            logging.warning(
                f"--- RUNNING IN DEBUG VISUALIZATION MODE FOR PIXEL ({debug_r}, {debug_c}) ---"
            )

            d_init, n_init = depth_map[debug_r, debug_c], normal_map[debug_r, debug_c]

            print("\n--- [DEBUG] 1. Initial State ---")
            self._debug_patch_visualization(
                debug_r,
                debug_c,
                d_init,
                n_init,
                ref_image,
                ref_pose,
                neighbor_views_data,
                "Initial State",
            )
            print(
                ">>> Press any key in a visualization window to continue to random samples..."
            )
            cv2.waitKey(0)

            print("\n--- [DEBUG] 2. Random Samples ---")
            d_range = (
                self.config.PATCHMATCH_BASE_ERROR
                + self.config.PATCHMATCH_ERROR_WEIGHT
                * initial_depth_error[debug_r, debug_c]
            )
            d_new = d_init + (np.random.rand() * 2 - 1) * d_range
            angle_rad = np.radians(
                (np.random.rand() * 2 - 1) * self.config.PATCHMATCH_NORMAL_SEARCH_ANGLE
            )
            rand_axis = np.random.randn(3)
            rand_axis /= np.linalg.norm(rand_axis)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            n_new = (
                n_init * cos_a
                + np.cross(rand_axis, n_init) * sin_a
                + rand_axis * np.dot(rand_axis, n_init) * (1 - cos_a)
            )

            self._debug_patch_visualization(
                debug_r,
                debug_c,
                d_new,
                n_new,
                ref_image,
                ref_pose,
                neighbor_views_data,
                f"Random Sample",
            )
            print(">>> Press any key to see the next sample...")
            cv2.waitKey(0)

            cv2.destroyAllWindows()
            logging.warning(
                "--- Exiting after debug visualization. Full optimization was SKIPPED. ---"
            )
            return initial_depth

        # --- 通常の高速な最適化処理 ---
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
        for i in range(self.config.PATCHMATCH_ITERATIONS):
            np.copyto(depth_map_prev, depth_map)
            logging.info(
                f"PatchMatch Iteration {i+1}/{self.config.PATCHMATCH_ITERATIONS}"
            )

            _patchmatch_iteration_jit(
                depth_map,
                normal_map,
                cost_map,
                initial_depth_error.astype(np.float32),
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
            )

            # --- イテレーションごとのデプスマップ保存処理 ---
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

            # --- 収束チェック ---
            diff = np.abs(depth_map_prev - depth_map)
            valid_pixels = np.isfinite(depth_map_prev)
            if np.sum(valid_pixels) > 0:
                mean_change = np.mean(diff[valid_pixels] / depth_map_prev[valid_pixels])
                logging.info(f"Average depth change: {mean_change:.5f}")
                if mean_change < convergence_threshold:
                    logging.info(
                        f"Convergence reached at iteration {i+1}. Stopping early."
                    )
                    break
        logging.info("PatchMatch MVS refinement finished.")
        return depth_map

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

        # (オプション) 初期デプスマップを保存
        if self.config.DEBUG_SAVE_DEPTH_MAPS:
            save_path = os.path.join(
                self.config.DEPTH_MAP_DIR, f"depth_vanilla_initial_{ref_idx:04d}.png"
            )
            logging.info(f"Saving vanilla initial depth map to {save_path}")
            save_depth_map_as_image(depth_map.copy(), save_path)

        # 4. PatchMatch反復ループ
        for i in range(self.config.PATCHMATCH_ITERATIONS):
            logging.info(
                f"Vanilla PatchMatch Iteration {i+1}/{self.config.PATCHMATCH_ITERATIONS}"
            )

            _patchmatch_iteration_vanilla_jit(
                depth_map,
                normal_map,
                cost_map,
                i,
                self.config.PATCHMATCH_PATCH_SIZE,
                self.config.TOP_K_COSTS,
                self.config.PATCHMATCH_VANILLA_INITIAL_SEARCH_RANGE,
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
            )

            # イテレーションごとのデプスマップ保存
            if self.config.DEBUG_SAVE_DEPTH_MAPS:
                save_path = os.path.join(
                    self.config.DEPTH_MAP_DIR,
                    f"depth_vanilla_iter_{ref_idx:04d}_{i+1:02d}.png",
                )
                logging.info(f"Saving vanilla intermediate depth map to {save_path}")
                save_depth_map_as_image(depth_map.copy(), save_path)

        logging.info("Vanilla PatchMatch MVS refinement finished.")
        return depth_map
