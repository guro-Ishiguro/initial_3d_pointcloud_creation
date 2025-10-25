# mvs/depth_optimization.py

import logging
import math
import os
import threading
import time

import config
import cv2
import numpy as np
from logging_setup import log_ndarray_stats, time_block
from numba import cuda, njit, prange
from numba.cuda.random import create_xoroshiro128p_states
from utils import (
    append_to_csv,
    clear_folder,
    compute_depth_metrics,
    initialize_csv,
    save_depth_map_as_image,
    save_error_map_as_image,
    save_normal_map_as_image,
)

# Constants for CUDA kernels
PATCHMATCH_PATCH_SIZE_CONST = config.PATCHMATCH_PATCH_SIZE
MAX_NEIGHBORS_CONST = config.MAX_NEIGHBORS
MAX_ACMH_HYPOTHESES_CONST = max(2, int(getattr(config, "ACMH_NUM_HYPOTHESES", 2)))


@cuda.jit(device=True)
def _bilinear_interpolate_cuda(image, y, x):
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


@cuda.jit(device=True)
def _compute_homography_cuda(
    H_out,
    K_ref,
    R_ref,
    T_ref,
    K_src,
    R_src,
    T_src,
    plane_point_3d_0,
    plane_point_3d_1,
    plane_point_3d_2,
    plane_normal_0,
    plane_normal_1,
    plane_normal_2,
):
    R_ref_inv = cuda.local.array((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            R_ref_inv[i, j] = R_ref[j, i]

    T_ref_inv_0 = -(
        R_ref_inv[0, 0] * T_ref[0]
        + R_ref_inv[0, 1] * T_ref[1]
        + R_ref_inv[0, 2] * T_ref[2]
    )
    T_ref_inv_1 = -(
        R_ref_inv[1, 0] * T_ref[0]
        + R_ref_inv[1, 1] * T_ref[1]
        + R_ref_inv[1, 2] * T_ref[2]
    )
    T_ref_inv_2 = -(
        R_ref_inv[2, 0] * T_ref[0]
        + R_ref_inv[2, 1] * T_ref[1]
        + R_ref_inv[2, 2] * T_ref[2]
    )

    p_ref_0 = (
        R_ref[0, 0] * plane_point_3d_0
        + R_ref[0, 1] * plane_point_3d_1
        + R_ref[0, 2] * plane_point_3d_2
        + T_ref[0]
    )
    p_ref_1 = (
        R_ref[1, 0] * plane_point_3d_0
        + R_ref[1, 1] * plane_point_3d_1
        + R_ref[1, 2] * plane_point_3d_2
        + T_ref[1]
    )
    p_ref_2 = (
        R_ref[2, 0] * plane_point_3d_0
        + R_ref[2, 1] * plane_point_3d_1
        + R_ref[2, 2] * plane_point_3d_2
        + T_ref[2]
    )

    n_ref_0 = (
        R_ref[0, 0] * plane_normal_0
        + R_ref[0, 1] * plane_normal_1
        + R_ref[0, 2] * plane_normal_2
    )
    n_ref_1 = (
        R_ref[1, 0] * plane_normal_0
        + R_ref[1, 1] * plane_normal_1
        + R_ref[1, 2] * plane_normal_2
    )
    n_ref_2 = (
        R_ref[2, 0] * plane_normal_0
        + R_ref[2, 1] * plane_normal_1
        + R_ref[2, 2] * plane_normal_2
    )

    R_rel = cuda.local.array((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            R_rel[i, j] = 0
            for k in range(3):
                R_rel[i, j] += R_src[i, k] * R_ref_inv[k, j]

    T_rel_0 = (
        R_src[0, 0] * T_ref_inv_0
        + R_src[0, 1] * T_ref_inv_1
        + R_src[0, 2] * T_ref_inv_2
        + T_src[0]
    )
    T_rel_1 = (
        R_src[1, 0] * T_ref_inv_0
        + R_src[1, 1] * T_ref_inv_1
        + R_src[1, 2] * T_ref_inv_2
        + T_src[1]
    )
    T_rel_2 = (
        R_src[2, 0] * T_ref_inv_0
        + R_src[2, 1] * T_ref_inv_1
        + R_src[2, 2] * T_ref_inv_2
        + T_src[2]
    )

    d = n_ref_0 * p_ref_0 + n_ref_1 * p_ref_1 + n_ref_2 * p_ref_2
    if abs(d) < 1e-8:
        H_out[0, 0] = 1.0
        H_out[0, 1] = 0.0
        H_out[0, 2] = 0.0
        H_out[1, 0] = 0.0
        H_out[1, 1] = 1.0
        H_out[1, 2] = 0.0
        H_out[2, 0] = 0.0
        H_out[2, 1] = 0.0
        H_out[2, 2] = 1.0
        return

    H = cuda.local.array((3, 3), dtype=np.float32)
    H[0, 0] = R_rel[0, 0] + T_rel_0 * n_ref_0 / d
    H[0, 1] = R_rel[0, 1] + T_rel_0 * n_ref_1 / d
    H[0, 2] = R_rel[0, 2] + T_rel_0 * n_ref_2 / d
    H[1, 0] = R_rel[1, 0] + T_rel_1 * n_ref_0 / d
    H[1, 1] = R_rel[1, 1] + T_rel_1 * n_ref_1 / d
    H[1, 2] = R_rel[1, 2] + T_rel_1 * n_ref_2 / d
    H[2, 0] = R_rel[2, 0] + T_rel_2 * n_ref_0 / d
    H[2, 1] = R_rel[2, 1] + T_rel_2 * n_ref_1 / d
    H[2, 2] = R_rel[2, 2] + T_rel_2 * n_ref_2 / d

    K_inv = cuda.local.array((3, 3), dtype=np.float32)
    fx = K_ref[0, 0]
    fy = K_ref[1, 1]
    cx = K_ref[0, 2]
    cy = K_ref[1, 2]
    # Inverse of intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
    K_inv[0, 0] = 1.0 / fx
    K_inv[0, 1] = 0.0
    K_inv[0, 2] = -cx / fx
    K_inv[1, 0] = 0.0
    K_inv[1, 1] = 1.0 / fy
    K_inv[1, 2] = -cy / fy
    K_inv[2, 0] = 0.0
    K_inv[2, 1] = 0.0
    K_inv[2, 2] = 1.0

    temp_mat = cuda.local.array((3, 3), dtype=np.float32)
    # temp_mat = K_src @ H
    for i in range(3):
        for j in range(3):
            val = 0.0
            for k in range(3):
                val += K_src[i, k] * H[k, j]
            temp_mat[i, j] = val

    # H_out = temp_mat @ K_inv
    for i in range(3):
        for j in range(3):
            val = 0.0
            for k in range(3):
                val += temp_mat[i, k] * K_inv[k, j]
            H_out[i, j] = val


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


@cuda.jit(device=True)
def _compute_weighted_zncc_cost_cuda(
    patch_ref, warped_patch_src, sigma_color, zncc_epsilon
):
    patch_size = patch_ref.shape[0]
    half = patch_size // 2
    center_val = patch_ref[half, half]
    weights = cuda.local.array(
        (PATCHMATCH_PATCH_SIZE_CONST, PATCHMATCH_PATCH_SIZE_CONST), dtype=np.float32
    )
    for r in range(patch_size):
        for c in range(patch_size):
            color_diff_sq = (patch_ref[r, c] - center_val) ** 2
            weights[r, c] = math.exp(-color_diff_sq / (2.0 * (sigma_color**2)))
    sum_w = 0.0
    for r in range(patch_size):
        for c in range(patch_size):
            sum_w += weights[r, c]
    if sum_w < 1e-6:
        return 1.0
    mean_ref = 0.0
    mean_src = 0.0
    for r in range(patch_size):
        for c in range(patch_size):
            mean_ref += patch_ref[r, c] * weights[r, c]
            mean_src += warped_patch_src[r, c] * weights[r, c]
    mean_ref /= sum_w
    mean_src /= sum_w
    var_ref = 0.0
    var_src = 0.0
    for r in range(patch_size):
        for c in range(patch_size):
            var_ref += weights[r, c] * (patch_ref[r, c] - mean_ref) ** 2
            var_src += weights[r, c] * (warped_patch_src[r, c] - mean_src) ** 2
    var_ref /= sum_w
    var_src /= sum_w
    std_ref = math.sqrt(var_ref)
    std_src = math.sqrt(var_src)
    if std_ref < zncc_epsilon or std_src < zncc_epsilon:
        return 1.0
    numerator = 0.0
    for r in range(patch_size):
        for c in range(patch_size):
            numerator += (
                weights[r, c]
                * (patch_ref[r, c] - mean_ref)
                * (warped_patch_src[r, c] - mean_src)
            )
    numerator /= sum_w
    denominator = std_ref * std_src
    return (1.0 - (numerator / denominator)) / 2.0


@njit(fastmath=True)
def _compute_weighted_zncc_cost_jit(
    patch_ref, warped_patch_src, sigma_color, zncc_epsilon
):
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

    if std_ref < zncc_epsilon or std_src < zncc_epsilon:
        return 1.0

    # 3. 重み付きZNCCを計算
    numerator = (
        np.sum(weights * (patch_ref - mean_ref) * (warped_patch_src - mean_src)) / sum_w
    )
    denominator = std_ref * std_src

    # ZNCC値は-1から1なので、コストを0から1の範囲に変換
    return (1.0 - (numerator / denominator)) / 2.0


@cuda.jit(device=True)
def _evaluate_cost_cuda(
    r,
    c,
    depth,
    normal_0,
    normal_1,
    normal_2,
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
    zncc_epsilon,
    use_median_top_k,
    cov_required,
    min_valid,
):
    h, w = ref_image_gray.shape
    half = patch_size // 2
    if r - half < 0 or r + half + 1 > h or c - half < 0 or c + half + 1 > w:
        return 1.0
    x_cam = (c - ref_pose_K[0, 2]) * depth / ref_pose_K[0, 0]
    y_cam = (r - ref_pose_K[1, 2]) * depth / ref_pose_K[1, 1]
    point_3d_cam_0 = x_cam
    point_3d_cam_1 = y_cam
    point_3d_cam_2 = depth
    R_ref_inv = cuda.local.array((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            R_ref_inv[i, j] = ref_pose_R[j, i]
    point_3d_world_0 = (
        R_ref_inv[0, 0] * (point_3d_cam_0 - ref_pose_T[0])
        + R_ref_inv[0, 1] * (point_3d_cam_1 - ref_pose_T[1])
        + R_ref_inv[0, 2] * (point_3d_cam_2 - ref_pose_T[2])
    )
    point_3d_world_1 = (
        R_ref_inv[1, 0] * (point_3d_cam_0 - ref_pose_T[0])
        + R_ref_inv[1, 1] * (point_3d_cam_1 - ref_pose_T[1])
        + R_ref_inv[1, 2] * (point_3d_cam_2 - ref_pose_T[2])
    )
    point_3d_world_2 = (
        R_ref_inv[2, 0] * (point_3d_cam_0 - ref_pose_T[0])
        + R_ref_inv[2, 1] * (point_3d_cam_1 - ref_pose_T[1])
        + R_ref_inv[2, 2] * (point_3d_cam_2 - ref_pose_T[2])
    )
    normal_world_0 = (
        R_ref_inv[0, 0] * normal_0
        + R_ref_inv[0, 1] * normal_1
        + R_ref_inv[0, 2] * normal_2
    )
    normal_world_1 = (
        R_ref_inv[1, 0] * normal_0
        + R_ref_inv[1, 1] * normal_1
        + R_ref_inv[1, 2] * normal_2
    )
    normal_world_2 = (
        R_ref_inv[2, 0] * normal_0
        + R_ref_inv[2, 1] * normal_1
        + R_ref_inv[2, 2] * normal_2
    )
    patch_ref = cuda.local.array(
        (PATCHMATCH_PATCH_SIZE_CONST, PATCHMATCH_PATCH_SIZE_CONST), dtype=np.float32
    )
    for pr in range(patch_size):
        for pc in range(patch_size):
            patch_ref[pr, pc] = ref_image_gray[r - half + pr, c - half + pc]
    num_neighbors = src_images_gray.shape[0]
    costs = cuda.local.array(MAX_NEIGHBORS_CONST, dtype=np.float32)
    valid = cuda.local.array(MAX_NEIGHBORS_CONST, dtype=np.int32)
    for ii in range(MAX_NEIGHBORS_CONST):
        costs[ii] = 1.0
        valid[ii] = 0
    H = cuda.local.array((3, 3), dtype=np.float32)
    for i in range(num_neighbors):
        _compute_homography_cuda(
            H,
            ref_pose_K,
            ref_pose_R,
            ref_pose_T,
            src_K[i],
            src_R[i],
            src_T[i],
            point_3d_world_0,
            point_3d_world_1,
            point_3d_world_2,
            normal_world_0,
            normal_world_1,
            normal_world_2,
        )
        invalid_H = False
        for ii in range(3):
            for jj in range(3):
                valH = H[ii, jj]
                if math.isnan(valH) or math.isinf(valH):
                    invalid_H = True
        if invalid_H:
            costs[i] = 1.0
            continue
        warped_patch = cuda.local.array(
            (PATCHMATCH_PATCH_SIZE_CONST, PATCHMATCH_PATCH_SIZE_CONST), dtype=np.float32
        )
        inside = 0
        for pr in range(patch_size):
            for pc in range(patch_size):
                u_ref, v_ref = c - half + pc, r - half + pr
                p_ref_h_0 = u_ref
                p_ref_h_1 = v_ref
                p_ref_h_2 = 1.0
                p_src_h_0 = (
                    H[0, 0] * p_ref_h_0 + H[0, 1] * p_ref_h_1 + H[0, 2] * p_ref_h_2
                )
                p_src_h_1 = (
                    H[1, 0] * p_ref_h_0 + H[1, 1] * p_ref_h_1 + H[1, 2] * p_ref_h_2
                )
                p_src_h_2 = (
                    H[2, 0] * p_ref_h_0 + H[2, 1] * p_ref_h_1 + H[2, 2] * p_ref_h_2
                )
                if abs(p_src_h_2) < 1e-8:
                    warped_patch[pr, pc] = 0.0
                    continue
                u_src, v_src = p_src_h_0 / p_src_h_2, p_src_h_1 / p_src_h_2
                src_h = src_images_gray[i].shape[0]
                src_w = src_images_gray[i].shape[1]
                if 0 <= v_src < src_h and 0 <= u_src < src_w:
                    warped_patch[pr, pc] = _bilinear_interpolate_cuda(
                        src_images_gray[i], v_src, u_src
                    )
                    inside += 1
                else:
                    warped_patch[pr, pc] = 0.0
        coverage = inside / float(patch_size * patch_size)
        if coverage >= cov_required:
            valid[i] = 1
            costs[i] = _compute_weighted_zncc_cost_cuda(
                patch_ref, warped_patch, adaptive_weight_sigma_color, zncc_epsilon
            )
        else:
            costs[i] = 1.0
    valid_count = 0
    for i in range(num_neighbors):
        if valid[i] == 1:
            valid_count += 1
    if valid_count < min_valid:
        return 1.0
    for i in range(num_neighbors):
        for j in range(i + 1, num_neighbors):
            if costs[i] > costs[j]:
                tmp = costs[i]
                costs[i] = costs[j]
                costs[j] = tmp
    top_k = min(top_k_costs, num_neighbors)
    if use_median_top_k != 0:
        mid = top_k // 2
        return costs[mid]
    total_cost = 0.0
    for i in range(top_k):
        total_cost += costs[i]
    return total_cost / top_k


@njit(fastmath=True)
def _evaluate_cost_jit(
    r,
    c,
    depth,
    normal_0,
    normal_1,
    normal_2,
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
    zncc_epsilon,
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
    normal = np.array([normal_0, normal_1, normal_2], dtype=np.float32)
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
            patch_ref, warped_patch, adaptive_weight_sigma_color, zncc_epsilon
        )

    costs = np.sort(costs)
    top_k = min(top_k_costs, len(costs))
    if getattr(config, "USE_MEDIAN_TOP_K", 0):
        return np.median(costs[:top_k])
    return np.mean(costs[:top_k])


@cuda.jit
def _propagate_spatial_one_color_cuda(
    depth_map,
    normal_map,
    cost_map,
    propagation_mask,
    neighbors_dr,
    neighbors_dc,
    color,
    patch_size,
    top_k_costs,
    adaptive_weight_sigma_color,
    zncc_epsilon,
    ref_image_gray,
    ref_pose_K,
    ref_pose_R,
    ref_pose_T,
    src_images_gray,
    src_K,
    src_R,
    src_T,
    use_median_top_k,
    cov_required,
    min_valid,
):
    c, r = cuda.grid(2)
    h, w = depth_map.shape

    if r >= h or c >= w:
        return
    if not propagation_mask[r, c]:
        return
    if (r + c) % 2 != color:
        return
    # 無効深度の画素はスキップ（CPU基準の挙動に統一）
    if math.isnan(depth_map[r, c]) or math.isinf(depth_map[r, c]):
        return

    for i in range(len(neighbors_dr)):
        dr = neighbors_dr[i]
        dc = neighbors_dc[i]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < h and 0 <= nc < w and propagation_mask[nr, nc]):
            continue

        neighbor_depth = depth_map[nr, nc]
        if math.isnan(neighbor_depth) or math.isinf(neighbor_depth):
            continue

        neighbor_normal = normal_map[nr, nc]
        new_cost = _evaluate_cost_cuda(
            r,
            c,
            neighbor_depth,
            neighbor_normal[0],
            neighbor_normal[1],
            neighbor_normal[2],
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
            zncc_epsilon,
            use_median_top_k,
            cov_required,
            min_valid,
        )

        if new_cost < cost_map[r, c]:
            depth_map[r, c] = neighbor_depth
            normal_map[r, c, 0] = neighbor_normal[0]
            normal_map[r, c, 1] = neighbor_normal[1]
            normal_map[r, c, 2] = neighbor_normal[2]
            cost_map[r, c] = new_cost


@cuda.jit
def _propagate_bucket_push_dir_cuda(
    depth_map,
    normal_map,
    cost_map,
    propagation_mask,
    bin_rs,
    bin_cs,
    dir_code,
    patch_size,
    top_k_costs,
    adaptive_weight_sigma_color,
    zncc_epsilon,
    ref_image_gray,
    ref_pose_K,
    ref_pose_R,
    ref_pose_T,
    src_images_gray,
    src_K,
    src_R,
    src_T,
    update_counter,
    use_median_top_k,
    cov_required,
    min_valid,
):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = bin_rs.shape[0]
    for idx in range(start, n, stride):
        r = int(bin_rs[idx])
        c = int(bin_cs[idx])
        h, w = depth_map.shape
        if not (0 <= r < h and 0 <= c < w):
            continue
        if not propagation_mask[r, c]:
            continue

        src_depth = depth_map[r, c]
        if math.isnan(src_depth) or math.isinf(src_depth):
            continue
        src_normal = normal_map[r, c]

        # 4-neighborhood: up, down, left, right
        if dir_code == 0:
            nr = r - 1
            nc = c
        elif dir_code == 1:
            nr = r + 1
            nc = c
        elif dir_code == 2:
            nr = r
            nc = c - 1
        else:
            nr = r
            nc = c + 1

        if not (0 <= nr < h and 0 <= nc < w and propagation_mask[nr, nc]):
            continue

        new_cost = _evaluate_cost_cuda(
            nr,
            nc,
            src_depth,
            src_normal[0],
            src_normal[1],
            src_normal[2],
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
            zncc_epsilon,
            use_median_top_k,
            cov_required,
            min_valid,
        )

        if new_cost < cost_map[nr, nc]:
            depth_map[nr, nc] = src_depth
            normal_map[nr, nc, 0] = src_normal[0]
            normal_map[nr, nc, 1] = src_normal[1]
            normal_map[nr, nc, 2] = src_normal[2]
            cost_map[nr, nc] = new_cost
            cuda.atomic.add(update_counter, 0, 1)


@cuda.jit
def _propagate_bucket_push4_cuda(
    depth_map,
    normal_map,
    cost_map,
    propagation_mask,
    bin_rs,
    bin_cs,
    patch_size,
    top_k_costs,
    adaptive_weight_sigma_color,
    zncc_epsilon,
    ref_image_gray,
    ref_pose_K,
    ref_pose_R,
    ref_pose_T,
    src_images_gray,
    src_K,
    src_R,
    src_T,
    use_median_top_k,
    cov_required,
    min_valid,
):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = bin_rs.shape[0]
    for idx in range(start, n, stride):
        r = int(bin_rs[idx])
        c = int(bin_cs[idx])
        h, w = depth_map.shape
        if not (0 <= r < h and 0 <= c < w):
            continue
        if not propagation_mask[r, c]:
            continue

        src_depth = depth_map[r, c]
        if math.isnan(src_depth) or math.isinf(src_depth):
            continue
        src_normal = normal_map[r, c]

        # Four or eight directions based on config
        for d in range(8 if config.PROPAGATION_NEIGHBOR_DIRECTIONS == 8 else 4):
            if d == 0:
                nr = r - 1
                nc = c
            elif d == 1:
                nr = r + 1
                nc = c
            elif d == 2:
                nr = r
                nc = c - 1
            elif d == 3:
                nr = r
                nc = c + 1
            elif d == 4:
                nr = r - 1
                nc = c - 1
            elif d == 5:
                nr = r - 1
                nc = c + 1
            elif d == 6:
                nr = r + 1
                nc = c - 1
            else:
                nr = r + 1
                nc = c + 1

            if not (0 <= nr < h and 0 <= nc < w and propagation_mask[nr, nc]):
                continue

            new_cost = _evaluate_cost_cuda(
                nr,
                nc,
                src_depth,
                src_normal[0],
                src_normal[1],
                src_normal[2],
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
                zncc_epsilon,
                use_median_top_k,
                cov_required,
                min_valid,
            )

            if new_cost < cost_map[nr, nc]:
                depth_map[nr, nc] = src_depth
                normal_map[nr, nc, 0] = src_normal[0]
                normal_map[nr, nc, 1] = src_normal[1]
                normal_map[nr, nc, 2] = src_normal[2]
                cost_map[nr, nc] = new_cost


@njit(parallel=True, fastmath=True)
def _propagate_spatial_one_color_jit(
    depth_map,
    normal_map,
    cost_map,  # 更新対象のマップ
    propagation_mask,
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
            if not propagation_mask[r, c]:
                continue

            # 対象の色（赤 or 黒）のピクセルのみを処理
            if (r + c) % 2 != color:
                continue

            if math.isnan(depth_map[r, c]) or math.isinf(depth_map[r, c]):
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
                if math.isnan(neighbor_depth) or math.isinf(neighbor_depth):
                    continue

                # 現在のピクセル(r, c)の位置で、隣接ピクセル(nr, nc)の平面を評価
                neighbor_normal = normal_map[nr, nc]
                new_cost = _evaluate_cost_jit(
                    r,
                    c,
                    neighbor_depth,
                    neighbor_normal[0],
                    neighbor_normal[1],
                    neighbor_normal[2],
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
                    np.float32(config.ZNCC_EPSILON),
                )

                # コストが改善されれば、現在のピクセルの平面を更新
                if new_cost < cost_map[r, c]:
                    depth_map[r, c] = neighbor_depth
                    normal_map[r, c] = neighbor_normal
                    cost_map[r, c] = new_cost


@njit(parallel=True, fastmath=True)
def _propagate_bucket_jit(
    depth_map,
    normal_map,
    cost_map,
    initial_depth_error,
    propagation_mask,
    num_bins,
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
    コストをビンに分割し、低コストのビンから優先的に並列伝播を実行する。
    """
    h, w = depth_map.shape
    neighbors_dr = np.array([-1, 1, 0, 0], dtype=np.int8)
    neighbors_dc = np.array([0, 0, -1, 1], dtype=np.int8)

    # --- 1. 有効なピクセルを抽出し、コストに基づいてビンに分類 ---
    valid_pixels_coords = np.empty((h * w, 2), dtype=np.int32)
    valid_pixels_costs = np.empty(h * w, dtype=np.float32)
    valid_pixel_count = 0
    for r in range(h):
        for c in range(w):
            cost = initial_depth_error[r, c]
            if propagation_mask[r, c] and np.isfinite(cost):
                valid_pixels_coords[valid_pixel_count, 0] = r
                valid_pixels_coords[valid_pixel_count, 1] = c
                valid_pixels_costs[valid_pixel_count] = cost
                valid_pixel_count += 1

    if valid_pixel_count == 0:
        return

    # コストの最小値と最大値からビンの幅を計算
    min_cost = np.inf
    max_cost = -np.inf
    for i in range(valid_pixel_count):
        cost = valid_pixels_costs[i]
        if cost < min_cost:
            min_cost = cost
        if cost > max_cost:
            max_cost = cost

    if max_cost - min_cost < 1e-6:
        bin_width = 1.0
        num_bins = 1
    else:
        bin_width = (max_cost - min_cost) / num_bins

    # 各有効ピクセルがどのビンに属するかを計算
    pixel_bin_indices = np.floor(
        (valid_pixels_costs[:valid_pixel_count] - min_cost) / bin_width
    ).astype(np.int32)
    pixel_bin_indices[pixel_bin_indices >= num_bins] = num_bins - 1

    # --- 2. バケット伝播ループ ---
    # 優先度の高い（コストの低い）ビンから順番に処理
    for bin_idx in range(num_bins):
        # 現在のビンに属するピクセルを並列で処理
        for i in prange(valid_pixel_count):
            # このピクセルが現在の処理対象ビンに属しているかチェック
            if pixel_bin_indices[i] != bin_idx:
                continue

            # 伝播元となるピクセル（source）の情報を取得
            r_source, c_source = valid_pixels_coords[i]
            source_depth = depth_map[r_source, c_source]
            if math.isnan(source_depth) or math.isinf(source_depth):
                continue
            source_normal = normal_map[r_source, c_source]

            # 4方向の近傍ピクセル（target）へ伝播
            for j in range(len(neighbors_dr)):
                r_target, c_target = (
                    r_source + neighbors_dr[j],
                    c_source + neighbors_dc[j],
                )

                # 伝播先が画像範囲内で、かつマスク内かチェック
                if not (
                    0 <= r_target < h
                    and 0 <= c_target < w
                    and propagation_mask[r_target, c_target]
                ):
                    continue

                # 伝播元の平面を、伝播先の位置で評価し、新しいコストを計算
                new_cost = _evaluate_cost_jit(
                    r_target,
                    c_target,
                    source_depth,
                    source_normal[0],
                    source_normal[1],
                    source_normal[2],
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
                    np.float32(config.ZNCC_EPSILON),
                )

                # コストが改善される場合、伝播先の平面情報を更新
                if new_cost < cost_map[r_target, c_target]:
                    depth_map[r_target, c_target] = source_depth
                    normal_map[r_target, c_target] = source_normal
                    cost_map[r_target, c_target] = new_cost


@cuda.jit
def _random_search_cuda(
    depth_map,
    normal_map,
    cost_map,
    search_mask,
    iteration,
    patch_size,
    top_k_costs,
    decay_rate,
    normal_search_angle,
    zncc_epsilon,
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
    random_states,
    use_median_top_k,
    cov_required,
    min_valid,
):
    c, r = cuda.grid(2)
    h, w = depth_map.shape
    thread_id = r * w + c

    if r >= h or c >= w:
        return
    if not search_mask[r, c]:
        return
    d_current = depth_map[r, c]
    if math.isnan(d_current) or math.isinf(d_current) or d_current <= 0:
        return
    d_range = depth_range_map[r, c]
    if math.isnan(d_range) or math.isinf(d_range) or d_range <= 0:
        return
    d_new = (
        d_current
        + (cuda.random.xoroshiro128p_uniform_float32(random_states, thread_id) * 2 - 1)
        * d_range
    )
    if d_new <= 0:
        return

    n_current = normal_map[r, c]
    angle_rad = (
        (
            (
                cuda.random.xoroshiro128p_uniform_float32(random_states, thread_id) * 2
                - 1
            )
            * normal_search_angle
            * (decay_rate**iteration)
        )
        * math.pi
        / 180.0
    )

    rand_axis_x = cuda.random.xoroshiro128p_normal_float32(random_states, thread_id)
    rand_axis_y = cuda.random.xoroshiro128p_normal_float32(random_states, thread_id)
    rand_axis_z = cuda.random.xoroshiro128p_normal_float32(random_states, thread_id)
    norm = (rand_axis_x**2 + rand_axis_y**2 + rand_axis_z**2) ** 0.5
    rand_axis_x /= norm
    rand_axis_y /= norm
    rand_axis_z /= norm

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    one_minus_cos_a = 1.0 - cos_a

    dot_product = (
        rand_axis_x * n_current[0]
        + rand_axis_y * n_current[1]
        + rand_axis_z * n_current[2]
    )

    cross_product_x = rand_axis_y * n_current[2] - rand_axis_z * n_current[1]
    cross_product_y = rand_axis_z * n_current[0] - rand_axis_x * n_current[2]
    cross_product_z = rand_axis_x * n_current[1] - rand_axis_y * n_current[0]

    n_new_x = (
        n_current[0] * cos_a
        + cross_product_x * sin_a
        + rand_axis_x * dot_product * one_minus_cos_a
    )
    n_new_y = (
        n_current[1] * cos_a
        + cross_product_y * sin_a
        + rand_axis_y * dot_product * one_minus_cos_a
    )
    n_new_z = (
        n_current[2] * cos_a
        + cross_product_z * sin_a
        + rand_axis_z * dot_product * one_minus_cos_a
    )

    # If current normal is invalid, start from a random unit vector around Z
    if (
        math.isnan(n_current[0])
        or math.isinf(n_current[0])
        or math.isnan(n_current[1])
        or math.isinf(n_current[1])
        or math.isnan(n_current[2])
        or math.isinf(n_current[2])
    ):
        n_current[0] = 0.0
        n_current[1] = 0.0
        n_current[2] = 1.0
    norm_new = (n_new_x**2 + n_new_y**2 + n_new_z**2) ** 0.5
    n_new_x /= norm_new
    n_new_y /= norm_new
    n_new_z /= norm_new

    n_new = cuda.local.array(3, dtype=np.float32)
    n_new[0] = n_new_x
    n_new[1] = n_new_y
    n_new[2] = n_new_z

    new_cost = _evaluate_cost_cuda(
        r,
        c,
        d_new,
        n_new[0],
        n_new[1],
        n_new[2],
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
        zncc_epsilon,
        use_median_top_k,
        cov_required,
        min_valid,
    )
    if new_cost < cost_map[r, c]:
        depth_map[r, c] = d_new
        normal_map[r, c, 0] = n_new[0]
        normal_map[r, c, 1] = n_new[1]
        normal_map[r, c, 2] = n_new[2]
        cost_map[r, c] = new_cost


@njit(parallel=True, fastmath=True)
def _random_search_jit(
    depth_map,
    normal_map,
    cost_map,
    search_mask,
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
            if not search_mask[r, c]:
                continue

            if math.isnan(depth_map[r, c]) or math.isinf(depth_map[r, c]):
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
                n_new[0],
                n_new[1],
                n_new[2],
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
                np.float32(config.ZNCC_EPSILON),
            )
            if new_cost < cost_map[r, c]:
                depth_map[r, c], normal_map[r, c], cost_map[r, c] = (
                    d_new,
                    n_new,
                    new_cost,
                )


@njit(fastmath=True)
def _check_geometric_consistency_jit(
    point_3d_world, neighbor_K_np, neighbor_R_np, neighbor_T_np, neighbor_depth_maps_np
):
    """
    単一の3Dポイントが、近傍ビューの深度マップと幾何学的に一貫しているかチェックする
    """
    consistent_views = 0
    h, w = neighbor_depth_maps_np[0].shape

    # 各近傍ビューでチェック
    for i in range(len(neighbor_K_np)):
        K_src = neighbor_K_np[i]
        R_src = neighbor_R_np[i]
        T_src = neighbor_T_np[i]
        depth_map_src = neighbor_depth_maps_np[i]

        # 近傍ビューのカメラ座標に変換
        p_src_cam = R_src @ point_3d_world + T_src

        # 近傍ビューの画像座標に投影
        p_src_img_h = K_src @ p_src_cam
        d_proj_src = p_src_img_h[2]

        # ゼロ除算を防ぎ、カメラの後ろにある点も無視する
        if d_proj_src < 1e-6:
            continue

        u_src, v_src = p_src_img_h[0] / d_proj_src, p_src_img_h[1] / d_proj_src

        # 画像範囲外かチェック
        if not (0 <= u_src < w and 0 <= v_src < h):
            continue

        # 最も近いピクセルの深度値を取得
        r_src, c_src = int(round(v_src)), int(round(u_src))

        if not (0 <= c_src < w and 0 <= r_src < h):
            continue

        d_actual_src = depth_map_src[r_src, c_src]

        # 近傍ビューの深度が有効かチェック
        # ゼロ除算を避けるために、非常に小さい正の値も除外
        if not np.isfinite(d_actual_src) or d_actual_src < 1e-6:
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
        R_src, T_src = np.ascontiguousarray(neighbor_R[i]), neighbor_T[i]

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


@cuda.jit
def _initialize_normals_from_depth_cuda(normals, depth_map, K):
    c, r = cuda.grid(2)
    h, w = depth_map.shape
    if r >= 1 and r < h - 1 and c >= 1 and c < w - 1:
        val = depth_map[r, c]
        if math.isnan(val) or math.isinf(val):
            return

        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]

        p_center_x = (c - cx) * depth_map[r, c] / fx
        p_center_y = (r - cy) * depth_map[r, c] / fy
        p_center_z = depth_map[r, c]

        p_right_x = (c + 1 - cx) * depth_map[r, c + 1] / fx
        p_right_y = (r - cy) * depth_map[r, c + 1] / fy
        p_right_z = depth_map[r, c + 1]

        p_down_x = (c - cx) * depth_map[r + 1, c] / fx
        p_down_y = (r + 1 - cy) * depth_map[r + 1, c] / fy
        p_down_z = depth_map[r + 1, c]

        if (
            (math.isnan(p_right_x) or math.isinf(p_right_x))
            or (math.isnan(p_right_y) or math.isinf(p_right_y))
            or (math.isnan(p_right_z) or math.isinf(p_right_z))
            or (math.isnan(p_down_x) or math.isinf(p_down_x))
            or (math.isnan(p_down_y) or math.isinf(p_down_y))
            or (math.isnan(p_down_z) or math.isinf(p_down_z))
        ):
            return

        v_c_x = p_right_x - p_center_x
        v_c_y = p_right_y - p_center_y
        v_c_z = p_right_z - p_center_z

        v_r_x = p_down_x - p_center_x
        v_r_y = p_down_y - p_center_y
        v_r_z = p_down_z - p_center_z

        normal_x = v_r_y * v_c_z - v_r_z * v_c_y
        normal_y = v_r_z * v_c_x - v_r_x * v_c_z
        normal_z = v_r_x * v_c_y - v_r_y * v_c_x

        norm = (normal_x**2 + normal_y**2 + normal_z**2) ** 0.5
        if norm > 1e-6:
            normals[r, c, 0] = normal_x / norm
            normals[r, c, 1] = normal_y / norm
            normals[r, c, 2] = normal_z / norm
        else:
            normals[r, c, 0] = 0.0
            normals[r, c, 1] = 0.0
            normals[r, c, 2] = 1.0


@cuda.jit
def _propagate_spatial_one_color_acmhH_cuda(
    depth_H,
    normal_H,
    cost_H,
    H_count,
    propagation_mask,
    neighbors_dr,
    neighbors_dc,
    color,
    patch_size,
    top_k_costs,
    adaptive_weight_sigma_color,
    zncc_epsilon,
    ref_image_gray,
    ref_pose_K,
    ref_pose_R,
    ref_pose_T,
    src_images_gray,
    src_K,
    src_R,
    src_T,
    joint_view_selection,
    joint_top_k,
    use_median_top_k,
    cov_required,
    min_valid,
):
    c, r = cuda.grid(2)
    Hmax, h, w = depth_H.shape
    if r >= h or c >= w:
        return
    if not propagation_mask[r, c]:
        return
    if (r + c) % 2 != color:
        return

    # 少なくとも1つの有効仮説が必要
    has_valid = False
    for s in range(H_count):
        if not (
            math.isnan(depth_H[s, r, c])
            or math.isinf(depth_H[s, r, c])
            or depth_H[s, r, c] <= 0
        ):
            has_valid = True
            break
    if not has_valid:
        return

    # Evaluate current H slots
    for s in range(H_count):
        d = depth_H[s, r, c]
        if math.isnan(d) or math.isinf(d) or d <= 0:
            continue
        n0 = normal_H[s, r, c, 0]
        n1 = normal_H[s, r, c, 1]
        n2 = normal_H[s, r, c, 2]
        if joint_view_selection == 1:
            cost = _evaluate_cost_cuda(
                r,
                c,
                d,
                n0,
                n1,
                n2,
                patch_size,
                ref_image_gray,
                ref_pose_K,
                ref_pose_R,
                ref_pose_T,
                src_images_gray,
                src_K,
                src_R,
                src_T,
                joint_top_k,
                adaptive_weight_sigma_color,
                zncc_epsilon,
                use_median_top_k,
                cov_required,
                min_valid,
            )
        else:
            cost = _evaluate_cost_cuda(
                r,
                c,
                d,
                n0,
                n1,
                n2,
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
                zncc_epsilon,
                use_median_top_k,
                cov_required,
                min_valid,
            )
        cost_H[s, r, c] = cost

    # Neighbor proposals
    max_neighbors = neighbors_dr.shape[0]
    for i in range(max_neighbors):
        nr = r + neighbors_dr[i]
        nc = c + neighbors_dc[i]
        if not (0 <= nr < h and 0 <= nc < w and propagation_mask[nr, nc]):
            continue
        for s in range(H_count):
            d = depth_H[s, nr, nc]
            if math.isnan(d) or math.isinf(d) or d <= 0:
                continue
            n0 = normal_H[s, nr, nc, 0]
            n1 = normal_H[s, nr, nc, 1]
            n2 = normal_H[s, nr, nc, 2]
            if joint_view_selection == 1:
                cost = _evaluate_cost_cuda(
                    r,
                    c,
                    d,
                    n0,
                    n1,
                    n2,
                    patch_size,
                    ref_image_gray,
                    ref_pose_K,
                    ref_pose_R,
                    ref_pose_T,
                    src_images_gray,
                    src_K,
                    src_R,
                    src_T,
                    joint_top_k,
                    adaptive_weight_sigma_color,
                    zncc_epsilon,
                    use_median_top_k,
                    cov_required,
                    min_valid,
                )
            else:
                cost = _evaluate_cost_cuda(
                    r,
                    c,
                    d,
                    n0,
                    n1,
                    n2,
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
                    zncc_epsilon,
                    use_median_top_k,
                    cov_required,
                    min_valid,
                )
            # replace worst among current H if better
            worst_slot = 0
            worst_cost = cost_H[0, r, c]
            for ss in range(1, H_count):
                cval = cost_H[ss, r, c]
                if cval > worst_cost:
                    worst_cost = cval
                    worst_slot = ss
            if cost < worst_cost:
                depth_H[worst_slot, r, c] = d
                normal_H[worst_slot, r, c, 0] = n0
                normal_H[worst_slot, r, c, 1] = n1
                normal_H[worst_slot, r, c, 2] = n2
                cost_H[worst_slot, r, c] = cost


@cuda.jit
def _invalidate_by_validview_cuda(
    depth_map,
    normal_map,
    cost_map,
    propagation_mask,
    patch_size,
    ref_pose_K,
    ref_pose_R,
    ref_pose_T,
    src_images_gray,
    src_K,
    src_R,
    src_T,
    cov_required,
    min_valid,
):
    c, r = cuda.grid(2)
    h, w = depth_map.shape
    if r >= h or c >= w:
        return
    if not propagation_mask[r, c]:
        return
    d = depth_map[r, c]
    if math.isnan(d) or math.isinf(d) or d <= 0:
        return
    half = patch_size // 2
    if r - half < 0 or r + half + 1 > h or c - half < 0 or c + half + 1 > w:
        return
    # backproject
    x_cam = (c - ref_pose_K[0, 2]) * d / ref_pose_K[0, 0]
    y_cam = (r - ref_pose_K[1, 2]) * d / ref_pose_K[1, 1]
    point_3d_cam_0 = x_cam
    point_3d_cam_1 = y_cam
    point_3d_cam_2 = d
    R_ref_inv = cuda.local.array((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            R_ref_inv[i, j] = ref_pose_R[j, i]
    point_3d_world_0 = (
        R_ref_inv[0, 0] * (point_3d_cam_0 - ref_pose_T[0])
        + R_ref_inv[0, 1] * (point_3d_cam_1 - ref_pose_T[1])
        + R_ref_inv[0, 2] * (point_3d_cam_2 - ref_pose_T[2])
    )
    point_3d_world_1 = (
        R_ref_inv[1, 0] * (point_3d_cam_0 - ref_pose_T[0])
        + R_ref_inv[1, 1] * (point_3d_cam_1 - ref_pose_T[1])
        + R_ref_inv[1, 2] * (point_3d_cam_2 - ref_pose_T[2])
    )
    point_3d_world_2 = (
        R_ref_inv[2, 0] * (point_3d_cam_0 - ref_pose_T[0])
        + R_ref_inv[2, 1] * (point_3d_cam_1 - ref_pose_T[1])
        + R_ref_inv[2, 2] * (point_3d_cam_2 - ref_pose_T[2])
    )
    n = normal_map[r, c]
    normal_world_0 = (
        R_ref_inv[0, 0] * n[0] + R_ref_inv[0, 1] * n[1] + R_ref_inv[0, 2] * n[2]
    )
    normal_world_1 = (
        R_ref_inv[1, 0] * n[0] + R_ref_inv[1, 1] * n[1] + R_ref_inv[1, 2] * n[2]
    )
    normal_world_2 = (
        R_ref_inv[2, 0] * n[0] + R_ref_inv[2, 1] * n[1] + R_ref_inv[2, 2] * n[2]
    )
    valid_views = 0
    H = cuda.local.array((3, 3), dtype=np.float32)
    for i in range(src_K.shape[0]):
        _compute_homography_cuda(
            H,
            ref_pose_K,
            ref_pose_R,
            ref_pose_T,
            src_K[i],
            src_R[i],
            src_T[i],
            point_3d_world_0,
            point_3d_world_1,
            point_3d_world_2,
            normal_world_0,
            normal_world_1,
            normal_world_2,
        )
        invalid_H = False
        for ii in range(3):
            for jj in range(3):
                valH = H[ii, jj]
                if math.isnan(valH) or math.isinf(valH):
                    invalid_H = True
        if invalid_H:
            continue
        inside = 0
        for pr in range(patch_size):
            for pc in range(patch_size):
                u_ref = c - half + pc
                v_ref = r - half + pr
                p0 = H[0, 0] * u_ref + H[0, 1] * v_ref + H[0, 2]
                p1 = H[1, 0] * u_ref + H[1, 1] * v_ref + H[1, 2]
                p2 = H[2, 0] * u_ref + H[2, 1] * v_ref + H[2, 2]
                if abs(p2) < 1e-8:
                    continue
                u_src = p0 / p2
                v_src = p1 / p2
                src_h = src_images_gray[i].shape[0]
                src_w = src_images_gray[i].shape[1]
                if 0 <= v_src < src_h and 0 <= u_src < src_w:
                    inside += 1
        coverage = inside / float(patch_size * patch_size)
        if coverage >= cov_required:
            valid_views += 1
    if valid_views < min_valid:
        depth_map[r, c] = np.float32(np.nan)
        cost_map[r, c] = np.float32(1.0)


class DepthOptimization:
    def __init__(self, config):
        self.config = config
        self._gpu_cum_start_nojit = None  # set after first GPU kernel finishes
        if not hasattr(self.config, "ADAPTIVE_WEIGHT_SIGMA_COLOR"):
            logging.warning(
                "ADAPTIVE_WEIGHT_SIGMA_COLOR not found in config. Using default value 10.0."
            )
            self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR = 10.0
        if not hasattr(self.config, "BUCKET_PROPAGATION_BINS"):
            logging.warning(
                "BUCKET_PROPAGATION_BINS not found in config. Using default value 16."
            )

        # Asynchronous CUDA JIT warm-up to hide initial compile latency during I/O
        try:
            import os

            if os.getenv("PM_GPU_WARMUP", "1") == "1":
                t = threading.Thread(target=self._async_warmup, daemon=True)
                t.start()
        except Exception as e:
            logging.debug(f"GPU warm-up thread not started: {e}")

    def _async_warmup(self):
        try:
            self._warmup_kernels()
        except Exception as e:
            logging.debug(f"GPU warm-up failed: {e}")

    def _warmup_kernels(self):
        """
        Launch tiny dummy kernels once to trigger CUDA JIT compilation ahead of time.
        This runs in a background thread to overlap with image pre-loading.
        """
        h, w = 32, 32
        depth_map = np.ones((h, w), dtype=np.float32)
        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        cost_map = np.full((h, w), np.inf, dtype=np.float32)
        propagation_mask = np.ones((h, w), dtype=np.bool_)
        ref_image_gray = np.ones((h, w), dtype=np.float32)
        ref_pose_K = np.array(
            [[500.0, 0.0, w / 2], [0.0, 500.0, h / 2], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        ref_pose_R = np.eye(3, dtype=np.float32)
        ref_pose_T = np.zeros(3, dtype=np.float32)
        src_images_gray = np.stack([ref_image_gray, ref_image_gray], axis=0)
        src_K = np.stack([ref_pose_K, ref_pose_K], axis=0)
        src_R = np.stack([ref_pose_R, ref_pose_R], axis=0)
        src_T = np.stack([ref_pose_T, ref_pose_T], axis=0)

        # Device copies
        d_depth_map = cuda.to_device(depth_map)
        d_normal_map = cuda.to_device(normal_map)
        d_cost_map = cuda.to_device(cost_map)
        d_propagation_mask = cuda.to_device(propagation_mask)
        d_ref_image_gray = cuda.to_device(ref_image_gray)
        d_ref_pose_K = cuda.to_device(ref_pose_K)
        d_ref_pose_R = cuda.to_device(ref_pose_R)
        d_ref_pose_T = cuda.to_device(ref_pose_T)
        d_src_images_gray = cuda.to_device(src_images_gray)
        d_src_K = cuda.to_device(src_K)
        d_src_R = cuda.to_device(src_R)
        d_src_T = cuda.to_device(src_T)
        d_depth_range_map = cuda.to_device(np.full((h, w), 1.0, dtype=np.float32))
        rng_states = create_xoroshiro128p_states(16 * 16, seed=1)

        threadsperblock = (16, 16)
        blockspergrid = ((w + 15) // 16, (h + 15) // 16)

        # Initialize normals kernel
        _initialize_normals_from_depth_cuda[blockspergrid, threadsperblock](
            d_normal_map, d_depth_map, d_ref_pose_K
        )

        # Propagation kernels based on method
        if (
            getattr(self.config, "CHOICED_PROPAGATION_METHOD", "checkerboard")
            == "checkerboard"
        ):
            neighbors_dr = np.array([-1, 1, 0, 0], dtype=np.int8)
            neighbors_dc = np.array([0, 0, -1, 1], dtype=np.int8)
            d_neighbors_dr = cuda.to_device(neighbors_dr)
            d_neighbors_dc = cuda.to_device(neighbors_dc)
            for j in [0, 1]:
                _propagate_spatial_one_color_cuda[blockspergrid, threadsperblock](
                    d_depth_map,
                    d_normal_map,
                    d_cost_map,
                    d_propagation_mask,
                    d_neighbors_dr,
                    d_neighbors_dc,
                    j,
                    7,
                    3,
                    10,
                    np.float32(self.config.ZNCC_EPSILON),
                    d_ref_image_gray,
                    d_ref_pose_K,
                    d_ref_pose_R,
                    d_ref_pose_T,
                    d_src_images_gray,
                    d_src_K,
                    d_src_R,
                    d_src_T,
                    np.int32(config.USE_MEDIAN_TOP_K),
                    np.float32(getattr(config, "MIN_WARP_COVERAGE_RATIO", 0.0)),
                    np.int32(getattr(config, "MIN_VALID_VIEWS", 0)),
                )
        else:
            # Bucket push directions kernel (use small bin arrays)
            bin_rs = cuda.to_device(np.array([8, 16], dtype=np.int32))
            bin_cs = cuda.to_device(np.array([8, 16], dtype=np.int32))
            update_counter = cuda.to_device(np.array([0], dtype=np.int32))
            # auto-tune launch dims based on bin size
            threads_1d = 256
            blocks_1d = (bin_rs.size + threads_1d - 1) // threads_1d
            blocks_1d = max(1, blocks_1d)
            for dir_code in range(4):
                _propagate_bucket_push_dir_cuda[blocks_1d, threads_1d](
                    d_depth_map,
                    d_normal_map,
                    d_cost_map,
                    d_propagation_mask,
                    bin_rs,
                    bin_cs,
                    dir_code,
                    7,
                    3,
                    10,
                    np.float32(self.config.ZNCC_EPSILON),
                    d_ref_image_gray,
                    d_ref_pose_K,
                    d_ref_pose_R,
                    d_ref_pose_T,
                    d_src_images_gray,
                    d_src_K,
                    d_src_R,
                    d_src_T,
                    update_counter,
                    np.int32(config.USE_MEDIAN_TOP_K),
                    np.float32(getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)),
                    np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0)),
                )

        # Random search kernel
        _random_search_cuda[blockspergrid, threadsperblock](
            d_depth_map,
            d_normal_map,
            d_cost_map,
            d_propagation_mask,
            0,
            7,
            3,
            0.9,
            20.0,
            np.float32(config.ZNCC_EPSILON),
            d_ref_image_gray,
            d_ref_pose_K,
            d_ref_pose_R,
            d_ref_pose_T,
            d_src_images_gray,
            d_src_K,
            d_src_R,
            d_src_T,
            10.0,
            d_depth_range_map,
            rng_states,
            np.int32(config.USE_MEDIAN_TOP_K),
            np.float32(getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)),
            np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0)),
        )
        cuda.synchronize()

    def _initialize_normals_gpu(self, depth_map, K):
        h, w = depth_map.shape
        d_depth_map = cuda.to_device(depth_map.astype(np.float32))
        d_normals = cuda.device_array((h, w, 3), dtype=np.float32)
        d_K = cuda.to_device(K.astype(np.float32))

        threadsperblock = (16, 16)
        blockspergrid_x = (w + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (h + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        _initialize_normals_from_depth_cuda[blockspergrid, threadsperblock](
            d_normals, d_depth_map, d_K
        )
        return d_normals.copy_to_host()

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
            "注目画像",
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
            "注目画像のパッチ",
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
                f"投影変換されたパッチ",
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
                f"参照画像",
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
        gt_depth,
        ref_idx=0,
    ):
        logging.info(
            f"Starting PatchMatch MVS depth refinement using '{self.config.CHOICED_PROPAGATION_METHOD}' method..."
        )

        h, w = initial_depth.shape
        depth_map = initial_depth.astype(np.float32)
        normal_map = self._initialize_normals_gpu(
            depth_map, ref_pose["K"].astype(np.float32)
        )

        # 初期の視差(深度)が有効な画素の近傍も含めたマスク（CPUに合わせてダイレーション）
        valid_initial_mask = np.isfinite(initial_depth)
        kernel = np.ones((7, 7), np.uint8)
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
        # 初期コストを計算
        for r in range(h):
            for c in range(w):
                if propagation_mask[r, c] and np.isfinite(depth_map[r, c]):
                    cost_map[r, c] = _evaluate_cost_jit(
                        r,
                        c,
                        depth_map[r, c],
                        normal_map[r, c, 0],
                        normal_map[r, c, 1],
                        normal_map[r, c, 2],
                        self.config.PATCHMATCH_PATCH_SIZE,
                        ref_image_gray,
                        ref_pose_K,
                        ref_pose_R,
                        ref_pose_T,
                        src_images_gray,
                        src_K,
                        src_R,
                        src_T,
                        self.config.TOP_K_COSTS,
                        self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                        np.float32(self.config.ZNCC_EPSILON),
                    )

        # Early-stop state will be managed on-the-fly without predeclared thresholds

        if gt_depth is not None:
            save_each_csv_dir = os.path.join(config.CSV_DIR, f"csv_{ref_idx:04d}")
            os.makedirs(save_each_csv_dir, exist_ok=True)
            clear_folder(save_each_csv_dir)
            csv_files = {
                "rmse": os.path.join(
                    save_each_csv_dir,
                    f"rmse_{self.config.CHOICED_PROPAGATION_METHOD}.csv",
                ),
                "mae": os.path.join(
                    save_each_csv_dir,
                    f"mae_{self.config.CHOICED_PROPAGATION_METHOD}.csv",
                ),
                "abs_rel": os.path.join(
                    save_each_csv_dir,
                    f"abs_rel_{self.config.CHOICED_PROPAGATION_METHOD}.csv",
                ),
                "rmse_log": os.path.join(
                    save_each_csv_dir,
                    f"rmse_log_{self.config.CHOICED_PROPAGATION_METHOD}.csv",
                ),
                "delta1": os.path.join(
                    save_each_csv_dir,
                    f"delta1_{self.config.CHOICED_PROPAGATION_METHOD}.csv",
                ),
                "delta2": os.path.join(
                    save_each_csv_dir,
                    f"delta2_{self.config.CHOICED_PROPAGATION_METHOD}.csv",
                ),
                "delta3": os.path.join(
                    save_each_csv_dir,
                    f"delta3_{self.config.CHOICED_PROPAGATION_METHOD}.csv",
                ),
            }
            for metric, path in csv_files.items():
                initialize_csv(path, ["time", metric])

        # start_refinement_time removed (unused)

        # --- PatchMatch反復ループ (GPU) ---
        save_each_depth_dir = None
        if config.DEBUG_SAVE_DEPTH_MAPS:
            save_each_depth_dir = os.path.join(
                config.DEPTH_IMAGE_DIR, f"depth_{ref_idx:04d}"
            )
            os.makedirs(save_each_depth_dir, exist_ok=True)
            # 初期深度も保存（iter_00）
            save_path0 = os.path.join(save_each_depth_dir, f"depth_iter_00.png")
            logging.info(f"Saving initial depth map to {save_path0}")
            save_depth_map_as_image(depth_map, save_path0)
        iter_times_gpu = []
        depth_map, normal_map, cost_map = self._propagate_and_search_gpu(
            depth_map,
            normal_map,
            cost_map,
            propagation_mask,
            initial_depth_error,
            ref_image_gray,
            ref_pose_K,
            ref_pose_R,
            ref_pose_T,
            src_images_gray,
            src_K,
            src_R,
            src_T,
            save_per_iter=config.DEBUG_SAVE_DEPTH_MAPS,
            save_dir=save_each_depth_dir,
            gt_depth=gt_depth,
            iter_times=iter_times_gpu,
            csv_files=csv_files if gt_depth is not None else None,
        )

        # --- Debug: cost_map statistics and CPU/GPU cost consistency check on samples ---
        try:
            mask = propagation_mask & np.isfinite(depth_map)
            if np.any(mask):
                valid_costs = cost_map[mask]
                logging.info(
                    f"[GPU Debug] cost_map stats (valid): min={np.nanmin(valid_costs):.6f}, max={np.nanmax(valid_costs):.6f}, mean={np.nanmean(valid_costs):.6f}"
                )
                h, w = depth_map.shape
                samples = [
                    (h // 2, w // 2),
                    (h // 4, w // 4),
                    (h // 4, 3 * w // 4),
                    (3 * h // 4, w // 4),
                    (3 * h // 4, 3 * w // 4),
                ]
                for rr, cc in samples:
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                        d = float(depth_map[rr, cc])
                        n0 = float(normal_map[rr, cc, 0])
                        n1 = float(normal_map[rr, cc, 1])
                        n2 = float(normal_map[rr, cc, 2])
                        cpu_cost = _evaluate_cost_jit(
                            rr,
                            cc,
                            d,
                            n0,
                            n1,
                            n2,
                            self.config.PATCHMATCH_PATCH_SIZE,
                            ref_image_gray,
                            ref_pose_K,
                            ref_pose_R,
                            ref_pose_T,
                            src_images_gray,
                            src_K,
                            src_R,
                            src_T,
                            self.config.TOP_K_COSTS,
                            self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                            np.float32(self.config.ZNCC_EPSILON),
                        )
                        gpu_cost = float(cost_map[rr, cc])
                        logging.info(
                            f"[GPU Debug] sample (r={rr}, c={cc}): gpu_cost={gpu_cost:.6f}, cpu_cost={cpu_cost:.6f}, diff={abs(gpu_cost - cpu_cost):.6f}"
                        )
        except Exception as e:
            logging.warning(f"[GPU Debug] cost comparison failed: {e}")

        logging.info("PatchMatch MVS refinement finished.")
        final_depth_map = depth_map.copy()
        final_depth_map[~propagation_mask] = np.nan
        # Save per-iteration timing plot (optional)
        try:
            import matplotlib.pyplot as _plt  # local import to avoid hard dependency

            if save_each_depth_dir is not None and len(iter_times_gpu) > 0:
                fig_path = os.path.join(save_each_depth_dir, f"iter_times_gpu.png")
                _plt.figure(figsize=(6, 4))
                _plt.plot(
                    np.arange(1, len(iter_times_gpu) + 1), iter_times_gpu, marker="o"
                )
                _plt.xlabel("Iteration")
                _plt.ylabel("Time (s)")
                _plt.title("GPU PatchMatch Iteration Times")
                _plt.grid(True)
                _plt.tight_layout()
                _plt.savefig(fig_path)
                _plt.close()
        except Exception as e:
            logging.debug(f"Skip plotting GPU iteration time: {e}")
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

        if config.DEBUG_SAVE_DEPTH_MAPS:
            save_each_depth_dir = os.path.join(
                config.DEPTH_IMAGE_DIR, f"depth_{ref_idx:04d}"
            )
            save_depth_path = os.path.join(save_each_depth_dir, f"depth_iter_00.png")
            logging.info(f"Saving initial depth map to {save_depth_path}")
            save_depth_map_as_image(depth_map, save_depth_path)

        # 2. 法線マップを初期化
        normal_map = self._initialize_normals_gpu(
            depth_map, ref_pose["K"].astype(np.float32)
        )

        if self.config.DEBUG_SAVE_NORMAL_MAPS:
            save_each_normal_dir = os.path.join(
                config.NORMAL_IMAGE_DIR, f"normal_{ref_idx:04d}"
            )
            save_path_normal = os.path.join(save_each_normal_dir, f"normal_iter_00.png")
            logging.info(f"Saving initial normal map to {save_path_normal}")
            save_normal_map_as_image(normal_map.copy(), save_path_normal)

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
        propagation_mask = np.full((h, w), True, dtype=np.bool_)

        # ダミーの initial_depth_error を作成
        initial_depth_error = np.full_like(
            depth_map,
            self.config.PATCHMATCH_VANILLA_INITIAL_SEARCH_RANGE,
            dtype=np.float32,
        )

        # 4. PatchMatch反復ループ (GPU)
        depth_map, normal_map, cost_map = self._propagate_and_search_gpu(
            depth_map,
            normal_map,
            cost_map,
            propagation_mask,
            initial_depth_error,
            ref_image_gray,
            ref_pose_K,
            ref_pose_R,
            ref_pose_T,
            src_images_gray,
            src_K,
            src_R,
            src_T,
            save_per_iter=config.DEBUG_SAVE_DEPTH_MAPS,
            save_dir=(
                os.path.join(config.DEPTH_IMAGE_DIR, f"depth_{ref_idx:04d}")
                if config.DEBUG_SAVE_DEPTH_MAPS
                else None
            ),
        )

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
        initial_valid = int(np.sum(np.isfinite(ref_depth_map) & (ref_depth_map > 0)))

        K_ref = ref_pose["K"].astype(np.float32)
        R_ref = ref_pose["R"].astype(np.float32)
        T_ref = ref_pose["T"].astype(np.float32)

        if abs(K_ref[0, 0]) < 1e-6 or abs(K_ref[1, 1]) < 1e-6:
            logging.error("Focal length is zero. Aborting geometric consistency check.")
            return ref_depth_map

        neighbor_K_list = []
        neighbor_R_list = []
        neighbor_T_list = []
        neighbor_depth_maps_list = []

        for view in neighbor_views_data:
            view_idx = view["image_idx"]
            if view_idx in all_optimized_depths:
                neighbor_K_list.append(view["K"].astype(np.float32))
                neighbor_R_list.append(view["R"].astype(np.float32))
                neighbor_T_list.append(view["T"].astype(np.float32))
                neighbor_depth_maps_list.append(
                    all_optimized_depths[view_idx].astype(np.float32)
                )

        if not neighbor_depth_maps_list:
            logging.warning(
                "No neighbor depth maps available for geometric consistency check."
            )
            return filtered_depth_map

        # JIT関数に渡すためにリストをNumPy配列に変換
        neighbor_K_np = np.stack(neighbor_K_list)
        neighbor_R_np = np.stack(neighbor_R_list)
        neighbor_T_np = np.stack(neighbor_T_list)
        neighbor_depth_maps_np = np.stack(neighbor_depth_maps_list)

        failures = 0
        for r in range(h):
            for c in range(w):
                d_ref = filtered_depth_map[r, c]
                if not np.isfinite(d_ref) or d_ref <= 0:
                    continue

                # 3Dポイントへの逆投影をループの外で一度だけ行う
                x_cam_ref = (c - K_ref[0, 2]) * d_ref / K_ref[0, 0]
                y_cam_ref = (r - K_ref[1, 2]) * d_ref / K_ref[1, 1]
                point_3d_cam_ref = np.array(
                    [x_cam_ref, y_cam_ref, d_ref], dtype=np.float32
                )
                point_3d_world = R_ref.T @ (point_3d_cam_ref - T_ref)

                # NumPy配列をJIT関数に渡す
                consistent_views = _check_geometric_consistency_jit(
                    point_3d_world,
                    neighbor_K_np,
                    neighbor_R_np,
                    neighbor_T_np,
                    neighbor_depth_maps_np,
                )

                if consistent_views < self.config.GEOMETRIC_MIN_CONSISTENT_VIEWS:
                    filtered_depth_map[r, c] = np.nan
                    failures += 1

        logging.info(
            f"{failures} points ({failures/(h*w)*100:.2f}%) invalidated by geometric consistency check."
        )
        if initial_valid > 0 and failures >= 0.9 * initial_valid:
            logging.warning(
                "Geometric filter invalidated >90% of valid pixels. Returning unfiltered depth."
            )
            return ref_depth_map
        return filtered_depth_map

    def filter_depth_map_by_photometric_consistency(
        self, depth_map, ref_image, ref_pose, neighbor_views_data
    ):
        """
        光度一貫性に基づいて深度マップをフィルタリングする
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

        # 光度一貫性チェック
        consistency_failures = 0
        for r in range(h):
            for c in range(w):
                if math.isnan(filtered_depth_map[r, c]) or math.isinf(
                    filtered_depth_map[r, c]
                ):
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

    def _propagate_and_search_gpu(
        self,
        depth_map,
        normal_map,
        cost_map,
        propagation_mask,
        initial_depth_error,
        ref_image_gray,
        ref_pose_K,
        ref_pose_R,
        ref_pose_T,
        src_images_gray,
        src_K,
        src_R,
        src_T,
        save_per_iter=False,
        save_dir=None,
        gt_depth=None,
        iter_times=None,
        csv_files=None,
    ):
        h, w = depth_map.shape

        # CPU実装に合わせ、invalid depthの強制シードは行わず、そのまま扱う
        depth_map = depth_map.copy()
        threadsperblock = (16, 16)
        blockspergrid_x = (w + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (h + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Device arrays
        d_depth_map = cuda.to_device(depth_map)
        d_normal_map = cuda.to_device(normal_map)
        d_cost_map = cuda.to_device(cost_map)
        d_propagation_mask = cuda.to_device(propagation_mask)
        d_ref_image_gray = cuda.to_device(ref_image_gray)
        d_ref_pose_K = cuda.to_device(ref_pose_K)
        d_ref_pose_R = cuda.to_device(ref_pose_R)
        d_ref_pose_T = cuda.to_device(ref_pose_T)
        d_src_images_gray = cuda.to_device(np.ascontiguousarray(src_images_gray))
        d_src_K = cuda.to_device(np.ascontiguousarray(src_K))
        d_src_R = cuda.to_device(np.ascontiguousarray(src_R))
        d_src_T = cuda.to_device(np.ascontiguousarray(src_T))
        d_depth_range_map = cuda.device_array_like(depth_map)
        rng_states = create_xoroshiro128p_states(
            threadsperblock[0] * threadsperblock[1] * blockspergrid_x * blockspergrid_y,
            seed=1,
        )

        # depth_prev_for_conv = depth_map.copy()  # early stopping disabled
        for i in range(self.config.PATCHMATCH_ITERATIONS):
            iter_start_time = time.time()
            logging.info(
                f"PatchMatch GPU Iteration {i+1}/{self.config.PATCHMATCH_ITERATIONS}"
            )

            # Propagation
            if self.config.CHOICED_PROPAGATION_METHOD == "checkerboard":
                with time_block("GPU propagate checkerboard"):
                    use_acmh = int(getattr(self.config, "ACMH_ENABLE", 0))
                    if config.PROPAGATION_NEIGHBOR_DIRECTIONS == 8:
                        neighbors_dr = np.array(
                            [-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int8
                        )
                        neighbors_dc = np.array(
                            [0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int8
                        )
                    else:
                        neighbors_dr = np.array([-1, 1, 0, 0], dtype=np.int8)
                        neighbors_dc = np.array([0, 0, -1, 1], dtype=np.int8)
                    d_neighbors_dr = cuda.to_device(neighbors_dr)
                    d_neighbors_dc = cuda.to_device(neighbors_dc)
                    if use_acmh:
                        # ACMH-H path
                        H_count = int(getattr(self.config, "ACMH_NUM_HYPOTHESES", 2))
                        joint_view_sel = int(
                            getattr(self.config, "ACMH_JOINT_VIEW_SELECTION", 1)
                        )
                        joint_top_k = int(
                            getattr(
                                self.config,
                                "ACMH_JOINT_TOP_K",
                                max(1, self.config.TOP_K_COSTS),
                            )
                        )
                        use_median_top_k = np.int32(
                            getattr(self.config, "USE_MEDIAN_TOP_K", 0)
                        )
                        cov_required = np.float32(
                            getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)
                        )
                        min_valid = np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0))
                        # H tensors on device
                        d_depth_H = cuda.device_array((H_count, h, w), dtype=np.float32)
                        d_normal_H = cuda.device_array(
                            (H_count, h, w, 3), dtype=np.float32
                        )
                        d_cost_H = cuda.device_array((H_count, h, w), dtype=np.float32)
                        # init slot0 from current maps
                        cuda.to_device(d_depth_map.copy_to_host(), to=d_depth_H[0])
                        cuda.to_device(d_normal_map.copy_to_host(), to=d_normal_H[0])
                        cuda.to_device(d_cost_map.copy_to_host(), to=d_cost_H[0])
                        for s in range(1, H_count):
                            cuda.to_device(d_depth_map.copy_to_host(), to=d_depth_H[s])
                            cuda.to_device(
                                d_normal_map.copy_to_host(), to=d_normal_H[s]
                            )
                            cuda.to_device(d_cost_map.copy_to_host(), to=d_cost_H[s])
                        for j in [0, 1]:
                            _propagate_spatial_one_color_acmhH_cuda[
                                blockspergrid, threadsperblock
                            ](
                                d_depth_H,
                                d_normal_H,
                                d_cost_H,
                                np.int32(H_count),
                                d_propagation_mask,
                                d_neighbors_dr,
                                d_neighbors_dc,
                                j,
                                np.int32(self.config.PATCHMATCH_PATCH_SIZE),
                                np.int32(self.config.TOP_K_COSTS),
                                np.int32(self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR),
                                np.float32(self.config.ZNCC_EPSILON),
                                d_ref_image_gray,
                                d_ref_pose_K,
                                d_ref_pose_R,
                                d_ref_pose_T,
                                d_src_images_gray,
                                d_src_K,
                                d_src_R,
                                d_src_T,
                                np.int32(joint_view_sel),
                                np.int32(joint_top_k),
                                use_median_top_k,
                                cov_required,
                                min_valid,
                            )
                        # reflect best slot (0) back
                        cuda.to_device(d_depth_H[0].copy_to_host(), to=d_depth_map)
                        cuda.to_device(d_normal_H[0].copy_to_host(), to=d_normal_map)
                        cuda.to_device(d_cost_H[0].copy_to_host(), to=d_cost_map)
                    else:
                        use_median_top_k = np.int32(
                            getattr(self.config, "USE_MEDIAN_TOP_K", 0)
                        )
                        cov_required = np.float32(
                            getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)
                        )
                        min_valid = np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0))
                        for j in [0, 1]:
                            _propagate_spatial_one_color_cuda[
                                blockspergrid, threadsperblock
                            ](
                                d_depth_map,
                                d_normal_map,
                                d_cost_map,
                                d_propagation_mask,
                                d_neighbors_dr,
                                d_neighbors_dc,
                                j,
                                np.int32(self.config.PATCHMATCH_PATCH_SIZE),
                                np.int32(self.config.TOP_K_COSTS),
                                np.int32(self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR),
                                np.float32(self.config.ZNCC_EPSILON),
                                d_ref_image_gray,
                                d_ref_pose_K,
                                d_ref_pose_R,
                                d_ref_pose_T,
                                d_src_images_gray,
                                d_src_K,
                                d_src_R,
                                d_src_T,
                                use_median_top_k,
                                cov_required,
                                min_valid,
                            )
            else:
                # Priority/bucket propagation path (GPU native)
                if i > 0:
                    with time_block("GPU propagate priority"):
                        mask = propagation_mask & np.isfinite(initial_depth_error)
                        rs, cs = np.nonzero(mask)
                        if rs.size > 0:
                            costs = initial_depth_error[rs, cs].astype(np.float32)
                            log_ndarray_stats("priority/bin_costs", costs)
                            num_bins = getattr(
                                self.config, "BUCKET_PROPAGATION_BINS", 4
                            )
                            cmin = float(np.min(costs))
                            cmax = float(np.max(costs))
                            if cmax - cmin < 1e-6:
                                bin_indices = np.zeros_like(costs, dtype=np.int32)
                                num_bins_effective = 1
                            else:
                                bin_width = (cmax - cmin) / num_bins
                                bin_indices = np.floor(
                                    (costs - cmin) / bin_width
                                ).astype(np.int32)
                                bin_indices[bin_indices >= num_bins] = num_bins - 1
                                num_bins_effective = num_bins
                            for b in range(num_bins_effective):
                                sel = bin_indices == b
                                if not np.any(sel):
                                    continue
                                bin_rs = rs[sel].astype(np.int32)
                                bin_cs = cs[sel].astype(np.int32)
                                # 固定順（行優先→列）で並べ替え、決定的な処理順を担保
                                order = np.lexsort((bin_cs, bin_rs))
                                bin_rs_sorted = bin_rs[order]
                                bin_cs_sorted = bin_cs[order]
                                d_bin_rs = cuda.to_device(bin_rs_sorted)
                                d_bin_cs = cuda.to_device(bin_cs_sorted)
                                threads_1d = 256
                                blocks_1d = (
                                    bin_rs_sorted.size + threads_1d - 1
                                ) // threads_1d
                                max_inner_sweeps = int(
                                    getattr(self.config, "PRIORITY_MAX_SWEEPS", 8)
                                )
                                for _ in range(max_inner_sweeps):
                                    d_update_counter = cuda.to_device(
                                        np.array([0], dtype=np.int32)
                                    )
                                    for dir_code in range(4):
                                        _propagate_bucket_push_dir_cuda[
                                            blocks_1d, threads_1d
                                        ](
                                            d_depth_map,
                                            d_normal_map,
                                            d_cost_map,
                                            d_propagation_mask,
                                            d_bin_rs,
                                            d_bin_cs,
                                            dir_code,
                                            self.config.PATCHMATCH_PATCH_SIZE,
                                            self.config.TOP_K_COSTS,
                                            self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                                            np.float32(self.config.ZNCC_EPSILON),
                                            d_ref_image_gray,
                                            d_ref_pose_K,
                                            d_ref_pose_R,
                                            d_ref_pose_T,
                                            d_src_images_gray,
                                            d_src_K,
                                            d_src_R,
                                            d_src_T,
                                            d_update_counter,
                                            np.int32(
                                                getattr(
                                                    self.config, "USE_MEDIAN_TOP_K", 0
                                                )
                                            ),
                                            np.float32(
                                                getattr(
                                                    self.config,
                                                    "MIN_WARP_COVERAGE_RATIO",
                                                    0.0,
                                                )
                                            ),
                                            np.int32(
                                                getattr(
                                                    self.config, "MIN_VALID_VIEWS", 0
                                                )
                                            ),
                                        )
                                        cuda.synchronize()
                                    updates = d_update_counter.copy_to_host()[0]
                                    if updates == 0:
                                        break

            # Random Search
            depth_range_map = (
                initial_depth_error.astype(np.float32)
                * (self.config.PATCHMATCH_DECAY_RATE**i)
            ).astype(np.float32)
            cuda.to_device(depth_range_map, to=d_depth_range_map)

            with time_block("GPU random_search"):
                _random_search_cuda[blockspergrid, threadsperblock](
                    d_depth_map,
                    d_normal_map,
                    d_cost_map,
                    d_propagation_mask,
                    i,
                    self.config.PATCHMATCH_PATCH_SIZE,
                    self.config.TOP_K_COSTS,
                    self.config.PATCHMATCH_DECAY_RATE,
                    self.config.PATCHMATCH_NORMAL_SEARCH_ANGLE,
                    np.float32(self.config.ZNCC_EPSILON),
                    d_ref_image_gray,
                    d_ref_pose_K,
                    d_ref_pose_R,
                    d_ref_pose_T,
                    d_src_images_gray,
                    d_src_K,
                    d_src_R,
                    d_src_T,
                    self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR,
                    d_depth_range_map,
                    rng_states,
                    np.int32(getattr(self.config, "USE_MEDIAN_TOP_K", 0)),
                    np.float32(getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)),
                    np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0)),
                )
            cuda.synchronize()

            # Invalidate by valid views gating (turn to NaN -> black in visualization)
            cov_required = np.float32(
                getattr(self.config, "MIN_WARP_COVERAGE_RATIO", 0.0)
            )
            min_valid = np.int32(getattr(self.config, "MIN_VALID_VIEWS", 0))
            if cov_required > 0 and min_valid > 0:
                _invalidate_by_validview_cuda[blockspergrid, threadsperblock](
                    d_depth_map,
                    d_normal_map,
                    d_cost_map,
                    d_propagation_mask,
                    np.int32(self.config.PATCHMATCH_PATCH_SIZE),
                    d_ref_pose_K,
                    d_ref_pose_R,
                    d_ref_pose_T,
                    d_src_images_gray,
                    d_src_K,
                    d_src_R,
                    d_src_T,
                    cov_required,
                    min_valid,
                )
                cuda.synchronize()

            # Mark cumulative timer start after first successful kernel run (exclude initial JIT)
            if self._gpu_cum_start_nojit is None:
                self._gpu_cum_start_nojit = time.time()

            # Save depth per-iteration if requested
            if save_per_iter and save_dir is not None:
                depth_tmp = d_depth_map.copy_to_host()
                save_path = os.path.join(save_dir, f"depth_iter_{i+1:02d}.png")
                logging.info(f"Saving depth map at iteration {i+1} to {save_path}")
                save_depth_map_as_image(depth_tmp, save_path)
            else:
                depth_tmp = None
            # Record iteration duration
            if iter_times is not None:
                iter_times.append(time.time() - iter_start_time)
                if gt_depth is not None:
                    err_path = os.path.join(save_dir, f"error_iter_{i+1:02d}.png")
                    save_error_map_as_image(depth_tmp, gt_depth, err_path)
            # Log per-iteration metrics and elapsed time (and cumulative since after JIT)
            iter_duration = time.time() - iter_start_time
            cum_txt = ""
            if self._gpu_cum_start_nojit is not None:
                cum_txt = f" | cum={time.time() - self._gpu_cum_start_nojit:.2f}s"
            if gt_depth is not None:
                if depth_tmp is None:
                    depth_host = d_depth_map.copy_to_host()
                else:
                    depth_host = depth_tmp
                try:
                    metrics = compute_depth_metrics(depth_host, gt_depth)
                    logging.info(
                        f"[GPU] Iter {i+1}: {iter_duration:.2f}s{cum_txt} | MAE={metrics['mae']:.4f}, AbsRel={metrics['abs_rel']:.4f}, "
                        f"RMSE={metrics['rmse']:.4f}, RMSElog={metrics['rmse_log']:.4f}, d1={metrics['delta1']:.4f}, d2={metrics['delta2']:.4f}, d3={metrics['delta3']:.4f}"
                    )
                    # CSV 出力
                    if csv_files is not None:
                        current_time = (
                            float(np.sum(iter_times))
                            if iter_times is not None
                            else float(i + 1)
                        )
                        for metric_key, value in metrics.items():
                            if metric_key in csv_files:
                                append_to_csv(
                                    csv_files[metric_key], [current_time, value]
                                )
                except Exception as e:
                    logging.warning(
                        f"[GPU] Could not compute metrics at iter {i+1}: {e}"
                    )
            else:
                logging.info(f"[GPU] Iter {i+1}: {iter_duration:.2f}s{cum_txt}")

            # Early convergence check disabled
            # depth_curr = d_depth_map.copy_to_host()
            # valid_mask = (
            #     np.isfinite(depth_prev_for_conv)
            #     & (depth_prev_for_conv != 0)
            #     & np.isfinite(depth_curr)
            # )
            # if np.any(valid_mask):
            #     mean_change = np.mean(
            #         np.abs(depth_prev_for_conv[valid_mask] - depth_curr[valid_mask])
            #         / np.maximum(1e-6, np.abs(depth_prev_for_conv[valid_mask]))
            #     )
            #     logging.info(f"[GPU] Average depth change: {mean_change:.5f}")
            #     if mean_change < 0.001:
            #         depth_map = depth_curr
            #         normal_map = d_normal_map.copy_to_host()
            #         cost_map = d_cost_map.copy_to_host()
            #         return depth_map, normal_map, cost_map
            # depth_prev_for_conv = depth_curr

        depth_map = d_depth_map.copy_to_host()
        normal_map = d_normal_map.copy_to_host()
        cost_map = d_cost_map.copy_to_host()

        return depth_map, normal_map, cost_map
