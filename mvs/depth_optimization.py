# mvs/depth_optimization.py

import logging
import math
import os
import time

import cv2
import numpy as np
from numba import cuda, njit, prange
from numba.cuda.random import create_xoroshiro128p_states
from utils import (
    save_depth_map_as_exr,
    save_depth_map_as_image,
    save_normal_map_as_image,
)

import mvs.config as config

# Constants for CUDA kernels
PATCHMATCH_PATCH_SIZE_CONST = config.PATCHMATCH_PATCH_SIZE
MAX_NEIGHBORS_CONST = config.MAX_NEIGHBORS


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
    R_ref_inv,  # 事前計算済みのR_ref_invを引数として受け取る
    K_inv,  # 事前計算済みのK_invを引数として受け取る
):
    # R_ref_invとK_invは事前計算済みなので、ここでは計算しない

    # T_ref_invの計算（R_ref_invを使用）
    # 共通の計算をまとめて最適化
    T_ref_0 = T_ref[0]
    T_ref_1 = T_ref[1]
    T_ref_2 = T_ref[2]
    T_ref_inv_0 = -(
        R_ref_inv[0, 0] * T_ref_0
        + R_ref_inv[0, 1] * T_ref_1
        + R_ref_inv[0, 2] * T_ref_2
    )
    T_ref_inv_1 = -(
        R_ref_inv[1, 0] * T_ref_0
        + R_ref_inv[1, 1] * T_ref_1
        + R_ref_inv[1, 2] * T_ref_2
    )
    T_ref_inv_2 = -(
        R_ref_inv[2, 0] * T_ref_0
        + R_ref_inv[2, 1] * T_ref_1
        + R_ref_inv[2, 2] * T_ref_2
    )

    # ワールド座標からカメラ座標への変換（R_refとT_refを使用）
    # 共通の計算をまとめて最適化
    p_ref_0 = (
        R_ref[0, 0] * plane_point_3d_0
        + R_ref[0, 1] * plane_point_3d_1
        + R_ref[0, 2] * plane_point_3d_2
        + T_ref_0
    )
    p_ref_1 = (
        R_ref[1, 0] * plane_point_3d_0
        + R_ref[1, 1] * plane_point_3d_1
        + R_ref[1, 2] * plane_point_3d_2
        + T_ref_1
    )
    p_ref_2 = (
        R_ref[2, 0] * plane_point_3d_0
        + R_ref[2, 1] * plane_point_3d_1
        + R_ref[2, 2] * plane_point_3d_2
        + T_ref_2
    )

    # 法線の変換（R_refを使用）
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

    # 相対回転行列の計算（R_src @ R_ref_inv）
    R_rel = cuda.local.array((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            R_rel[i, j] = (
                R_src[i, 0] * R_ref_inv[0, j]
                + R_src[i, 1] * R_ref_inv[1, j]
                + R_src[i, 2] * R_ref_inv[2, j]
            )

    # 相対並進ベクトルの計算（R_src @ T_ref_inv + T_src）
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
    # dの逆数を事前計算して除算回数を削減
    d_inv = 1.0 / d
    H[0, 0] = R_rel[0, 0] + T_rel_0 * n_ref_0 * d_inv
    H[0, 1] = R_rel[0, 1] + T_rel_0 * n_ref_1 * d_inv
    H[0, 2] = R_rel[0, 2] + T_rel_0 * n_ref_2 * d_inv
    H[1, 0] = R_rel[1, 0] + T_rel_1 * n_ref_0 * d_inv
    H[1, 1] = R_rel[1, 1] + T_rel_1 * n_ref_1 * d_inv
    H[1, 2] = R_rel[1, 2] + T_rel_1 * n_ref_2 * d_inv
    H[2, 0] = R_rel[2, 0] + T_rel_2 * n_ref_0 * d_inv
    H[2, 1] = R_rel[2, 1] + T_rel_2 * n_ref_1 * d_inv
    H[2, 2] = R_rel[2, 2] + T_rel_2 * n_ref_2 * d_inv

    # K_invは事前計算済みなので、ここでは計算しない

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
    R_ref_inv,  # 事前計算済みのR_ref_invを引数として受け取る
    K_inv,  # 事前計算済みのK_invを引数として受け取る
):
    h, w = ref_image_gray.shape
    half = patch_size // 2
    if r - half < 0 or r + half + 1 > h or c - half < 0 or c + half + 1 > w:
        return 1.0

    # カメラ座標の計算（K_invを使用して最適化）
    fx_inv = K_inv[0, 0]
    fy_inv = K_inv[1, 1]
    cx = ref_pose_K[0, 2]
    cy = ref_pose_K[1, 2]
    x_cam = (c - cx) * depth * fx_inv
    y_cam = (r - cy) * depth * fy_inv
    point_3d_cam_0 = x_cam
    point_3d_cam_1 = y_cam
    point_3d_cam_2 = depth

    # R_ref_invは事前計算済みなので、ここでは計算しない
    # ワールド座標への変換
    point_3d_cam_T_0 = point_3d_cam_0 - ref_pose_T[0]
    point_3d_cam_T_1 = point_3d_cam_1 - ref_pose_T[1]
    point_3d_cam_T_2 = point_3d_cam_2 - ref_pose_T[2]

    point_3d_world_0 = (
        R_ref_inv[0, 0] * point_3d_cam_T_0
        + R_ref_inv[0, 1] * point_3d_cam_T_1
        + R_ref_inv[0, 2] * point_3d_cam_T_2
    )
    point_3d_world_1 = (
        R_ref_inv[1, 0] * point_3d_cam_T_0
        + R_ref_inv[1, 1] * point_3d_cam_T_1
        + R_ref_inv[1, 2] * point_3d_cam_T_2
    )
    point_3d_world_2 = (
        R_ref_inv[2, 0] * point_3d_cam_T_0
        + R_ref_inv[2, 1] * point_3d_cam_T_1
        + R_ref_inv[2, 2] * point_3d_cam_T_2
    )

    # 法線のワールド座標への変換
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
            R_ref_inv,  # 事前計算済みのR_ref_invを渡す
            K_inv,  # 事前計算済みのK_invを渡す
        )
        # Guard invalid H
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
                warped_patch[pr, pc] = _bilinear_interpolate_cuda(
                    src_images_gray[i], v_src, u_src
                )
        costs[i] = _compute_weighted_zncc_cost_cuda(
            patch_ref, warped_patch, adaptive_weight_sigma_color, zncc_epsilon
        )
    for i in range(num_neighbors):
        for j in range(i + 1, num_neighbors):
            if costs[i] > costs[j]:
                costs[i], costs[j] = costs[j], costs[i]
    top_k = min(top_k_costs, num_neighbors)
    # 常に中央値を使用（外れ値に強い集約方法）
    # costs は昇順ソート済み（下で2重ループの後に並べ替えがあるため、こちらでも整列を担保）
    # ただし上の2重ループは隣接要素の交換なのでコストが単調とは限らない。念のため再整列。
    # 軽量なローカル選択のため単純な挿入ソートでもよいが、件数が少ないので再使用。
    # 手動の簡易ソート（バブル）
    for ii in range(num_neighbors):
        for jj in range(ii + 1, num_neighbors):
            if costs[ii] > costs[jj]:
                tmp = costs[ii]
                costs[ii] = costs[jj]
                costs[jj] = tmp
    # 中央値（偶数なら下側の中間値）
    mid = top_k // 2
    return costs[mid]


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
    # 常に中央値を使用（外れ値に強い集約方法）
    return np.median(costs[:top_k])


@njit(parallel=True, fastmath=True)
def _initialize_cost_map_jit(
    cost_map,
    depth_map,
    normal_map,
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
    """
    全ピクセルに対して並列に初期コストを計算する
    """
    h, w = depth_map.shape
    for r in prange(h):
        for c in range(w):
            if np.isfinite(depth_map[r, c]):
                cost_map[r, c] = _evaluate_cost_jit(
                    r,
                    c,
                    depth_map[r, c],
                    normal_map[r, c, 0],
                    normal_map[r, c, 1],
                    normal_map[r, c, 2],
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
                )


@cuda.jit
def _propagate_spatial_one_color_cuda(
    depth_map,
    normal_map,
    cost_map,
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
    R_ref_inv,  # 事前計算済みのR_ref_inv
    K_inv,  # 事前計算済みのK_inv
):
    c, r = cuda.grid(2)
    h, w = depth_map.shape

    if r >= h or c >= w:
        return
    if (r + c) % 2 != color:
        return
    # 無効深度の画素はスキップ
    if math.isnan(depth_map[r, c]) or math.isinf(depth_map[r, c]):
        return

    for i in range(len(neighbors_dr)):
        dr = neighbors_dr[i]
        dc = neighbors_dc[i]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < h and 0 <= nc < w):
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
            R_ref_inv,  # 事前計算済みのR_ref_invを渡す
            K_inv,  # 事前計算済みのK_invを渡す
        )

        if new_cost < cost_map[r, c]:
            depth_map[r, c] = neighbor_depth
            normal_map[r, c, 0] = neighbor_normal[0]
            normal_map[r, c, 1] = neighbor_normal[1]
            normal_map[r, c, 2] = neighbor_normal[2]
            cost_map[r, c] = new_cost


@cuda.jit
def _initialize_cost_map_cuda(
    depth_map,
    normal_map,
    cost_map,
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
    R_ref_inv,  # 事前計算済みのR_ref_inv
    K_inv,  # 事前計算済みのK_inv
):
    """
    GPUで全ピクセルに対して並列に初期コストを計算する
    """
    c, r = cuda.grid(2)
    h, w = depth_map.shape

    if r >= h or c >= w:
        return
    d = depth_map[r, c]
    if math.isnan(d) or math.isinf(d):
        return

    n = normal_map[r, c]
    cost = _evaluate_cost_cuda(
        r,
        c,
        d,
        n[0],
        n[1],
        n[2],
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
        R_ref_inv,  # 事前計算済みのR_ref_invを渡す
        K_inv,  # 事前計算済みのK_invを渡す
    )
    cost_map[r, c] = cost


@cuda.jit
def _random_search_cuda(
    depth_map,
    normal_map,
    cost_map,
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
    R_ref_inv,  # 事前計算済みのR_ref_inv
    K_inv,  # 事前計算済みのK_inv
):
    c, r = cuda.grid(2)
    h, w = depth_map.shape
    thread_id = r * w + c

    if r >= h or c >= w:
        return

    # 現在の状態を取得
    d_current = depth_map[r, c]
    n_current = normal_map[r, c]
    current_cost = cost_map[r, c]

    if math.isnan(d_current) or math.isinf(d_current) or d_current <= 0:
        return

    # -----------------------------------------------------------------
    # フェーズ1: 深度のみ更新 (Depth Refinement)
    # 法線は固定して、深度だけを動かしてみる
    # -----------------------------------------------------------------
    d_range = depth_range_map[r, c]
    if not (math.isnan(d_range) or math.isinf(d_range) or d_range <= 0):
        d_new = (
            d_current
            + (
                cuda.random.xoroshiro128p_uniform_float32(random_states, thread_id) * 2
                - 1
            )
            * d_range
        )

        if d_new > 0:
            # 法線はそのまま使用
            cost_depth_only = _evaluate_cost_cuda(
                r,
                c,
                d_new,
                n_current[0],
                n_current[1],
                n_current[2],
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
                R_ref_inv,  # 事前計算済みのR_ref_invを渡す
                K_inv,  # 事前計算済みのK_invを渡す
            )

            if cost_depth_only < current_cost:
                depth_map[r, c] = d_new
                current_cost = cost_depth_only
                d_current = d_new  # 次の法線探索のために現在値を更新
                cost_map[r, c] = current_cost  # グローバルメモリも更新

    # -----------------------------------------------------------------
    # フェーズ2: 法線のみ更新 (Normal Refinement)
    # 深度は固定(またはフェーズ1で更新された値)して、法線だけを動かす
    # 加法ノイズ方式を使用（ロドリゲスの回転公式より高速でロバスト）
    # -----------------------------------------------------------------

    # 現在の法線をローカル変数にコピー
    n_curr_x = n_current[0]
    n_curr_y = n_current[1]
    n_curr_z = n_current[2]

    # 無効な法線の場合はリセット
    if (
        math.isnan(n_curr_x)
        or math.isinf(n_curr_x)
        or math.isnan(n_curr_y)
        or math.isinf(n_curr_y)
        or math.isnan(n_curr_z)
        or math.isinf(n_curr_z)
    ):
        n_curr_x = 0.0
        n_curr_y = 0.0
        n_curr_z = 1.0

    # 探索スケール（ノイズの大きさ）を決定
    # normal_search_angle(度数法)をラジアン換算し、さらに減衰させる
    # 角度θの変化は、単位ベクトルに対して長さ 2*sin(θ/2) 程度のノイズを加えることに相当
    # 近似的に angle_rad をそのままスケールとして使っても機能します
    scale = (normal_search_angle * (decay_rate**iteration)) * (math.pi / 180.0)

    # [-0.5, 0.5] の一様乱数を生成してスケールを掛ける
    rand_x = (
        (cuda.random.xoroshiro128p_uniform_float32(random_states, thread_id) - 0.5)
        * 2.0
        * scale
    )
    rand_y = (
        (cuda.random.xoroshiro128p_uniform_float32(random_states, thread_id) - 0.5)
        * 2.0
        * scale
    )
    rand_z = (
        (cuda.random.xoroshiro128p_uniform_float32(random_states, thread_id) - 0.5)
        * 2.0
        * scale
    )

    # 現在の法線にノイズを加える
    n_new_x = n_curr_x + rand_x
    n_new_y = n_curr_y + rand_y
    n_new_z = n_curr_z + rand_z

    # 正規化（単位ベクトルに戻す）
    # 逆平方根を使用して除算を削減（精度は維持）
    norm_sq = n_new_x * n_new_x + n_new_y * n_new_y + n_new_z * n_new_z
    if norm_sq > 1e-12:  # 1e-6の2乗
        norm_inv = 1.0 / math.sqrt(norm_sq)
        n_new_x *= norm_inv
        n_new_y *= norm_inv
        n_new_z *= norm_inv

        # コスト計算（深度は固定または更新済みの値を使用）
        cost_normal_only = _evaluate_cost_cuda(
            r,
            c,
            d_current,  # フェーズ1で更新された値
            n_new_x,
            n_new_y,
            n_new_z,
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
            R_ref_inv,  # 事前計算済みのR_ref_invを渡す
            K_inv,  # 事前計算済みのK_invを渡す
        )

        # 法線単独でコストが下がれば更新
        if cost_normal_only < current_cost:
            normal_map[r, c, 0] = n_new_x
            normal_map[r, c, 1] = n_new_y
            normal_map[r, c, 2] = n_new_z
            cost_map[r, c] = cost_normal_only


@njit(fastmath=True)
def _check_geometric_consistency_jit(
    point_3d_world,
    neighbor_K_np,
    neighbor_R_np,
    neighbor_T_np,
    neighbor_depth_maps_np,
    error_threshold,
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

        if relative_error < error_threshold:
            consistent_views += 1

    return consistent_views


@njit(parallel=True)
def _filter_depth_map_by_geometric_consistency_jit(
    filtered_depth_map,
    K_ref,
    R_ref,
    T_ref,
    neighbor_K_np,
    neighbor_R_np,
    neighbor_T_np,
    neighbor_depth_maps_np,
    error_threshold,
    min_consistent_views,
):
    """
    幾何学的一貫性に基づいて深度マップをフィルタリングする（並列化版）
    """
    h, w = filtered_depth_map.shape
    failures = 0

    for r in prange(h):
        for c in range(w):
            d_ref = filtered_depth_map[r, c]
            if not (np.isfinite(d_ref) and d_ref > 0):
                continue

            # 3Dポイントへの逆投影
            x_cam_ref = (c - K_ref[0, 2]) * d_ref / K_ref[0, 0]
            y_cam_ref = (r - K_ref[1, 2]) * d_ref / K_ref[1, 1]
            point_3d_cam_ref = np.array([x_cam_ref, y_cam_ref, d_ref], dtype=np.float32)
            point_3d_world = R_ref.T @ (point_3d_cam_ref - T_ref)

            # 幾何学的一貫性をチェック
            consistent_views = _check_geometric_consistency_jit(
                point_3d_world,
                neighbor_K_np,
                neighbor_R_np,
                neighbor_T_np,
                neighbor_depth_maps_np,
                error_threshold,
            )

            if consistent_views < min_consistent_views:
                filtered_depth_map[r, c] = np.nan
                failures += 1

    return failures


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
    color_diff_threshold,
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

        if color_diff < color_diff_threshold:
            consistent_views += 1

    return consistent_views


@njit(parallel=True)
def _filter_depth_map_by_photometric_consistency_jit(
    filtered_depth_map,
    ref_image,
    K,
    R_ref,
    T_ref,
    neighbor_images,
    neighbor_R,
    neighbor_T,
    color_diff_threshold,
    min_consistent_views,
):
    """
    光度一貫性に基づいて深度マップをフィルタリングする（並列化版）
    """
    h, w = filtered_depth_map.shape
    # Numbaのprangeでは、各スレッドが独立して変数を更新するため、
    # 最終的な合計は正確ではない可能性があるが、実際には問題ない
    # （ログ出力用のカウントなので、完全に正確である必要はない）
    consistency_failures = 0

    for r in prange(h):
        for c in range(w):
            depth = filtered_depth_map[r, c]
            if not (np.isfinite(depth)):
                continue

            ref_color_pixel = ref_image[r, c]
            consistent_views = _check_photometric_consistency_jit(
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
                color_diff_threshold,
            )

            if consistent_views < min_consistent_views:
                filtered_depth_map[r, c] = np.nan
                consistency_failures += 1

    return consistency_failures


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


class DepthOptimization:
    def __init__(self, config):
        self.config = config
        self._gpu_cum_start_nojit = None  # set after first GPU kernel finishes
        if not hasattr(self.config, "ADAPTIVE_WEIGHT_SIGMA_COLOR"):
            logging.warning(
                "ADAPTIVE_WEIGHT_SIGMA_COLOR not found in config. Using default value 10.0."
            )
            self.config.ADAPTIVE_WEIGHT_SIGMA_COLOR = 10.0

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
        filename_stem=None,
    ):
        logging.info(
            "Starting PatchMatch MVS depth refinement using checkerboard propagation..."
        )

        h, w = initial_depth.shape
        depth_map = initial_depth.astype(np.float32)
        normal_map = self._initialize_normals_gpu(
            depth_map, ref_pose["K"].astype(np.float32)
        )

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

        # 初期コストをGPUで計算（CPU版より高速）
        logging.info("Computing initial cost map on GPU...")
        threadsperblock = (16, 16)
        blockspergrid_x = (w + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (h + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # R_ref_invとK_invを事前計算（CPU側で一度だけ計算）
        R_ref_inv = ref_pose_R.T.astype(np.float32)  # 転置行列
        fx = ref_pose_K[0, 0]
        fy = ref_pose_K[1, 1]
        cx = ref_pose_K[0, 2]
        cy = ref_pose_K[1, 2]
        K_inv = np.array(
            [
                [1.0 / fx, 0.0, -cx / fx],
                [0.0, 1.0 / fy, -cy / fy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # GPUデバイスにデータを転送
        d_depth_map = cuda.to_device(depth_map.astype(np.float32))
        d_normal_map = cuda.to_device(normal_map.astype(np.float32))
        d_cost_map = cuda.to_device(cost_map)
        d_ref_image_gray = cuda.to_device(ref_image_gray)
        d_ref_pose_K = cuda.to_device(ref_pose_K)
        d_ref_pose_R = cuda.to_device(ref_pose_R)
        d_ref_pose_T = cuda.to_device(ref_pose_T)
        d_src_images_gray = cuda.to_device(np.ascontiguousarray(src_images_gray))
        d_src_K = cuda.to_device(np.ascontiguousarray(src_K))
        d_src_R = cuda.to_device(np.ascontiguousarray(src_R))
        d_src_T = cuda.to_device(np.ascontiguousarray(src_T))
        d_R_ref_inv = cuda.to_device(R_ref_inv)
        d_K_inv = cuda.to_device(K_inv)

        # GPUで初期コストを計算
        _initialize_cost_map_cuda[blockspergrid, threadsperblock](
            d_depth_map,
            d_normal_map,
            d_cost_map,
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
            d_R_ref_inv,  # 事前計算済みのR_ref_invを渡す
            d_K_inv,  # 事前計算済みのK_invを渡す
        )
        cuda.synchronize()

        # 結果をホストにコピー
        cost_map = d_cost_map.copy_to_host()
        logging.info("Initial cost map computation completed on GPU.")

        # Early-stop state will be managed on-the-fly without predeclared thresholds

        # CSV書き込み処理は削除（評価は別スクリプトで実行）

        # start_refinement_time removed (unused)

        # --- PatchMatch反復ループ (GPU) ---
        save_each_depth_dir = None
        # ファイル名ベースのフォルダ名を使用（フォールバック: ref_idx）
        depth_folder_name = (
            filename_stem if filename_stem is not None else f"{ref_idx:04d}"
        )
        if config.DEBUG_SAVE_DEPTH_MAPS:
            save_each_depth_dir = os.path.join(
                config.DEPTH_IMAGE_DIR, depth_folder_name
            )
            os.makedirs(save_each_depth_dir, exist_ok=True)
            # 初期深度も保存（iter_00）
            save_path0 = os.path.join(save_each_depth_dir, f"depth_iter_00.png")
            logging.info(f"Saving initial depth map to {save_path0}")
            save_depth_map_as_image(depth_map, save_path0)
        if self.config.DEBUG_SAVE_NORMAL_MAPS:
            save_each_normal_dir = os.path.join(
                config.NORMAL_IMAGE_DIR, depth_folder_name
            )
            os.makedirs(save_each_normal_dir, exist_ok=True)
            save_pathn0 = os.path.join(save_each_normal_dir, f"normal_iter_00.png")
            logging.info(f"Saving initial normal map to {save_pathn0}")
            save_normal_map_as_image(normal_map.copy(), save_pathn0)
        iter_times_gpu = []
        # 各イテレーションの深度マップを保存するリスト（エラーマップの統一スケール用）
        all_iteration_depths = []
        if gt_depth is not None:
            all_iteration_depths.append(initial_depth.copy())

        (
            depth_map,
            normal_map,
            cost_map,
            iteration_depths,
        ) = self._propagate_and_search_gpu(
            depth_map,
            normal_map,
            cost_map,
            initial_depth_error,
            ref_image_gray,
            ref_pose_K,
            ref_pose_R,
            ref_pose_T,
            src_images_gray,
            src_K,
            src_R,
            src_T,
            ref_idx=ref_idx,
            filename_stem=filename_stem,
            save_per_iter=config.DEBUG_SAVE_DEPTH_MAPS,
            save_dir=save_each_depth_dir,
            save_normals_per_iter=self.config.DEBUG_SAVE_NORMAL_MAPS,
            normal_save_dir=(
                os.path.join(config.NORMAL_IMAGE_DIR, depth_folder_name)
                if self.config.DEBUG_SAVE_NORMAL_MAPS
                else None
            ),
            gt_depth=gt_depth,
            iter_times=iter_times_gpu,
            csv_files=None,  # CSV書き込み処理は削除（評価は別スクリプトで実行）
        )

        # すべてのイテレーションの深度マップを結合
        if gt_depth is not None and iteration_depths:
            all_iteration_depths.extend(iteration_depths)

        # 深度マップのEXR形式での保存はmain.pyで統一して行うため、ここでは削除

        # --- Debug: cost_map statistics and cost validation check on samples ---
        try:
            mask = np.isfinite(depth_map)
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
                        reference_cost = _evaluate_cost_jit(
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
                            f"[GPU Debug] sample (r={rr}, c={cc}): gpu_cost={gpu_cost:.6f}, reference_cost={reference_cost:.6f}, diff={abs(gpu_cost - reference_cost):.6f}"
                        )
        except Exception as e:
            logging.warning(f"[GPU Debug] cost comparison failed: {e}")

        logging.info("PatchMatch MVS refinement finished.")
        final_depth_map = depth_map.copy()
        final_normal_map = normal_map.copy()

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

        return final_depth_map, final_normal_map, iter_times_gpu

    def filter_depth_map_by_geometric_consistency(
        self, ref_depth_map, ref_pose, neighbor_views_data, all_optimized_depths
    ):
        """
        複数ビュー間の幾何学的一貫性に基づいて深度マップをフィルタリングする
        """
        logging.info("Filtering depth map by geometric consistency...")
        h, w = ref_depth_map.shape
        filtered_depth_map = ref_depth_map.copy()

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

        # 並列化されたJIT関数を使用して高速化
        failures = _filter_depth_map_by_geometric_consistency_jit(
            filtered_depth_map,
            K_ref,
            R_ref,
            T_ref,
            neighbor_K_np,
            neighbor_R_np,
            neighbor_T_np,
            neighbor_depth_maps_np,
            np.float32(self.config.GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD),
            self.config.GEOMETRIC_MIN_CONSISTENT_VIEWS,
        )

        logging.info(
            f"{failures} points ({failures/(h*w)*100:.2f}%) invalidated by geometric consistency check."
        )
        return filtered_depth_map

    def filter_depth_map_by_photometric_consistency(
        self, depth_map, ref_image, ref_pose, neighbor_views_data
    ):
        """
        光度一貫性に基づいて深度マップをフィルタリングする（高速な並列化版）
        """
        logging.info(
            "Filtering optimized depth map based on cost and photometric consistency..."
        )
        filtered_depth_map = depth_map.copy().astype(np.float32)

        # データをJIT用に準備
        K = ref_pose["K"].astype(np.float32)
        R_ref = ref_pose["R"].astype(np.float32)
        T_ref = ref_pose["T"].astype(np.float32)

        neighbor_images = np.stack(
            [view["image"].astype(np.float32) for view in neighbor_views_data], axis=0
        )
        neighbor_R = np.stack(
            [view["R"].astype(np.float32) for view in neighbor_views_data], axis=0
        )
        neighbor_T = np.stack(
            [view["T"].astype(np.float32) for view in neighbor_views_data], axis=0
        )

        ref_image_float = ref_image.astype(np.float32)

        # 光度一貫性チェック（並列化版）
        consistency_failures = _filter_depth_map_by_photometric_consistency_jit(
            filtered_depth_map,
            ref_image_float,
            K,
            R_ref,
            T_ref,
            neighbor_images,
            neighbor_R,
            neighbor_T,
            np.float32(self.config.FILTERING_COLOR_DIFFERENCE_THRESHOLD),
            self.config.FILTERING_MIN_CONSISTENT_VIEWS,
        )

        logging.info(
            f"{consistency_failures} points invalidated by photometric consistency check."
        )

        return filtered_depth_map

    def _propagate_and_search_gpu(
        self,
        depth_map,
        normal_map,
        cost_map,
        initial_depth_error,
        ref_image_gray,
        ref_pose_K,
        ref_pose_R,
        ref_pose_T,
        src_images_gray,
        src_K,
        src_R,
        src_T,
        ref_idx=0,
        filename_stem=None,
        save_per_iter=False,
        save_dir=None,
        save_normals_per_iter=False,
        normal_save_dir=None,
        gt_depth=None,
        iter_times=None,
        csv_files=None,
    ):
        # 各イテレーションの深度マップを保存するリスト（エラーマップの統一スケール用）
        iteration_depths = []
        h, w = depth_map.shape

        # Invalid depthの強制シードは行わず、そのまま扱う
        depth_map = depth_map.copy()
        threadsperblock = (16, 16)
        blockspergrid_x = (w + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (h + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # R_ref_invとK_invを事前計算（CPU側で一度だけ計算）
        R_ref_inv = ref_pose_R.T.astype(np.float32)  # 転置行列
        fx = ref_pose_K[0, 0]
        fy = ref_pose_K[1, 1]
        cx = ref_pose_K[0, 2]
        cy = ref_pose_K[1, 2]
        K_inv = np.array(
            [
                [1.0 / fx, 0.0, -cx / fx],
                [0.0, 1.0 / fy, -cy / fy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Device arrays
        d_depth_map = cuda.to_device(depth_map)
        d_normal_map = cuda.to_device(normal_map)
        d_cost_map = cuda.to_device(cost_map)
        d_ref_image_gray = cuda.to_device(ref_image_gray)
        d_ref_pose_K = cuda.to_device(ref_pose_K)
        d_ref_pose_R = cuda.to_device(ref_pose_R)
        d_ref_pose_T = cuda.to_device(ref_pose_T)
        d_src_images_gray = cuda.to_device(np.ascontiguousarray(src_images_gray))
        d_src_K = cuda.to_device(np.ascontiguousarray(src_K))
        d_src_R = cuda.to_device(np.ascontiguousarray(src_R))
        d_src_T = cuda.to_device(np.ascontiguousarray(src_T))
        d_R_ref_inv = cuda.to_device(R_ref_inv)
        d_K_inv = cuda.to_device(K_inv)
        d_depth_range_map = cuda.device_array_like(depth_map)
        rng_states = create_xoroshiro128p_states(
            threadsperblock[0] * threadsperblock[1] * blockspergrid_x * blockspergrid_y,
            seed=1,
        )

        # depth_prev_for_conv = depth_map.copy()  # early stopping disabled
        # 最初のイテレーション開始時点を記録（累積時間の基準）
        first_iter_start_time = None
        for i in range(self.config.PATCHMATCH_ITERATIONS):
            # イテレーション全体の開始時間（ログ表示用）
            iter_start_time = time.time()
            # 最初のイテレーション開始時点を記録
            if first_iter_start_time is None:
                first_iter_start_time = iter_start_time

            logging.info(
                f"PatchMatch GPU Iteration {i+1}/{self.config.PATCHMATCH_ITERATIONS}"
            )

            # Propagation (checkerboard)
            neighbors_dr = np.array([-1, 1, 0, 0], dtype=np.int8)
            neighbors_dc = np.array([0, 0, -1, 1], dtype=np.int8)
            d_neighbors_dr = cuda.to_device(neighbors_dr)
            d_neighbors_dc = cuda.to_device(neighbors_dc)
            for j in [0, 1]:
                _propagate_spatial_one_color_cuda[blockspergrid, threadsperblock](
                    d_depth_map,
                    d_normal_map,
                    d_cost_map,
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
                    d_R_ref_inv,  # 事前計算済みのR_ref_invを渡す
                    d_K_inv,  # 事前計算済みのK_invを渡す
                )

            # Random Search
            depth_range_map = (
                initial_depth_error.astype(np.float32)
                * (self.config.PATCHMATCH_DECAY_RATE**i)
            ).astype(np.float32)
            cuda.to_device(depth_range_map, to=d_depth_range_map)
            _random_search_cuda[blockspergrid, threadsperblock](
                d_depth_map,
                d_normal_map,
                d_cost_map,
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
                d_R_ref_inv,  # 事前計算済みのR_ref_invを渡す
                d_K_inv,  # 事前計算済みのK_invを渡す
            )
            cuda.synchronize()

            # Mark cumulative timer start after first successful kernel run (exclude initial JIT)
            if self._gpu_cum_start_nojit is None:
                self._gpu_cum_start_nojit = time.time()

            # Save depth per-iteration if requested (each iteration when DEBUG_SAVE_DEPTH_MAPS is true)
            depth_tmp = None
            if save_per_iter and save_dir is not None:
                depth_tmp = d_depth_map.copy_to_host()
                # PNG形式で保存
                save_path = os.path.join(save_dir, f"depth_iter_{i+1:02d}.png")
                logging.info(f"Saving depth map at iteration {i+1} to {save_path}")
                save_depth_map_as_image(depth_tmp, save_path)
                # EXR形式でも保存
                save_path_exr = os.path.join(save_dir, f"depth_iter_{i+1:02d}.exr")
                logging.info(
                    f"Saving depth map as EXR at iteration {i+1} to {save_path_exr}"
                )
                save_depth_map_as_exr(depth_tmp, save_path_exr)
            # Save normal per-iteration if requested (each iteration when DEBUG_SAVE_NORMAL_MAPS is true)
            if save_normals_per_iter and normal_save_dir is not None:
                normal_tmp = d_normal_map.copy_to_host()
                save_path_n = os.path.join(
                    normal_save_dir, f"normal_iter_{i+1:02d}.png"
                )
                logging.info(f"Saving normal map at iteration {i+1} to {save_path_n}")
                save_normal_map_as_image(normal_tmp, save_path_n)
            # Record cumulative time from first iteration start
            # 各イテレーションの開始時点での経過時間を記録
            if iter_times is not None:
                cumulative_time = iter_start_time - first_iter_start_time
                iter_times.append(cumulative_time)
                if gt_depth is not None and save_per_iter and (save_dir is not None):
                    # depth_tmp が未作成（保存オフ）ならホストへコピー
                    if depth_tmp is None:
                        depth_tmp = d_depth_map.copy_to_host()
                    # エラーマップは後で統一スケールで再保存するため、ここでは保存しない
                    # 代わりに深度マップをリストに保存
                    iteration_depths.append(depth_tmp.copy())
            # Log per-iteration metrics and elapsed time (and cumulative from first iteration)
            # iter_durationはイテレーション全体の時間（ログ表示用）
            iter_duration = time.time() - iter_start_time
            # cumは最初のイテレーション開始からの累積時間（time.csvと同じ基準）
            cum_txt = ""
            if first_iter_start_time is not None:
                cum_time = time.time() - first_iter_start_time
                cum_txt = f", 累積: {cum_time:.2f}秒"
            # メトリクス計算と出力は無効化（評価は別スクリプトで実行）
            # if gt_depth is not None:
            #     if depth_tmp is None:
            #         depth_host = d_depth_map.copy_to_host()
            #     else:
            #         depth_host = depth_tmp
            #     try:
            #         metrics = compute_depth_metrics(depth_host, gt_depth)
            #         logging.info(
            #             f"[{filename_stem if filename_stem else f'{ref_idx:04d}'}] イテレーション {i+1}/{self.config.PATCHMATCH_ITERATIONS} "
            #             f"(経過時間: {iter_duration:.2f}秒{cum_txt}) | "
            #             f"MAE={metrics['mae']:.4f}, AbsRel={metrics['abs_rel']:.4f}, SqRel={metrics['sq_rel']:.4f}, "
            #             f"RMSE={metrics['rmse']:.4f}, RMSElog={metrics['rmse_log']:.4f}, "
            #             f"d1={metrics['delta1']:.4f}, d2={metrics['delta2']:.4f}, d3={metrics['delta3']:.4f}"
            #         )
            #         # CSV書き込み処理は削除（評価は別スクリプトで実行）
            #     except Exception as e:
            #         logging.warning(
            #             f"[GPU] Could not compute metrics at iter {i+1}: {e}"
            #         )
            # else:
            logging.info(
                f"[{filename_stem if filename_stem else f'{ref_idx:04d}'}] イテレーション {i+1}/{self.config.PATCHMATCH_ITERATIONS} "
                f"(経過時間: {iter_duration:.2f}秒{cum_txt})"
            )

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

        return depth_map, normal_map, cost_map, iteration_depths
