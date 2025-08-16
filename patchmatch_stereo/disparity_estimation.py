# tests/disparity_estimation.py

import numpy as np
import cv2
from numba import njit, prange
import logging
import time
import config
from utils import save_depth_map_as_image

@njit(fastmath=True)
def _compute_zncc_cost_jit(patch_ref, patch_src):
    """ZNCCコストを計算する"""
    mean_ref = np.mean(patch_ref)
    mean_src = np.mean(patch_src)
    
    std_ref = np.std(patch_ref)
    std_src = np.std(patch_src)

    if std_ref < 1e-6 or std_src < 1e-6:
        return 1.0  # コスト最大

    numerator = np.mean((patch_ref - mean_ref) * (patch_src - mean_src))
    denominator = std_ref * std_src
    
    # ZNCC値は-1から1なので、コストを0から1の範囲に変換
    return (1.0 - numerator) / 2.0


# ★ 修正点: parallel=True を削除
@njit(fastmath=True)
def _patchmatch_stereo_iteration(
    disp_map,
    cost_map,
    left_img_gray,
    right_img_gray,
    patch_size,
    min_disp,
    max_disp,
    is_odd_iteration
):
    """PatchMatch Stereoの1イテレーション（伝播）を実行する"""
    h, w = disp_map.shape
    half = patch_size // 2
    
    # 伝播方向を決定 (奇数/偶数イテレーションで変える)
    if is_odd_iteration:
        # Numbaでrangeをprangeに置き換えても、parallel=Trueがなければ通常のforループとして動作します
        row_range = prange(1, h - half)
        col_range = prange(1, w - half)
        dr, dc = -1, -1 # 左上から伝播
    else:
        row_range = prange(h - half - 1, -1, -1)
        col_range = prange(w - half - 1, -1, -1)
        dr, dc = 1, 1 # 右下から伝播

    for r in row_range:
        for c in col_range:
            # 境界チェック
            if not (half <= r < h - half and half <= c < w - half):
                continue
                
            # --- 空間伝播 ---
            d_neighbor_row = disp_map[r + dr, c]
            d_neighbor_col = disp_map[r, c + dc]

            current_cost = cost_map[r, c]

            # 隣(行方向)の視差でコスト計算
            c_right_row = int(c - d_neighbor_row)
            if c_right_row - half >= 0 and c_right_row + half < w:
                patch_left = left_img_gray[r-half:r+half+1, c-half:c+half+1]
                patch_right = right_img_gray[r-half:r+half+1, c_right_row-half:c_right_row+half+1]
                cost_row = _compute_zncc_cost_jit(patch_left, patch_right)
                if cost_row < current_cost:
                    disp_map[r, c] = d_neighbor_row
                    current_cost = cost_row

            # 隣(列方向)の視差でコスト計算
            c_right_col = int(c - d_neighbor_col)
            if c_right_col - half >= 0 and c_right_col + half < w:
                patch_left = left_img_gray[r-half:r+half+1, c-half:c+half+1]
                patch_right = right_img_gray[r-half:r+half+1, c_right_col-half:c_right_col+half+1]
                cost_col = _compute_zncc_cost_jit(patch_left, patch_right)
                if cost_col < current_cost:
                    disp_map[r, c] = d_neighbor_col
                    current_cost = cost_col

            cost_map[r, c] = current_cost


@njit(parallel=True, fastmath=True)
def _random_search_stereo_jit(
    disp_map,
    cost_map,
    left_img_gray,
    right_img_gray,
    patch_size,
    min_disp,
    max_disp,
    search_range
):
    """ランダムサーチを実行する"""
    h, w = disp_map.shape
    half = patch_size // 2

    # ランダムサーチは各ピクセルが独立しているので並列化可能
    for r in prange(half, h - half):
        for c in prange(half, w - half):
            current_d = disp_map[r, c]
            
            random_d = current_d + (np.random.rand() * 2 - 1) * search_range
            
            if not (min_disp <= random_d < max_disp):
                continue

            c_right = int(c - random_d)
            if c_right - half >= 0 and c_right + half < w:
                patch_left = left_img_gray[r-half:r+half+1, c-half:c+half+1]
                patch_right = right_img_gray[r-half:r+half+1, c_right-half:c_right+half+1]
                new_cost = _compute_zncc_cost_jit(patch_left, patch_right)

                if new_cost < cost_map[r, c]:
                    disp_map[r, c] = random_d
                    cost_map[r, c] = new_cost


class DisparityEstimator:
    def __init__(self, config):
        self.config = config

    def refine_disparity_with_patchmatch_stereo(self, left_image, right_image, ref_idx=0):
        logging.info("Starting PatchMatch Stereo refinement...")
        
        h, w, _ = left_image.shape
        left_img_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        right_img_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        min_disp = self.config.PATCHMATCH_STEREO_MIN_DISP
        max_disp = self.config.PATCHMATCH_STEREO_MAX_DISP
        disp_map = np.random.uniform(min_disp, max_disp, (h, w)).astype(np.float32)
        
        cost_map = np.full((h, w), 1.0, dtype=np.float32) # コストの最大値で初期化
        patch_size = self.config.PATCHMATCH_STEREO_PATCH_SIZE
        half = patch_size // 2
        for r in range(half, h - half):
            for c in range(half, w - half):
                c_right = int(c - disp_map[r, c])
                if c_right - half >=0 and c_right + half < w:
                     patch_left = left_img_gray[r-half:r+half+1, c-half:c+half+1]
                     patch_right = right_img_gray[r-half:r+half+1, c_right-half:c_right+half+1]
                     cost_map[r,c] = _compute_zncc_cost_jit(patch_left, patch_right)

        search_range = max_disp / 2.0
        for i in range(self.config.PATCHMATCH_STEREO_ITERATIONS):
            iteration_start_time = time.time()
            logging.info(f"PatchMatch Stereo Iteration {i+1}/{self.config.PATCHMATCH_STEREO_ITERATIONS}")
            
            is_odd = (i % 2 == 1)
            _patchmatch_stereo_iteration(disp_map, cost_map, left_img_gray, right_img_gray, patch_size, min_disp, max_disp, is_odd)

            _random_search_stereo_jit(disp_map, cost_map, left_img_gray, right_img_gray, patch_size, min_disp, max_disp, search_range)

            search_range *= self.config.PATCHMATCH_STEREO_DECAY_RATE
            
            iteration_end_time = time.time()
            logging.info(f"Iteration {i+1} took {iteration_end_time - iteration_start_time:.2f} seconds.")

        logging.info("PatchMatch Stereo refinement finished.")
        return disp_map