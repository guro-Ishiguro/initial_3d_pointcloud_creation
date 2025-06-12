import numpy as np
import cv2
from numba import njit
import logging


class DepthEstimator:
    def __init__(self, config):
        self.config = config

    def disparity_to_depth(self, disparity):
        """視差マップを深度マップに変換する"""
        depth = self.config.B * self.config.focal_length / (disparity + 1e-6)
        # 不適切な深度値をNaNに設定
        depth[(depth < 0) | (depth > 50) | ((depth > 8.5) & (depth < 9.5))] = np.nan
        return depth

    @staticmethod
    @njit
    def compute_depth_error_cost_jit(disparity, depth, B, focal, block_size):
        """深度誤差コストを計算する (Numbaによる高速化)"""
        half = block_size // 2
        rows, cols = disparity.shape
        cost = np.full((rows, cols), np.nan, np.float32)
        for y in range(half, rows - half):
            for x in range(half, cols - half):
                d = disparity[y, x]
                if not np.isnan(depth[y, x]):
                    cost[y, x] = (B * focal) / (d * d + 1e-6)  # 深度の二乗誤差
        return cost

    def compute_depth_error_cost(self, disparity, depth, block_size):
        """深度誤差コストを計算するラッパー"""
        return self.compute_depth_error_cost_jit(
            disparity, depth, self.config.B, self.config.focal_length, block_size
        )

    @staticmethod
    def depth_to_world(depth_map, color_image, K, R, T):
        """深度マップとカラー画像をワールド座標の点群に変換する"""
        h, w = depth_map.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h))

        valid = np.isfinite(depth_map)
        i, j, depth = i[valid], j[valid], depth_map[valid]
        colors = color_image[j, i].astype(np.float32) / 255.0

        x = (i - K[0, 2]).astype(np.float32) * depth / K[0, 0]
        y = (j - K[1, 2]).astype(np.float32) * depth / K[1, 1]
        z = depth.astype(np.float32)

        cam_coords = np.vstack((x, y, z)).T

        world_coords = (R.T @ (cam_coords - T).T).T
        return world_coords.astype(np.float32), colors.astype(np.float32)
