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
    def to_orthographic_projection(depth, color_image, camera_height):
        """深度マップとカラー画像をオルソ画像に変換する"""
        rows, cols = depth.shape
        mid_x, mid_y = cols // 2, rows // 2
        ri, ci = np.indices((rows, cols))

        valid = np.isfinite(depth)
        shift_x = np.zeros_like(depth, int)
        shift_y = np.zeros_like(depth, int)

        # 深度に基づいてピクセルシフトを計算
        shift_x[valid] = (
            (camera_height - depth[valid]) * (mid_x - ci[valid]) / camera_height
        ).astype(int)
        shift_y[valid] = (
            (camera_height - depth[valid]) * (mid_y - ri[valid]) / camera_height
        ).astype(int)

        nx = ci + shift_x
        ny = ri + shift_y

        # 有効な範囲内のピクセルをマスク
        mask = valid & (nx >= 0) & (nx < cols) & (ny >= 0) & (ny < rows)

        depths = depth[mask]
        colors = color_image[ri[mask], ci[mask]]

        # フラットなインデックスに変換してソート（Z順）
        flat = ny[mask] * cols + nx[mask]
        order = np.lexsort((depths, flat))  # 同じ(ny, nx)の場合、より近い深度を優先
        flat_s, depth_s, col_s = flat[order], depths[order], colors[order]

        # 重複するピクセルを処理（最も近い深度を保持）
        uniq, idx = np.unique(flat_s, return_index=True)
        uy, ux = uniq // cols, uniq % cols

        ortho_d = np.full_like(depth, np.nan, float)
        ortho_c = np.full_like(color_image, np.nan, float)

        ortho_d[uy, ux] = depth_s[idx]
        ortho_c[uy, ux] = col_s[idx]

        return ortho_d, ortho_c

    @staticmethod
    def depth_to_world(depth_map, color_image, K, R, T, pixel_size):
        """深度マップとカラー画像をワールド座標の点群に変換する"""
        h, w = depth_map.shape
        # グリッドのX, Y座標を生成
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")

        # ピクセル座標からカメラ座標におけるX, Yに変換
        x = (i - w // 2) * pixel_size
        y = -(j - h // 2) * pixel_size  # Y軸は通常、画像では下方向、3Dでは上方向なので反転
        z = depth_map  # 深度マップの値をZ座標として使用

        # カメラ座標における3D点 (X, Y, Z)
        loc = np.stack((x, y, z), -1).reshape(-1, 3)

        # ワールド座標への変換: R @ (p_cam) + T
        world = (R @ loc.T).T + T
        cols = color_image.reshape(-1, 3) / 255.0  # 色を0-1に正規化

        # 無効な深度や閾値以下の深度の点をNaNとしてマーク
        valid = np.isfinite(z.flatten()) & (z.flatten() > 5)
        world[~valid] = np.nan
        cols[~valid] = np.nan
        return world, cols
