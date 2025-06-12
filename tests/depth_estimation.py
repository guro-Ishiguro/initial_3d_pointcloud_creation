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
    
    def to_orthographic_projection(self, depth, color_image, camera_height):
        """中心投影の深度マップを正射投影に変換する"""
        rows, cols = depth.shape
        mid_x, mid_y = cols // 2, rows // 2
        ri, ci = np.indices((rows, cols))

        valid = np.isfinite(depth)
        shift_x = np.zeros_like(depth, dtype=int)
        shift_y = np.zeros_like(depth, dtype=int)

        shift_x[valid] = (
            (camera_height - depth[valid]) * (mid_x - ci[valid]) / camera_height
        ).astype(int)
        shift_y[valid] = (
            (camera_height - depth[valid]) * (mid_y - ri[valid]) / camera_height
        ).astype(int)

        nx = ci + shift_x
        ny = ri + shift_y

        mask = valid & (nx >= 0) & (nx < cols) & (ny >= 0) & (ny < rows)

        # マスクされた有効なピクセル情報のみを抽出
        depths = depth[mask]
        colors = color_image[ri[mask], ci[mask]]
        flat = ny[mask] * cols + nx[mask]

        # 深度が浅い（手前にある）点を優先するためにソート
        order = np.lexsort((depths, flat))
        flat_s, depth_s, col_s = flat[order], depths[order], colors[order]

        # 新しい座標で重複する点を削除（手前の点を保持）
        uniq, idx = np.unique(flat_s, return_index=True)
        uy, ux = uniq // cols, uniq % cols

        # 正射投影された深度マップとカラー画像を生成
        ortho_d = np.full_like(depth, np.nan, dtype=float)
        ortho_c = np.full_like(color_image, np.nan, dtype=float)
        ortho_d[uy, ux] = depth_s[idx]
        ortho_c[uy, ux] = col_s[idx]

        return ortho_d, ortho_c

    def perspective_to_orthographic(self, depth_map, color_image, K):
        """
        中心投影深度マップを正射投影深度マップ・カラー画像に変換する。
        """
        h, w = depth_map.shape
        pixel_size = self.config.pixel_size

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # ピクセルインデックスを生成
        v, u = np.indices((h, w))
        valid_mask = np.isfinite(depth_map)

        # 有効なピクセルのみを抽出
        depth_vals = depth_map[valid_mask]
        u_vals = u[valid_mask]
        v_vals = v[valid_mask]

        # カメラ座標系での3D点に逆投影
        x_cam = (u_vals - cx) * depth_vals / fx
        y_cam = (v_vals - cy) * depth_vals / fy
        z_cam = depth_vals

        # 正射投影の画像座標に変換 (画像中心が原点)
        ortho_u = (x_cam / pixel_size) + (w / 2)
        ortho_v = (y_cam / pixel_size) + (h / 2)  # Y軸は上下反転

        # 画像範囲内の座標のみを対象とする
        coord_mask = (ortho_u >= 0) & (ortho_u < w) & (ortho_v >= 0) & (ortho_v < h)

        ortho_u_int = ortho_u[coord_mask].astype(int)
        ortho_v_int = ortho_v[coord_mask].astype(int)

        ortho_depths = z_cam[coord_mask]
        # 対応する色を抽出
        original_colors = color_image.reshape(-1, 3)[valid_mask.flatten()]
        ortho_colors = original_colors[coord_mask]

        # オクルージョン解決: 深度が小さい(手前)の点を優先する
        # フラットなインデックスを作成し、深度が小さい順にソートする
        flat_indices = ortho_v_int * w + ortho_u_int
        sort_order = np.lexsort((ortho_depths, flat_indices))

        # ソート後のインデックスからユニークなピクセル位置のインデックスを取得
        sorted_flat_indices = flat_indices[sort_order]
        _, unique_indices_idx = np.unique(sorted_flat_indices, return_index=True)
        final_indices = sort_order[unique_indices_idx]

        # 正射投影マップを初期化
        ortho_depth_map = np.full((h, w), np.nan, dtype=np.float32)
        ortho_color_map = np.full((h, w, 3), 0, dtype=np.uint8)

        # ユニークなピクセル位置を取得
        unique_v, unique_u = np.divmod(flat_indices[final_indices], w)

        # マップに値を書き込む
        ortho_depth_map[unique_v, unique_u] = ortho_depths[final_indices]
        ortho_color_map[unique_v, unique_u] = ortho_colors[final_indices]

        return ortho_depth_map, ortho_color_map

    @staticmethod
    def ortho_depth_to_world(depth_map, color_image, R, T, pixel_size):
        """
        正射投影深度マップをワールド座標の点群に変換する。
        """
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid_mask = np.isfinite(depth_map)

        # ローカルな正射投影座標系での3D点（カメラ視点基準）
        x_local = (u[valid_mask] - w / 2) * pixel_size
        y_local = (v[valid_mask] - h / 2) * pixel_size
        z_local = depth_map[valid_mask]

        local_coords = np.vstack((x_local, y_local, z_local)).T

        # ワールド座標系へ変換
        world_coords = (R.T @ (local_coords - T).T).T

        colors = color_image.reshape(-1, 3)[valid_mask.flatten()] / 255.0

        return world_coords.astype(np.float32), colors.astype(np.float32)
