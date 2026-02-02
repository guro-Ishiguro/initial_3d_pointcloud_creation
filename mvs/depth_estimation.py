"""視差→深度変換、深度誤差コスト、正射投影・透視投影→ワールド座標変換を提供するモジュール。"""

import numpy as np
from numba import njit


class DepthEstimator:
    """視差から深度への変換と深度誤差コスト、正射・透視投影からワールド座標への変換を行うクラス。"""

    def __init__(self, config):
        self.config = config

    def disparity_to_depth(self, disparity):
        """視差マップを B・焦点距離から深度マップに変換し、範囲外や特定区間を NaN にする。"""
        depth = self.config.B * self.config.focal_length / (disparity + 1e-6)
        depth[
            (depth < 0)
            | (depth > self.config.camera_height)
            | ((depth > 8.5) & (depth < 9.5))
        ] = np.nan
        return depth

    @staticmethod
    @njit
    def compute_depth_error_cost_jit(disparity, depth, B, focal, block_size):
        """視差・深度から深度の二乗誤差に比例するコストを計算する。"""
        half = block_size // 2
        rows, cols = disparity.shape
        cost = np.full((rows, cols), np.nan, np.float32)
        for y in range(half, rows - half):
            for x in range(half, cols - half):
                d = disparity[y, x]
                if not np.isnan(depth[y, x]):
                    cost[y, x] = (B * focal) * 2 / (d * d + 1e-6)
        return cost

    def compute_depth_error_cost(self, disparity, depth, block_size):
        """config の B・焦点距離を使って深度誤差コストを計算するラッパー。"""
        return self.compute_depth_error_cost_jit(
            disparity, depth, self.config.B, self.config.focal_length, block_size
        )

    def to_orthographic_projection(self, depth, color_image, camera_height):
        """深度とカメラ高さから正射投影の深度・色マップを作り、オクルージョンは手前優先で解決する。"""
        rows, cols = depth.shape
        mid_x, mid_y = cols // 2, rows // 2
        ri, ci = np.indices((rows, cols))
        valid = np.isfinite(depth)
        depth_valid = depth[valid]
        ci_valid = ci[valid]
        ri_valid = ri[valid]
        shift_x_float = (
            (camera_height - depth_valid) * (mid_x - ci_valid) / camera_height
        )
        shift_y_float = (
            (camera_height - depth_valid) * (mid_y - ri_valid) / camera_height
        )
        shift_x_float[~np.isfinite(shift_x_float)] = 0
        shift_y_float[~np.isfinite(shift_y_float)] = 0
        shift_x = shift_x_float.astype(int)
        shift_y = shift_y_float.astype(int)
        nx, ny = ci.copy(), ri.copy()
        nx[valid] += shift_x
        ny[valid] += shift_y
        mask = valid & (nx >= 0) & (nx < cols) & (ny >= 0) & (ny < rows)
        flat = ny[mask] * cols + nx[mask]
        depths = depth[mask]
        cols_masked = color_image[ri[mask], ci[mask]]
        order = np.lexsort((depths, flat))
        flat_s, depth_s, col_s = flat[order], depths[order], cols_masked[order]
        uniq, idx = np.unique(flat_s, return_index=True)
        uy, ux = uniq // cols, uniq % cols
        ortho_d = np.full_like(depth, np.nan)
        ortho_c = np.full_like(color_image, np.nan)
        ortho_d[uy, ux] = depth_s[idx]
        ortho_c[uy, ux] = col_s[idx]
        return ortho_d, ortho_c

    def perspective_to_orthographic(self, depth_map, color_image, K):
        """透視投影の深度マップを逆投影し、pixel_size で正射投影の深度・色マップに変換する。"""
        h, w = depth_map.shape
        pixel_size = self.config.pixel_size

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        v, u = np.indices((h, w))
        valid_mask = np.isfinite(depth_map)

        depth_vals = depth_map[valid_mask]
        u_vals = u[valid_mask]
        v_vals = v[valid_mask]

        x_cam = (u_vals - cx) * depth_vals / fx
        y_cam = (v_vals - cy) * depth_vals / fy
        z_cam = depth_vals

        ortho_u = (x_cam / pixel_size) + (w / 2)
        ortho_v = (y_cam / pixel_size) + (h / 2)

        coord_mask = (ortho_u >= 0) & (ortho_u < w) & (ortho_v >= 0) & (ortho_v < h)

        ortho_u_int = ortho_u[coord_mask].astype(int)
        ortho_v_int = ortho_v[coord_mask].astype(int)

        ortho_depths = z_cam[coord_mask]
        original_colors = color_image.reshape(-1, 3)[valid_mask.flatten()]
        ortho_colors = original_colors[coord_mask]

        flat_indices = ortho_v_int * w + ortho_u_int
        sort_order = np.lexsort((ortho_depths, flat_indices))

        sorted_flat_indices = flat_indices[sort_order]
        _, unique_indices_idx = np.unique(sorted_flat_indices, return_index=True)
        final_indices = sort_order[unique_indices_idx]

        ortho_depth_map = np.full((h, w), np.nan, dtype=np.float32)
        ortho_color_map = np.full((h, w, 3), 0, dtype=np.uint8)

        unique_v, unique_u = np.divmod(flat_indices[final_indices], w)

        ortho_depth_map[unique_v, unique_u] = ortho_depths[final_indices]
        ortho_color_map[unique_v, unique_u] = ortho_colors[final_indices]

        return ortho_depth_map, ortho_color_map

    @staticmethod
    def depth_to_world(depth_map, color_image, K, R, T, normal_map=None):
        """透視投影の深度マップを逆投影し、R・T でワールド座標の点群・色に変換する。"""
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid_mask = np.isfinite(depth_map) & (depth_map > 0)

        if not np.any(valid_mask):
            empty_points = np.empty((0, 3), dtype=np.float32)
            empty_colors = np.empty((0, 3), dtype=np.float32)
            if normal_map is not None:
                empty_normals = np.empty((0, 3), dtype=np.float32)
                return empty_points, empty_colors, empty_normals
            return empty_points, empty_colors

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u_vals = u[valid_mask]
        v_vals = v[valid_mask]
        z_vals = depth_map[valid_mask]

        x_cam = (u_vals - cx) * z_vals / fx
        y_cam = (v_vals - cy) * z_vals / fy
        z_cam = z_vals

        pts_cam = np.vstack((x_cam, y_cam, z_cam))  # (3, N)
        T_reshaped = T.reshape(3, 1) if T.ndim == 1 else T
        pts_world = (R.T @ (pts_cam - T_reshaped)).T  # (N, 3)

        colors = color_image.reshape(-1, 3)[valid_mask.flatten()] / 255.0

        if normal_map is not None:
            normals_cam = normal_map.reshape(-1, 3)[valid_mask.flatten()]
            normals_world = (R.T @ normals_cam.T).T
            norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
            norms = np.where(norms > 1e-6, norms, 1.0)
            normals_world = normals_world / norms
            return (
                pts_world.astype(np.float32),
                colors.astype(np.float32),
                normals_world.astype(np.float32),
            )
        else:
            return pts_world.astype(np.float32), colors.astype(np.float32)

    @staticmethod
    def ortho_depth_to_world(depth_map, color_image, R, T, pixel_size):
        """正射投影の深度マップを pixel_size で3Dにし、R・T でワールド座標の点群・色に変換する。"""
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid_mask = np.isfinite(depth_map)

        x_local = (u[valid_mask] - w / 2) * pixel_size
        y_local = (v[valid_mask] - h / 2) * pixel_size
        z_local = depth_map[valid_mask]

        local_coords = np.vstack((x_local, y_local, z_local)).T

        world_coords = (R.T @ (local_coords - T).T).T

        colors = color_image.reshape(-1, 3)[valid_mask.flatten()] / 255.0

        return world_coords.astype(np.float32), colors.astype(np.float32)
