import numpy as np
from numba import njit

@njit
def compute_depth_error_cost_jit(disparity, B, focal_length, block_size):
    half = block_size // 2
    rows, cols = disparity.shape
    cost_image = np.zeros((rows, cols), dtype=np.float32)
    for y in range(half, rows - half):
        for x in range(half, cols - half):
            d = disparity[y, x]
            if d > 0:
                cost_image[y, x] = (B * focal_length) / (d * d + 1e-6)
            else:
                cost_image[y, x] = 0.0
    return cost_image

class DepthCreator:
    def __init__(self, B, focal_length):
        self.B = B
        self.focal_length = focal_length

    def compute_depth(self, disparity):
        depth = self.B * self.focal_length / (disparity + 1e-6)
        # 深度範囲の制限
        depth[(depth < 10) | (depth > 40)] = 0
        return depth

    def to_orthographic_projection(self, depth, color_image, confidence, camera_height):
        rows, cols = depth.shape
        mid_x = cols // 2
        mid_y = rows // 2

        row_indices, col_indices = np.indices((rows, cols))
        valid = (depth > 0) & (depth < 40)

        shift_x = np.zeros_like(depth, dtype=int)
        shift_y = np.zeros_like(depth, dtype=int)
        shift_x[valid] = ((camera_height - depth[valid]) * (mid_x - col_indices[valid]) / camera_height).astype(int)
        shift_y[valid] = ((camera_height - depth[valid]) * (mid_y - row_indices[valid]) / camera_height).astype(int)

        new_x = col_indices + shift_x
        new_y = row_indices + shift_y

        valid_mask = valid & (new_x >= 0) & (new_x < cols) & (new_y >= 0) & (new_y < rows)

        valid_depths = depth[valid_mask]
        valid_color = color_image[row_indices[valid_mask], col_indices[valid_mask]]
        valid_conf = confidence[valid_mask]

        flat_indices = new_y[valid_mask] * cols + new_x[valid_mask]
        order = np.lexsort((valid_depths, flat_indices))
        sorted_flat_indices = flat_indices[order]
        sorted_depths = valid_depths[order]
        sorted_color = valid_color[order]
        sorted_conf = valid_conf[order]

        unique_flat_indices, unique_idx = np.unique(sorted_flat_indices, return_index=True)
        chosen_depth = sorted_depths[unique_idx]
        chosen_color = sorted_color[unique_idx]
        chosen_conf = sorted_conf[unique_idx]

        unique_new_y = unique_flat_indices // cols
        unique_new_x = unique_flat_indices % cols

        ortho_depth = np.zeros_like(depth)
        ortho_color = np.zeros_like(color_image)
        ortho_conf = np.zeros_like(confidence)

        ortho_depth[unique_new_y, unique_new_x] = chosen_depth
        ortho_color[unique_new_y, unique_new_x] = chosen_color
        ortho_conf[unique_new_y, unique_new_x] = chosen_conf

        return ortho_depth, ortho_color, ortho_conf

    def depth_to_world(self, depth_map, color_image, confidence_image, K, R, T, pixel_size):
        height, width = depth_map.shape
        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
        x_coords = (i - width // 2) * pixel_size
        y_coords = -(j - height // 2) * pixel_size
        z_coords = depth_map
        local_coords = np.stack((x_coords, y_coords, z_coords), axis=-1).reshape(-1, 3)
        world_coords = (R @ local_coords.T).T + T
        valid_mask = (depth_map > 10).reshape(-1)
        colors = color_image.reshape(-1, 3)[valid_mask] / 255.0
        conf = confidence_image.reshape(-1)[valid_mask]
        return world_coords[valid_mask], colors, conf

    # 高速化した深度誤差コストの計算（numba JIT コンパイル済みの関数を利用）
    def compute_depth_error_cost(self, disparity, block_size):
        return compute_depth_error_cost_jit(disparity, self.B, self.focal_length, block_size)

    # 信頼度計算：コスト画像と深度誤差コスト画像を組み合わせる
    def compute_confidence(self, cost, depth_cost, alpha=0.5, beta=0.5):
        cost_norm = (cost - np.min(cost)) / (np.max(cost) - np.min(cost) + 1e-6)
        depth_norm = (depth_cost - np.min(depth_cost)) / (np.max(depth_cost) - np.min(depth_cost) + 1e-6)
        confidence = np.exp(-(alpha * cost_norm + beta * depth_norm))
        return confidence
