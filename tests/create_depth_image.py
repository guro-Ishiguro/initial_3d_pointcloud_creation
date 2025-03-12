import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import config
import numba


def create_disparity_image(image_L, image_R, window_size, min_disp, num_disp):
    """左・右画像から視差画像を生成する"""
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=16 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disparity


# SAD ベースでコスト画像を計算（各ピクセルにおける左右ブロックの差分和）
@numba.njit(parallel=True)
def compute_disparity_cost(left, right, disparity, block_size):
    half = block_size // 2
    rows, cols = left.shape
    cost_image = np.zeros((rows, cols), dtype=np.float32)

    for y in numba.prange(half, rows - half):
        for x in range(half, cols - half):
            disp_value = disparity[y, x]
            d = int(np.round(disp_value))
            if d >= 0:
                xr = x - d
                if (xr - half >= 0) and (xr + half < cols):
                    sad = 0.0
                    for i in range(-half, half):
                        for j in range(-half, half):
                            diff = abs(
                                float(left[y + i, x + j]) - float(right[y + i, xr + j])
                            )
                            sad += diff
                    cost_image[y, x] = sad
    return cost_image


@numba.njit(parallel=True)
def compute_depth_error_cost(disparity, block_size, B, c):
    """
    視差画像から深度誤差を考慮したコスト画像を生成する関数
    """
    half = block_size // 2
    rows, cols = disparity.shape
    cost_image = np.zeros((rows, cols), dtype=np.float32)

    for y in numba.prange(half, rows - half):
        for x in range(half, cols - half):
            d = disparity[y, x]
            if d > 0:
                cost_image[y, x] = (B * c) / (d * d + 1e-6)
            else:
                cost_image[y, x] = 0.0
    return cost_image


def compute_confidence(sad_cost, depth_cost, alpha=0.5, beta=0.5):
    sad_norm = (sad_cost - np.min(sad_cost)) / (
        np.max(sad_cost) - np.min(sad_cost) + 1e-6
    )
    depth_norm = (depth_cost - np.min(depth_cost)) / (
        np.max(depth_cost) - np.min(depth_cost) + 1e-6
    )
    confidence = np.exp(-(alpha * sad_norm + beta * depth_norm))
    return confidence


def to_orthographic_projection(depth, color_image, confidence, camera_height):
    """
    中心投影から正射投影への変換を適用し、カラー画像を正射投影にマッピングする関数
    """
    rows, cols = depth.shape
    mid_x = cols // 2
    mid_y = rows // 2

    # 各ピクセルの行, 列インデックスを生成
    row_indices, col_indices = np.indices((rows, cols))

    # 有効な深度領域 (0 < depth < 40) のマスク
    valid = (depth > 0) & (depth < 40)

    # 有効領域に対するシフト量を計算
    shift_x = np.zeros_like(depth, dtype=int)
    shift_y = np.zeros_like(depth, dtype=int)
    shift_x[valid] = (
        (camera_height - depth[valid]) * (mid_x - col_indices[valid]) / camera_height
    ).astype(int)
    shift_y[valid] = (
        (camera_height - depth[valid]) * (mid_y - row_indices[valid]) / camera_height
    ).astype(int)

    # 新しい座標の計算
    new_x = col_indices + shift_x
    new_y = row_indices + shift_y

    # 座標が画像内にあり、かつ有効な領域のみ対象
    valid_mask = valid & (new_x >= 0) & (new_x < cols) & (new_y >= 0) & (new_y < rows)

    # 各入力画像の有効領域の値を抽出
    valid_depths = depth[valid_mask]
    valid_color = color_image[
        row_indices[valid_mask], col_indices[valid_mask]
    ]  # shape=(N,3)
    valid_conf = confidence[valid_mask]

    # 再配置先の1次元インデックスを計算
    flat_indices = new_y[valid_mask] * cols + new_x[valid_mask]

    order = np.lexsort((valid_depths, flat_indices))
    sorted_flat_indices = flat_indices[order]
    sorted_depths = valid_depths[order]
    sorted_color = valid_color[order]
    sorted_conf = valid_conf[order]

    # 同じ flat_indices 内で一意な値を取得（最小深度を採用）
    unique_flat_indices, unique_idx = np.unique(sorted_flat_indices, return_index=True)
    chosen_depth = sorted_depths[unique_idx]
    chosen_color = sorted_color[unique_idx]
    chosen_conf = sorted_conf[unique_idx]

    # flat_indices を2次元の座標に変換
    unique_new_y = unique_flat_indices // cols
    unique_new_x = unique_flat_indices % cols

    # 出力画像の初期化
    ortho_depth = np.zeros_like(depth)
    # カラー画像の場合は元のチャネル数を維持
    ortho_color = np.zeros_like(color_image)
    ortho_conf = np.zeros_like(confidence)

    # 各再配置先に選ばれた値を割り当て
    ortho_depth[unique_new_y, unique_new_x] = chosen_depth
    ortho_color[unique_new_y, unique_new_x] = chosen_color
    ortho_conf[unique_new_y, unique_new_x] = chosen_conf

    return ortho_depth, ortho_color, ortho_conf


def save_depth_colormap(depth, output_path):
    """視差画像をカラーマップで保存する"""
    depth_normalized = cv2.normalize(
        depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # カラーマップの範囲を表示
    plt.imshow(depth_colormap)
    plt.colorbar(label="Depth (normalized)")
    plt.title("Disparity Colormap with Depth Values")
    plt.savefig(output_path)


# パラメータの読み込み
(B, focal_length, camera_height, K, R) = (
    config.B,
    config.focal_length,
    config.camera_height,
    config.K,
    config.R,
)
window_size, min_disp, num_disp = config.window_size, config.min_disp, config.num_disp

# 左右の画像を読み込み
img_id = 110
left_image = cv2.imread(
    os.path.join(config.IMAGE_DIR, f"left_{str(img_id).zfill(6)}.png")
)
left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_image = cv2.imread(
    os.path.join(config.IMAGE_DIR, f"right_{str(img_id).zfill(6)}.png"),
    cv2.IMREAD_GRAYSCALE,
)

# 視差画像を生成
disparity = create_disparity_image(
    left_image_gray,
    right_image,
    window_size=window_size,
    min_disp=min_disp,
    num_disp=num_disp,
)

# コスト画像の計算（SAD 値）
disparity_cost = compute_disparity_cost(
    left_image_gray, right_image, disparity, window_size
)

# 深度誤差ベースのコスト画像を計算
depth_cost = compute_depth_error_cost(disparity, window_size, B, focal_length)

confidence = compute_confidence(disparity_cost, depth_cost)

# 深度画像を生成
depth = B * focal_length / (disparity + 1e-6)
depth[(depth < 10) | (depth > 40)] = 0
ortho_depth, ortho_color, ortho_conf = to_orthographic_projection(
    depth, left_image_gray, confidence, camera_height
)

# 深度画像をカラーマップとして保存
output_path = os.path.join(config.DEPTH_IMAGE_DIR, f"depth_{img_id}.png")
save_depth_colormap(ortho_conf, output_path)
