import os
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import config
import numba


def create_disparity_image(image_L, image_R, window_size, min_disp, num_disp):
    """左・右画像から視差画像を生成する"""
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=16 * 3 * window_size ** 2,
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
                    # 内側の2重ループを numba でコンパイル
                    for i in range(-half, half):
                        for j in range(-half, half):
                            diff = abs(
                                float(left[y + i, x + j]) - float(right[y + i, xr + j])
                            )
                            sad += diff
                    cost_image[y, x] = sad
    return cost_image


def to_orthographic_projection(depth, color_image, camera_height):
    """
    中心投影から正射投影への変換を適用し、カラー画像を正射投影にマッピングする関数
    """
    rows, cols = depth.shape
    mid_x = cols // 2
    mid_y = rows // 2

    # 行・列のインデックス配列を生成
    row_indices, col_indices = np.indices((rows, cols))

    # 深度が有効な領域 (0 < depth < 40) のマスク
    valid = (depth > 0) & (depth < 40)

    # 有効な領域に対してシフトを計算（無効領域は0）
    shift_x = np.zeros_like(depth, dtype=int)
    shift_y = np.zeros_like(depth, dtype=int)
    shift_x[valid] = (
        (camera_height - depth[valid]) * (mid_x - col_indices[valid]) / camera_height
    ).astype(int)
    shift_y[valid] = (
        (camera_height - depth[valid]) * (mid_y - row_indices[valid]) / camera_height
    ).astype(int)

    # 新しい座標を計算
    new_x = col_indices + shift_x
    new_y = row_indices + shift_y

    # 投影先が画像内にあるかつ有効な深度の画素のみ対象
    valid_mask = valid & (new_x >= 0) & (new_x < cols) & (new_y >= 0) & (new_y < rows)

    # 有効な元画素から投影先座標、深度、色を抽出
    valid_new_x = new_x[valid_mask]
    valid_new_y = new_y[valid_mask]
    valid_depths = depth[valid_mask]
    valid_colors = color_image[row_indices[valid_mask], col_indices[valid_mask]]

    # 投影先座標を一意に管理するため、1次元のインデックスに変換
    flat_indices = valid_new_y * cols + valid_new_x

    # flat_indicesでグループ化し、各グループ内で深度が最小となる画素を選ぶためにソート
    # np.lexsortでは、最初のキー（ここではvalid_depths）が第2優先、最後のキー（flat_indices）が第1優先となるため、
    # 同じflat_indices内で有効深度が小さいものが前方に並びます
    order = np.lexsort((valid_depths, flat_indices))
    sorted_flat_indices = flat_indices[order]
    sorted_depths = valid_depths[order]
    sorted_colors = valid_colors[order]

    # 同じflat_indicesごとに、最初（＝最小深度）のインデックスを取得
    unique_flat_indices, unique_idx = np.unique(sorted_flat_indices, return_index=True)
    min_depths = sorted_depths[unique_idx]
    min_colors = sorted_colors[unique_idx]

    # 1次元インデックスを2次元の座標に戻す
    unique_new_y = unique_flat_indices // cols
    unique_new_x = unique_flat_indices % cols

    # 出力用の画像を初期化
    ortho_depth = np.zeros_like(depth)
    ortho_color_image = np.zeros_like(color_image)

    # 最小深度と対応する色を割り当て
    ortho_depth[unique_new_y, unique_new_x] = min_depths
    ortho_color_image[unique_new_y, unique_new_x] = min_colors

    return ortho_depth, ortho_color_image


def depth_to_world(depth_map, K, R, T, pixel_size):
    """深度マップをワールド座標に変換する"""
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    x_coords = (j - width // 2) * pixel_size
    y_coords = (i - height // 2) * pixel_size
    z_coords = -(camera_height - depth_map)
    local_coords = np.stack((x_coords, y_coords, z_coords), axis=-1).reshape(-1, 3)
    world_coords = (R @ local_coords.T).T + T
    return world_coords


def convert_right_to_left_hand_coordinates(data):
    """右手座標系から左手座標系へ変換する"""
    data[:, 2] = -data[:, 2]
    return data


# パラメータの読み込み
B, height, focal_length, camera_height, K, R, pixel_size = (
    config.B,
    config.height,
    config.focal_length,
    config.camera_height,
    config.K,
    config.R,
    config.pixel_size,
)
T = np.array([0, 0, 0])
window_size, min_disp, num_disp = config.window_size, config.min_disp, config.num_disp

# 左右の画像を読み込み
img_id = 70
left_image = cv2.imread(
    os.path.join(config.IMAGE_DIR, f"left_{str(img_id).zfill(6)}.png"),
    cv2.IMREAD_GRAYSCALE,
)
right_image = cv2.imread(
    os.path.join(config.IMAGE_DIR, f"right_{str(img_id).zfill(6)}.png"),
    cv2.IMREAD_GRAYSCALE,
)

# 視差画像を生成
disparity = create_disparity_image(
    left_image,
    right_image,
    window_size=window_size,
    min_disp=min_disp,
    num_disp=num_disp,
)

# コスト画像の計算（SAD 値）
disparity_cost = compute_disparity_cost(left_image, right_image, disparity, window_size)

# 深度画像を生成
depth = B * focal_length / (disparity + 1e-6)
depth[(depth < 0) | (depth > 23)] = 0
depth = to_orthographic_projection(depth, camera_height)

world_coords = convert_right_to_left_hand_coordinates(
    depth_to_world(depth, K, R, T, pixel_size)
)
world_coords = world_coords[depth.reshape(-1) > 0]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(world_coords)
o3d.visualization.draw_geometries([pcd])
