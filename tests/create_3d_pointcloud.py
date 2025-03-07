import os
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import config


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


def to_orthographic_projection(depth, camera_height):
    """中心投影から正射投影への変換を適用する"""
    rows, cols = depth.shape
    mid_idx = cols // 2
    mid_idy = rows // 2
    col_indices = np.vstack([np.arange(cols)] * rows)
    row_indices = np.vstack([np.arange(rows)] * cols).transpose()
    shift_x = np.where(
        (depth > 23) | (depth < 0),
        0,
        ((camera_height - depth) * (mid_idx - col_indices) / camera_height).astype(int),
    )
    shift_y = np.where(
        (depth > 23) | (depth < 0),
        0,
        ((camera_height - depth) * (mid_idy - row_indices) / camera_height).astype(int),
    )
    new_x = np.clip(col_indices + shift_x, 0, cols - 1)
    new_y = np.clip(row_indices + shift_y, 0, rows - 1)
    ortho_depth = np.full_like(depth, np.inf)
    np.minimum.at(ortho_depth, (new_y, new_x), depth)
    ortho_depth[ortho_depth == np.inf] = 0
    return ortho_depth


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
img_id = 300
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
