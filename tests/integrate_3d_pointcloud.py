import logging
import os
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import matching
import config


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数を回転行列に変換"""
    # 回転行列を計算
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    return R


def create_disparity_image(image_L, image_R, img_id, window_size, min_disp, num_disp):
    """左・右画像から視差画像を生成する"""
    if image_L is None or image_R is None:
        logging.error(f"Error: One or both images for ID {img_id} are None.")
        return None
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
    x_coords = (i - width // 2) * pixel_size
    y_coords = -(j - height // 2) * pixel_size
    z_coords = camera_height - depth_map
    local_coords = np.stack((x_coords, y_coords, z_coords), axis=-1).reshape(-1, 3)
    print(local_coords)
    world_coords = (R @ local_coords.T).T + T
    return world_coords


def grid_sampling(point_cloud, grid_size):
    """三次元点群のグリッドサンプリングを行う"""
    rounded_coords = np.floor(point_cloud / grid_size).astype(int)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    return point_cloud[unique_indices]


def write_ply(filename, vertices):
    """頂点データをPLYファイルに書き込む"""
    header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, vertices, fmt="%f %f %f")


# パラメータの読み込み
B, height, focal_length, camera_height, K, pixel_size = (
    config.B,
    config.height,
    config.focal_length,
    config.camera_height,
    config.K,
    config.pixel_size,
)
window_size, min_disp, num_disp = config.window_size, config.min_disp, config.num_disp

drone_image_list = matching.read_file_list(config.DRONE_IMAGE_LOG)
orb_slam_pose_list = matching.read_file_list(config.ORB_SLAM_LOG)

# マッチング開始
matches = matching.associate(drone_image_list, orb_slam_pose_list, 0.0, 0.02)
print(len(matches))

# 3Dマップ生成開始
cumulative_world_coords = None
cumulative_colors = None

for i in range(29, 31):
    img_id = int(matches[i][1][0])
    dx = float(matches[i][2][0])
    dy = float(matches[i][2][1])
    dz = float(matches[i][2][2])
    qx = float(matches[i][2][3])
    qy = float(matches[i][2][4])
    qz = float(matches[i][2][5])
    qw = float(matches[i][2][6])
    T = np.array([dx, dy, dz], dtype=np.float32)
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    left_image = cv2.imread(
        os.path.join(config.IMAGE_DIR, f"left_{str(img_id).zfill(6)}.png")
    )
    left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    left_image_gray = cv2.flip(left_image_gray, 0)
    right_image = cv2.imread(
        os.path.join(config.IMAGE_DIR, f"right_{str(img_id).zfill(6)}.png")
    )
    right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    right_image_gray = cv2.flip(right_image_gray, 0)

    if left_image_gray is None or right_image_gray is None:
        continue

    disparity = create_disparity_image(
        left_image_gray,
        right_image_gray,
        i,
        window_size=window_size,
        min_disp=min_disp,
        num_disp=num_disp,
    )
    if disparity is None:
        continue

    depth = B * focal_length / (disparity + 1e-6)
    depth[(depth < 0) | (depth > 23)] = 0
    depth = to_orthographic_projection(depth, camera_height)
    world_coords = depth_to_world(depth, K, R, T, pixel_size)
    world_coords = world_coords[depth.reshape(-1) > 0]
    world_coords = grid_sampling(world_coords, 0.1)

    if cumulative_world_coords is None:
        cumulative_world_coords = world_coords
    else:
        cumulative_world_coords = np.vstack((cumulative_world_coords, world_coords))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cumulative_world_coords)
# pcd = pcd.voxel_down_sample(voxel_size=0.05)
# pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
write_ply(config.POINT_CLOUD_FILE_PATH, np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])
