import logging
import os
import cv2
import numpy as np
import open3d as o3d
import config


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数を回転行列に変換"""
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
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disparity


def to_orthographic_projection(depth, color_image, camera_height):
    """中心投影から正射投影への変換を適用し、カラー画像を正射投影にマッピング"""
    rows, cols = depth.shape
    mid_idx = cols // 2
    mid_idy = rows // 2
    col_indices = np.vstack([np.arange(cols)] * rows)
    row_indices = np.vstack([np.arange(rows)] * cols).transpose()

    shift_x = np.where(
        (depth > 40) | (depth < 0),
        0,
        ((camera_height - depth) * (mid_idx - col_indices) / camera_height).astype(int),
    )
    shift_y = np.where(
        (depth > 40) | (depth < 0),
        0,
        ((camera_height - depth) * (mid_idy - row_indices) / camera_height).astype(int),
    )

    new_x = col_indices + shift_x
    new_y = row_indices + shift_y

    valid_mask = (new_x >= 0) & (new_x < cols) & (new_y >= 0) & (new_y < rows)
    new_x = np.where(valid_mask, new_x, -1)
    new_y = np.where(valid_mask, new_y, -1)

    ortho_color_image = np.zeros_like(color_image)
    valid_depth_mask = (depth > 0) & (depth < 40) & valid_mask

    ortho_color_image[new_y[valid_depth_mask], new_x[valid_depth_mask]] = color_image[
        row_indices[valid_depth_mask], col_indices[valid_depth_mask]
    ]

    ortho_depth = np.full_like(depth, np.inf)
    np.minimum.at(ortho_depth, (new_y[valid_mask], new_x[valid_mask]), depth[valid_mask])
    ortho_depth[ortho_depth == np.inf] = 0

    ortho_color_image[ortho_depth == 0] = [0, 0, 0]

    return ortho_depth, ortho_color_image


def depth_to_world(depth_map, color_image, K, R, T, pixel_size):
    """深度マップをワールド座標に変換し、テクスチャを適用する"""
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    x_coords = (i - width // 2) * pixel_size
    y_coords = -(j - height // 2) * pixel_size
    z_coords = depth_map
    local_coords = np.stack((x_coords, y_coords, z_coords), axis=-1).reshape(-1, 3)
    world_coords = (R @ local_coords.T).T + T
    valid_mask = (depth_map > 0).reshape(-1)
    colors = color_image.reshape(-1, 3)[valid_mask] / 255.0
    return world_coords[valid_mask], colors


def grid_sampling(point_cloud, colors, grid_size):
    """三次元点群のグリッドサンプリングを行う"""
    rounded_coords = np.floor(point_cloud / grid_size).astype(int)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    return point_cloud[unique_indices], colors[unique_indices]


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

drone_image_list = config.DRONE_IMAGE_LOG

camera_data = []

with open(drone_image_list, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        file_name = parts[0]
        position = tuple(map(float, parts[1:4]))  
        quaternion = tuple(map(float, parts[4:8]))  
        camera_data.append((file_name, position, quaternion))

# 3Dマップ生成開始
cumulative_world_coords = None
cumulative_colors = None

for i in range(63, 64):
    dx = float(camera_data[i][1][0])
    dy = float(camera_data[i][1][1])
    dz = float(camera_data[i][1][2])
    qx = float(camera_data[i][2][0])
    qy = float(camera_data[i][2][1])
    qz = float(camera_data[i][2][2])
    qw = float(camera_data[i][2][3])
    T = np.array([dx, dy, dz], dtype=np.float32)
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    left_image = cv2.imread(
        os.path.join(config.IMAGE_DIR, f"left_{str(i).zfill(6)}.png")
    )
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image = cv2.imread(
        os.path.join(config.IMAGE_DIR, f"right_{str(i).zfill(6)}.png")
    )
    right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    disparity = create_disparity_image(
        left_image_gray,
        right_image_gray,
        i,
        window_size=window_size,
        min_disp=min_disp,
        num_disp=num_disp,
    )

    depth = B * focal_length / (disparity + 1e-6)
    depth[(depth < 0) | (depth > 40)] = 0

    # 境界値の処理
    valid_area = (depth > 0)
    boundary_mask = valid_area & (
        ~np.roll(valid_area, 10, axis=0) | ~np.roll(valid_area, -10, axis=0) |
        ~np.roll(valid_area, 10, axis=1) | ~np.roll(valid_area, -10, axis=1)
    )
    depth[boundary_mask] = 0

    depth, ortho_color_image = to_orthographic_projection(
        depth, left_image, camera_height
    )
    world_coords, colors = depth_to_world(depth, ortho_color_image, K, R, T, pixel_size)
    world_coords, colors = grid_sampling(world_coords, colors, 0.1)

    # 点群をOpen3DのPointCloud形式に変換
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ノイズ除去を適用
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)

    # 点群を統合
    if cumulative_world_coords is None:
        cumulative_world_coords = np.asarray(pcd.points)
        cumulative_colors = np.asarray(pcd.colors)
    else:
        cumulative_world_coords = np.vstack((cumulative_world_coords, np.asarray(pcd.points)))
        cumulative_colors = np.vstack((cumulative_colors, np.asarray(pcd.colors)))

final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(cumulative_world_coords)
final_pcd.colors = o3d.utility.Vector3dVector(cumulative_colors)

# ボクセルダウンサンプリング
final_pcd = final_pcd.voxel_down_sample(voxel_size=0.05)

# 統合点群の可視化
o3d.visualization.draw_geometries([final_pcd])
