import os
import shutil
import cv2
import numpy as np
import open3d as o3d
import logging
import time
import config
import argparse
from numba import njit
from concurrent.futures import ProcessPoolExecutor


# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 録画の設定
video_filename = os.path.join(config.VIDEO_DIR, "3d_map_visualization.avi")
video_fps = 10
video_width, video_height = 640, 480


def parse_arguments():
    """コマンド引数の値を受け取る"""
    parser = argparse.ArgumentParser(description="3D Point Cloud Creater")
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Show viewer during point cloud generation.",
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Record viewer output to video."
    )
    return parser.parse_args()


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数を回転行列に変換"""
    R = np.array(
        [
            [
                1 - 2 * (qy ** 2 + qz ** 2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx ** 2 + qz ** 2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx ** 2 + qy ** 2),
            ],
        ]
    )
    return R


def clear_folder(dir_path):
    """指定フォルダの中身を削除する"""
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                logging.info(f"{dir_path} is cleared.")
            except Exception as e:
                logging.error(f"Error deleting {file_path}: {e}")
    else:
        logging.info(f"The folder {dir_path} does not exist.")


def to_orthographic_projection(depth, color_image, camera_height):
    """
    中心投影から正射投影への変換を適用し、カラー画像を正射投影にマッピングする関数
    """
    rows, cols = depth.shape
    mid_x = cols // 2
    mid_y = rows // 2

    row_indices, col_indices = np.indices((rows, cols))

    valid = np.isfinite(depth)

    shift_x = np.zeros_like(depth, dtype=int)
    shift_y = np.zeros_like(depth, dtype=int)
    shift_x[valid] = (
        (camera_height - depth[valid]) * (mid_x - col_indices[valid]) / camera_height
    ).astype(int)
    shift_y[valid] = (
        (camera_height - depth[valid]) * (mid_y - row_indices[valid]) / camera_height
    ).astype(int)

    new_x = col_indices + shift_x
    new_y = row_indices + shift_y

    valid_mask = valid & (new_x >= 0) & (new_x < cols) & (new_y >= 0) & (new_y < rows)

    valid_depths = depth[valid_mask]
    valid_color = color_image[
        row_indices[valid_mask], col_indices[valid_mask]
    ] 

    flat_indices = new_y[valid_mask] * cols + new_x[valid_mask]

    order = np.lexsort((valid_depths, flat_indices))
    sorted_flat_indices = flat_indices[order]
    sorted_depths = valid_depths[order]
    sorted_color = valid_color[order]

    unique_flat_indices, unique_idx = np.unique(sorted_flat_indices, return_index=True)
    chosen_depth = sorted_depths[unique_idx]
    chosen_color = sorted_color[unique_idx]

    unique_new_y = unique_flat_indices // cols
    unique_new_x = unique_flat_indices % cols

    ortho_depth = np.full_like(depth, np.nan, dtype=float)
    ortho_color = np.full_like(color_image, np.nan, dtype=float)

    ortho_depth[unique_new_y, unique_new_x] = chosen_depth
    ortho_color[unique_new_y, unique_new_x] = chosen_color

    return ortho_depth, ortho_color


def depth_to_world(depth_map, color_image, K, R, T, pixel_size):
    """
    深度マップ、カラー画像、信頼度画像を用いて、ワールド座標とカラー情報、信頼度値を取得する関数
    """
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    x_coords = (i - width // 2) * pixel_size
    y_coords = -(j - height // 2) * pixel_size
    z_coords = depth_map
    local_coords = np.stack((x_coords, y_coords, z_coords), axis=-1).reshape(-1, 3)
    world_coords = (R @ local_coords.T).T + T
    colors = color_image.reshape(-1, 3) / 255.0
    valid = np.isfinite(z_coords.flatten()) & (z_coords.flatten() > 5)
    invalid = ~valid
    world_coords[invalid, :] = np.nan
    colors[invalid, :] = np.nan
    return world_coords, colors


def create_disparity(image_L, image_R, img_id):
    if image_L is None or image_R is None:
        logging.error(f"Error: One or both images for ID {img_id} are None.")
        return None
    stereo = cv2.StereoSGBM_create(
        minDisparity=config.min_disp,
        numDisparities=config.num_disp,
        blockSize=config.window_size,
        P1=8 * 3 * config.window_size ** 2,
        P2=16 * 3 * config.window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disparity

@njit
def census_transform_numba(img, window_size=7):
    half = window_size // 2
    rows, cols = img.shape
    census = np.zeros((rows, cols), dtype=np.uint64)
    for y in range(half, rows - half):
        for x in range(half, cols - half):
            center = img[y, x]
            descriptor = 0
            for i in range(-half, half + 1):
                for j in range(-half, half + 1):
                    if i == 0 and j == 0:
                        continue
                    descriptor = descriptor << 1
                    if img[y + i, x + j] < center:
                        descriptor |= 1
            census[y, x] = descriptor
    return census

@njit
def compute_census_cost(
    left, right, disparity, census_left, census_right, window_size
):
    """
    単独のセンサス変換によるコスト計算（ハミング距離のみ）
    """
    half = window_size // 2
    rows, cols = left.shape
    cost_image = np.zeros((rows, cols), dtype=np.float32)
    for y in range(half, rows - half):
        for x in range(half, cols - half):
            d_val = disparity[y, x]
            d = int(np.round(d_val))
            if d >= 0:
                xr = x - d
                if (xr - half >= 0) and (xr + half < cols):
                    cost = 0
                    xor_val = census_left[y, x] ^ census_right[y, xr]
                    while xor_val:
                        cost += xor_val & 1
                        xor_val = xor_val >> 1
                    cost_image[y, x] = cost
    return cost_image

@njit
def compute_depth_error_cost_jit(
    disparity, depth, B, focal_length, block_size, threshold=0.1
):
    half = block_size // 2
    rows, cols = disparity.shape
    cost_image = np.full((rows, cols), np.nan, dtype=np.float32)
    for y in range(half, rows - half):
        for x in range(half, cols - half):
            d = disparity[y, x]
            if not np.isnan(depth[y, x]):
                cost = (B * focal_length * 1.0) / (d * d + 1e-6)
                if cost > threshold:
                    aggregated_cost = threshold + np.log((cost - threshold) + 1)
                    cost_image[y, x] = aggregated_cost
                else:
                    cost_image[y, x] = cost
    return cost_image

def compute_depth_error_cost(disparity, depth, block_size):
    return compute_depth_error_cost_jit(
        disparity, depth, config.B, config.focal_length, block_size
    )


def grid_sampling(point_cloud, colors, grid_size):
    """
    各画像ペア内でのグリッドサンプリング
    """
    rounded_coords = np.floor(point_cloud / grid_size).astype(int)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    return (
        point_cloud[unique_indices],
        colors[unique_indices],
    )


def process_point_cloud(points, colors):
    """統合点群に対してボクセルダウンサンプリングとアウトライヤ除去を実施"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
    return pcd


def voxel_downsample_points(points, colors, voxel_size):
    """点群と色情報を指定サイズでボクセルダウンサンプリングする"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd_down.points), np.asarray(pcd_down.colors)


def process_image_pair(image_data):
    """
    画像ペアを処理してワールド座標の点群を生成する。
    """
    (
        img_id,
        T,
        left_image_path,
        right_image_path,
        B,
        focal_length,
        K,
        R,
        camera_height,
        pixel_size,
        window_size,
        min_disp,
        num_disp,
    ) = image_data

    left_image = cv2.imread(left_image_path)
    if left_image is None:
        logging.error(f"Left image not found for ID {img_id}: {left_image_path}")
        return None
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image = cv2.imread(right_image_path)
    if right_image is None:
        logging.error(f"Right image not found for ID {img_id}: {right_image_path}")
        return None

    left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    disparity = create_disparity(
        left_image_gray, right_image_gray, img_id
    )
    if disparity is None:
        logging.error(f"Disparity computation failed for ID {img_id}")
        return None

    census_left = census_transform_numba(
        left_image_gray, window_size
    )
    census_right = census_transform_numba(
        right_image_gray, window_size
    )
    census_cost = compute_census_cost(
        left_image_gray, right_image_gray, disparity, census_left, census_right, window_size
    )
    census_cost = census_cost.flatten()

    depth = B * focal_length / (disparity + 1e-6)
    depth[(depth < 0) | (depth > 50) | ((depth > 8.5) & (depth < 9.5))] = np.nan

    depth_cost = compute_depth_error_cost(disparity, depth, window_size)
    depth_cost = depth_cost.flatten()

    valid_area = np.isfinite(depth)
    boundary_mask = valid_area & (
        ~np.roll(valid_area, 10, axis=0)
        | ~np.roll(valid_area, -10, axis=0)
        | ~np.roll(valid_area, 10, axis=1)
        | ~np.roll(valid_area, -10, axis=1)
    )
    depth[boundary_mask] = np.nan

    ortho_depth, ortho_color = to_orthographic_projection(
        depth, left_image, camera_height
    )
    world_coords, colors = depth_to_world(
        ortho_depth, ortho_color, K, R, T, pixel_size
    )

    world_coords, colors = filter_points_by_depth(world_coords, colors, depth_cost, census_cost)

    world_coords, colors = grid_sampling(world_coords, colors, 0.05)

    world_coords, colors = filter_points_by_support(
        world_coords,
        colors,
        img_id,
        camera_data,
        K,
        config.IMAGE_DIR,
        support_threshold=2,
    )

    return world_coords, colors, img_id


def filter_points_by_depth(world_coords, colors, depth_cost, census_cost):
    before_n = world_coords.shape[0]
    keep_mask = depth_cost < config.DEPTH_ERROR_THRESHOLD
    kept_world = world_coords[keep_mask]
    kept_colors = colors[keep_mask]
    after_n = kept_world.shape[0]

    dropped_indices = np.where(~keep_mask)[0]
    good_disp_mask = census_cost[dropped_indices] < config.DISP_COST_THRESHOLD
    revived = world_coords[dropped_indices][good_disp_mask]
    revived_cols = colors[dropped_indices][good_disp_mask]
    revived_n = revived.shape[0]

    final_n = after_n + revived_n

    logging.info(
        f"Depth filtering: {before_n} -> {after_n} after depth filter, "
        f"revived {revived_n}, total {final_n} points"
    )

    merged_world = np.vstack([kept_world, revived])
    merged_colors = np.vstack([kept_colors, revived_cols])
    return merged_world, merged_colors

def compute_color_difference(image1, image2):
    """
    image1: ドローン画像 (BGR, 0-255)
    image2: 再投影画像 (RGB, 0-255) → BGR に変換済みと仮定
    背景（黒色）を除外した後、2値化した色差画像を返す
    """
    mask = np.any(image2 != 0, axis=-1)
    diff = cv2.absdiff(image1, image2)
    diff[~mask] = 0
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_binary = cv2.threshold(diff_gray, 120, 255, cv2.THRESH_BINARY)
    return diff_binary


def write_ply(filename, vertices, colors):
    """頂点データと色データをPLYファイルに書き込む"""
    assert vertices.shape[0] == colors.shape[0], "頂点数と色データの数が一致しません。"
    header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        data = np.hstack((vertices, (colors * 255).astype(np.uint8)))
        np.savetxt(f, data, fmt="%f %f %f %d %d %d")


def compute_support_for_view(
    points, colors, position, quaternion, K, neighbor_image, color_threshold=50
):
    """
    各点を指定のカメラパラメータ（position, quaternion）から再投影し、
    隣接画像（neighbor_image）上での色差がしきい値以下ならば1を返す配列を返す。
    points: (N,3) 、colors: (N,3) [RGB, 0-1]
    """
    N = points.shape[0]
    support = np.zeros(N, dtype=np.int32)
    T = np.array(position).reshape(3, 1)
    R = quaternion_to_rotation_matrix(*quaternion)
    p_cam = (R.T @ (points.T - T)).T  
    if np.median(p_cam[:, 2]) < 0:
        p_cam = -p_cam
    valid_mask = p_cam[:, 2] > 0
    if not np.any(valid_mask):
        return support
    p_valid = p_cam[valid_mask]
    u = (K[0, 0] * (p_valid[:, 0] / p_valid[:, 2]) + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * (p_valid[:, 1] / p_valid[:, 2]) + K[1, 2]).astype(np.int32)
    height, width = neighbor_image.shape[:2]
    v = height - 1 - v

    valid_indices = np.where(valid_mask)[0]
    for idx_local, idx in enumerate(valid_indices):
        if (
            u[idx_local] < 0
            or u[idx_local] >= width
            or v[idx_local] < 0
            or v[idx_local] >= height
        ):
            continue
        neighbor_color = neighbor_image[v[idx_local], u[idx_local], :].astype(
            np.float32
        )
        point_color = colors[idx] * 255.0
        diff = np.linalg.norm(point_color - neighbor_color)
        if diff < color_threshold:
            support[idx] = 1
    return support


def filter_points_by_support(
    points,
    colors,
    current_index,
    camera_data,
    K,
    image_dir,
    support_threshold=2,
):
    n = len(camera_data)
    neighbor_indices = []
    if current_index - 2 >= 0:
        neighbor_indices.append(camera_data[current_index - 2])
    if current_index - 1 >= 0:
        neighbor_indices.append(camera_data[current_index - 1])
    if current_index + 1 < n:
        neighbor_indices.append(camera_data[current_index + 1])
    if current_index + 2 < n:
        neighbor_indices.append(camera_data[current_index + 2])

    if len(neighbor_indices) < support_threshold:
        support_threshold = len(neighbor_indices)

    support_counts = np.zeros(points.shape[0], dtype=np.int32)

    for neighbor_cam in neighbor_indices:
        neighbor_position = neighbor_cam[1]
        neighbor_quaternion = neighbor_cam[2]
        neighbor_image_path = os.path.join(image_dir, neighbor_cam[0])
        print(neighbor_image_path)
        neighbor_image = cv2.imread(neighbor_image_path)
        if neighbor_image is None:
            logging.warning(f"Neighbor image not found: {neighbor_image_path}")
            continue
        neighbor_image = cv2.cvtColor(neighbor_image, cv2.COLOR_BGR2RGB)
        support_view = compute_support_for_view(
            points, colors, neighbor_position, neighbor_quaternion, K, neighbor_image
        )
        support_counts += support_view

    keep_indices = support_counts >= support_threshold
    logging.info(
        f"Support filtering: {points.shape[0]} -> {np.sum(keep_indices)} points (threshold: {support_threshold}, neighbor count: {len(neighbor_indices)})"
    )
    return (
        points[keep_indices],
        colors[keep_indices],
    )


if __name__ == "__main__":
    args = parse_arguments()

    logging.info(f"Image directory is {config.IMAGE_DIR}")
    logging.info(f"Show viewer flag is {args.show_viewer}")
    logging.info(f"Video capture flag is {args.record_video}")

    start_time = time.time()

    B = config.B
    focal_length = config.focal_length
    camera_height = config.camera_height
    K = config.K
    pixel_size = config.pixel_size
    window_size = config.window_size
    min_disp = config.min_disp
    num_disp = config.num_disp

    drone_image_list = config.DRONE_IMAGE_LOG
    camera_data = []
    with open(drone_image_list, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            file_name = parts[0]
            position = tuple(map(float, parts[1:4]))
            quaternion = tuple(map(float, parts[4:8]))
            camera_data.append((file_name, position, quaternion))

    logging.info("Starting 3D map creation")
    map_start = time.time()
    image_pairs_data = [
        (
            idx,
            np.array(camera_data[idx][1], dtype=np.float32),
            os.path.join(config.IMAGE_DIR, f"left_{str(idx).zfill(6)}.png"),
            os.path.join(config.IMAGE_DIR, f"right_{str(idx).zfill(6)}.png"),
            B,
            focal_length,
            K,
            quaternion_to_rotation_matrix(*camera_data[idx][2]),
            camera_height,
            pixel_size,
            window_size,
            min_disp,
            num_disp,
        )
        for idx in range(len(camera_data))
    ]

    merged_points_list = []
    merged_colors_list = []
    last_pair = image_pairs_data[-1][0]
    with ProcessPoolExecutor(max_workers=6) as exe:
        futures = exe.map(process_image_pair, image_pairs_data)
        for result in futures:
            if result is not None:
                points, colors, current_idx = result
                merged_points_list.append(points)
                merged_colors_list.append(colors)
                logging.info(f"Processed {current_idx} out of {last_pair} image pairs.")
            else:
                logging.warning(f"Result is None for image pair {current_idx}.")

    if merged_points_list:
        all_points = np.vstack(merged_points_list)
        all_colors = np.vstack(merged_colors_list)

        logging.info(
            f"3D map creation completed. Total points before weighted integration: {len(all_points)}"
        )

        all_points[:, 2] = -all_points[:, 2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
        logging.info(f"After outlier removal: {len(pcd.points)} points.")
        write_ply(
            config.POINT_CLOUD_FILE_PATH, np.asarray(pcd.points), np.asarray(pcd.colors)
        )

        o3d.visualization.draw_geometries([pcd])
    else:
        logging.warning("No points to merge. The point cloud might be empty.")
