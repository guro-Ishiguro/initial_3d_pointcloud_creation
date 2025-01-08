import os
import shutil
import sys
import cv2
import numpy as np
import open3d as o3d
import logging
import matching
import time
import config
import argparse
from utils.viewer import ViewerRecorder
import concurrent.futures
import multiprocessing

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 録画の設定
video_writer = None
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
    # 回転行列を計算
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
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
    """中心投影から正射投影への変換を適用し、カラー画像を正射投影にマッピング"""
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

    ortho_color_image = np.zeros_like(color_image)
    valid_mask = (depth > 0) & (depth < 23)
    ortho_color_image[new_y[valid_mask], new_x[valid_mask]] = color_image[
        row_indices[valid_mask], col_indices[valid_mask]
    ]

    ortho_depth = np.full_like(depth, np.inf)
    np.minimum.at(ortho_depth, (new_y, new_x), depth)
    ortho_depth[ortho_depth == np.inf] = 0

    ortho_color_image[ortho_depth == 0] = [0, 0, 0]

    return ortho_depth, ortho_color_image


def depth_to_world(depth_map, color_image, K, R, T, pixel_size):
    """深度マップをワールド座標に変換し、テクスチャを適用する"""
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    x_coords = (i - width // 2) * pixel_size
    y_coords = -(j - height // 2) * pixel_size
    z_coords = camera_height - depth_map
    local_coords = np.stack((x_coords, y_coords, z_coords), axis=-1).reshape(-1, 3)
    world_coords = (R @ local_coords.T).T + T
    valid_mask = (depth_map > 0).reshape(-1)
    colors = color_image.reshape(-1, 3)[valid_mask] / 255.0
    return world_coords[valid_mask], colors


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


def grid_sampling(point_cloud, colors, grid_size):
    """三次元点群のグリッドサンプリングを行う"""
    rounded_coords = np.floor(point_cloud / grid_size).astype(int)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    return point_cloud[unique_indices], colors[unique_indices]


def process_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
    write_ply(
        config.POINT_CLOUD_FILE_PATH, np.asarray(pcd.points), np.asarray(pcd.colors)
    )
    return pcd


def process_image_pair(image_data):
    """画像ペアを処理してワールド座標を生成する"""
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
    ) = image_data

    left_image = cv2.imread(left_image_path)
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image = cv2.imread(right_image_path)

    # パス確認
    if not os.path.exists(left_image_path):
        logging.error(f"Left image path does not exist: {left_image_path}")
    if not os.path.exists(right_image_path):
        logging.error(f"Right image path does not exist: {right_image_path}")

    # 画像読み込み確認
    if left_image is None or right_image is None:
        logging.error(
            f"Failed to load images for ID {img_id}. Paths: {left_image_path}, {right_image_path}"
        )
        return None

    left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    left_image = cv2.flip(left_image, 0)
    left_image_gray = cv2.flip(left_image_gray, 0)
    right_image_gray = cv2.flip(right_image_gray, 0)

    disparity = create_disparity_image(
        left_image_gray,
        right_image_gray,
        img_id,
        window_size=window_size,
        min_disp=min_disp,
        num_disp=num_disp,
    )

    depth = B * focal_length / (disparity + 1e-6)
    depth, ortho_color_image = to_orthographic_projection(
        depth, left_image, camera_height
    )
    depth[(depth < 0) | (depth > camera_height + 2)] = 0
    world_coords, colors = depth_to_world(depth, ortho_color_image, K, R, T, pixel_size)
    world_coords[:, 1] = -world_coords[:, 1]
    world_coords, colors = grid_sampling(world_coords, colors, 0.05)

    filename = os.path.join(config.POINT_CLOUD_DIR, f"point_cloud_{img_id}.ply")
    write_ply(filename, world_coords, colors)
    logging.info(f"Point cloud saved to {filename}")
    return world_coords, colors


def create_mesh_from_point_cloud(pcd):
    """点群からメッシュを生成"""
    try:
        # 法線を計算
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=20)
        )
        logging.info("Normals estimated for the point cloud.")

        # ポアソン表面再構成でメッシュを作成
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=12
        )
        logging.info(f"Mesh created with {len(mesh.triangles)} triangles.")

        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        logging.info("Bounding box cleaned the mesh.")

        return mesh
    except Exception as e:
        logging.error(f"Error creating mesh: {e}")
        return None


def save_mesh(mesh, output_path):
    """生成されたメッシュをファイルに保存"""
    try:
        o3d.io.write_triangle_mesh(output_path, mesh)
        logging.info(f"Mesh saved to {output_path}.")
    except Exception as e:
        logging.error(f"Error saving mesh: {e}")


if __name__ == "__main__":
    args = parse_arguments()

    logging.info(f"Image directory is {config.DRONE_IMAGE_DIR}")
    logging.info(f"Show view flag is {args.show_viewer}")
    logging.info(f"Video capture flag is {args.record_video}")

    max_workers = multiprocessing.cpu_count()

    start_time = time.time()

    viewer_recorder = None
    if args.show_viewer:
        viewer_recorder = ViewerRecorder(
            width=video_width,
            height=video_height,
            video_filename=video_filename,
            fps=video_fps,
            record_video=args.record_video,
        )

    # パラメータの読み込み
    B, height, focal_length, camera_height, K, pixel_size = (
        config.B,
        config.height,
        config.focal_length,
        config.camera_height,
        config.K,
        config.pixel_size,
    )
    window_size, min_disp, num_disp = (
        config.window_size,
        config.min_disp,
        config.num_disp,
    )

    drone_image_list = matching.read_file_list(config.DRONE_IMAGE_LOG)
    orb_slam_pose_list = matching.read_file_list(config.ORB_SLAM_LOG)

    # マッチング開始
    logging.info("Starting timestamp matching...")
    match_start = time.time()
    matches = matching.associate(drone_image_list, orb_slam_pose_list, 0.0, 0.02)

    # 暫定的な処理
    matches = matches[::2]

    logging.info(
        f"Timestamp matching completed in {time.time() - match_start:.2f} seconds"
    )
    if len(matches) < 2:
        logging.error("Couldn't find matching timestamp pairs.")
        sys.exit(1)
    logging.info(f"{len(matches)} points matched.")

    # 3Dマップ生成開始
    logging.info(f"Starting 3D map creation...")
    map_start = time.time()
    image_pairs_data = [
        (
            int(match[1][0]),
            np.array(
                [float(match[2][0]), float(match[2][1]), float(match[2][2])],
                dtype=np.float32,
            ),
            os.path.join(
                config.DRONE_IMAGE_DIR, f"left_{str(match[1][0]).zfill(6)}.png"
            ),
            os.path.join(
                config.DRONE_IMAGE_DIR, f"right_{str(match[1][0]).zfill(6)}.png"
            ),
            B,
            focal_length,
            K,
            quaternion_to_rotation_matrix(
                float(match[2][3]),
                float(match[2][4]),
                float(match[2][5]),
                float(match[2][6]),
            ),
            camera_height,
            pixel_size,
        )
        for match in matches
    ]

    total_tasks = len(image_pairs_data)
    completed_tasks = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_image_pair, data) for data in image_pairs_data
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    world_coords, colors = result
                    completed_tasks += 1
                    progress = (completed_tasks / total_tasks) * 100
                    logging.info(
                        f"Progress: {completed_tasks}/{total_tasks} tasks completed ({progress:.2f}%)"
                    )
                else:
                    logging.warning("Result is None.")
            except Exception as e:
                logging.error(f"Error occurred while processing image pair: {e}")
    logging.info("All point clouds have been saved individually.")

    point_cloud_files = [
        os.path.join(config.POINT_CLOUD_DIR, f)
        for f in os.listdir(config.POINT_CLOUD_DIR)
        if f.endswith(".ply") and f.startswith("point")
    ]
    all_points = []
    all_colors = []
    for pc_file in point_cloud_files:
        pcd = o3d.io.read_point_cloud(pc_file)
        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))

    if all_points:
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        logging.info(
            f"3D map creation completed in {time.time() - map_start:.2f} seconds"
        )
        logging.info(
            f"Number of points in the cumulative point cloud: {len(all_points)}"
        )
        # PointCloudフィルタリング
        logging.info("Starting pointcloud filtering...")
        filter_start = time.time()
        pcd = process_point_cloud(all_points, all_colors)
        total_time = time.time() - start_time
        logging.info(
            f"Pointcloud filtering completed in {time.time() - filter_start:.2f} seconds"
        )
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        logging.info(
            f"Final number of points in the point cloud: {len(np.asarray(pcd.points))}"
        )
        o3d.visualization.draw_geometries([pcd])

        for pc_file in point_cloud_files:
            try:
                os.remove(pc_file)
                logging.info(f"Deleted file: {pc_file}")
            except Exception as e:
                logging.error(f"Error deleting file {pc_file}: {e}")

        # 点群からメッシュを生成する
        mesh = create_mesh_from_point_cloud(pcd)

        # 生成したメッシュを保存する
        save_mesh(mesh, config.MESH_FILE_PATH)

        # メッシュの表示
        o3d.visualization.draw_geometries([mesh], window_name="Generated Mesh")
    else:
        logging.warning("No points to merge. Point cloud files might be empty.")
