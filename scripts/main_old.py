import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
import logging
from sklearn.neighbors import NearestNeighbors
import matching
import time
import config
import argparse
from utils.viewer import ViewerRecorder


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


def disparity_image(disparity, img_id):
    """視差画像を表示する"""
    plt.figure(figsize=(10, 8))
    plt.imshow(disparity, cmap="jet")
    plt.colorbar(label="Image")
    plt.title(f"Image {img_id}")
    plt.axis("off")
    plt.show()


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


def convert_right_to_left_hand_coordinates(data):
    """右手座標系から左手座標系へ変換する"""
    data[:, 2] = -data[:, 2]
    return data


def display_neighbors(points, k=4):
    """近傍点のヒストグラムを表示し、外れ値の閾値を示す"""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    threshold = np.mean(mean_distances) + 2 * np.std(mean_distances)
    plt.hist(mean_distances, bins=1000, alpha=0.9)
    plt.axvline(np.mean(mean_distances), color="r", linestyle="dashed", linewidth=2)
    plt.axvline(threshold, color="g", linestyle="dashed", linewidth=2)
    plt.xlabel("Mean Distance to k Nearest Neighbors")
    plt.ylabel("Frequency")
    plt.title("Histogram of Mean Distances")
    plt.xlim(0, 0.1)
    plt.legend()
    plt.savefig("figure.png")


def grid_sampling(point_cloud, grid_size):
    """三次元点群のグリッドサンプリングを行う"""
    rounded_coords = np.floor(point_cloud / grid_size).astype(int)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    return point_cloud[unique_indices]


if __name__ == "__main__":
    args = parse_arguments()

    logging.info(f"Image directory is {config.DRONE_IMAGE_DIR}")
    logging.info(f"Show view flag is {args.show_viewer}")
    logging.info(f"Video capture flag is {args.record_video}")

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
    B, height, focal_length, camera_height, K, R, pixel_size = (
        config.B,
        config.height,
        config.focal_length,
        config.camera_height,
        config.K,
        config.R,
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
    logging.info(
        f"Timestamp matching completed in {time.time() - match_start:.2f} seconds"
    )

    if len(matches) < 2:
        logging.error("Couldn't find matching timestamp pairs.")
        sys.exit(1)

    logging.info(f"{len(matches)} points matched.")

    # 3Dマップ生成開始
    logging.info("Starting 3D map creation...")
    map_start = time.time()
    cumulative_world_coords = None
    for i in range(len(matches)):
        img_id = int(matches[i][1][0])
        dx = float(matches[i][2][0])
        dy = float(matches[i][2][1])
        dz = float(matches[i][2][2])
        T = np.array([dy, dx, dz], dtype=np.float32)
        left_image = cv2.imread(
            os.path.join(config.DRONE_IMAGE_DIR, f"left_{img_id}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        right_image = cv2.imread(
            os.path.join(config.DRONE_IMAGE_DIR, f"right_{img_id}.png"),
            cv2.IMREAD_GRAYSCALE,
        )

        if left_image is None or right_image is None:
            logging.error(f"Failed to load images for ID {img_id}")
            continue

        disparity = create_disparity_image(
            left_image,
            right_image,
            i,
            window_size=window_size,
            min_disp=min_disp,
            num_disp=num_disp,
        )
        if disparity is None:
            continue

        depth = B * focal_length / (disparity + 1e-6)
        depth = to_orthographic_projection(depth, camera_height)
        depth[(depth < 0) | (depth > 23)] = 0
        world_coords = convert_right_to_left_hand_coordinates(
            depth_to_world(depth, K, R, T, pixel_size)
        )
        world_coords = world_coords[depth.reshape(-1) > 0]
        world_coords = grid_sampling(world_coords, 0.1)

        if cumulative_world_coords is None:
            cumulative_world_coords = world_coords
        else:
            cumulative_world_coords = np.vstack((cumulative_world_coords, world_coords))

        if viewer_recorder:
            viewer_recorder.update_viewer(cumulative_world_coords)

    logging.info(f"3D map creation completed in {time.time() - map_start:.2f} seconds")
    logging.info(
        f"Number of points in the cumulative point cloud: {len(cumulative_world_coords)}"
    )

    # PointCloudフィルタリング
    logging.info("Starting pointcloud filtering...")
    filter_start = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cumulative_world_coords)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    write_ply(config.POINT_CLOUD_FILE_PATH, np.asarray(pcd.points))
    logging.info(
        f"Pointcloud filtering completed in {time.time() - filter_start:.2f} seconds"
    )
    total_time = time.time() - start_time
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(
        f"Final number of points in the point cloud: {len(np.asarray(pcd.points))}"
    )
    if viewer_recorder:
        viewer_recorder.update_viewer(np.asarray(pcd.points))
    if viewer_recorder:
        viewer_recorder.close()
    o3d.visualization.draw_geometries([pcd])
