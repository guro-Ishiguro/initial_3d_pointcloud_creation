import os
import shutil
import cv2
import numpy as np
import open3d as o3d
import logging
import time
import config
import argparse
import concurrent.futures

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 録画の設定
video_filename = os.path.join(config.VIDEO_DIR, "3d_map_visualization.avi")
video_fps = 10
video_width, video_height = 640, 480

def parse_arguments():
    """コマンド引数の値を受け取る"""
    parser = argparse.ArgumentParser(description="3D Point Cloud Creater")
    parser.add_argument("--show-viewer", action="store_true", help="Show viewer during point cloud generation.")
    parser.add_argument("--record-video", action="store_true", help="Record viewer output to video.")
    return parser.parse_args()

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数を回転行列に変換"""
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ])
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
    valid_mask = (depth_map > 10).reshape(-1)
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
        P2=16 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disparity

def grid_sampling(point_cloud, colors, grid_size):
    """各画像ペア内でのグリッドサンプリング（単純にユニークなグリッドセルごとに1点を残す）"""
    rounded_coords = np.floor(point_cloud / grid_size).astype(int)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    return point_cloud[unique_indices], colors[unique_indices]

def process_point_cloud(points, colors):
    """統合点群に対してボクセルダウンサンプリングとアウトライヤ除去を実施"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # ボクセルダウンサンプリング
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    # 統計的外れ値除去
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
    各画像ペアごとに事前にボクセル化を行ってから結果を返す。
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

    disparity = create_disparity_image(
        left_image_gray, right_image_gray, img_id, window_size, min_disp, num_disp
    )
    if disparity is None:
        logging.error(f"Disparity computation failed for ID {img_id}")
        return None

    depth = B * focal_length / (disparity + 1e-6)
    depth[(depth < 10) | (depth > 40)] = 0

    # 境界部分の除去
    valid_area = (depth > 0)
    boundary_mask = valid_area & (
        ~np.roll(valid_area, 10, axis=0)
        | ~np.roll(valid_area, -10, axis=0)
        | ~np.roll(valid_area, 10, axis=1)
        | ~np.roll(valid_area, -10, axis=1)
    )
    depth[boundary_mask] = 0

    depth, ortho_color_image = to_orthographic_projection(depth, left_image, camera_height)
    world_coords, colors = depth_to_world(depth, ortho_color_image, K, R, T, pixel_size)
    # 各画像ペア内でのグリッドサンプリング
    world_coords, colors = grid_sampling(world_coords, colors, 0.1)
    
    return world_coords, colors

def reproject_points(points, K, R, T):
    """
    3D世界座標の点群をカメラパラメータを用いて画像平面へ射影する。
    カメラ座標系への変換は X_cam = R^T * (X_world - T) とする。
    """
    # (points - T) は各点からカメラ中心へのベクトル
    points_cam = (points - T) @ R.T  # shape: (N,3)
    projected = (K @ points_cam.T).T  # shape: (N,3)
    projected_xy = projected[:, :2] / projected[:, 2:3]  # 正規化
    
    return projected_xy, points_cam[:, 2]  # 返り値：画像平面上の座標と深度

def filter_inconsistent_points(points, colors, test_camera_params, diff_threshold=0.1):
    """
    テスト用カメラパラメータに基づき、各テスト画像に再投影して色差が大きい点をフィルタリングする。
    - points: 点群 (N,3)
    - colors: 各点のRGB (N,3) (0～1の範囲)
    - test_camera_params: 辞書のリスト。各辞書は 'K', 'R', 'T', 'image'（0～1のfloat画像）を含む。
    - diff_threshold: 色差の閾値（0～1のスケール）
    """
    N = points.shape[0]
    # 各点の最大色差を初期化（各テストカメラでの色差の最大値）
    max_diff = np.zeros(N)
    for cam in test_camera_params:
        K_cam = cam['K']
        R_cam = cam['R']
        T_cam = cam['T']
        test_img = cam['image']  # float32, 0～1, shape: (H, W, 3)
        h, w, _ = test_img.shape
        proj_xy, depth = reproject_points(points, K_cam, R_cam, T_cam)
        # 画像内に入っていて、かつ深度が正の点を対象とする
        valid_mask = (depth > 0) & \
                     (proj_xy[:, 0] >= 0) & (proj_xy[:, 0] < w) & \
                     (proj_xy[:, 1] >= 0) & (proj_xy[:, 1] < h)
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            continue
        proj_valid = proj_xy[valid_indices]
        proj_valid_int = np.round(proj_valid).astype(int)
        # インデックスが画像サイズの範囲を超えないようにクリッピング
        proj_valid_int[:, 0] = np.clip(proj_valid_int[:, 0], 0, w - 1)
        proj_valid_int[:, 1] = np.clip(proj_valid_int[:, 1], 0, h - 1)
        # 画像インデックスは (row, col) で取得
        test_colors = test_img[proj_valid_int[:, 1], proj_valid_int[:, 0], :]
        # 各点の色との差（ユークリッド距離）を計算
        diff = np.linalg.norm(colors[valid_indices] - test_colors, axis=1)
        max_diff[valid_indices] = np.maximum(max_diff[valid_indices], diff)
    # 最大色差が閾値以下の点のみ残す
    consistent_mask = max_diff <= diff_threshold
    return points[consistent_mask], colors[consistent_mask]


if __name__ == "__main__":
    args = parse_arguments()

    logging.info(f"Image directory is {config.IMAGE_DIR}")
    logging.info(f"Show viewer flag is {args.show_viewer}")
    logging.info(f"Video capture flag is {args.record_video}")

    start_time = time.time()

    # パラメータの読み込み
    B = config.B
    focal_length = config.focal_length
    camera_height = config.camera_height
    K = config.K
    pixel_size = config.pixel_size
    window_size = config.window_size
    min_disp = config.min_disp
    num_disp = config.num_disp

    # ドローン画像ログの読み込み
    drone_image_list = config.DRONE_IMAGE_LOG
    camera_data = []
    with open(drone_image_list, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            file_name = parts[0]
            position = tuple(map(float, parts[1:4]))
            quaternion = tuple(map(float, parts[4:8]))
            camera_data.append((file_name, position, quaternion))
    
    test_camera_indices = list(range(0, len(camera_data), 15))
    test_camera_params = []
    for idx in test_camera_indices:
        T_test = np.array(camera_data[idx][1], dtype=np.float32)
        R_test = quaternion_to_rotation_matrix(*camera_data[idx][2])
        test_image_path = os.path.join(config.IMAGE_DIR, f"left_{str(idx).zfill(6)}.png")
        test_img = cv2.imread(test_image_path)
        if test_img is None:
            logging.warning(f"Test image not found for index {idx}: {test_image_path}")
            continue
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = test_img.astype(np.float32) / 255.0
        test_camera_params.append({
            'K': K,
            'R': R_test,
            'T': T_test,
            'image': test_img
        })


    logging.info("Starting 3D map creation (training set)...")
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
        for idx in range(150)
    ]

    merged_points_list = []
    merged_colors_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_image_pair, data) for data in image_pairs_data]
        completed_tasks = 0
        total_tasks = len(futures)
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    points, colors = result
                    merged_points_list.append(points)
                    merged_colors_list.append(colors)
                    completed_tasks += 1
                    progress = (completed_tasks / total_tasks) * 100
                    logging.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed ({progress:.2f}%)")
                else:
                    logging.warning("Result is None.")
            except Exception as e:
                logging.error(f"Error occurred while processing image pair: {e}")

    if merged_points_list:
        # --- 統合 ---
        all_points = np.vstack(merged_points_list)
        all_colors = np.vstack(merged_colors_list)

        # Z軸反転
        all_points[:, 2] = -all_points[:, 2]

        logging.info(f"3D map creation completed. Total points before filtering: {len(all_points)}")

        # --- 完全重複点の削除 ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # --- ボクセルダウンサンプリング（統一ボクセルグリッドで均一化） ---
        voxel_size = 0.02  # ボクセルサイズを調整
        pcd = pcd.voxel_down_sample(voxel_size)
        logging.info(f"Voxel downsampling applied. New count: {len(pcd.points)}")

        # --- 統計的外れ値除去（ノイズ除去） ---
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
        logging.info(f"Outlier removal applied. Count after outlier removal: {len(pcd.points)}")

        # ---------- テストカメラによる再投影と色不整合点のフィルタリング ----------
        logging.info("Starting reprojection-based color consistency filtering...")
        all_points_np = np.asarray(pcd.points)
        all_colors_np = np.asarray(pcd.colors)
        filtered_points, filtered_colors = filter_inconsistent_points(all_points_np, all_colors_np, test_camera_params, diff_threshold=0.1)
        logging.info(f"After filtering, remaining points: {len(filtered_points)}")
        
        # フィルタリング結果を新たな点群として生成
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        # --- 可視化 ----------
        o3d.visualization.draw_geometries([filtered_pcd])
    else:
        logging.warning("No points to merge. The point cloud might be empty.")
