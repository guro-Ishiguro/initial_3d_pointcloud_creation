import os
import shutil
import cv2
import numpy as np
import open3d as o3d
import logging
import time
import config
import argparse
from numba import njit, prange

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
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
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


def depth_to_world(depth_map, color_image, confidence_image, K, R, T, pixel_size):
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
    # ここでは、深度が有効な領域を (depth_map > 10) としている
    valid_mask = (depth_map > 10).reshape(-1)
    colors = color_image.reshape(-1, 3)[valid_mask] / 255.0
    conf = confidence_image.reshape(-1)[valid_mask]
    return world_coords[valid_mask], colors, conf


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


# SAD ベースでコスト画像を計算（各ピクセルにおける左右ブロックの差分和）
@njit
def compute_disparity_cost(left, right, disparity, block_size):
    half = block_size // 2
    rows, cols = left.shape
    cost_image = np.zeros((rows, cols), dtype=np.float32)
    for y in range(half, rows - half):
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


@njit
def compute_depth_error_cost(disparity, block_size, B, c):
    """
    視差画像から深度誤差を考慮したコスト画像を生成する関数
    """
    half = block_size // 2
    rows, cols = disparity.shape
    cost_image = np.zeros((rows, cols), dtype=np.float32)
    for y in range(half, rows - half):
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


def grid_sampling(point_cloud, colors, confidences, grid_size):
    """
    各画像ペア内でのグリッドサンプリング
    """
    rounded_coords = np.floor(point_cloud / grid_size).astype(int)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    return (
        point_cloud[unique_indices],
        colors[unique_indices],
        confidences[unique_indices],
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

    disparity = create_disparity_image(
        left_image_gray, right_image_gray, img_id, window_size, min_disp, num_disp
    )
    if disparity is None:
        logging.error(f"Disparity computation failed for ID {img_id}")
        return None

    # コスト画像の計算（SAD 値）
    disparity_cost = compute_disparity_cost(
        left_image_gray, right_image_gray, disparity, window_size
    )

    # 深度誤差ベースのコスト画像を計算
    depth_cost = compute_depth_error_cost(disparity, window_size, B, focal_length)

    confidence = compute_confidence(disparity_cost, depth_cost)

    depth = B * focal_length / (disparity + 1e-6)
    depth[(depth < 10) | (depth > 40)] = 0

    # 境界部分の除去
    valid_area = depth > 0
    boundary_mask = valid_area & (
        ~np.roll(valid_area, 10, axis=0)
        | ~np.roll(valid_area, -10, axis=0)
        | ~np.roll(valid_area, 10, axis=1)
        | ~np.roll(valid_area, -10, axis=1)
    )
    depth[boundary_mask] = 0

    ortho_depth, ortho_color, ortho_conf = to_orthographic_projection(
        depth, left_image, confidence, camera_height
    )
    world_coords, colors, conf = depth_to_world(
        ortho_depth, ortho_color, ortho_conf, K, R, T, pixel_size
    )

    world_coords, colors, conf = grid_sampling(world_coords, colors, conf, 0.1)

    return world_coords, colors, conf


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


def filter_point_cloud_by_color_diff_image(pcd, position, quaternion, K, color_diff):
    """
    指定カメラパラメータから点群を再投影し、
    色差画像 (バイナリ画像：255が色差大) に該当する点を除外して新たな点群を返す関数
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    height, width = color_diff.shape[:2]

    T = np.array(position).reshape(3, 1)
    R = quaternion_to_rotation_matrix(*quaternion)
    p_cam = (R.T @ (points.T - T)).T

    if np.median(p_cam[:, 2]) < 0:
        p_cam = -p_cam

    valid_mask = p_cam[:, 2] > 0
    keep_mask = np.ones(points.shape[0], dtype=bool)
    valid_indices = np.where(valid_mask)[0]
    z_valid = p_cam[valid_mask, 2]
    u = (K[0, 0] * (p_cam[valid_mask, 0] / z_valid) + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * (p_cam[valid_mask, 1] / z_valid) + K[1, 2]).astype(np.int32)
    v = height - 1 - v

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u_in = u[in_bounds]
    v_in = v[in_bounds]
    valid_indices_in = valid_indices[in_bounds]

    diff_condition = color_diff[v_in, u_in] == 255
    keep_mask[valid_indices_in[diff_condition]] = False

    new_points = points[keep_mask]
    new_colors = colors[keep_mask]
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    logging.info(f"Filtered point cloud: {len(points)} -> {len(new_points)} points.")
    return new_pcd


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
    points, colors, position, quaternion, K, neighbor_image, color_threshold=30
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
    # カメラ座標系に変換
    p_cam = (R.T @ (points.T - T)).T  # shape (N,3)
    if np.median(p_cam[:, 2]) < 0:
        p_cam = -p_cam
    valid_mask = p_cam[:, 2] > 0
    if not np.any(valid_mask):
        return support
    p_valid = p_cam[valid_mask]
    u = (K[0, 0] * (p_valid[:, 0] / p_valid[:, 2]) + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * (p_valid[:, 1] / p_valid[:, 2]) + K[1, 2]).astype(np.int32)
    height, width = neighbor_image.shape[:2]
    # 画像座標系は上下反転
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
        # 隣接画像（RGBに変換済み）の該当画素の色
        neighbor_color = neighbor_image[v[idx_local], u[idx_local], :].astype(
            np.float32
        )
        # 点の色（0-1の範囲）を255倍して比較
        point_color = colors[idx] * 255.0
        diff = np.linalg.norm(point_color - neighbor_color)
        if diff < color_threshold:
            support[idx] = 1
    return support


def filter_points_by_support(
    points,
    colors,
    confidences,
    current_index,
    camera_data,
    K,
    image_dir,
    support_threshold=3,
):
    """
    現在の画像ペアから得られた各点について、隣接ビューからの支持数を計算し、
    支持数がしきい値以上の点のみを残す。
    - camera_data: ドローン画像ログで、各要素は (file_name, position, quaternion)
    - current_index: 現在の画像ペアのインデックス
    - image_dir: 画像ディレクトリパス
    ここでは、前後2枚ずつ、計4ビューで再投影支持数を計算する。
    """
    n = len(camera_data)
    neighbor_indices = []
    # 前後2枚ずつ取得（存在する場合）
    if current_index - 2 >= 0:
        neighbor_indices.append(camera_data[current_index - 2])
    if current_index - 1 >= 0:
        neighbor_indices.append(camera_data[current_index - 1])
    if current_index + 1 < n:
        neighbor_indices.append(camera_data[current_index + 1])
    if current_index + 2 < n:
        neighbor_indices.append(camera_data[current_index + 2])

    # もし利用可能な隣接ビューが support_threshold 未満なら、thresholdを利用可能枚数に調整
    if len(neighbor_indices) < support_threshold:
        support_threshold = len(neighbor_indices)

    # 各点の支持数を初期化
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
        # 色比較のためRGBに変換
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
        confidences[keep_indices],
        support_counts[keep_indices],
    )


def median_integration(points, colors, voxel_size):
    """
    複数の点群を、各ボクセル内の点位置と色の中央値で統合する。
    """
    # 各点のボクセルインデックスを計算
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    # ユニークなボクセルごとにグループ化するための逆インデックスを取得
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    num_voxels = np.max(inverse_indices) + 1

    # グループごとにソートするためのインデックスを取得
    sorted_idx = np.argsort(inverse_indices)
    sorted_groups = inverse_indices[sorted_idx]

    # グループの開始位置と各グループの要素数を計算
    group_start = np.empty(num_voxels, dtype=np.int64)
    group_length = np.empty(num_voxels, dtype=np.int64)
    current_group = sorted_groups[0]
    group_start[0] = 0
    count = 1
    group_idx = 0
    for i in range(1, sorted_groups.shape[0]):
        if sorted_groups[i] == current_group:
            count += 1
        else:
            group_length[group_idx] = count
            group_idx += 1
            current_group = sorted_groups[i]
            group_start[group_idx] = i
            count = 1
    group_length[group_idx] = count

    # NumPy配列をNumbaで高速に中央値計算
    integrated_points, integrated_colors = _median_integration_numba(
        points, colors, sorted_idx, group_start, group_length, num_voxels
    )
    return integrated_points, integrated_colors


@njit(parallel=True)
def _median_integration_numba(
    points, colors, sorted_idx, group_start, group_length, num_voxels
):
    n_dims = points.shape[1]
    integrated_points = np.empty((num_voxels, n_dims), dtype=points.dtype)
    integrated_colors = np.empty((num_voxels, colors.shape[1]), dtype=colors.dtype)

    for i in prange(num_voxels):
        start = group_start[i]
        length = group_length[i]
        # 一時的な配列に該当グループの点と色をコピー
        group_points = np.empty((length, n_dims), dtype=points.dtype)
        group_colors = np.empty((length, colors.shape[1]), dtype=colors.dtype)
        for j in range(length):
            group_points[j, :] = points[sorted_idx[start + j]]
            group_colors[j, :] = colors[sorted_idx[start + j]]
        # 各次元ごとに中央値を計算
        for d in range(n_dims):
            sorted_vals = np.sort(group_points[:, d])
            if length % 2 == 1:
                integrated_points[i, d] = sorted_vals[length // 2]
            else:
                integrated_points[i, d] = (
                    sorted_vals[length // 2 - 1] + sorted_vals[length // 2]
                ) / 2.0
        for d in range(colors.shape[1]):
            sorted_vals = np.sort(group_colors[:, d])
            if length % 2 == 1:
                integrated_colors[i, d] = sorted_vals[length // 2]
            else:
                integrated_colors[i, d] = (
                    sorted_vals[length // 2 - 1] + sorted_vals[length // 2]
                ) / 2.0
    return integrated_points, integrated_colors


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
    merged_confidences_list = []
    last_pair = image_pairs_data[-1][0]
    for data in image_pairs_data:
        current_idx = data[0]
        result = process_image_pair(data)
        if result is not None:
            points, colors, conf = result
            points, colors, conf, support = filter_points_by_support(
                points,
                colors,
                conf,
                current_idx,
                camera_data,
                K,
                config.IMAGE_DIR,
                support_threshold=4,
            )
            merged_points_list.append(points)
            merged_colors_list.append(colors)
            merged_confidences_list.append(conf)
            logging.info(f"Processed {current_idx} out of {last_pair} image pairs.")
        else:
            logging.warning(f"Result is None for image pair {current_idx}.")

    if merged_points_list:
        all_points = np.vstack(merged_points_list)
        all_colors = np.vstack(merged_colors_list)
        all_confidences = np.concatenate(merged_confidences_list)

        logging.info(
            f"3D map creation completed. Total points before weighted integration: {len(all_points)}"
        )

        integrated_points, integrated_colors = median_integration(
            all_points, all_colors, voxel_size=0.02
        )

        integrated_points[:, 2] = -integrated_points[:, 2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(integrated_points)
        pcd.colors = o3d.utility.Vector3dVector(integrated_colors)

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
        logging.info(f"After outlier removal: {len(pcd.points)} points.")
        write_ply(
            config.OLD_POINT_CLOUD_FILE_PATH,
            np.asarray(pcd.points),
            np.asarray(pcd.colors),
        )

        o3d.visualization.draw_geometries([pcd])
    else:
        logging.warning("No points to merge. The point cloud might be empty.")
