import os
import cv2
import numpy as np
import open3d as o3d
import logging
import config
import numba
from numba import njit, prange

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    R = np.array([
        [1 - 2 * (qy**2 + qz**2),     2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2),     2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return R

def load_point_cloud(file_path):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        logging.info(f"Loaded point cloud from {file_path} with {len(pcd.points)} points.")
        
        transform_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        pcd.transform(transform_matrix)
        
        return pcd
    except Exception as e:
        logging.error(f"Error loading point cloud: {e}")
        return None

def reproject_point_cloud(pcd, position, quaternion, K, image_shape, splat_radius=3):
    """
    点群 pcd を指定カメラ姿勢から再投影し、ポイントスプラッティングと zバッファ法を適用する関数
    """
    R = quaternion_to_rotation_matrix(*quaternion)
    C = np.array(position).reshape(3, 1)
    
    points = np.asarray(pcd.points)      # (N,3)
    colors = np.asarray(pcd.colors)      # (N,3) [0,1] 範囲と仮定

    p_cam = (R.T @ (points.T - C)).T

    # カメラ前方の定義が逆の場合に備えて
    if np.median(p_cam[:, 2]) < 0:
        p_cam = -p_cam

    valid_idx = p_cam[:, 2] > 0
    p_cam = p_cam[valid_idx]
    colors = colors[valid_idx]
    z_all = p_cam[:, 2]  # 正の深度

    # 画像平面への投影
    u = (K[0, 0] * (p_cam[:, 0] / z_all) + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * (p_cam[:, 1] / z_all) + K[1, 2]).astype(np.int32)

    height, width = image_shape[:2]
    v = height - 1 - v  # v 座標を上下反転

    # zバッファと再投影画像の初期化
    reprojected_img = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)

    # colors を uint8 型に変換（Numba 関数で扱いやすいように）
    colors_uint8 = (colors * 255).astype(np.uint8)

    # Numba を使ったスプラッティング処理
    splat_points_numba(u, v, z_all.astype(np.float32), colors_uint8, splat_radius, width, height, z_buffer, reprojected_img)
    
    return reprojected_img

@njit(parallel=True)
def splat_points_numba(u, v, z_all, colors, splat_radius, width, height, z_buffer, reprojected_img):
    for i in prange(u.shape[0]):
        for dy in range(-splat_radius, splat_radius + 1):
            for dx in range(-splat_radius, splat_radius + 1):
                if dx * dx + dy * dy <= splat_radius * splat_radius:
                    uu = u[i] + dx
                    vv = v[i] + dy
                    if 0 <= uu < width and 0 <= vv < height:
                        if z_all[i] < z_buffer[vv, uu]:
                            z_buffer[vv, uu] = z_all[i]
                            reprojected_img[vv, uu, 0] = colors[i, 0]
                            reprojected_img[vv, uu, 1] = colors[i, 1]
                            reprojected_img[vv, uu, 2] = colors[i, 2]

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
    _, diff_binary = cv2.threshold(diff_gray, 100, 255, cv2.THRESH_BINARY)
    return diff_binary

def filter_point_cloud_by_color_diff_image(pcd, position, quaternion, K, color_diff):
    """
    指定カメラパラメータから点群を再投影し、
    色差画像 (バイナリ画像：255が色差大) に該当する点を除外して新たな点群を返す関数
    """
    # 点群の座標と色を取得
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    height, width = color_diff.shape[:2]

    # カメラ座標系へ変換（reproject_point_cloud と同様の処理）
    R = quaternion_to_rotation_matrix(*quaternion)
    C = np.array(position).reshape(3, 1)
    p_cam = (R.T @ (points.T - C)).T

    # カメラ前方の定義が逆の場合に備えて
    if np.median(p_cam[:, 2]) < 0:
        p_cam = -p_cam

    # 正の深度を持つ点のみ対象
    valid_mask = p_cam[:, 2] > 0

    # フィルタリング用のマスク（初期状態は全点保持）
    keep_mask = np.ones(points.shape[0], dtype=bool)

    # 正の深度を持つ点について、画像上のピクセル位置を計算
    valid_indices = np.where(valid_mask)[0]
    u = (K[0, 0] * (p_cam[valid_mask, 0] / p_cam[valid_mask, 2]) + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * (p_cam[valid_mask, 1] / p_cam[valid_mask, 2]) + K[1, 2]).astype(np.int32)
    # reproject_point_cloud と同様に、v 座標を上下反転
    v = height - 1 - v

    # 各有効な点について、画像内で色差が大きい領域に属するかをチェック
    for idx, (px, py) in zip(valid_indices, zip(u, v)):
        if 0 <= px < width and 0 <= py < height:
            if color_diff[py, px] == 255:
                keep_mask[idx] = False

    # フィルタリング後の点群作成
    new_points = points[keep_mask]
    new_colors = colors[keep_mask]
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
    
    logging.info(f"Filtered point cloud: {len(points)} -> {len(new_points)} points.")
    return new_pcd


if __name__ == '__main__':
    pcd = load_point_cloud(config.POINT_CLOUD_FILE_PATH)
    o3d.visualization.draw_geometries([pcd])
    if pcd is None:
        exit(1)

    camera_data = []
    with open(config.DRONE_IMAGE_LOG, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            file_name = parts[0]
            position = (float(parts[1]), float(parts[2]), float(parts[3]))
            quaternion = tuple(map(float, parts[4:8]))
            camera_data.append((file_name, position, quaternion))
    
    if len(camera_data) == 0:
        logging.error("No camera data found.")
        exit(1)

    camera_id = 90
    file_name, position, quaternion = camera_data[camera_id]
    logging.info(f"Using camera data: {file_name}, position: {position}, quaternion: {quaternion}")
    
    file_path = os.path.join(config.IMAGE_DIR, f"left_{str(camera_id).zfill(6)}.png")
    drone_image = cv2.imread(file_path)
    if drone_image is None:
        logging.error(f"Failed to load drone image {file_name}")
        exit(1)
    image_shape = drone_image.shape

    K = config.K

    # 再投影画像の生成
    reprojected_image = reproject_point_cloud(pcd, position, quaternion, K, image_shape, splat_radius=3)
    # OpenCV は BGR 表示なので、RGB→BGR に変換
    reprojected_image = cv2.cvtColor(reprojected_image, cv2.COLOR_RGB2BGR)
    logging.info("Saved reprojected image")
    cv2.imwrite("reprojected_image.png", reprojected_image)

    # 色差2値画像の計算
    color_diff = compute_color_difference(drone_image, reprojected_image)
    cv2.imwrite("color_diff.png", color_diff)

    # 色差画像を用いた点群フィルタリング
    filtered_pcd = filter_point_cloud_by_color_diff_image(pcd, position, quaternion, K, color_diff)
    o3d.visualization.draw_geometries([filtered_pcd])