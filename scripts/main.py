import os, sys
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
            except Exception as e:
                logging.error(f"Error deleting {file_path}: {e}")
    else:
        logging.info(f"The folder {dir_path} does not exist.")


def to_orthographic_projection(depth, color_image, camera_height):
    rows, cols = depth.shape
    mid_x, mid_y = cols // 2, rows // 2
    ri, ci = np.indices((rows, cols))
    valid = np.isfinite(depth)
    shift_x = np.zeros_like(depth, int)
    shift_y = np.zeros_like(depth, int)
    shift_x[valid] = (
        (camera_height - depth[valid]) * (mid_x - ci[valid]) / camera_height
    ).astype(int)
    shift_y[valid] = (
        (camera_height - depth[valid]) * (mid_y - ri[valid]) / camera_height
    ).astype(int)
    nx = ci + shift_x
    ny = ri + shift_y
    mask = valid & (nx >= 0) & (nx < cols) & (ny >= 0) & (ny < rows)
    depths = depth[mask]
    colors = color_image[ri[mask], ci[mask]]
    flat = ny[mask] * cols + nx[mask]
    order = np.lexsort((depths, flat))
    flat_s, depth_s, col_s = flat[order], depths[order], colors[order]
    uniq, idx = np.unique(flat_s, return_index=True)
    uy, ux = uniq // cols, uniq % cols
    ortho_d = np.full_like(depth, np.nan, float)
    ortho_c = np.full_like(color_image, np.nan, float)
    ortho_d[uy, ux] = depth_s[idx]
    ortho_c[uy, ux] = col_s[idx]
    return ortho_d, ortho_c


def depth_to_world(depth_map, color_image, K, R, T, pixel_size):
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    x = (i - w // 2) * pixel_size
    y = -(j - h // 2) * pixel_size
    z = depth_map
    loc = np.stack((x, y, z), -1).reshape(-1, 3)
    world = (R @ loc.T).T + T
    cols = color_image.reshape(-1, 3) / 255.0
    valid = np.isfinite(z.flatten()) & (z.flatten() > 5)
    world[~valid] = np.nan
    cols[~valid] = np.nan
    return world, cols


def create_disparity(image_L, image_R, img_id):
    stereo = cv2.StereoSGBM_create(
        minDisparity=config.min_disp,
        numDisparities=config.num_disp,
        blockSize=config.window_size,
        P1=8 * 3 * config.window_size**2,
        P2=16 * 3 * config.window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disp = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disp


@njit
def census_transform_numba(img, window_size=7):
    half = window_size // 2
    rows, cols = img.shape
    census = np.zeros((rows, cols), np.uint64)
    for y in range(half, rows - half):
        for x in range(half, cols - half):
            center = img[y, x]
            desc = 0
            for i in range(-half, half + 1):
                for j in range(-half, half + 1):
                    if i == 0 and j == 0:
                        continue
                    desc <<= 1
                    if img[y + i, x + j] < center:
                        desc |= 1
            census[y, x] = desc
    return census


@njit
def compute_census_cost(left, right, disparity, c_left, c_right, window_size):
    half = window_size // 2
    rows, cols = left.shape
    cost = np.zeros((rows, cols), np.float32)
    for y in range(half, rows - half):
        for x in range(half, cols - half):
            d = int(round(disparity[y, x]))
            if d >= 0:
                xr = x - d
                if xr - half >= 0 and xr + half < cols:
                    xorv = c_left[y, x] ^ c_right[y, xr]
                    c = 0
                    while xorv:
                        c += xorv & 1
                        xorv >>= 1
                    cost[y, x] = c
    return cost


@njit
def compute_depth_error_cost_jit(disparity, depth, B, focal, block_size):
    half = block_size // 2
    rows, cols = disparity.shape
    cost = np.full((rows, cols), np.nan, np.float32)
    for y in range(half, rows - half):
        for x in range(half, cols - half):
            d = disparity[y, x]
            if not np.isnan(depth[y, x]):
                cost[y, x] = (B * focal) / (d * d + 1e-6)
    return cost


def compute_depth_error_cost(disparity, depth, block_size):
    return compute_depth_error_cost_jit(
        disparity, depth, config.B, config.focal_length, block_size
    )


def grid_sampling(points, colors, grid_size):
    ids = np.floor(points / grid_size).astype(int)
    _, idxs = np.unique(ids, axis=0, return_index=True)
    return points[idxs], colors[idxs]


def process_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
    return pcd


def voxel_downsample_points(points, colors, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down.points), np.asarray(down.colors)


def process_image_pair(data):
    (
        idx,
        T,
        left_path,
        right_path,
        B,
        focal,
        K,
        R,
        cam_h,
        pix_sz,
        w_size,
        min_d,
        num_d,
    ) = data
    li = cv2.imread(left_path)
    ri = cv2.imread(right_path)
    li_rgb = cv2.cvtColor(li, cv2.COLOR_BGR2RGB)
    ri_rgb = cv2.cvtColor(ri, cv2.COLOR_BGR2RGB)
    li_gray = cv2.cvtColor(li, cv2.COLOR_BGR2GRAY)
    ri_gray = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)
    disp = create_disparity(li_gray, ri_gray, idx)
    cen_l = census_transform_numba(li_gray, w_size)
    cen_r = census_transform_numba(ri_gray, w_size)
    cen_cost = compute_census_cost(
        li_gray, ri_gray, disp, cen_l, cen_r, w_size
    ).flatten()
    depth = B * focal / (disp + 1e-6)
    depth[(depth < 0) | (depth > 50) | ((depth > 8.5) & (depth < 9.5))] = np.nan
    d_cost = compute_depth_error_cost(disp, depth, w_size).flatten()
    valid = np.isfinite(depth)
    bmask = valid & (
        ~np.roll(valid, 10, 0)
        | ~np.roll(valid, -10, 0)
        | ~np.roll(valid, 10, 1)
        | ~np.roll(valid, -10, 1)
    )
    depth[bmask] = np.nan
    ortho_d, ortho_c = to_orthographic_projection(depth, li_rgb, cam_h)
    pts, cols = depth_to_world(ortho_d, ortho_c, K, R, T, pix_sz)
    # pts, cols = filter_points_by_depth(wpts, cols, d_cost, cen_cost)
    pts, cols = grid_sampling(pts, cols, 0.1)
    # pts, cols = filter_points_by_support(
    #     pts, cols, idx, camera_data, K, config.IMAGE_DIR, 3
    # )
    logging.info(f"{idx}番目のカメラにおける初期点群が生成されました。")
    return pts, cols, idx, d_cost


def filter_points_by_depth(world_coords, colors, depth_cost, census_cost):
    before = world_coords.shape[0]
    km = depth_cost < config.DEPTH_ERROR_THRESHOLD
    kept = world_coords[km]
    kcols = colors[km]
    dropped = np.where(~km)[0]
    revive_mask = census_cost[dropped] < config.DISP_COST_THRESHOLD
    rev = world_coords[dropped][revive_mask]
    rev_c = colors[dropped][revive_mask]
    merged = np.vstack([kept, rev])
    return merged, np.vstack([kcols, rev_c])


def compute_support_for_view(points, colors, pos, quat, K, nbr_img, c_thresh=30):
    N = points.shape[0]
    support = np.zeros(N, int)
    T = np.array(pos)
    R = quaternion_to_rotation_matrix(*quat)
    p_cam = (R.T @ (points - T).T).T
    if np.median(p_cam[:, 2]) < 0:
        p_cam = -p_cam
    mask = p_cam[:, 2] > 0
    uv_x = (K[0, 0] * (p_cam[mask, 0] / p_cam[mask, 2]) + K[0, 2]).astype(int)
    uv_y = (K[1, 1] * (p_cam[mask, 1] / p_cam[mask, 2]) + K[1, 2]).astype(int)
    uv_y = nbr_img.shape[0] - 1 - uv_y
    idxs = np.where(mask)[0]
    for l, i in enumerate(idxs):
        u, v = uv_x[l], uv_y[l]
        if 0 <= u < nbr_img.shape[1] and 0 <= v < nbr_img.shape[0]:
            diff = np.linalg.norm(colors[i] * 255 - nbr_img[v, u].astype(np.float32))
            if diff < c_thresh:
                support[i] = 1
    return support


def filter_points_by_support(
    points, colors, cur_idx, camera_data, K, image_dir, support_threshold=3
):
    n = len(camera_data)
    neighbors = []
    for off in (-2, -1, 1, 2):
        idx = cur_idx + off
        if 0 <= idx < n:
            neighbors.append(camera_data[idx])
    support_threshold = min(support_threshold, len(neighbors))
    counts = np.zeros(points.shape[0], int)
    for fname, pos, quat in neighbors:
        nbr = cv2.cvtColor(
            cv2.imread(os.path.join(image_dir, fname)), cv2.COLOR_BGR2RGB
        )
        sv = compute_support_for_view(points, colors, pos, quat, K, nbr)
        counts += sv
    mask = counts >= support_threshold
    return points[mask], colors[mask]


def integrate_depth_maps_median(points_list, colors_list, voxel_size=0.1):
    all_pts = np.vstack(points_list)
    all_cols = np.vstack(colors_list)
    valid = np.isfinite(all_pts).all(axis=1)
    pts = all_pts[valid]
    cols = all_cols[valid]
    vids = np.floor(pts / voxel_size).astype(int)
    voxel_dict = {}
    for i, vid in enumerate(map(tuple, vids)):
        voxel_dict.setdefault(vid, []).append(i)
    med_pts, med_cols = [], []
    for idxs in voxel_dict.values():
        voxel_pts = pts[idxs]
        voxel_cols = cols[idxs]
        med_pts.append(np.median(voxel_pts, axis=0))
        med_cols.append(np.median(voxel_cols, axis=0))
    return np.array(med_pts), np.array(med_cols)


def compute_color_difference(image1, image2):
    mask = np.any(image2 != 0, axis=-1)
    diff = cv2.absdiff(image1, image2)
    diff[~mask] = 0
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, binar = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    return binar


def write_ply(filename, vertices, colors):
    assert vertices.shape[0] == colors.shape[0]
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


@njit
def compute_multiview_cost(
    point_3d,
    ref_color,
    camera_K,
    neighbor_R,
    neighbor_T,
    neighbor_image,
    cost_type="SSD",
    color_threshold=10,
):
    """
    単一の3D点に対する複数視点コストを計算する。
    Args:
        point_3d (np.ndarray): 3D点のワールド座標 (x, y, z).
        ref_color (np.ndarray): 参照画像での3D点の色 (R, G, B) / 255.0.
        camera_K (np.ndarray): カメラの内部パラメータ行列 K.
        neighbor_R (np.ndarray): 隣接カメラの回転行列 R.
        neighbor_T (np.ndarray): 隣接カメラの並進ベクトル T.
        neighbor_image (np.ndarray): 隣接カメラの画像 (H, W, 3).
        cost_type (str): 使用する光度コストのタイプ ("SSD", "SAD", "NCC", "CENSUS").
        color_threshold (int): カラー差の閾値 (0-255).

    Returns:
        float: 計算されたコスト (小さいほど良い).
        float: 逆投影誤差 (小さいほど良い).
    """
    # ワールド座標から隣接カメラ座標へ
    p_cam = (neighbor_R.T @ (point_3d - neighbor_T).T).T
    # 負の深度（カメラの後ろ）なら高コスト
    if p_cam[2] <= 0.1:  # 非常に小さい正の値でZ > 0を保証
        return 1e9, 1e9  # 高いコストと誤差

    # カメラ座標から画像座標へ
    uv_hom = camera_K @ p_cam
    u, v = uv_hom[0] / uv_hom[2], uv_hom[1] / uv_hom[2]
    v = neighbor_image.shape[0] - 1 - v

    # 画像の境界チェック
    h, w, _ = neighbor_image.shape
    if not (0 <= u < w and 0 <= v < h):
        return 1e9, 1e9  # 範囲外なら高コスト

    # 逆投影誤差
    reprojection_error = 0.0

    # 画像のピクセル値取得
    v_int, u_int = int(round(v)), int(round(u))
    neighbor_color = neighbor_image[v_int, u_int].astype(np.float32) / 255.0

    # 光度コスト計算
    if cost_type == "SSD":
        photometric_cost = np.sum((ref_color - neighbor_color) ** 2)
    elif cost_type == "SAD":
        photometric_cost = np.sum(np.abs(ref_color - neighbor_color))
    else:
        photometric_cost = np.sum((ref_color - neighbor_color) ** 2)  # デフォルトはSSD

    # カラー閾値による追加フィルタリング
    diff_sq = np.sum(((ref_color - neighbor_color) * 255) ** 2)
    color_diff_norm = np.sqrt(diff_sq)
    if color_diff_norm > color_threshold:
        photometric_cost += 1000
    else:
        photometric_cost += color_diff_norm

    return photometric_cost, reprojection_error


def refine_points_with_multiview_cost(
    cur_idx,
    points,
    colors,
    depth_costs,
    camera_data,
    K,
    image_dir,
    max_search_range=0.5,
    num_steps=20,
    cost_weight_photo=1.0,
    cost_weight_reproj=0.1,
):
    optimized_points = np.copy(points)
    optimized_colors = np.copy(colors)
    num_points = points.shape[0]

    # 各画像のR, T行列を事前に計算・キャッシュ
    camera_poses = {}
    neighbor_images = {}

    # 各点について最適化を試みる
    print("Type:", type(depth_costs))
    print("Shape:", depth_costs.shape)
    print("Min:", np.nanmin(depth_costs))
    print("Max:", np.nanmax(depth_costs))
    print("Mean:", np.nanmean(depth_costs))
    print("Has NaNs:", np.isnan(depth_costs).any())

    # 近隣画像の取得
    n = len(camera_data)
    neighbors_info = []

    for off in (-2, -1, 1, 2):
        idx = cur_idx + off
        if 0 <= idx < n:
            fname, pos, quat = camera_data[idx]
            R = quaternion_to_rotation_matrix(*quat).astype(np.float32)
            T = np.array(pos, dtype=np.float32)
            camera_poses[idx] = (R, T)
            neighbor_image_path = os.path.join(image_dir, fname)
            neighbor_images[idx] = cv2.cvtColor(
                cv2.imread(neighbor_image_path), cv2.COLOR_BGR2RGB
            )
            neighbors_info.append((idx, fname, pos, quat))

    logging.info(
        f"{num_points}点ある{cur_idx}番目から生成される点群を複数視点コストに基づいて深度を最適化中..."
    )

    # 複数視点最適化のコアロジック
    for p_idx in range(num_points):

        # 現在の点の情報を取得
        current_point = points[p_idx]
        current_color = colors[p_idx]

        # 深度誤差に応じて探索範囲を動的に変える
        # search_range = min(depth_costs[p_idx], max_search_range)
        # if not np.isnan(depth_costs[p_idx]):
        #     print(search_range)
        search_range = config.DEPTH_SEARCH_RANGE

        # この点がどの画像から来たかを特定 (idxはprocess_image_pairから渡されるが、ここでは全点に適用)
        # 厳密には、各点がどの左右画像ペアから生成されたかを追跡する必要があるが、
        best_cost = float("inf")
        best_point = current_point

        # 深度探索範囲 (現在のZ値 ± search_range)
        z_start = current_point[2] - search_range
        z_end = current_point[2] + search_range

        # 深度候補を生成
        z_candidates = np.linspace(z_start, z_end, num_steps, dtype=np.float32)

        for z_cand in z_candidates:
            candidate_point = np.array(
                [current_point[0], current_point[1], z_cand], dtype=np.float32
            )
            total_current_cost = 0.0
            valid_neighbor_views = 0

            # 近隣画像をループ
            # 実際には、current_pointが可視である可能性のある近隣カメラのみを選ぶべき
            for neighbor_idx, _, _, _ in neighbors_info:
                neighbor_R, neighbor_T = camera_poses[neighbor_idx]
                neighbor_image = neighbor_images[neighbor_idx]

                # compute_multiview_cost を呼び出し
                photo_cost, reproj_error = compute_multiview_cost(
                    candidate_point,
                    current_color,
                    K,
                    neighbor_R,
                    neighbor_T,
                    neighbor_image,
                )

                # 有効なコストの場合のみ加算
                if photo_cost < 1e8:  # 非常に高いコスト（範囲外など）は無視
                    total_current_cost += (
                        cost_weight_photo * photo_cost
                        + cost_weight_reproj * reproj_error
                    )
                    valid_neighbor_views += 1

            # 複数の視点からの平均コスト
            if valid_neighbor_views > 0:
                total_current_cost /= valid_neighbor_views
            else:
                total_current_cost = float("inf")

            # 最もコストの低い深度を保持
            if total_current_cost < best_cost:
                best_cost = total_current_cost
                best_point = candidate_point

        # 最適化された点を更新
        optimized_points[p_idx] = best_point
        optimized_colors[p_idx] = current_color
    num_points = optimized_points.shape[0]
    logging.info(
        f"{num_points}点ある{cur_idx}番目の画像から生成される点群の複数視点最適化が終了"
    )
    return optimized_points, optimized_colors


if __name__ == "__main__":
    args = parse_arguments()
    logging.info(f"Image dir: {config.IMAGE_DIR}")
    logging.info(f"Show viewer: {args.show_viewer}")
    logging.info(f"Record video: {args.record_video}")
    start = time.time()
    B = config.B
    focal = config.focal_length
    cam_h = config.camera_height
    K = config.K.astype(np.float32)
    pix_sz = config.pixel_size
    w_size = config.window_size
    min_d, num_d = config.min_disp, config.num_disp
    camera_data = []
    with open(config.DRONE_IMAGE_LOG) as f:
        for line in f:
            fn, x, y, z, qx, qy, qz, qw = line.strip().split(",")
            pos = (float(x), float(y), float(z))
            quat = (float(qx), float(qy), float(qz), float(qw))
            camera_data.append((fn, pos, quat))
    pairs = []
    for idx in range(len(camera_data)):
        fn, pos, quat = camera_data[idx]
        left = os.path.join(config.IMAGE_DIR, f"left_{str(idx).zfill(6)}.png")
        right = os.path.join(config.IMAGE_DIR, f"right_{str(idx).zfill(6)}.png")
        R = quaternion_to_rotation_matrix(*quat).astype(np.float32)
        pairs.append(
            (
                idx,
                np.array(pos, dtype=np.float32),
                left,
                right,
                B,
                focal,
                K,
                R,
                cam_h,
                pix_sz,
                w_size,
                min_d,
                num_d,
            )
        )
    merged_pts_list, merged_cols_list = [], []
    with ProcessPoolExecutor(max_workers=1) as exe:
        for res in exe.map(process_image_pair, pairs):
            if res:
                pts, cols, idx, d_cost = res
                pts = pts.astype(np.float32)
                logging.info(f"idx {pts.shape[0]}")
                # pts, cols = refine_points_with_multiview_cost(
                #     idx,
                #     pts,
                #     cols,
                #     d_cost,
                #     camera_data,
                #     K,
                #     config.IMAGE_DIR,
                # )
                merged_pts_list.append(pts)
                merged_cols_list.append(cols)
                logging.info(f"Processed {idx}/{len(pairs)-1}")
    if merged_pts_list:
        pts, cols = integrate_depth_maps_median(
            merged_pts_list, merged_cols_list, voxel_size=0.1
        )
        pts[:, 2] = -pts[:, 2]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        print(np.asarray(pcd.points).shape[0])
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
        write_ply(
            config.OLD_POINT_CLOUD_FILE_PATH, np.asarray(pcd.points), np.asarray(pcd.colors)
        )
        o3d.visualization.draw_geometries([pcd])
    else:
        logging.warning("No points to merge.")
    logging.info(f"Total time: {time.time()-start:.2f}s")
