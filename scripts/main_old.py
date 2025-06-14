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
from numba import njit, prange

# --- ログ設定: コンソール出力とファイル出力両方を設定 ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# コンソールハンドラ
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# ファイルハンドラ
fh = logging.FileHandler("3d_map_creation.log", mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

# 録画の設定
video_filename = os.path.join(config.VIDEO_DIR, "3d_map_visualization.avi")
video_fps = 10
video_width, video_height = 640, 480


def parse_arguments():
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
    return np.array(
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


def create_disparity_image(image_L, image_R, img_id):
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


def to_orthographic_projection(depth, color_image, camera_height):
    rows, cols = depth.shape
    mid_x, mid_y = cols // 2, rows // 2
    ri, ci = np.indices((rows, cols))
    valid = np.isfinite(depth)
    shift_x = (
        (camera_height - depth[valid]) * (mid_x - ci[valid]) / camera_height
    ).astype(int)
    shift_y = (
        (camera_height - depth[valid]) * (mid_y - ri[valid]) / camera_height
    ).astype(int)
    nx, ny = ci.copy(), ri.copy()
    nx[valid] += shift_x
    ny[valid] += shift_y
    mask = valid & (nx >= 0) & (nx < cols) & (ny >= 0) & (ny < rows)
    flat = ny[mask] * cols + nx[mask]
    depths = depth[mask]
    cols_masked = color_image[ri[mask], ci[mask]]
    order = np.lexsort((depths, flat))
    flat_s, depth_s, col_s = flat[order], depths[order], cols_masked[order]
    uniq, idx = np.unique(flat_s, return_index=True)
    uy, ux = uniq // cols, uniq % cols
    ortho_d = np.full_like(depth, np.nan)
    ortho_c = np.full_like(color_image, np.nan)
    ortho_d[uy, ux] = depth_s[idx]
    ortho_c[uy, ux] = col_s[idx]
    return ortho_d, ortho_c


def depth_to_world(depth_map, color_image, K, R, T, pixel_size):
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    loc = np.stack(
        ((i - w // 2) * pixel_size, -(j - h // 2) * pixel_size, depth_map), axis=-1
    ).reshape(-1, 3)
    world = (R @ loc.T).T + T
    cols = color_image.reshape(-1, 3) / 255.0
    valid = np.isfinite(loc[:, 2]) & (loc[:, 2] > 5)
    world[~valid] = np.nan
    cols[~valid] = np.nan
    return world, cols


def grid_sampling(points, colors, grid_size):
    ids = np.floor(points / grid_size).astype(int)
    _, idxs = np.unique(ids, axis=0, return_index=True)
    return points[idxs], colors[idxs]


def integrate_depth_maps_median(points_list, colors_list, voxel_size=0.1):
    logger.info("Integrating depth maps with median fusion...")
    all_pts = np.vstack(points_list)
    all_cols = np.vstack(colors_list)
    valid = np.isfinite(all_pts).all(axis=1)
    pts, cols = all_pts[valid], all_cols[valid]
    vids = np.floor(pts / voxel_size).astype(int)
    voxel_dict = {}
    for i, vid in enumerate(map(tuple, vids)):
        voxel_dict.setdefault(vid, []).append(i)
    med_pts, med_cols = [], []
    for idxs in voxel_dict.values():
        med_pts.append(np.median(pts[idxs], axis=0))
        med_cols.append(np.median(cols[idxs], axis=0))
    logger.info(f"Median integration produced {len(med_pts)} points.")
    return np.array(med_pts), np.array(med_cols)


def save_depth_map_as_image(depth_map, file_path):
    """
    デプスマップ（float配列）を視覚的に確認可能なカラー画像として保存する。
    """
    try:
        valid_mask = np.isfinite(depth_map)
        if not valid_mask.any():
            h, w = depth_map.shape
            black_image = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.imwrite(file_path, black_image)
            return
        min_val = np.min(depth_map[valid_mask])
        max_val = np.max(depth_map[valid_mask])
        if max_val - min_val > 1e-6:
            normalized_map = 255.0 * (1.0 - (depth_map - min_val) / (max_val - min_val))
        else:
            normalized_map = np.full(depth_map.shape, 128, dtype=np.float32)
        vis_map = np.nan_to_num(normalized_map).astype(np.uint8)
        colored_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
        colored_map[~valid_mask] = [0, 0, 0]

        cv2.imwrite(file_path, colored_map)
    except Exception as e:
        logging.error(f"Failed to save depth map to {file_path}: {e}")


def process_image_pair(data):
    img_id, T, lpath, rpath, B, focal, K, R, cam_h, pix_sz, w_size, min_d, num_d = data
    logger.info(f"Processing pair {img_id}...")
    li, ri = cv2.imread(lpath), cv2.imread(rpath)
    if li is None or ri is None:
        logger.error(f"Missing images for ID {img_id}")
        return None
    li_g, ri_g = cv2.cvtColor(li, cv2.COLOR_BGR2GRAY), cv2.cvtColor(
        ri, cv2.COLOR_BGR2GRAY
    )
    disp = create_disparity_image(li_g, ri_g, img_id)
    depth = B * focal / (disp + 1e-6)
    bad = (depth < 0) | (depth > 50) | ((depth > 8.5) & (depth < 9.5))
    depth[bad] = np.nan
    valid = np.isfinite(depth)
    bmask = valid & (
        ~np.roll(valid, 10, 0)
        | ~np.roll(valid, -10, 0)
        | ~np.roll(valid, 10, 1)
        | ~np.roll(valid, -10, 1)
    )
    depth[bmask] = np.nan
    ortho_d, ortho_c = to_orthographic_projection(
        depth, cv2.cvtColor(li, cv2.COLOR_BGR2RGB), cam_h
    )
    pts, cols = depth_to_world(ortho_d, ortho_c, K, R, T, pix_sz)
    pts2, cols2 = grid_sampling(
        pts[~np.isnan(pts).any(axis=1)], cols[~np.isnan(pts).any(axis=1)], 0.1
    )
    logger.info(f"Pair {img_id}: {len(pts2)} points after sampling.")
    return pts2, cols2


def write_ply(filename, vertices, colors):
    logger.info(f"Writing PLY to {filename}...")
    assert vertices.shape[0] == colors.shape[0]
    header = (
        ("ply\nformat ascii 1.0\nelement vertex %d\n" % len(vertices))
        + "property float x\nproperty float y\nproperty float z\n"
        + "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(filename, "w") as f:
        f.write(header)
        data = np.hstack((vertices, (colors * 255).astype(np.uint8)))
        np.savetxt(f, data, fmt="%f %f %f %d %d %d")
    logger.info("PLY write complete.")


def main():
    args = parse_arguments()
    logger.info(
        f"Starting 3D map creation. Images: {config.IMAGE_DIR}, Show: {args.show_viewer}, Record: {args.record_video}"
    )
    start = time.time()
    B, focal, cam_h = config.B, config.focal_length, config.camera_height
    K, pix_sz = config.K, config.pixel_size
    w_size, min_d, num_d = config.window_size, config.min_disp, config.num_disp
    camera_data = []
    with open(config.DRONE_IMAGE_LOG) as f:
        for line in f:
            fn, *vals = line.strip().split(",")
            pos, quat = tuple(map(float, vals[:3])), tuple(map(float, vals[3:7]))
            camera_data.append((fn, np.array(pos, dtype=np.float32), quat))
    pairs = [
        (
            i,
            cam[1],
            os.path.join(config.IMAGE_DIR, f"left_{i:06d}.png"),
            os.path.join(config.IMAGE_DIR, f"right_{i:06d}.png"),
            B,
            focal,
            K,
            quaternion_to_rotation_matrix(*cam[2]),
            cam_h,
            pix_sz,
            w_size,
            min_d,
            num_d,
        )
        for i, cam in enumerate(camera_data)
    ]
    pts_list, cols_list = [], []
    pairs = [pair for pair in pairs if pair[0] in [39]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as exe:
        for res in exe.map(process_image_pair, pairs):
            if res:
                pts_list.append(res[0])
                cols_list.append(res[1])
    med_pts, med_cols = integrate_depth_maps_median(pts_list, cols_list, 0.1)
    med_pts[:, 2] = -med_pts[:, 2]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(med_pts)
    pcd.colors = o3d.utility.Vector3dVector(med_cols)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
    write_ply(
        config.OLD_POINT_CLOUD_FILE_PATH, np.asarray(pcd.points), np.asarray(pcd.colors)
    )
    o3d.visualization.draw_geometries([pcd])
    elapsed = time.time() - start
    logger.info(f"3D map creation finished in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
