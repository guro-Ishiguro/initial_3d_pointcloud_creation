# tests/main.py

import os
import config
import logging
import time
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from point_cloud_filtering import PointCloudFilter
from utils import parse_arguments, clear_folder
from data_loader import DataLoader
from image_processing import ImageProcessor
from depth_estimation import DepthEstimator
from point_cloud_integrator import PointCloudIntegrator


if __name__ == "__main__":
    args = parse_arguments()
    start_time = time.time()

    # --- 初期化 ---
    data_loader = DataLoader(config.IMAGE_DIR, config.DRONE_IMAGE_LOG)
    image_processor = ImageProcessor(config)
    depth_estimator = DepthEstimator(config)
    point_cloud_filter = PointCloudFilter(config)
    point_cloud_integrator = PointCloudIntegrator(config)

    os.makedirs(config.POINT_CLOUD_DIR, exist_ok=True)
    clear_folder(config.POINT_CLOUD_DIR)

    all_pairs_data = data_loader.get_all_camera_pairs(config.K)
    target_indices = list(range(len(all_pairs_data))) 

    # --- パフォーマンス向上のため、必要な画像を事前に一括ロード ---
    image_indices_to_load = set()
    for idx in target_indices:
        image_indices_to_load.add(idx)
        for offset in (-2, -1, 1, 2): # 近傍ビューもロード対象
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(all_pairs_data):
                image_indices_to_load.add(neighbor_idx)
            
    logging.info("Pre-loading images...")
    loaded_images = {}
    for idx in sorted(list(image_indices_to_load)):
        left_path, _ = data_loader.get_image_paths(idx)
        img = cv2.imread(left_path)
        if img is not None:
            loaded_images[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    merged_pts_list, merged_cols_list = [], []

    # --- 各画像ペアの処理ループ ---
    for idx in target_indices:
        if idx not in loaded_images:
            logging.warning(f"Image for index {idx} could not be loaded. Skipping.")
            continue
            
        _, T_pos, left_path, right_path, R_mat = all_pairs_data[idx]
        logging.info(f"Processing image pair {idx}...")

        try:
            li_bgr = cv2.imread(left_path)
            ri_bgr = cv2.imread(right_path)
            if li_bgr is None or ri_bgr is None: continue

            li_rgb = loaded_images[idx]
            li_gray = cv2.cvtColor(li_bgr, cv2.COLOR_BGR2GRAY)
            ri_gray = cv2.cvtColor(ri_bgr, cv2.COLOR_BGR2GRAY)

            # 1. 初期深度マップと深度誤差コストを計算
            disp = image_processor.create_disparity(li_gray, ri_gray)
            initial_depth = depth_estimator.disparity_to_depth(disp)
            d_cost = depth_estimator.compute_depth_error_cost(disp, initial_depth, config.window_size)
            
            # 境界領域や無効な深度をNaNでマスク
            valid_mask = np.isfinite(initial_depth)
            bmask = valid_mask & (~np.roll(valid_mask, 10, 0) | ~np.roll(valid_mask, -10, 0) |
                                  ~np.roll(valid_mask, 10, 1) | ~np.roll(valid_mask, -10, 1))
            initial_depth[bmask] = np.nan
            d_cost[bmask] = np.nan

            # 2. PatchMatchによる深度マップの最適化
            optimized_depth = point_cloud_filter.refine_depth_with_patchmatch()

            # 3. 最適化された深度マップをオルソ化し、3D点群に変換
            ortho_d, ortho_c = depth_estimator.to_orthographic_projection(
                optimized_depth, li_rgb, config.camera_height
            )
            pts, cols = depth_estimator.depth_to_world(
                ortho_d, ortho_c, config.K, R_mat, T_pos, config.pixel_size
            )
            
            # 有効な点のみを抽出
            valid_mask = ~np.isnan(pts).any(axis=1)
            final_pts = pts[valid_mask]
            final_cols = cols[valid_mask]
            
            logging.info(f"Generated {final_pts.shape[0]} points for image {idx} after refinement.")
            
            # 最適化された点群を表示する
            if config.DEBUG_VISUALIZATION:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(final_pts)
                pcd.colors = o3d.utility.Vector3dVector(final_cols)
                o3d.visualization.draw_geometries([pcd], window_name="Final Integrated Point Cloud")

            merged_pts_list.append(final_pts)
            merged_cols_list.append(final_cols)

        except Exception as e:
            logging.error(f"Error processing image pair {idx}: {e}", exc_info=True)

    # --- 全点群の統合と保存 ---
    if merged_pts_list:
        logging.info("Integrating all point clouds...")
        final_pcd = point_cloud_integrator.process_and_save_final_point_cloud(
            merged_pts_list, merged_cols_list, config.POINT_CLOUD_FILE_PATH
        )
        if final_pcd and len(final_pcd.points) > 0:
            logging.info("Showing final integrated point cloud. Close the window to exit.")
            o3d.visualization.draw_geometries([final_pcd], window_name="Final Integrated Point Cloud")
    else:
        logging.warning("No point clouds were generated.")

    end_time = time.time()
    logging.info(f"Total point cloud generation time: {end_time - start_time:.2f}s")