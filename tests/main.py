import os
import config
import logging
import time
import cv2
import open3d as o3d
import numpy as np

# 各モジュールをインポート
# visualize_reprojection をインポートする
from point_cloud_filtering import PointCloudFilter, MAX_SEARCH_RANGE, visualize_reprojection
from utils import parse_arguments, clear_folder, quaternion_to_rotation_matrix # quaternion_to_rotation_matrix を追加
from data_loader import DataLoader
from image_processing import ImageProcessor
from depth_estimation import DepthEstimator
from point_cloud_integrator import PointCloudIntegrator


if __name__ == "__main__":
    args = parse_arguments()
    start_time = time.time()
    
    subsample_factor = 10
    
    data_loader = DataLoader(config.IMAGE_DIR, config.DRONE_IMAGE_LOG)
    image_processor = ImageProcessor(config)
    depth_estimator = DepthEstimator(config)
    point_cloud_filter = PointCloudFilter(config)
    point_cloud_integrator = PointCloudIntegrator(config)

    os.makedirs(config.POINT_CLOUD_DIR, exist_ok=True)
    clear_folder(config.POINT_CLOUD_DIR)

    all_pairs_data = data_loader.get_all_camera_pairs(config.K)
    target_indices = list(range(45, 46)) # 処理範囲を少し広げてテスト
    
    image_indices_to_load = set()
    for idx in target_indices:
        image_indices_to_load.add(idx)
        for offset in (-2, -1, 1, 2):
            image_indices_to_load.add(idx + offset)
            
    logging.info("Pre-loading images...")
    loaded_images = {}
    for idx in image_indices_to_load:
        if 0 <= idx < len(all_pairs_data):
            left_path, _ = data_loader.get_image_paths(idx)
            img = cv2.imread(left_path)
            if img is not None:
                loaded_images[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    merged_pts_list, merged_cols_list = [], []

    for idx in target_indices:
        if idx not in loaded_images:
            continue
            
        data_tuple = all_pairs_data[idx]
        _, T_pos, left_path, right_path, R_mat = data_tuple
        logging.info(f"Processing image pair {idx}...")

        try:
            li = cv2.imread(left_path)
            ri = cv2.imread(right_path)
            if li is None or ri is None: continue

            li_rgb = loaded_images[idx]
            li_gray, ri_gray = cv2.cvtColor(li, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)

            disp = image_processor.create_disparity(li_gray, ri_gray)
            depth = depth_estimator.disparity_to_depth(disp)
            d_cost = depth_estimator.compute_depth_error_cost(disp, depth, config.window_size)
            
            valid = np.isfinite(depth)
            bmask = valid & (~np.roll(valid, 10, 0) | ~np.roll(valid, -10, 0) |
                             ~np.roll(valid, 10, 1) | ~np.roll(valid, -10, 1))
            depth[bmask] = np.nan
            d_cost[bmask] = np.nan

            # PatchMatchMVSで深度マップを深度コストを探索範囲として最適化する
            optimized_pts, optimized_cols = point_cloud_filter.refine_points_with_patchmatch(
                idx, final_pts, final_cols, final_uvs, final_depth_costs, loaded_images,
                data_loader.camera_data, config.K
            )

            # PatchMatchで最適化された深度マップを元にオルソ化する

            ortho_d, ortho_c = depth_estimator.to_orthographic_projection(depth, li_rgb, config.camera_height)
            pts, cols = depth_estimator.depth_to_world(ortho_d, ortho_c, config.K, R_mat, T_pos, config.pixel_size)
            
            
            valid_mask = np.isfinite(ortho_d.flatten())
            valid_indices = np.where(valid_mask)[0]
            subsampled_indices = valid_indices[::subsample_factor]
            
            final_pts = pts[subsampled_indices]
            final_cols = cols[subsampled_indices]
            
            logging.info(f"Initial point cloud generated. Shape after subsampling: {final_pts.shape[0]}")

            initial_pts = np.copy(final_pts)

            pcd_initial = o3d.geometry.PointCloud()
            pcd_initial.points = o3d.utility.Vector3dVector(initial_pts)
            pcd_initial.paint_uniform_color([1, 0, 0])

            pcd_optimized = o3d.geometry.PointCloud()
            pcd_optimized.points = o3d.utility.Vector3dVector(optimized_pts)
            pcd_optimized.paint_uniform_color([0, 1, 0])

            # 可視化が不要な場合は以下のdraw_geometriesの引数を [pcd_optimized] に変更
            logging.info("Visualizing comparison: Red=Initial, Green=Optimized. Close the window to continue.")
            o3d.visualization.draw_geometries(
                [pcd_optimized, pcd_initial],
                window_name=f"Optimization Result for Image {idx}"
            )
            
            merged_pts_list.append(optimized_pts)
            merged_cols_list.append(optimized_cols)
            logging.info(f"Processed and refined for image {idx}")

        except Exception as e:
            logging.error(f"Error processing image pair {idx}: {e}", exc_info=True)

    if merged_pts_list:
        all_pts = np.vstack(merged_pts_list)
        all_cols = np.vstack(merged_cols_list)
        final_pcd = point_cloud_integrator.process_and_save_final_point_cloud(
            all_pts, all_cols, config.POINT_CLOUD_FILE_PATH
        )
        if final_pcd and len(final_pcd.points) > 0:
            logging.info("Showing final integrated point cloud.")
            o3d.visualization.draw_geometries([final_pcd], window_name="Final Integrated Point Cloud")

    end_time = time.time()
    logging.info(f"Total point cloud generation time: {end_time - start_time:.2f}s")