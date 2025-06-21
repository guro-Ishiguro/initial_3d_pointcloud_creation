# tests/main.py

import os
import config
import logging
import time
import cv2
import open3d as o3d
import numpy as np

from depth_optimization import DepthOptimization
from utils import parse_arguments, clear_folder, save_depth_map_as_image
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
    depth_optimization = DepthOptimization(config)
    point_cloud_integrator = PointCloudIntegrator(config)

    os.makedirs(config.POINT_CLOUD_DIR, exist_ok=True)
    clear_folder(config.POINT_CLOUD_DIR)

    if config.DEBUG_SAVE_DEPTH_MAPS:
        os.makedirs(config.DEPTH_IMAGE_DIR, exist_ok=True)
        clear_folder(config.DEPTH_IMAGE_DIR)

    all_pairs_data = data_loader.get_all_camera_pairs(config.K)

    # config.pyでTARGET_INDICESが定義されていればそれを使う
    if hasattr(config, "TARGET_INDICES") and config.TARGET_INDICES:
        target_indices = config.TARGET_INDICES
    else:
        target_indices = list(range(len(all_pairs_data)))

    logging.info(f"Targeting specific image indices for processing: {target_indices}")

    # --- パフォーマンス向上のため、必要な画像を事前に一括ロード ---
    image_indices_to_load = set()
    neighbor_view_offsets = (-2, -1, 1, 2)  # 近傍ビューのオフセット
    for idx in target_indices:
        image_indices_to_load.add(idx)
        for offset in neighbor_view_offsets:
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(all_pairs_data):
                image_indices_to_load.add(neighbor_idx)
    logging.info("Pre-loading images...")
    loaded_images = {}
    for idx in sorted(list(image_indices_to_load)):
        left_path, right_path = data_loader.get_image_paths(idx)
        img = cv2.imread(left_path)
        if img is not None:
            loaded_images[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- ステップ1: 各ビューの深度マップを最適化 & 光度フィルタリング ---
    logging.info(
        "\n--- Step 1: Optimizing depth maps and applying photometric filter ---"
    )
    all_optimized_depths = {}

    for idx in target_indices:
        if idx not in loaded_images:
            logging.warning(f"Image for index {idx} could not be loaded. Skipping.")
            continue

        _, T_pos, left_path, right_path, R_mat = all_pairs_data[idx]
        logging.info(f"Optimizing depth map for image pair {idx}...")

        try:
            li_bgr = cv2.imread(left_path)
            ri_bgr = cv2.imread(right_path)
            if li_bgr is None or ri_bgr is None:
                continue

            li_rgb = loaded_images[idx]
            li_gray = cv2.cvtColor(li_bgr, cv2.COLOR_BGR2GRAY)
            ri_gray = cv2.cvtColor(ri_bgr, cv2.COLOR_BGR2GRAY)

            # 1. 初期深度マップと深度誤差コストを計算
            disp = image_processor.create_disparity(li_gray, ri_gray)
            initial_depth = depth_estimator.disparity_to_depth(disp)
            d_cost = depth_estimator.compute_depth_error_cost(
                disp, initial_depth, config.window_size
            )

            # 境界領域や無効な深度をNaNでマスク
            valid_mask = np.isfinite(initial_depth)
            bmask = valid_mask & (
                ~np.roll(valid_mask, 10, 0)
                | ~np.roll(valid_mask, -10, 0)
                | ~np.roll(valid_mask, 10, 1)
                | ~np.roll(valid_mask, -10, 1)
            )
            initial_depth[bmask] = np.nan
            d_cost[bmask] = np.nan
            d_cost[np.isnan(d_cost)] = 1.0

            # 2. PatchMatchによる深度マップの最適化
            neighbor_views_data = []
            for offset in neighbor_view_offsets:
                neighbor_idx = idx + offset
                if (
                    0 <= neighbor_idx < len(all_pairs_data)
                    and neighbor_idx in loaded_images
                ):
                    _, T_n, _, _, R_n = all_pairs_data[neighbor_idx]
                    neighbor_views_data.append(
                        {
                            "image": loaded_images[neighbor_idx],
                            "image_idx": neighbor_idx,  # 幾何学フィルタで使うためインデックスを追加
                            "R": R_n,
                            "T": T_n,
                            "K": config.K,
                        }
                    )

            if config.DEBUG_SAVE_DEPTH_MAPS:
                save_each_depth_dir = os.path.join(
                    config.DEPTH_IMAGE_DIR, f"depth_{idx:04d}"
                )
                os.makedirs(save_each_depth_dir, exist_ok=True)
                clear_folder(save_each_depth_dir)
                save_initial_depth_path = os.path.join(
                    save_each_depth_dir, f"initial_depth.png"
                )
                logging.info(f"Saving initial depth map to {save_initial_depth_path}")
                save_depth_map_as_image(initial_depth, save_initial_depth_path)

            # PatchMatchを実行
            optimized_depth = depth_optimization.refine_depth_with_patchmatch(
                initial_depth=initial_depth,
                initial_depth_error=d_cost,
                ref_image=li_rgb,
                ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
                neighbor_views_data=neighbor_views_data,
                ref_idx=idx,
            )

            # 3. 光度一貫性フィルタリング
            photometrically_filtered_depth = depth_optimization.filter_depth_map(
                optimized_depth,
                li_rgb,
                {"R": R_mat, "T": T_pos, "K": config.K},
                neighbor_views_data,
            )

            if config.DEBUG_SAVE_DEPTH_MAPS:
                save_photometric_filtered_depth_path = os.path.join(
                    save_each_depth_dir, f"photometric_filtered_depth.png"
                )
                logging.info(
                    f"Saving photometrically filtered depth map to {save_photometric_filtered_depth_path}"
                )
                save_depth_map_as_image(
                    photometrically_filtered_depth, save_photometric_filtered_depth_path
                )

            all_optimized_depths[idx] = photometrically_filtered_depth
            logging.info(f"Stored photometrically filtered depth map for index {idx}.")

        except Exception as e:
            logging.error(f"Error in Step 1 for image pair {idx}: {e}", exc_info=True)

    # --- ステップ2: 幾何学的一貫性フィルタリングと点群生成 ---
    logging.info(
        "\n--- Step 2: Applying geometric consistency filter and generating point clouds ---"
    )
    merged_pts_list, merged_cols_list = [], []

    for idx in target_indices:
        if idx not in all_optimized_depths:
            logging.warning(
                f"No optimized depth map for index {idx}. Skipping point cloud generation."
            )
            continue

        logging.info(f"Generating point cloud for image pair {idx}...")

        try:
            _, T_pos, _, _, R_mat = all_pairs_data[idx]
            li_rgb = loaded_images[idx]

            ref_depth_map = all_optimized_depths[idx]

            # 近傍ビューのデータ準備 (再)
            neighbor_views_data = []
            for offset in neighbor_view_offsets:
                neighbor_idx = idx + offset
                if (
                    0 <= neighbor_idx < len(all_pairs_data)
                    and neighbor_idx in loaded_images
                ):
                    _, T_n, _, _, R_n = all_pairs_data[neighbor_idx]
                    neighbor_views_data.append(
                        {
                            "image": loaded_images[neighbor_idx],
                            "image_idx": neighbor_idx,
                            "R": R_n,
                            "T": T_n,
                            "K": config.K,
                        }
                    )

            # 4. 幾何学的一貫性フィルタリング
            geometrically_filtered_depth = depth_optimization.filter_depth_map_by_geometric_consistency(
                ref_depth_map=ref_depth_map,
                ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
                neighbor_views_data=neighbor_views_data,
                all_optimized_depths=all_optimized_depths,
            )

            if config.DEBUG_SAVE_DEPTH_MAPS:
                save_each_depth_dir = os.path.join(
                    config.DEPTH_IMAGE_DIR, f"depth_{idx:04d}"
                )
                os.makedirs(save_each_depth_dir, exist_ok=True)
                save_geometric_filtered_depth_path = os.path.join(
                    save_each_depth_dir, f"geometric_filtered_depth.png"
                )
                logging.info(
                    f"Saving geometrically filtered depth map to {save_geometric_filtered_depth_path}"
                )
                save_depth_map_as_image(
                    geometrically_filtered_depth, save_geometric_filtered_depth_path
                )

            # 5. 中心投影深度マップを正射投影深度マップに変換
            (
                ortho_depth_map,
                ortho_color_map,
            ) = depth_estimator.to_orthographic_projection(
                geometrically_filtered_depth, li_rgb, config.camera_height
            )

            if config.DEBUG_SAVE_DEPTH_MAPS:
                save_ortho_optimized_depth = os.path.join(
                    save_each_depth_dir, f"depth_orthographic.png"
                )
                logging.info(
                    f"Saving orthographic depth map to {save_ortho_optimized_depth}"
                )
                save_depth_map_as_image(ortho_depth_map, save_ortho_optimized_depth)

            # 6. 正射投影深度マップをワールド座標の点群に変換
            world_points, world_colors = depth_estimator.ortho_depth_to_world(
                ortho_depth_map, ortho_color_map, R_mat, T_pos, config.pixel_size
            )
            logging.info(
                f"Generated {world_points.shape[0]} points for image {idx} after geometric filtering."
            )

            merged_pts_list.append(world_points)
            merged_cols_list.append(world_colors)

        except Exception as e:
            logging.error(f"Error in Step 2 for image pair {idx}: {e}", exc_info=True)

    # --- ステップ3: 全点群の統合と保存 ---
    logging.info("\n--- Step 3: Integrating and saving the final point cloud ---")
    if merged_pts_list:
        logging.info("Integrating all point clouds...")
        (merged_pts, merged_cols) = point_cloud_integrator.integrate_depth_maps_median(
            merged_pts_list, merged_cols_list
        )
        final_pcd = point_cloud_integrator.process_and_save_final_point_cloud(
            merged_pts, merged_cols, config.POINT_CLOUD_FILE_PATH
        )
        if final_pcd and len(final_pcd.points) > 0:
            logging.info(
                "Showing final integrated point cloud. Close the window to exit."
            )
            o3d.visualization.draw_geometries([final_pcd])
    else:
        logging.warning("No point clouds were generated.")

    end_time = time.time()
    logging.info(f"Total point cloud generation time: {end_time - start_time:.2f}s")
