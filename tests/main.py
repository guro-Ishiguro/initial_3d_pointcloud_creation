# tests/main.py

import os
import config
import logging
import time
import cv2
import open3d as o3d
import numpy as np
import csv

from utils import (
    parse_arguments,
    clear_folder,
    save_depth_map_as_image,
    read_exr_depth,
    compute_depth_metrics,
    save_error_map_as_image,
)
from data_loader import DataLoader
from image_processing import ImageProcessor
from depth_estimation import DepthEstimator
from point_cloud_integrator import PointCloudIntegrator
from depth_optimization import DepthOptimization


if __name__ == "__main__":
    args = parse_arguments()
    start_time = time.time()

    # --- 初期化 ---
    data_loader = DataLoader(config.STEREO_IMAGE_DIR, config.DRONE_IMAGE_LOG)
    image_processor = ImageProcessor(config)
    depth_estimator = DepthEstimator(config)
    depth_optimization = DepthOptimization(config)
    point_cloud_integrator = PointCloudIntegrator(config)

    os.makedirs(config.POINT_CLOUD_DIR, exist_ok=True)
    clear_folder(config.POINT_CLOUD_DIR)

    os.makedirs(config.CSV_DIR, exist_ok=True)
    clear_folder(config.CSV_DIR)

    if config.DEBUG_SAVE_DEPTH_MAPS:
        os.makedirs(config.DEPTH_IMAGE_DIR, exist_ok=True)
        clear_folder(config.DEPTH_IMAGE_DIR)

    all_pairs_data = data_loader.get_all_camera_pairs(config.K)

    if hasattr(config, "TARGET_INDICES") and config.TARGET_INDICES:
        target_indices = config.TARGET_INDICES
    else:
        target_indices = list(range(len(all_pairs_data)))

    evaluation_results = []

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

        view_metrics = {"image_index": idx}

        save_each_depth_dir = os.path.join(config.DEPTH_IMAGE_DIR, f"depth_{idx:04d}")
        os.makedirs(save_each_depth_dir, exist_ok=True)

        # --- Ground Truth Depthの読み込み ---
        gt_depth_path = os.path.join(
            config.LABEL_DEPTH_IMAGE_DIR, f"depth_{idx:06d}.exr"
        )
        if not os.path.exists(gt_depth_path):
            logging.warning(
                f"Ground truth depth file not found for index {idx}, skipping evaluation for this view."
            )
            gt_depth = None
        else:
            gt_depth = read_exr_depth(gt_depth_path)
            if gt_depth is not None:
                h, w, _ = loaded_images[idx].shape
                if gt_depth.shape != (h, w):
                    gt_depth = cv2.resize(
                        gt_depth, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                clear_folder(save_each_depth_dir)
                save_depth_map_as_image(
                    gt_depth,
                    os.path.join(save_each_depth_dir, f"gt_depth_{idx:04d}.png"),
                )

        try:
            li_bgr = cv2.imread(left_path)
            ri_bgr = cv2.imread(right_path)
            if li_bgr is None or ri_bgr is None:
                continue

            li_rgb = loaded_images[idx]
            li_gray = cv2.cvtColor(li_bgr, cv2.COLOR_BGR2GRAY)
            ri_gray = cv2.cvtColor(ri_bgr, cv2.COLOR_BGR2GRAY)

            # 初期深度マップと深度誤差コストを計算
            start_time_initial_depth = time.time()
            disp = image_processor.create_disparity(li_gray, ri_gray)
            initial_depth = depth_estimator.disparity_to_depth(disp)
            end_time_initial_depth = time.time()
            logging.info(
                f"Initial depth calculation time for image {idx}: {end_time_initial_depth - start_time_initial_depth:.4f} seconds"
            )

            image_processor.save_disparity_image(
                disp, os.path.join(config.DISPARITY_IMAGE_DIR, f"disp_{idx:04d}.png")
            )

            # 深度誤差コストを計算
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

            # 初期深度を保存
            if config.DEBUG_SAVE_DEPTH_MAPS:
                save_initial_depth_path = os.path.join(
                    save_each_depth_dir, f"initial_depth.png"
                )
                logging.info(f"Saving initial depth map to {save_initial_depth_path}")
                save_depth_map_as_image(initial_depth, save_initial_depth_path)

            # 初期深度を評価
            if gt_depth is not None:
                metrics = compute_depth_metrics(initial_depth, gt_depth)
                logging.info(
                    f"[Initial Depth] RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, AbsRel: {metrics['abs_rel']:.4f}"
                )
                view_metrics["rmse_initial"] = metrics["rmse"]
                view_metrics["mae_initial"] = metrics["mae"]
                view_metrics["abs_rel_initial"] = metrics["abs_rel"]
                save_error_map_as_image(
                    initial_depth,
                    gt_depth,
                    os.path.join(save_each_depth_dir, "error_map_initial.png"),
                )

            # PatchMatchによる深度マップの最適化
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

            # PatchMatchを実行
            optimized_depth = depth_optimization.refine_depth_with_patchmatch(
                initial_depth=initial_depth,
                initial_depth_error=d_cost,
                ref_image=li_rgb,
                ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
                neighbor_views_data=neighbor_views_data,
                ref_idx=idx,
            )

            # 最適化後の深度を評価
            if gt_depth is not None:
                metrics = compute_depth_metrics(optimized_depth, gt_depth)
                logging.info(
                    f"[Optimized Depth] RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, AbsRel: {metrics['abs_rel']:.4f}"
                )
                view_metrics["rmse_optimized"] = metrics["rmse"]
                view_metrics["mae_optimized"] = metrics["mae"]
                view_metrics["abs_rel_optimized"] = metrics["abs_rel"]
                save_error_map_as_image(
                    optimized_depth,
                    gt_depth,
                    os.path.join(save_each_depth_dir, "error_map_optimized.png"),
                )

            # 光度一貫性フィルタリング
            photometrically_filtered_depth = (
                depth_optimization.filter_depth_map_by_photometric_consistency(
                    optimized_depth,
                    li_rgb,
                    {"R": R_mat, "T": T_pos, "K": config.K},
                    neighbor_views_data,
                )
            )

            # 光度フィルタリング後の深度を評価
            if gt_depth is not None:
                metrics = compute_depth_metrics(
                    photometrically_filtered_depth, gt_depth
                )
                logging.info(
                    f"  [Photometric Filtered] RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, AbsRel: {metrics['abs_rel']:.4f}"
                )
                view_metrics["rmse_photometric"] = metrics["rmse"]
                view_metrics["mae_photometric"] = metrics["mae"]
                view_metrics["abs_rel_photometric"] = metrics["abs_rel"]
                save_error_map_as_image(
                    photometrically_filtered_depth,
                    gt_depth,
                    os.path.join(save_each_depth_dir, "error_map_photometric.png"),
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

        evaluation_results.append(view_metrics)

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

            # 幾何学的一貫性フィルタリング
            geometrically_filtered_depth = (
                depth_optimization.filter_depth_map_by_geometric_consistency(
                    ref_depth_map=ref_depth_map,
                    ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
                    neighbor_views_data=neighbor_views_data,
                    all_optimized_depths=all_optimized_depths,
                )
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

            # 中心投影深度マップを正射投影深度マップに変換
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

            # 正射投影深度マップをワールド座標の点群に変換
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

    # --- 評価サマリの出力 ---
    if evaluation_results:
        # CSVファイルへの書き出し
        output_csv_path = os.path.join(config.CSV_DIR, "evaluation_summary.csv")
        logging.info(f"\n--- Evaluation Summary ---")
        logging.info(f"Writing evaluation summary to {output_csv_path}")

        headers = [
            "image_index",
            "rmse_initial",
            "mae_initial",
            "abs_rel_initial",
            "rmse_optimized",
            "mae_optimized",
            "abs_rel_optimized",
            "rmse_photometric",
            "mae_photometric",
            "abs_rel_photometric",
            "rmse_geometric",
            "mae_geometric",
            "abs_rel_geometric",
        ]

        try:
            with open(output_csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for row in evaluation_results:
                    # 各行のデータが存在しないキーをNoneで埋める
                    safe_row = {header: row.get(header) for header in headers}
                    writer.writerow(safe_row)
        except IOError as e:
            logging.error(f"Could not write to CSV file {output_csv_path}: {e}")

        # 平均値の計算とコンソールへの表示
        avg_metrics = {}
        for key in headers:
            if key == "image_index":
                continue
            # NaNを無視して平均を計算
            valid_values = [
                d[key] for d in evaluation_results if key in d and np.isfinite(d[key])
            ]
            if valid_values:
                avg_metrics[key] = np.mean(valid_values)
            else:
                avg_metrics[key] = np.nan

        logging.info("Average metrics across all views:")
        log_msg = ""
        for key, value in avg_metrics.items():
            log_msg += f"{key}: {value:.4f} | "
        logging.info(log_msg)

    end_time = time.time()
    logging.info(f"Total point cloud generation time: {end_time - start_time:.2f}s")
