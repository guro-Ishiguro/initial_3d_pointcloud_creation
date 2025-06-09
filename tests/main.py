import os
import config
import logging
import time
import cv2
import open3d as o3d
import numpy as np

# 各モジュールをインポート
from utils import parse_arguments, clear_folder
from data_loader import DataLoader
from image_processing import ImageProcessor
from depth_estimation import DepthEstimator
from point_cloud_filtering import PointCloudFilter
from point_cloud_integrator import PointCloudIntegrator


if __name__ == "__main__":
    args = parse_arguments()

    print("--- Starting 3D Point Cloud Generation ---")
    print(f"Image directory: {config.IMAGE_DIR}")
    print(f"Show viewer: {args.show_viewer}")
    print(f"Record video: {args.record_video}")
    print("-----------------------------------------")

    start_time = time.time()

    # 各クラスのインスタンスをメインプロセスで生成
    # これらは逐次処理なので、ProcessPoolExecutorのエラーは発生しません
    data_loader = DataLoader(config.IMAGE_DIR, config.DRONE_IMAGE_LOG)
    image_processor = ImageProcessor(config)
    depth_estimator = DepthEstimator(config)
    point_cloud_filter = PointCloudFilter(config)
    point_cloud_integrator = PointCloudIntegrator(config)

    # 出力ディレクトリの準備
    os.makedirs(config.VIDEO_DIR, exist_ok=True)
    os.makedirs(config.POINT_CLOUD_DIR, exist_ok=True)
    clear_folder(config.POINT_CLOUD_DIR)  # 既存の点群をクリア

    # すべてのカメラペアのデータを準備
    all_pairs_data = data_loader.get_all_camera_pairs(config.K)

    merged_pts_list, merged_cols_list = [], []

    # ここから逐次処理のループに変更
    for i, data_tuple in enumerate(all_pairs_data[45:46]):
        idx, T_pos, left_path, right_path, R_mat = data_tuple

        logging.info(f"Processing image pair {idx}...")

        try:
            li = cv2.imread(left_path)
            ri = cv2.imread(right_path)

            if li is None or ri is None:
                logging.error(f"Failed to load images for index {idx}. Skipping.")
                continue  # 次のペアへスキップ

            li_rgb = cv2.cvtColor(li, cv2.COLOR_BGR2RGB)
            # ri_rgb は使用しないため、変換は不要

            li_gray = cv2.cvtColor(li, cv2.COLOR_BGR2GRAY)
            ri_gray = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)

            # 視差マップ生成
            disp = image_processor.create_disparity(li_gray, ri_gray)

            # Census変換とコスト計算
            cen_l = image_processor.census_transform_numba(li_gray, config.window_size)
            cen_r = image_processor.census_transform_numba(ri_gray, config.window_size)
            cen_cost = image_processor.compute_census_cost(
                li_gray, ri_gray, disp, cen_l, cen_r, config.window_size
            ).flatten()

            # 深度マップ生成
            depth = depth_estimator.disparity_to_depth(disp)

            # 深度誤差コスト計算
            d_cost = depth_estimator.compute_depth_error_cost(
                disp, depth, config.window_size
            ).flatten()

            # 深度マップの境界マスク処理
            valid = np.isfinite(depth)
            bmask = valid & (
                ~np.roll(valid, 10, 0)
                | ~np.roll(valid, -10, 0)
                | ~np.roll(valid, 10, 1)
                | ~np.roll(valid, -10, 1)
            )
            depth[bmask] = np.nan

            # オルソ画像変換
            ortho_d, ortho_c = depth_estimator.to_orthographic_projection(
                depth, li_rgb, config.camera_height
            )

            # ワールド座標への変換
            pts, cols = depth_estimator.depth_to_world(
                ortho_d, ortho_c, config.K, R_mat, T_pos, config.pixel_size
            )

            # grid_samplingによる初期ダウンサンプリング
            pts, cols = point_cloud_filter.grid_sampling(pts, cols, 0.1)

            logging.info(f"{idx}番目のカメラにおける初期点群が生成されました。形状: {pts.shape[0]}")

            # 複数視点コストに基づく深度最適化
            pts, cols = point_cloud_filter.refine_points_with_multiview_cost(
                idx,
                pts,
                cols,
                data_loader.camera_data,  # 全カメラデータが必要
                config.K,
                config.IMAGE_DIR,
            )

            merged_pts_list.append(pts)
            merged_cols_list.append(cols)
            logging.info(
                f"Processed and refined for image {idx}/{len(all_pairs_data)-1}"
            )

        except Exception as e:
            logging.error(f"Error processing image pair {idx}: {e}")

    # すべての点群が処理された後、統合
    if merged_pts_list:
        # pts_med, cols_med = point_cloud_integrator.integrate_depth_maps_median(
        #     merged_pts_list, merged_cols_list, voxel_size=0.1
        # )

        # 最終点群の処理と保存、可視化
        final_pcd = point_cloud_integrator.process_and_save_final_point_cloud(
            pts, cols, config.POINT_CLOUD_FILE_PATH
        )

        o3d.visualization.draw_geometries([final_pcd])

        if args.record_video:
            logging.warning(
                "Video recording is not fully implemented in this refactored structure."
            )
            logging.warning(
                "Please refer to Open3D documentation for advanced visualization and recording."
            )

    else:
        logging.warning("No points generated to merge.")

    end_time = time.time()
    logging.info(f"Total point cloud generation time: {end_time - start_time:.2f}s")
