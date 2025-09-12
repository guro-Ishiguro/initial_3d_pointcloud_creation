# tests/main.py

import os
import config
import logging
import time
import cv2
import open3d as o3d
import numpy as np

from utils import (
    parse_arguments,
    save_depth_map_as_image,
    read_exr_depth,
    compute_depth_metrics,
    save_error_map_as_image,
    save_disparity_map_with_colorbar,
    clear_folder,
)
from data_loader import DataLoader
from disparity_estimation import ImageProcessor
from depth_estimation import DepthEstimator
from point_cloud_integrator import PointCloudIntegrator
from depth_fusion import CameraPlaneMedianFuser, OrthoDepthMedianFuser, WorldOrthoMedianFuser
from depth_optimization import DepthOptimization, is_gpu_enabled


if __name__ == "__main__":
    args = parse_arguments()
    start_time = time.time()

    # --- 初期化 ---
    data_loader = DataLoader(config.STEREO_IMAGE_DIR, config.DRONE_IMAGE_LOG)
    image_processor = ImageProcessor(config)
    depth_estimator = DepthEstimator(config)
    depth_optimization = DepthOptimization(config)
    logging.info(f"DepthOptimization backend: {'GPU' if is_gpu_enabled() else 'CPU'}")
    point_cloud_integrator = PointCloudIntegrator(config)

    os.makedirs(config.POINT_CLOUD_DIR, exist_ok=True)

    os.makedirs(config.CSV_DIR, exist_ok=True)

    os.makedirs(config.DISPARITY_IMAGE_DIR, exist_ok=True)

    if config.DEBUG_SAVE_DEPTH_MAPS:
        os.makedirs(config.DEPTH_IMAGE_DIR, exist_ok=True)

    if config.DEBUG_SAVE_NORMAL_MAPS:
        os.makedirs(config.NORMAL_IMAGE_DIR, exist_ok=True)

    all_pairs_data = data_loader.get_all_camera_pairs(config.K)

    if hasattr(config, "TARGET_INDICES") and config.TARGET_INDICES:
        target_indices = config.TARGET_INDICES
    else:
        target_indices = list(range(len(all_pairs_data)))

    evaluation_results = []

    logging.info(f"Targeting specific image indices for processing: {target_indices}")

    # --- パフォーマンス向上のため、必要な画像を事前に一括ロード ---
    image_indices_to_load = set()
    neighbor_view_offsets = (-3, -2, -1, 1, 2, 3)
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

    # --- ステップ1: 各ビューの深度マップを最適化 & 光度フィルタリング & 逐次点群統合/表示 ---
    logging.info(
        "\n--- Step 1: Optimizing depth maps and applying photometric filter ---"
    )
    all_optimized_depths = {}
    live_pcd = None
    vis = None
    added = False
    if getattr(config, "STREAMING_VIEWER", False):
        live_pcd = o3d.geometry.PointCloud()
    last_integ_pts, last_integ_cols = None, None
    merged_pts_list, merged_cols_list = [], []
    # 逐次深度融合: カメラ平面（参照ビュー）へワープして中央値融合
    cam_fuser = None
    ortho_fuser = OrthoDepthMedianFuser() if getattr(config, "DEPTH_FUSION_ENABLE", False) else None
    # Unityの鉛直下向き(Y-)視点に合わせ、平面=(X,Z)、深度=Y を採用（必要なら depth_invert=True）
    world_ortho_fuser = WorldOrthoMedianFuser(
        x_min=None, x_max=None, y_min=None, y_max=None, pixel_size=config.pixel_size,
        plane_axes=(0, 2), depth_axis=1, depth_invert=False
    ) if getattr(config, "DEPTH_FUSION_ENABLE", False) else None

    # 参照カメラ（最初のターゲット）で融合先を固定
    if getattr(config, "DEPTH_FUSION_ENABLE", False) and target_indices:
        ref_idx0 = target_indices[0]
        _, ref_T0, _, _, ref_R0 = all_pairs_data[ref_idx0]
        if ref_idx0 in loaded_images:
            ref_h0, ref_w0, _ = loaded_images[ref_idx0].shape
        else:
            # フォールバック：最初の読み込み済み画像サイズ
            any_idx = next(iter(loaded_images))
            ref_h0, ref_w0, _ = loaded_images[any_idx].shape
        cam_fuser = CameraPlaneMedianFuser(height=ref_h0, width=ref_w0, K=config.K, R_ref=ref_R0, T_ref=ref_T0)

    for idx in target_indices:
        if idx not in loaded_images:
            logging.warning(f"Image for index {idx} could not be loaded. Skipping.")
            continue

        _, T_pos, left_path, right_path, R_mat = all_pairs_data[idx]
        logging.info(f"Optimizing depth map for image pair {idx}...")

        view_metrics = {"image_index": idx}

        save_each_depth_dir = os.path.join(config.DEPTH_IMAGE_DIR, f"depth_{idx:04d}")
        os.makedirs(save_each_depth_dir, exist_ok=True)
        clear_folder(save_each_depth_dir)

        save_each_normal_dir = os.path.join(
            config.NORMAL_IMAGE_DIR, f"normal_{idx:04d}"
        )
        os.makedirs(save_each_normal_dir, exist_ok=True)
        clear_folder(save_each_normal_dir)

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

            save_disparity_map_with_colorbar(
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
                    save_each_depth_dir, f"depth_iter_00.png"
                )
                logging.info(f"Saving initial depth map to {save_initial_depth_path}")
                save_depth_map_as_image(initial_depth, save_initial_depth_path)

            # 初期深度を評価
            if gt_depth is not None:
                metrics = compute_depth_metrics(initial_depth, gt_depth)
                logging.info(
                    f"[Initial Depth] MAE: {metrics['mae']:.4f}, AbsRel: {metrics['abs_rel']:.4f}, RMSE: {metrics['rmse']:.4f}, RMSElog: {metrics['rmse_log']:.4f}, "
                    f"d1: {metrics['delta1']:.4f}, d2: {metrics['delta2']:.4f}, d3: {metrics['delta3']:.4f}"
                )
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
                gt_depth=gt_depth,
                ref_idx=idx,
            )
            # optimized_depth = depth_optimization.refine_depth_with_patchmatch_vanilla(
            #     ref_image=li_rgb,
            #     ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
            #     neighbor_views_data=neighbor_views_data,
            #     ref_idx=idx,
            # )

            # 最適化後の深度を評価
            if gt_depth is not None:
                metrics = compute_depth_metrics(optimized_depth, gt_depth)
                logging.info(
                    f"[Optimized Depth] MAE: {metrics['mae']:.4f}, AbsRel: {metrics['abs_rel']:.4f}, RMSE: {metrics['rmse']:.4f}, RMSElog: {metrics['rmse_log']:.4f}, "
                    f"d1: {metrics['delta1']:.4f}, d2: {metrics['delta2']:.4f}, d3: {metrics['delta3']:.4f}"
                )
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
                    f"  [Photometric Filtered] MAE: {metrics['mae']:.4f}, AbsRel: {metrics['abs_rel']:.4f}, RMSE: {metrics['rmse']:.4f}, RMSElog: {metrics['rmse_log']:.4f}, "
                    f"d1: {metrics['delta1']:.4f}, d2: {metrics['delta2']:.4f}, d3: {metrics['delta3']:.4f}"
                )
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

            # --- 幾何学的一貫性フィルタリングを即時適用（利用可能な近傍で） ---
            try:
                geometrically_filtered_depth = depth_optimization.filter_depth_map_by_geometric_consistency(
                    ref_depth_map=photometrically_filtered_depth,
                    ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
                    neighbor_views_data=neighbor_views_data,
                    all_optimized_depths=all_optimized_depths,
                )
            except Exception as e:
                logging.warning(f"Geometric consistency filtering skipped for {idx}: {e}")
                geometrically_filtered_depth = photometrically_filtered_depth

            # --- 逐次で点群へ変換し、これまでのものと統合して表示 ---
            (
                ortho_depth_map,
                ortho_color_map,
            ) = depth_estimator.to_orthographic_projection(
                geometrically_filtered_depth, li_rgb, config.camera_height
            )
            if config.DEBUG_SAVE_DEPTH_MAPS:
                save_ortho_depth_path = os.path.join(
                    save_each_depth_dir, f"ortho_depth.png"
                )
                logging.info(
                    f"Saving ortho depth map to {save_ortho_depth_path}"
                )
                save_depth_map_as_image(
                    ortho_depth_map, save_ortho_depth_path
                )
            world_points, world_colors = depth_estimator.ortho_depth_to_world(
                ortho_depth_map, ortho_color_map, R_mat, T_pos, config.pixel_size
            )
            merged_pts_list.append(world_points)
            merged_cols_list.append(world_colors)

            # 逐次深度融合
            if world_ortho_fuser is not None:
                fused_world_ortho = world_ortho_fuser.add_world_points(world_points)
                if config.DEBUG_SAVE_DEPTH_MAPS and fused_world_ortho is not None:
                    world_ortho_fuser.save_fused_depth(
                        os.path.join(config.DEPTH_IMAGE_DIR, f"depth_{idx:04d}", "fused_ortho_running.png"),
                        swap_axes=True, flip_y=True, flip_x=True
                    )

            integ_pts, integ_cols = point_cloud_integrator.integrate_depth_maps_median(
                merged_pts_list, merged_cols_list, voxel_size=0.1
            )
            if getattr(config, "STREAMING_VIEWER", False) and integ_pts.size > 0:
                try:
                    if vis is None:
                        vis = o3d.visualization.Visualizer()
                        vis.create_window(window_name="Streaming Point Cloud", width=1280, height=720, visible=True)
                        opt = vis.get_render_option()
                        opt.background_color = np.asarray([0, 0, 0])
                        added = False
                    live_pcd.points = o3d.utility.Vector3dVector(integ_pts)
                    live_pcd.colors = o3d.utility.Vector3dVector(integ_cols)
                    if not added:
                        vis.add_geometry(live_pcd)
                        # 初回のみカメラ姿勢を設定
                        ctr = vis.get_view_control()
                        front = np.asarray(getattr(config, "VIEWER_TOPDOWN_FRONT", [0.0, -1.0, 0.0]))
                        up = np.asarray(getattr(config, "VIEWER_TOPDOWN_UP", [0.0, 0.0, 1.0]))
                        # ロール回転（画面の回転）を up ベクトルに反映
                        roll_deg = float(getattr(config, "VIEWER_ROLL_DEG", 0.0))
                        if abs(roll_deg) > 1e-3:
                            theta = np.deg2rad(roll_deg)
                            # front 軸まわり回転（Rodrigues）
                            f = front / (np.linalg.norm(front) + 1e-9)
                            Kx = np.array([[0, -f[2], f[1]], [f[2], 0, -f[0]], [-f[1], f[0], 0]], dtype=float)
                            Rf = np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)
                            up = (Rf @ up.reshape(3, 1)).ravel()
                        center = np.mean(integ_pts, axis=0) if integ_pts.size > 0 else np.array([0, 0, 0], dtype=float)
                        zoom = float(getattr(config, "VIEWER_TOPDOWN_ZOOM", 0.7))
                        try:
                            ctr.set_front(front)
                            ctr.set_up(up)
                            ctr.set_lookat(center)
                            ctr.set_zoom(zoom)
                        except Exception:
                            pass
                        added = True
                    else:
                        vis.update_geometry(live_pcd)
                    vis.poll_events()
                    vis.update_renderer()
                except Exception as e:
                    logging.warning(f"Streaming viewer update failed: {e}")
            last_integ_pts, last_integ_cols = integ_pts, integ_cols

        except Exception as e:
            logging.error(f"Error in Step 1 for image pair {idx}: {e}", exc_info=True)

        evaluation_results.append(view_metrics)


    # --- 最終保存 ---
    logging.info("\n--- Final: Saving the last integrated point cloud ---")
    if merged_pts_list:
        merged_pts = last_integ_pts if last_integ_pts is not None else np.vstack(merged_pts_list)
        merged_cols = last_integ_cols if last_integ_cols is not None else np.vstack(merged_cols_list)
        final_pcd = point_cloud_integrator.process_and_save_final_point_cloud(
            merged_pts, merged_cols, config.POINT_CLOUD_FILE_PATH
        )
        if final_pcd and len(final_pcd.points) > 0:
            if getattr(config, "STREAMING_VIEWER", False):
                try:
                    live_pcd.points = o3d.utility.Vector3dVector(np.asarray(final_pcd.points))
                    live_pcd.colors = o3d.utility.Vector3dVector(np.asarray(final_pcd.colors))
                    vis.update_geometry(live_pcd)
                    logging.info("Final cloud shown in streaming window. Close to exit.")
                    vis.run()
                    vis.destroy_window()
                except Exception as e:
                    logging.warning(f"Could not finalize streaming window: {e}")
            else:
                logging.info(
                    "Showing final integrated point cloud. Close the window to exit."
                )
                o3d.visualization.draw_geometries([final_pcd])
    else:
        logging.warning("No point clouds were generated.")

    end_time = time.time()
    logging.info(f"Total point cloud generation time: {end_time - start_time:.2f}s")
