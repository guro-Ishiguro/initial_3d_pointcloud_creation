# mvs/main.py

import csv
import logging
import os
import re
import time
import bisect

import cv2
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
from depth_estimation import DepthEstimator
from depth_fusion import (
    CameraPlaneMedianFuser,
    OrthoDepthMedianFuser,
    WorldOrthoMedianFuser,
)
from depth_optimization import DepthOptimization, is_gpu_enabled
from disparity_estimation import ImageProcessor
from logging_setup import setup_logging
from point_cloud_integrator import PointCloudIntegrator
from scipy.spatial.transform import Rotation
from utils import (
    append_to_csv,
    clear_folder,
    compute_depth_metrics,
    initialize_csv,
    parse_arguments,
    read_exr_depth,
    save_depth_map_as_image,
    save_disparity_map_with_colorbar,
    save_error_map_as_image,
    save_normal_map_as_image,
)

import mvs.config as config
from app.data_loader import DataLoader


def _save_selected_pose_plot(
    *,
    data_loader: DataLoader,
    selected_indices: list,
    out_path: str,
    plane: str = "xz",
    arrow_stride: int = 5,
    arrow_scale: float = 0.25,
    title: str = "",
):
    """
    Plot camera positions (trajectory) and approximate viewing direction arrows for selected frames.

    - plane: "xz" (recommended for Unity-like top-down), "xy", "yz"
    - arrow_stride: draw direction arrows every N selected frames (>=1)
    - arrow_scale: arrow length multiplier in plot units
    """
    if not selected_indices:
        return

    plane = (plane or "xz").strip().lower()
    axes_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax_i, ax_j = axes_map.get(plane, (0, 2))
    axis_names = ["x", "y", "z"]

    xs, ys = [], []
    # forward direction (projected)
    dxs, dys = [], []
    arrow_points_x, arrow_points_y = [], []
    arrow_dxs, arrow_dys = [], []

    for k, idx in enumerate(selected_indices):
        fn, pos, quat = data_loader.get_camera_pose(idx)
        if pos is None or quat is None:
            continue
        p = np.array(pos, dtype=np.float64)
        xs.append(float(p[ax_i]))
        ys.append(float(p[ax_j]))

        # Direction arrow: assume camera forward is +Z in the pose coordinate.
        try:
            rot = Rotation.from_quat(np.array(quat, dtype=np.float64))
            forward = rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
            d = np.array([forward[ax_i], forward[ax_j]], dtype=np.float64)
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                d = d / n
        except Exception:
            d = np.array([np.nan, np.nan], dtype=np.float64)

        dxs.append(float(d[0]))
        dys.append(float(d[1]))

        if arrow_stride >= 1 and (k % arrow_stride == 0):
            arrow_points_x.append(xs[-1])
            arrow_points_y.append(ys[-1])
            arrow_dxs.append(dxs[-1])
            arrow_dys.append(dys[-1])

    if len(xs) < 2:
        return

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, "-", linewidth=1.0, alpha=0.8, label="trajectory")
    ax.scatter(xs, ys, s=6, alpha=0.8)

    # Start/end markers
    ax.scatter([xs[0]], [ys[0]], s=40, marker="o", label="start")
    ax.scatter([xs[-1]], [ys[-1]], s=40, marker="x", label="end")

    # Direction arrows
    if arrow_points_x:
        ax.quiver(
            arrow_points_x,
            arrow_points_y,
            arrow_dxs,
            arrow_dys,
            angles="xy",
            scale_units="xy",
            scale=1.0 / max(1e-6, arrow_scale),
            width=0.003,
            alpha=0.7,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(axis_names[ax_i])
    ax.set_ylabel(axis_names[ax_j])
    ax.set_title(title or f"Selected camera poses ({plane.upper()} plane)")
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _compute_normals_from_depth(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth_map.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            dc = depth_map[r, c]
            if not np.isfinite(dc):
                continue
            p_center = np.array(
                [(c - cx) * dc / fx, (r - cy) * dc / fy, dc], dtype=np.float32
            )
            dr = depth_map[r, c + 1]
            dd = depth_map[r + 1, c]
            if not (np.isfinite(dr) and np.isfinite(dd)):
                continue
            p_right = np.array(
                [(c + 1 - cx) * dr / fx, (r - cy) * dr / fy, dr], dtype=np.float32
            )
            p_down = np.array(
                [(c - cx) * dd / fx, (r + 1 - cy) * dd / fy, dd], dtype=np.float32
            )
            v_c = p_right - p_center
            v_r = p_down - p_center
            n = np.cross(v_r, v_c)
            norm = np.linalg.norm(n)
            if norm > 1e-6:
                normals[r, c] = n / norm
            else:
                normals[r, c] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


def run():
    args = parse_arguments()
    start_time = time.time()

    # --- 初期化 ---
    # ログ初期化
    try:
        setup_logging(
            getattr(config, "LOG_DIR", os.path.join(os.getcwd(), "logs")),
            getattr(config, "LOG_LEVEL", "INFO"),
            False,
        )
    except Exception:
        pass
    logging.info("init")
    logging.info(
        f"HOME_DIR={getattr(config, 'HOME_DIR', None)} DATA_DIR={getattr(config, 'DATA_DIR', None)} DATA_TYPE={getattr(config, 'DATA_TYPE', None)}"
    )

    data_loader = DataLoader(
        config.IMAGE_ROOT_DIR, getattr(config, "LEFT_CAMERA_POSES", None)
    )
    image_processor = ImageProcessor(config)
    depth_estimator = DepthEstimator(config)
    depth_optimization = DepthOptimization(config)
    logging.info(f"DepthOptimization backend: {'GPU' if is_gpu_enabled() else 'CPU'}")
    point_cloud_integrator = PointCloudIntegrator(config)

    os.makedirs(config.POINT_CLOUD_DIR, exist_ok=True)

    os.makedirs(config.CSV_DIR, exist_ok=True)
    # CSV 初期化（存在しない場合のみヘッダー作成）
    results_csv_path = os.path.join(config.CSV_DIR, "results.csv")
    if not os.path.exists(results_csv_path):
        initialize_csv(
            results_csv_path,
            [
                "index",
                "stage",
                "mae",
                "abs_rel",
                "sq_rel",
                "rmse",
                "rmse_log",
                "delta1",
                "delta2",
                "delta3",
            ],
        )

    os.makedirs(config.DISPARITY_IMAGE_DIR, exist_ok=True)

    if config.DEBUG_SAVE_DEPTH_MAPS:
        os.makedirs(config.DEPTH_IMAGE_DIR, exist_ok=True)

    if config.DEBUG_SAVE_NORMAL_MAPS:
        os.makedirs(config.NORMAL_IMAGE_DIR, exist_ok=True)

    # --- 追加: GT深度(EXR)の一括保存を各 depth/depth_XXXX 配下に作成 ---
    try:
        exr_files = [
            f
            for f in os.listdir(getattr(config, "LABEL_DEPTH_IMAGE_DIR", ""))
            if f.lower().endswith(".exr")
        ]
        for f in sorted(exr_files):
            # depth_000007.exr または 000007.exr のどちらにも対応
            m = re.match(r"^(?:depth_)?(\d+)\.exr$", f)
            if not m:
                continue
            idx = int(m.group(1))
            save_each_depth_dir = os.path.join(
                config.DEPTH_IMAGE_DIR, f"depth_{idx:04d}"
            )
            os.makedirs(save_each_depth_dir, exist_ok=True)

            src_path = os.path.join(config.LABEL_DEPTH_IMAGE_DIR, f)
            gt = read_exr_depth(src_path)
            if gt is None:
                continue

            h_vis, w_vis = int(getattr(config, "height", gt.shape[0])), int(
                getattr(config, "width", gt.shape[1])
            )
            if gt.shape != (h_vis, w_vis):
                gt_resized = cv2.resize(
                    gt, (w_vis, h_vis), interpolation=cv2.INTER_NEAREST
                )
            else:
                gt_resized = gt

            # 可視化PNGのみを保存
            save_depth_map_as_image(
                gt_resized,
                os.path.join(save_each_depth_dir, f"gt_depth_{idx:04d}.png"),
            )
        logging.info(
            f"Exported {len(exr_files)} GT depth files into per-view folders under {config.DEPTH_IMAGE_DIR}"
        )
    except Exception as e:
        logging.warning(f"GT per-view export skipped: {e}")

    # NOTE: all_pairs_data is a mapping from original frame index -> pair data.
    # Keeping original indices is important because many artifacts (depth_XXXX, GT exr names, etc.)
    # are keyed by the frame index coming from the dataset.
    all_pairs_data = data_loader.get_all_camera_pairs(config.K)
    if not all_pairs_data:
        logging.error(
            "No valid image pairs found. Check images under images/image_0 & image_1 and the txt/drone_image_log.txt."
        )
        return 1

    available_indices = sorted(list(all_pairs_data.keys()))

    # --- Log which images will be used (after subsampling & file existence checks) ---
    try:
        save_selected_csv = bool(getattr(config, "SAVE_SELECTED_FRAMES_CSV", True))
        if save_selected_csv:
            selected_csv_name = str(
                getattr(config, "SELECTED_FRAMES_CSV_NAME", "selected_frames.csv")
            ).strip() or "selected_frames.csv"
            selected_frames_csv_path = os.path.join(config.CSV_DIR, selected_csv_name)
            with open(selected_frames_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "left_path", "right_path"])
                for idx in available_indices:
                    _, _, left_path, right_path, _ = all_pairs_data[idx]
                    w.writerow([idx, left_path, right_path])
            logging.info(
                f"Selected frames CSV saved: {selected_frames_csv_path} (count={len(available_indices)})"
            )
            max_print = int(getattr(config, "LOG_SELECTED_FRAMES_MAX", 10) or 10)
            max_print = max(0, max_print)
            if max_print > 0:
                preview = available_indices[:max_print]
                logging.info(f"Selected frame indices (first {len(preview)}): {preview}")
    except Exception as e:
        logging.warning(f"Failed to write selected frames CSV: {e}")

    # --- Plot & save selected camera poses (trajectory) ---
    try:
        if bool(getattr(config, "SAVE_SELECTED_POSE_PLOT", True)):
            plots_dir = os.path.join(config.OUTPUT_TYPE_DIR, "plots")
            plot_name = str(
                getattr(config, "SELECTED_POSE_PLOT_NAME", "selected_camera_poses.png")
            ).strip() or "selected_camera_poses.png"
            plot_path = os.path.join(plots_dir, plot_name)
            plane = str(getattr(config, "POSE_PLOT_PLANE", "xz") or "xz")
            arrow_stride = int(getattr(config, "POSE_PLOT_ARROW_STRIDE", 5) or 5)
            arrow_stride = max(1, arrow_stride)
            arrow_scale = float(getattr(config, "POSE_PLOT_ARROW_SCALE", 0.25) or 0.25)
            _save_selected_pose_plot(
                data_loader=data_loader,
                selected_indices=available_indices,
                out_path=plot_path,
                plane=plane,
                arrow_stride=arrow_stride,
                arrow_scale=arrow_scale,
                title=f"Session={getattr(config, 'DATA_TYPE', '')} selected={len(available_indices)}",
            )
            logging.info(f"Selected pose plot saved: {plot_path}")
    except Exception as e:
        logging.warning(f"Failed to save selected pose plot: {e}")

    if hasattr(config, "TARGET_INDICES") and config.TARGET_INDICES:
        # Filter to actually available indices (after frame selection / missing file skips)
        requested = list(config.TARGET_INDICES)
        target_indices = [i for i in requested if i in all_pairs_data]
        missing = [i for i in requested if i not in all_pairs_data]
        if missing:
            logging.warning(
                f"Some TARGET_INDICES are not available (skipped/missing/filtered): {missing}"
            )
        if not target_indices:
            logging.error(
                "No TARGET_INDICES are available after filtering. Check FRAME_SELECTION_MODE or dataset integrity."
            )
            return 1
    else:
        target_indices = available_indices

    evaluation_results = []

    logging.info(f"Targeting specific image indices for processing: {target_indices}")

    # --- パフォーマンス向上のため、必要な画像を事前に一括ロード ---
    image_indices_to_load = set()
    # Neighbor selection must work even after subsampling (FRAME_SELECTION_MODE),
    # so we select neighbors as "adjacent selected keyframes" around the current idx.
    neighbor_each_side = int(getattr(config, "NEIGHBOR_KEYFRAMES_EACH_SIDE", 3) or 3)
    neighbor_each_side = max(0, neighbor_each_side)

    def _neighbor_indices(center_idx: int):
        if neighbor_each_side <= 0:
            return []
        pos = bisect.bisect_left(available_indices, center_idx)
        out = []
        for k in range(1, neighbor_each_side + 1):
            j = pos - k
            if j >= 0:
                out.append(available_indices[j])
        for k in range(1, neighbor_each_side + 1):
            j = pos + k
            if j < len(available_indices):
                out.append(available_indices[j])
        return out

    for idx in target_indices:
        image_indices_to_load.add(idx)
        for neighbor_idx in _neighbor_indices(idx):
            if neighbor_idx in all_pairs_data:
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
    # 各ステージの深度マップを保存（エラーマップの統一スケール用）
    all_stage_depths = {}
    # 各画像のポーズ情報を保存（幾何学的一貫性フィルタリング用）
    all_poses = {}
    all_images = {}
    all_gt_depths = {}
    # 逐次深度融合: カメラ平面（参照ビュー）へワープして中央値融合
    cam_fuser = None
    ortho_fuser = (
        OrthoDepthMedianFuser()
        if getattr(config, "DEPTH_FUSION_ENABLE", False)
        else None
    )
    # Unityの鉛直下向き(Y-)視点に合わせ、平面=(X,Z)、深度=Y を採用（必要なら depth_invert=True）
    world_ortho_fuser = (
        WorldOrthoMedianFuser(
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            pixel_size=config.pixel_size,
            plane_axes=(0, 2),
            depth_axis=1,
            depth_invert=False,
        )
        if getattr(config, "DEPTH_FUSION_ENABLE", False)
        else None
    )

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
        cam_fuser = CameraPlaneMedianFuser(
            height=ref_h0, width=ref_w0, K=config.K, R_ref=ref_R0, T_ref=ref_T0
        )

    for idx in target_indices:
        if idx not in loaded_images:
            logging.warning(f"Image for index {idx} could not be loaded. Skipping.")
            continue

        if idx not in all_pairs_data:
            logging.warning(
                f"Pair data for index {idx} not found (skipped/missing). Skipping."
            )
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
        # depth_######.exr と ######.exr の両方に対応
        gt_depth_path = os.path.join(
            config.LABEL_DEPTH_IMAGE_DIR, f"depth_{idx:06d}.exr"
        )
        if not os.path.exists(gt_depth_path):
            alt_path = os.path.join(config.LABEL_DEPTH_IMAGE_DIR, f"{idx:06d}.exr")
            gt_depth_path = alt_path if os.path.exists(alt_path) else ""

        if not gt_depth_path or not os.path.exists(gt_depth_path):
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
                # 可視化PNGのみを保存
                gt_vis_path = os.path.join(
                    save_each_depth_dir, f"gt_depth_{idx:04d}.png"
                )
                save_depth_map_as_image(gt_depth, gt_vis_path)

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
            if getattr(config, "DEBUG_SAVE_NORMAL_MAPS", False):
                init_normals = _compute_normals_from_depth(initial_depth, config.K)
                save_initial_normal_path = os.path.join(
                    save_each_normal_dir, f"normal_iter_00.png"
                )
                logging.info(f"Saving initial normal map to {save_initial_normal_path}")
                save_normal_map_as_image(init_normals, save_initial_normal_path)

            # 初期深度を評価（エラーマップは後で統一スケールで保存）
            if gt_depth is not None:
                metrics = compute_depth_metrics(initial_depth, gt_depth)
                logging.info(
                    f"[Initial Depth] MAE: {metrics['mae']:.4f}, AbsRel: {metrics['abs_rel']:.4f}, SqRel: {metrics['sq_rel']:.4f}, RMSE: {metrics['rmse']:.4f}, RMSElog: {metrics['rmse_log']:.4f}, "
                    f"d1: {metrics['delta1']:.4f}, d2: {metrics['delta2']:.4f}, d3: {metrics['delta3']:.4f}"
                )
                # エラーマップは後で統一スケールで保存するため、ここでは保存しない
                append_to_csv(
                    results_csv_path,
                    [
                        idx,
                        "initial",
                        metrics["mae"],
                        metrics["abs_rel"],
                        metrics["sq_rel"],
                        metrics["rmse"],
                        metrics["rmse_log"],
                        metrics["delta1"],
                        metrics["delta2"],
                        metrics["delta3"],
                    ],
                )

            # PatchMatchによる深度マップの最適化
            neighbor_views_data = []
            for neighbor_idx in _neighbor_indices(idx):
                if (
                    neighbor_idx in all_pairs_data
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

            # PatchMatchを実行（全体計測とイテレーション内計測は関数側で行う）
            refine_start = time.time()
            optimized_depth = depth_optimization.refine_depth_with_patchmatch(
                initial_depth=initial_depth,
                initial_depth_error=d_cost,
                ref_image=li_rgb,
                ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
                neighbor_views_data=neighbor_views_data,
                gt_depth=gt_depth,
                ref_idx=idx,
            )
            refine_elapsed = time.time() - refine_start
            logging.info(
                f"[Timing] refine_depth_with_patchmatch total time: {refine_elapsed:.2f}s for index {idx}"
            )
            # optimized_depth = depth_optimization.refine_depth_with_patchmatch_vanilla(
            #     ref_image=li_rgb,
            #     ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
            #     neighbor_views_data=neighbor_views_data,
            #     ref_idx=idx,
            # )

            # 最適化後の深度を評価（エラーマップは後で統一スケールで保存）
            if gt_depth is not None:
                valid_pixels_before_photo = np.sum(np.isfinite(optimized_depth))
                metrics = compute_depth_metrics(optimized_depth, gt_depth)
                logging.info(
                    f"[Optimized Depth] Valid pixels: {valid_pixels_before_photo}, "
                    f"MAE: {metrics['mae']:.4f}, AbsRel: {metrics['abs_rel']:.4f}, SqRel: {metrics['sq_rel']:.4f}, RMSE: {metrics['rmse']:.4f}, RMSElog: {metrics['rmse_log']:.4f}, "
                    f"d1: {metrics['delta1']:.4f}, d2: {metrics['delta2']:.4f}, d3: {metrics['delta3']:.4f}"
                )
                # エラーマップは後で統一スケールで保存するため、ここでは保存しない
                append_to_csv(
                    results_csv_path,
                    [
                        idx,
                        "optimized",
                        metrics["mae"],
                        metrics["abs_rel"],
                        metrics["sq_rel"],
                        metrics["rmse"],
                        metrics["rmse_log"],
                        metrics["delta1"],
                        metrics["delta2"],
                        metrics["delta3"],
                    ],
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
            if getattr(config, "DEBUG_SAVE_NORMAL_MAPS", False):
                normals_photo = _compute_normals_from_depth(
                    photometrically_filtered_depth, config.K
                )
                save_normal_map_as_image(
                    normals_photo,
                    os.path.join(
                        save_each_normal_dir, "photometric_filtered_normal.png"
                    ),
                )

            # 光度フィルタリング後の深度を評価
            if gt_depth is not None:
                valid_pixels_after_photo = np.sum(
                    np.isfinite(photometrically_filtered_depth)
                )
                pixels_filtered_photo = (
                    valid_pixels_before_photo - valid_pixels_after_photo
                )
                metrics = compute_depth_metrics(
                    photometrically_filtered_depth, gt_depth
                )
                logging.info(
                    f"  [Photometric Filtered] Valid pixels: {valid_pixels_after_photo} "
                    f"({pixels_filtered_photo} filtered, {pixels_filtered_photo/valid_pixels_before_photo*100:.2f}%), "
                    f"MAE: {metrics['mae']:.4f}, "
                    f"AbsRel: {metrics['abs_rel']:.4f}, SqRel: {metrics['sq_rel']:.4f}, "
                    f"RMSE: {metrics['rmse']:.4f}, RMSElog: {metrics['rmse_log']:.4f}, "
                    f"d1: {metrics['delta1']:.4f}, d2: {metrics['delta2']:.4f}, "
                    f"d3: {metrics['delta3']:.4f}"
                )
                # エラーマップは後で統一スケールで保存するため、ここでは保存しない
                append_to_csv(
                    results_csv_path,
                    [
                        idx,
                        "photometric",
                        metrics["mae"],
                        metrics["abs_rel"],
                        metrics["sq_rel"],
                        metrics["rmse"],
                        metrics["rmse_log"],
                        metrics["delta1"],
                        metrics["delta2"],
                        metrics["delta3"],
                    ],
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
            all_poses[idx] = {"R": R_mat, "T": T_pos, "K": config.K}
            all_images[idx] = li_rgb
            if gt_depth is not None:
                all_gt_depths[idx] = gt_depth
            # 各ステージの深度マップを保存（エラーマップの統一スケール用）
            if gt_depth is not None:
                if idx not in all_stage_depths:
                    all_stage_depths[idx] = {}
                all_stage_depths[idx]["initial"] = initial_depth.copy()
                all_stage_depths[idx]["optimized"] = optimized_depth.copy()
                all_stage_depths[idx][
                    "photometric"
                ] = photometrically_filtered_depth.copy()
                all_stage_depths[idx]["save_dir"] = save_each_depth_dir
            logging.info(f"Stored photometrically filtered depth map for index {idx}.")

            # --- 幾何学的一貫性フィルタリングは全画像処理後に実行（コメントアウト） ---
            # try:
            #     geometrically_filtered_depth = (
            #         depth_optimization.filter_depth_map_by_geometric_consistency(
            #             ref_depth_map=photometrically_filtered_depth,
            #             ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
            #             neighbor_views_data=neighbor_views_data,
            #             all_optimized_depths=all_optimized_depths,
            #         )
            #     )
            #     if gt_depth is not None:
            #         valid_pixels_after_geo = np.sum(
            #             np.isfinite(geometrically_filtered_depth)
            #         )
            #         pixels_filtered_geo = (
            #             valid_pixels_after_photo - valid_pixels_after_geo
            #         )
            #         metrics = compute_depth_metrics(
            #             geometrically_filtered_depth, gt_depth
            #         )
            #         logging.info(
            #             f"  [Geometric Filtered] Valid pixels: {valid_pixels_after_geo} "
            #             f"({pixels_filtered_geo} filtered, {pixels_filtered_geo/valid_pixels_after_photo*100:.2f}%), "
            #             f"MAE: {metrics['mae']:.4f}, "
            #             f"AbsRel: {metrics['abs_rel']:.4f}, SqRel: {metrics['sq_rel']:.4f}, "
            #             f"RMSE: {metrics['rmse']:.4f}, RMSElog: {metrics['rmse_log']:.4f}, "
            #             f"d1: {metrics['delta1']:.4f}, d2: {metrics['delta2']:.4f}, "
            #             f"d3: {metrics['delta3']:.4f}"
            #         )
            #         save_error_map_as_image(
            #             geometrically_filtered_depth,
            #             gt_depth,
            #             os.path.join(save_each_depth_dir, "error_map_geometric.png"),
            #         )
            #         append_to_csv(
            #             results_csv_path,
            #             [
            #                 idx,
            #                 "geometric",
            #                 metrics["mae"],
            #                 metrics["abs_rel"],
            #                 metrics["sq_rel"],
            #                 metrics["rmse"],
            #                 metrics["rmse_log"],
            #                 metrics["delta1"],
            #                 metrics["delta2"],
            #                 metrics["delta3"],
            #             ],
            #         )
            #     if config.DEBUG_SAVE_DEPTH_MAPS:
            #         save_geometrically_filtered_depth_path = os.path.join(
            #             save_each_depth_dir, f"geometrically_filtered_depth.png"
            #         )
            #         logging.info(
            #             f"Saving geometrically filtered depth map to {save_geometrically_filtered_depth_path}"
            #         )
            #         save_depth_map_as_image(
            #             geometrically_filtered_depth,
            #             save_geometrically_filtered_depth_path,
            #         )
            #     if getattr(config, "DEBUG_SAVE_NORMAL_MAPS", False):
            #         normals_geo = _compute_normals_from_depth(
            #             geometrically_filtered_depth, config.K
            #         )
            #         save_normal_map_as_image(
            #             normals_geo,
            #             os.path.join(
            #                 save_each_normal_dir, "geometrically_filtered_normal.png"
            #             ),
            #         )
            # except Exception as e:
            #     logging.warning(
            #         f"Geometric consistency filtering skipped for {idx}: {e}"
            #     )
            #     geometrically_filtered_depth = photometrically_filtered_depth

            # --- 点群への変換と統合は全画像処理後に実行（コメントアウト） ---
            # # 透視投影深度マップから直接ワールド座標の点群に変換（オルソ投影をスキップ）
            # world_points, world_colors = depth_estimator.depth_to_world(
            #     geometrically_filtered_depth, li_rgb, config.K, R_mat, T_pos
            # )
            # # オルソ投影を経由する旧方式（コメントアウト）
            # # (
            # #     ortho_depth_map,
            # #     ortho_color_map,
            # # ) = depth_estimator.to_orthographic_projection(
            # #     geometrically_filtered_depth, li_rgb, config.camera_height
            # # )
            # # if config.DEBUG_SAVE_DEPTH_MAPS:
            # #     save_ortho_depth_path = os.path.join(
            # #         save_each_depth_dir, f"ortho_depth.png"
            # #     )
            # #     logging.info(f"Saving ortho depth map to {save_ortho_depth_path}")
            # #     save_depth_map_as_image(ortho_depth_map, save_ortho_depth_path)
            # # world_points, world_colors = depth_estimator.ortho_depth_to_world(
            # #     ortho_depth_map, ortho_color_map, R_mat, T_pos, config.pixel_size
            # # )
            # merged_pts_list.append(world_points)
            # merged_cols_list.append(world_colors)

            # # 逐次深度融合
            # if world_ortho_fuser is not None:
            #     fused_world_ortho = world_ortho_fuser.add_world_points(world_points)
            #     if config.DEBUG_SAVE_DEPTH_MAPS and fused_world_ortho is not None:
            #         world_ortho_fuser.save_fused_depth(
            #             os.path.join(
            #                 config.DEPTH_IMAGE_DIR,
            #                 f"depth_{idx:04d}",
            #                 "fused_ortho_running.png",
            #             ),
            #             swap_axes=True,
            #             flip_y=True,
            #             flip_x=True,
            #         )

            # integ_pts, integ_cols = point_cloud_integrator.integrate_depth_maps_median(
            #     merged_pts_list, merged_cols_list, voxel_size=0.1
            # )
            # if getattr(config, "STREAMING_VIEWER", False) and integ_pts.size > 0:
            #     try:
            #         if vis is None:
            #             vis = o3d.visualization.Visualizer()
            #             vis.create_window(
            #                 window_name="Streaming Point Cloud",
            #                 width=1280,
            #                 height=720,
            #                 visible=True,
            #             )
            #             opt = vis.get_render_option()
            #             opt.background_color = np.asarray([0, 0, 0])
            #             added = False
            #         live_pcd.points = o3d.utility.Vector3dVector(integ_pts)
            #         live_pcd.colors = o3d.utility.Vector3dVector(integ_cols)
            #         if not added:
            #             vis.add_geometry(live_pcd)
            #             # 初回のみカメラ姿勢を設定
            #             ctr = vis.get_view_control()
            #             front = np.asarray(
            #                 getattr(config, "VIEWER_TOPDOWN_FRONT", [0.0, -1.0, 0.0])
            #             )
            #             up = np.asarray(
            #                 getattr(config, "VIEWER_TOPDOWN_UP", [0.0, 0.0, 1.0])
            #             )
            #             # ロール回転（画面の回転）を up ベクトルに反映
            #             roll_deg = float(getattr(config, "VIEWER_ROLL_DEG", 0.0))
            #             if abs(roll_deg) > 1e-3:
            #                 theta = np.deg2rad(roll_deg)
            #                 # front 軸まわり回転（Rodrigues）
            #                 f = front / (np.linalg.norm(front) + 1e-9)
            #                 Kx = np.array(
            #                     [[0, -f[2], f[1]], [f[2], 0, -f[0]], [-f[1], f[0], 0]],
            #                     dtype=float,
            #                 )
            #                 Rf = (
            #                     np.eye(3)
            #                     + np.sin(theta) * Kx
            #                     + (1 - np.cos(theta)) * (Kx @ Kx)
            #                 )
            #                 up = (Rf @ up.reshape(3, 1)).ravel()
            #             center = (
            #                 np.mean(integ_pts, axis=0)
            #                 if integ_pts.size > 0
            #                 else np.array([0, 0, 0], dtype=float)
            #             )
            #             zoom = float(getattr(config, "VIEWER_TOPDOWN_ZOOM", 0.7))
            #             try:
            #                 ctr.set_front(front)
            #                 ctr.set_up(up)
            #                 ctr.set_lookat(center)
            #                 ctr.set_zoom(zoom)
            #             except Exception:
            #                 pass
            #             added = True
            #         else:
            #             vis.update_geometry(live_pcd)
            #         vis.poll_events()
            #         vis.update_renderer()
            #     except Exception as e:
            #         logging.warning(f"Streaming viewer update failed: {e}")
            # last_integ_pts, last_integ_cols = integ_pts, integ_cols

        except Exception as e:
            logging.error(f"Error in Step 1 for image pair {idx}: {e}", exc_info=True)

        evaluation_results.append(view_metrics)

    # --- ステップ2: 全画像の深度マップが揃った状態で幾何学的一貫性フィルタリングを実行 ---
    logging.info(
        "\n--- Step 2: Applying geometric consistency filtering to all depth maps ---"
    )
    all_geometrically_filtered_depths = {}
    for idx in target_indices:
        if idx not in all_optimized_depths:
            continue

        save_each_depth_dir = os.path.join(config.DEPTH_IMAGE_DIR, f"depth_{idx:04d}")
        save_each_normal_dir = os.path.join(
            config.NORMAL_IMAGE_DIR, f"normal_{idx:04d}"
        )
        gt_depth = all_gt_depths.get(idx, None)
        photometrically_filtered_depth = all_optimized_depths[idx]
        ref_pose = all_poses[idx]

        # 近傍ビューのデータを準備
        neighbor_views_data = []
        for neighbor_idx in _neighbor_indices(idx):
            if (
                neighbor_idx in all_pairs_data
                and neighbor_idx in loaded_images
                and neighbor_idx in all_optimized_depths
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

        # 幾何学的一貫性フィルタリングを実行
        try:
            valid_pixels_after_photo = np.sum(
                np.isfinite(photometrically_filtered_depth)
            )
            geometrically_filtered_depth = (
                depth_optimization.filter_depth_map_by_geometric_consistency(
                    ref_depth_map=photometrically_filtered_depth,
                    ref_pose=ref_pose,
                    neighbor_views_data=neighbor_views_data,
                    all_optimized_depths=all_optimized_depths,
                )
            )
            all_geometrically_filtered_depths[idx] = geometrically_filtered_depth

            if gt_depth is not None:
                valid_pixels_after_geo = np.sum(
                    np.isfinite(geometrically_filtered_depth)
                )
                pixels_filtered_geo = valid_pixels_after_photo - valid_pixels_after_geo
                metrics = compute_depth_metrics(geometrically_filtered_depth, gt_depth)
                logging.info(
                    f"[Geometric Filtered {idx}] Valid pixels: {valid_pixels_after_geo} "
                    f"({pixels_filtered_geo} filtered, {pixels_filtered_geo/valid_pixels_after_photo*100:.2f}%), "
                    f"MAE: {metrics['mae']:.4f}, "
                    f"AbsRel: {metrics['abs_rel']:.4f}, SqRel: {metrics['sq_rel']:.4f}, "
                    f"RMSE: {metrics['rmse']:.4f}, RMSElog: {metrics['rmse_log']:.4f}, "
                    f"d1: {metrics['delta1']:.4f}, d2: {metrics['delta2']:.4f}, "
                    f"d3: {metrics['delta3']:.4f}"
                )
                # エラーマップは後で統一スケールで保存するため、ここでは保存しない
                # 幾何学フィルタリング後の深度マップも保存
                if idx in all_stage_depths:
                    all_stage_depths[idx][
                        "geometric"
                    ] = geometrically_filtered_depth.copy()
                append_to_csv(
                    results_csv_path,
                    [
                        idx,
                        "geometric",
                        metrics["mae"],
                        metrics["abs_rel"],
                        metrics["sq_rel"],
                        metrics["rmse"],
                        metrics["rmse_log"],
                        metrics["delta1"],
                        metrics["delta2"],
                        metrics["delta3"],
                    ],
                )
            if config.DEBUG_SAVE_DEPTH_MAPS:
                save_geometrically_filtered_depth_path = os.path.join(
                    save_each_depth_dir, f"geometrically_filtered_depth.png"
                )
                logging.info(
                    f"Saving geometrically filtered depth map to {save_geometrically_filtered_depth_path}"
                )
                save_depth_map_as_image(
                    geometrically_filtered_depth,
                    save_geometrically_filtered_depth_path,
                )
            if getattr(config, "DEBUG_SAVE_NORMAL_MAPS", False):
                normals_geo = _compute_normals_from_depth(
                    geometrically_filtered_depth, config.K
                )
                save_normal_map_as_image(
                    normals_geo,
                    os.path.join(
                        save_each_normal_dir, "geometrically_filtered_normal.png"
                    ),
                )
        except Exception as e:
            logging.warning(f"Geometric consistency filtering skipped for {idx}: {e}")
            all_geometrically_filtered_depths[idx] = photometrically_filtered_depth

    # --- ステップ2.5: すべてのエラーマップを統一スケールで再保存 ---
    # 各画像ごとに、その画像のすべてのステージ/イテレーションの誤差を集めて統一スケールを計算
    if all_stage_depths:
        logging.info(
            "\n--- Step 2.5: Re-saving all error maps with unified scale (per image) ---"
        )
        for idx in target_indices:
            if idx not in all_stage_depths:
                continue
            if idx not in all_gt_depths:
                continue

            stage_depths = all_stage_depths[idx]
            gt_depth = all_gt_depths[idx]
            save_each_depth_dir = stage_depths.get("save_dir")

            if save_each_depth_dir is None:
                continue

            # すべてのステージの誤差を収集してパーセンタイルを計算（外れ値に引っ張られないように）
            all_errors = []
            for stage_name, depth in stage_depths.items():
                if stage_name == "save_dir":
                    continue
                valid_mask = np.isfinite(depth) & np.isfinite(gt_depth) & (gt_depth > 0)
                if np.any(valid_mask):
                    errors = np.abs(depth[valid_mask] - gt_depth[valid_mask])
                    all_errors.extend(errors.tolist())

            if all_errors:
                # 95パーセンタイルを使用（外れ値に引っ張られない）
                error_percentile = getattr(config, "ERROR_MAP_PERCENTILE", 95.0)
                max_error_all = float(np.percentile(all_errors, error_percentile))
                # マージンを追加（5%）
                max_error_all = max_error_all * 1.05
                # 最大誤差も記録（参考用）
                max_error_actual = float(np.max(all_errors))
            else:
                max_error_all = 1.0
                max_error_actual = 1.0

            if max_error_all < 0.01:  # 最小値の設定
                max_error_all = 1.0

            logging.info(
                f"Re-saving error maps for image {idx} with unified scale "
                f"(percentile={getattr(config, 'ERROR_MAP_PERCENTILE', 95.0):.1f}%: {max_error_all:.4f} m, "
                f"max: {max_error_actual:.4f} m)"
            )

            # 各ステージのエラーマップを統一スケールで保存
            stage_map = {
                "initial": "error_map_initial.png",
                "optimized": "error_map_optimized.png",
                "photometric": "error_map_photometric.png",
                "geometric": "error_map_geometric.png",
            }
            for stage_name, depth in stage_depths.items():
                if stage_name == "save_dir":
                    continue
                if stage_name in stage_map:
                    save_error_map_as_image(
                        depth,
                        gt_depth,
                        os.path.join(save_each_depth_dir, stage_map[stage_name]),
                        max_error=max_error_all,
                    )

    # --- ステップ3: 点群への変換と統合 ---
    logging.info(
        "\n--- Step 3: Converting depth maps to point clouds and integrating ---"
    )
    merged_pts_list, merged_cols_list = [], []
    live_pcd = None
    vis = None
    added = False
    if getattr(config, "STREAMING_VIEWER", False):
        live_pcd = o3d.geometry.PointCloud()
    last_integ_pts, last_integ_cols = None, None

    for idx in target_indices:
        if idx not in all_geometrically_filtered_depths:
            continue

        geometrically_filtered_depth = all_geometrically_filtered_depths[idx]
        ref_pose = all_poses[idx]
        li_rgb = all_images[idx]
        R_mat = ref_pose["R"]
        T_pos = ref_pose["T"]

        # 透視投影深度マップから直接ワールド座標の点群に変換（オルソ投影をスキップ）
        world_points, world_colors = depth_estimator.depth_to_world(
            geometrically_filtered_depth, li_rgb, config.K, R_mat, T_pos
        )
        merged_pts_list.append(world_points)
        merged_cols_list.append(world_colors)

        # 逐次深度融合
        if world_ortho_fuser is not None:
            fused_world_ortho = world_ortho_fuser.add_world_points(world_points)
            if config.DEBUG_SAVE_DEPTH_MAPS and fused_world_ortho is not None:
                save_each_depth_dir = os.path.join(
                    config.DEPTH_IMAGE_DIR, f"depth_{idx:04d}"
                )
                world_ortho_fuser.save_fused_depth(
                    os.path.join(
                        save_each_depth_dir,
                        "fused_ortho_running.png",
                    ),
                    swap_axes=True,
                    flip_y=True,
                    flip_x=True,
                )

        integ_pts, integ_cols = point_cloud_integrator.integrate_depth_maps_median(
            merged_pts_list, merged_cols_list, voxel_size=0.1
        )
        if getattr(config, "STREAMING_VIEWER", False) and integ_pts.size > 0:
            try:
                if vis is None:
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(
                        window_name="Streaming Point Cloud",
                        width=1280,
                        height=720,
                        visible=True,
                    )
                    opt = vis.get_render_option()
                    opt.background_color = np.asarray([0, 0, 0])
                    added = False
                live_pcd.points = o3d.utility.Vector3dVector(integ_pts)
                live_pcd.colors = o3d.utility.Vector3dVector(integ_cols)
                if not added:
                    vis.add_geometry(live_pcd)
                    # 初回のみカメラ姿勢を設定
                    ctr = vis.get_view_control()
                    front = np.asarray(
                        getattr(config, "VIEWER_TOPDOWN_FRONT", [0.0, -1.0, 0.0])
                    )
                    up = np.asarray(
                        getattr(config, "VIEWER_TOPDOWN_UP", [0.0, 0.0, 1.0])
                    )
                    # ロール回転（画面の回転）を up ベクトルに反映
                    roll_deg = float(getattr(config, "VIEWER_ROLL_DEG", 0.0))
                    if abs(roll_deg) > 1e-3:
                        theta = np.deg2rad(roll_deg)
                        # front 軸まわり回転（Rodrigues）
                        f = front / (np.linalg.norm(front) + 1e-9)
                        Kx = np.array(
                            [[0, -f[2], f[1]], [f[2], 0, -f[0]], [-f[1], f[0], 0]],
                            dtype=float,
                        )
                        Rf = (
                            np.eye(3)
                            + np.sin(theta) * Kx
                            + (1 - np.cos(theta)) * (Kx @ Kx)
                        )
                        up = (Rf @ up.reshape(3, 1)).ravel()
                    center = (
                        np.mean(integ_pts, axis=0)
                        if integ_pts.size > 0
                        else np.array([0, 0, 0], dtype=float)
                    )
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

    # --- 最終保存 ---
    logging.info("\n--- Final: Saving the last integrated point cloud ---")
    if merged_pts_list:
        merged_pts = (
            last_integ_pts if last_integ_pts is not None else np.vstack(merged_pts_list)
        )
        merged_cols = (
            last_integ_cols
            if last_integ_cols is not None
            else np.vstack(merged_cols_list)
        )

        # 複数ビュー可視性フィルタリング（オプション）
        if getattr(config, "MULTI_VIEW_VISIBILITY_FILTER_ENABLED", False):
            logging.info(
                "\n--- Applying multi-view visibility filtering to point cloud ---"
            )
            visibility_threshold = getattr(config, "MULTI_VIEW_VISIBILITY_THRESHOLD", 2)
            geometric_error_threshold = getattr(
                config, "MULTI_VIEW_GEOMETRIC_ERROR_THRESHOLD", 0.05
            )
            merged_pts, merged_cols = (
                point_cloud_integrator.filter_points_by_multi_view_visibility(
                    merged_pts,
                    merged_cols,
                    all_poses,
                    all_geometrically_filtered_depths,
                    visibility_threshold=visibility_threshold,
                    geometric_error_threshold=geometric_error_threshold,
                )
            )

        final_pcd = point_cloud_integrator.process_and_save_final_point_cloud(
            merged_pts, merged_cols, config.POINT_CLOUD_FILE_PATH
        )
        if final_pcd and len(final_pcd.points) > 0:
            if getattr(config, "STREAMING_VIEWER", False):
                try:
                    live_pcd.points = o3d.utility.Vector3dVector(
                        np.asarray(final_pcd.points)
                    )
                    live_pcd.colors = o3d.utility.Vector3dVector(
                        np.asarray(final_pcd.colors)
                    )
                    vis.update_geometry(live_pcd)
                    logging.info(
                        "Final cloud shown in streaming window. Close to exit."
                    )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
