# mvs/main.py

import bisect
import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt  # noqa: E402
from depth_estimation import DepthEstimator  # noqa: E402
from depth_optimization import (  # noqa: E402
    DepthOptimization,
    _initialize_normals_from_depth_jit,
)
from disparity_estimation import ImageProcessor  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from point_cloud_integrator import PointCloudIntegrator  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from utils import (  # noqa: E402
    append_to_csv,
    clear_folder,
    initialize_csv,
    parse_arguments,
    read_exr_depth,
    save_depth_map_as_exr,
    save_depth_map_as_image,
    save_disparity_map_with_colorbar,
    save_normal_map_as_image,
)

import mvs.config as config  # noqa: E402
from app.data_loader import DataLoader  # noqa: E402


def _pose_unity_to_cv_RT(pos_unity, quat_unity):
    """
    Convert Unity-like pose (pos, quat) to CV extrinsics used in this pipeline.
    Matches the logic in app/data_loader.py.
    Returns (R_cv, T_cv) as np.float32.
    """
    pos_cv = np.array([pos_unity[0], -pos_unity[1], pos_unity[2]], dtype=np.float32)
    quat_cv = np.array(
        [-quat_unity[0], quat_unity[1], -quat_unity[2], quat_unity[3]],
        dtype=np.float32,
    )
    r = Rotation.from_quat(quat_cv)
    R_cv = r.as_matrix().astype(np.float32).T
    T_cv = (-R_cv @ pos_cv).astype(np.float32)
    return R_cv, T_cv


def _select_nearest_neighbors(
    *,
    ref_pos: tuple,
    candidates: list,
    count: int,
    r_min: float,
    r_max: float,
):
    """
    Select neighbors by distance from ref_pos (nearest first).
    - candidates: list of dicts with at least {"pos": (x,y,z), ...}
    Returns a list of candidate dicts (length <= count).
    """
    ref = np.array(ref_pos, dtype=np.float64)

    items = []
    for fr in candidates:
        p = fr.get("pos", None)
        if not p:
            continue
        v = np.array(p, dtype=np.float64) - ref
        r = float(np.linalg.norm(v))
        if r < max(0.0, r_min) or r > max(r_min, r_max):
            continue
        items.append((r, fr))

    if not items:
        return []

    items.sort(key=lambda t: t[0])  # by distance
    count = max(0, int(count))
    if count <= 0:
        return []
    return [fr for _, fr in items[:count]]


def _log_selected_neighbors(ref_idx: int, frames: list):
    """
    Log selected neighbor frames for a reference view.
    Controlled via YAML/env:
      - LOG_SELECTED_NEIGHBORS (bool)
      - LOG_SELECTED_NEIGHBORS_MAX_PER_REF (int)
      - LOG_SELECTED_NEIGHBORS_MAX_REFS (int)
      - LOG_SELECTED_NEIGHBORS_SHOW_PATHS (bool)
    """
    # Defaults:
    # - enabled
    # - log 10 neighbors per reference
    # - log for all reference views
    # - do not show paths
    if not bool(getattr(config, "LOG_SELECTED_NEIGHBORS", True)):
        return

    max_refs = int(getattr(config, "LOG_SELECTED_NEIGHBORS_MAX_REFS", 0) or 0)
    if max_refs > 0:
        c = getattr(_log_selected_neighbors, "_count", 0)
        if c >= max_refs:
            return
        setattr(_log_selected_neighbors, "_count", c + 1)  # noqa: B010

    max_per = int(getattr(config, "LOG_SELECTED_NEIGHBORS_MAX_PER_REF", 10) or 10)
    max_per = max(0, max_per)
    show_paths = bool(getattr(config, "LOG_SELECTED_NEIGHBORS_SHOW_PATHS", False))

    items = []
    for fr in (frames or [])[:max_per]:
        ni = fr.get("id", None)
        if show_paths:
            lp = fr.get("left_path", "")
            items.append(f"{ni} path={lp}")
        else:
            items.append(str(ni))

    mode = str(getattr(config, "NEIGHBOR_SELECTION_MODE", "")).strip()
    logging.info(
        f"[Neighbors] ref={ref_idx} mode={mode} count={len(frames or [])} -> {items}"
    )


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
    """
    深度マップから法線マップを計算する（高速なJITコンパイル版を使用）
    """
    return _initialize_normals_from_depth_jit(
        depth_map.astype(np.float32), K.astype(np.float32)
    )


def _process_single_view_cpu(
    idx: int,
    all_pairs_data: dict,
    loaded_images: dict,
    image_processor,
    depth_estimator,
    depth_optimization,
    _neighbors_for_ref,
    _get_neighbor_view,
    _log_selected_neighbors,
    _compute_normals_from_depth,
):
    """
    単一ビューのCPU処理部分を実行する（並列化可能）。

    この関数は、データ読み込み、視差推定、深度変換、光度フィルタリングまでを実行し、
    GPU処理（PatchMatch）に必要なデータを準備する。

    Returns:
        dict: 処理結果を含む辞書。キーは以下の通り:
            - 'idx': 画像インデックス
            - 'success': 処理が成功したかどうか
            - 'initial_depth': 初期深度マップ
            - 'd_cost': 深度誤差コスト
            - 'li_rgb': 左画像（RGB）
            - 'ref_pose': 参照ビューのポーズ
            - 'neighbor_views_data': 近傍ビューのデータ
            - 'gt_depth': 真値深度（存在する場合）
            - 'filename_stem': ファイル名（拡張子なし）
            - 'save_each_depth_dir': 深度マップ保存ディレクトリ
            - 'save_each_normal_dir': 法線マップ保存ディレクトリ
            - 'time_csv_path': time.csvのパス
            - 'view_metrics': 評価メトリクス
            - 'error': エラーメッセージ（失敗した場合）
    """
    result = {
        "idx": idx,
        "success": False,
        "initial_depth": None,
        "d_cost": None,
        "li_rgb": None,
        "ref_pose": None,
        "neighbor_views_data": None,
        "gt_depth": None,
        "filename_stem": None,
        "save_each_depth_dir": None,
        "save_each_normal_dir": None,
        "time_csv_path": None,
        "view_metrics": {"image_index": idx},
        "error": None,
    }

    try:
        if idx not in loaded_images:
            result["error"] = f"Image for index {idx} could not be loaded"
            return result

        if idx not in all_pairs_data:
            result["error"] = f"Pair data for index {idx} not found"
            return result

        _, T_pos, left_path, right_path, R_mat = all_pairs_data[idx]
        filename_stem = Path(left_path).stem

        # ディレクトリの準備
        save_each_depth_dir = os.path.join(config.DEPTH_IMAGE_DIR, filename_stem)
        os.makedirs(save_each_depth_dir, exist_ok=True)
        clear_folder(save_each_depth_dir)

        save_each_normal_dir = os.path.join(config.NORMAL_IMAGE_DIR, filename_stem)
        os.makedirs(save_each_normal_dir, exist_ok=True)
        clear_folder(save_each_normal_dir)

        # time.csvを初期化
        csv_subdir = os.path.join(config.CSV_DIR, filename_stem)
        os.makedirs(csv_subdir, exist_ok=True)
        time_csv_path = os.path.join(csv_subdir, "time.csv")
        if os.path.exists(time_csv_path):
            os.remove(time_csv_path)
        initialize_csv(time_csv_path, ["stage", "time"])

        # Ground Truth Depthの読み込み
        gt_depth_path = os.path.join(
            config.LABEL_DEPTH_IMAGE_DIR, f"depth_{idx:06d}.exr"
        )
        if not os.path.exists(gt_depth_path):
            alt_path = os.path.join(config.LABEL_DEPTH_IMAGE_DIR, f"{idx:06d}.exr")
            gt_depth_path = alt_path if os.path.exists(alt_path) else ""

        gt_depth = None
        if gt_depth_path and os.path.exists(gt_depth_path):
            gt_depth = read_exr_depth(gt_depth_path)
            if gt_depth is not None:
                h, w, _ = loaded_images[idx].shape
                if gt_depth.shape != (h, w):
                    gt_depth = cv2.resize(
                        gt_depth, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                if (
                    bool(getattr(config, "DEBUG_SAVE_GT_DEPTH_MAPS", True))
                    and config.DEBUG_SAVE_DEPTH_MAPS
                ):
                    gt_png_path = os.path.join(
                        save_each_depth_dir, f"gt_depth_{idx:04d}.png"
                    )
                    save_depth_map_as_image(gt_depth, gt_png_path)

        # 画像の読み込み
        li_bgr = cv2.imread(left_path)
        ri_bgr = cv2.imread(right_path)
        if li_bgr is None or ri_bgr is None:
            result["error"] = f"Failed to load images for index {idx}"
            return result

        li_rgb = loaded_images[idx]
        li_gray = cv2.cvtColor(li_bgr, cv2.COLOR_BGR2GRAY)
        ri_gray = cv2.cvtColor(ri_bgr, cv2.COLOR_BGR2GRAY)

        # 視差画像の生成
        disp_start = time.time()
        disp = image_processor.create_disparity(li_gray, ri_gray)
        disp_elapsed = time.time() - disp_start

        # 視差から深度への変換
        depth_conv_start = time.time()
        initial_depth = depth_estimator.disparity_to_depth(disp)
        depth_conv_elapsed = time.time() - depth_conv_start

        # 深度誤差コストを計算
        d_cost = depth_estimator.compute_depth_error_cost(
            disp, initial_depth, config.WINDOW_SIZE
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
            save_depth_map_as_image(initial_depth, save_initial_depth_path)
        if getattr(config, "DEBUG_SAVE_NORMAL_MAPS", False):
            init_normals = _compute_normals_from_depth(initial_depth, config.K)
            save_initial_normal_path = os.path.join(
                save_each_normal_dir, f"normal_iter_00.png"
            )
            save_normal_map_as_image(init_normals, save_initial_normal_path)

        # 視差マップを保存
        save_disparity_map_with_colorbar(
            disp, os.path.join(config.DISPARITY_IMAGE_DIR, f"disp_{idx:04d}.png")
        )

        # 近傍ビューのデータを準備
        neighbor_views_data = []
        neighbor_frames = _neighbors_for_ref(idx)
        # ログ出力はメインループで行うため、ここでは呼び出さない
        for fr in neighbor_frames:
            nv = _get_neighbor_view(idx, fr)
            if nv is not None:
                neighbor_views_data.append(nv)

        # 結果を保存
        result["success"] = True
        result["initial_depth"] = initial_depth
        result["d_cost"] = d_cost
        result["li_rgb"] = li_rgb
        result["ref_pose"] = {"R": R_mat, "T": T_pos, "K": config.K}
        result["neighbor_views_data"] = neighbor_views_data
        result["gt_depth"] = gt_depth
        result["filename_stem"] = filename_stem
        result["save_each_depth_dir"] = save_each_depth_dir
        result["save_each_normal_dir"] = save_each_normal_dir
        result["time_csv_path"] = time_csv_path
        result["disp_elapsed"] = disp_elapsed
        result["depth_conv_elapsed"] = depth_conv_elapsed

    except Exception as e:
        result["error"] = str(e)
        logging.error(
            f"Error in CPU processing for image pair {idx}: {e}", exc_info=True
        )

    return result


def _export_gt_depth_pngs_per_view(
    *,
    indices: list,
    label_depth_dir: str,
    out_depth_dir: str,
    all_pairs_data: dict,
):
    """
    Export GT depth EXR files to per-view folders as PNG visualizations.
    This can be expensive if run for all frames, so we allow passing only selected indices.
    GT depth maps are saved in the same folders as the generated depth maps (using filename_stem).
    """
    if not label_depth_dir or not os.path.isdir(label_depth_dir):
        return 0

    exported = 0
    for idx in indices:
        try:
            idx_int = int(idx)
        except Exception:
            continue

        # depth_######.exr と ######.exr の両方に対応
        src_path = os.path.join(label_depth_dir, f"depth_{idx_int:06d}.exr")
        if not os.path.exists(src_path):
            alt = os.path.join(label_depth_dir, f"{idx_int:06d}.exr")
            src_path = alt if os.path.exists(alt) else ""
        if not src_path:
            continue

        gt = read_exr_depth(src_path)
        if gt is None:
            continue

        h_vis, w_vis = int(getattr(config, "height", gt.shape[0])), int(
            getattr(config, "width", gt.shape[1])
        )
        if gt.shape != (h_vis, w_vis):
            gt_resized = cv2.resize(gt, (w_vis, h_vis), interpolation=cv2.INTER_NEAREST)
        else:
            gt_resized = gt

        # 生成された深度マップと同じフォルダを使用（filename_stem）
        if idx_int in all_pairs_data:
            _, _, left_path, _, _ = all_pairs_data[idx_int]
            filename_stem = Path(left_path).stem
        else:
            filename_stem = f"{idx_int:04d}"

        save_each_depth_dir = os.path.join(out_depth_dir, filename_stem)
        os.makedirs(save_each_depth_dir, exist_ok=True)
        save_depth_map_as_image(
            gt_resized, os.path.join(save_each_depth_dir, f"gt_depth_{idx_int:04d}.png")
        )
        exported += 1
    return exported


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
    point_cloud_integrator = PointCloudIntegrator(config)

    os.makedirs(config.POINT_CLOUD_DIR, exist_ok=True)

    os.makedirs(config.CSV_DIR, exist_ok=True)

    os.makedirs(config.DISPARITY_IMAGE_DIR, exist_ok=True)

    if config.DEBUG_SAVE_DEPTH_MAPS:
        os.makedirs(config.DEPTH_IMAGE_DIR, exist_ok=True)

    if config.DEBUG_SAVE_NORMAL_MAPS:
        os.makedirs(config.NORMAL_IMAGE_DIR, exist_ok=True)

    # NOTE: all_pairs_data is a mapping from original frame index -> pair data.
    # Keeping original indices is important because many artifacts (depth_XXXX, GT exr names, etc.)
    # are keyed by the frame index coming from the dataset.
    all_pairs_data = data_loader.get_all_camera_pairs(config.K)
    # --- Optional: Bundle Adjustment for noisy poses ---
    enable_ba = bool(getattr(config, "ENABLE_BUNDLE_ADJUSTMENT", False))
    pos_scale = float(getattr(config, "POSITION_ERROR_SCALE", 0.0) or 0.0)
    rot_scale = float(getattr(config, "ROTATION_ERROR_SCALE", 0.0) or 0.0)

    if enable_ba and (pos_scale > 0.0 or rot_scale > 0.0):
        logging.info("")
        logging.info("=" * 80)
        logging.info("バンドル調整 (Bundle Adjustment)")
        logging.info("=" * 80)
        logging.info(
            "Noise detected (Pos: %.4f, Rot: %.4f). Running Bundle Adjustment...",
            pos_scale,
            rot_scale,
        )
        from sfm.bundle_adjustment import run_bundle_adjustment

        all_pairs_data = run_bundle_adjustment(all_pairs_data, config.K)
        logging.info("=" * 80)
        logging.info("")
    elif enable_ba and pos_scale <= 0.0 and rot_scale <= 0.0:
        logging.info(
            "Bundle Adjustment is enabled but skipped (POSITION_ERROR_SCALE=%.4f, ROTATION_ERROR_SCALE=%.4f). "
            "Set error scales > 0 to run bundle adjustment.",
            pos_scale,
            rot_scale,
        )
    elif not enable_ba and (pos_scale > 0.0 or rot_scale > 0.0):
        logging.info(
            "Noise detected (Pos: %.4f, Rot: %.4f) but Bundle Adjustment is disabled. "
            "Set ENABLE_BUNDLE_ADJUSTMENT=true to enable.",
            pos_scale,
            rot_scale,
        )
    if not all_pairs_data:
        logging.error(
            "No valid image pairs found. Check images under images/image_0 & image_1 and the txt/camera_params.csv."
        )
        return 1

    available_indices = sorted(list(all_pairs_data.keys()))

    # --- Log which images will be used (after subsampling & file existence checks) ---
    try:
        selected_frames_csv_path = os.path.join(config.CSV_DIR, "selected_frames.csv")
        with open(selected_frames_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "left_path", "right_path"])
            for idx in available_indices:
                _, _, left_path, right_path, _ = all_pairs_data[idx]
                w.writerow([idx, left_path, right_path])
        logging.info(
            f"Selected frames CSV saved: {selected_frames_csv_path} (count={len(available_indices)})"
        )
    except Exception as e:
        logging.warning(f"Failed to write selected frames CSV: {e}")

    # --- Plot & save selected camera poses (trajectory) ---
    try:
        plots_dir = os.path.join(config.OUTPUT_TYPE_DIR, "plots")
        plot_path = os.path.join(plots_dir, "selected_camera_poses.png")
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

    # --- Target selection ---
    requested = None
    if hasattr(config, "TARGET_INDICES") and config.TARGET_INDICES:
        requested = list(config.TARGET_INDICES)

    if requested is not None:
        # Explicit request:
        # - [] means "skip this dataset"
        if len(requested) == 0:
            logging.info(
                "TARGET_INDICES specified as empty for this dataset; skipping processing."
            )
            return 0

        target_indices = [i for i in requested if i in all_pairs_data]
        missing = [i for i in requested if i not in all_pairs_data]
        if missing:
            logging.warning(
                f"Some TARGET_INDICES are not available (skipped/missing/filtered): {missing}"
            )
        if not target_indices:
            logging.error(
                "No TARGET_INDICES are available after filtering. Check FRAME_STRIDE or dataset integrity."
            )
            return 1
    else:
        target_indices = available_indices

    evaluation_results = []

    logging.info(f"Targeting specific image indices for processing: {target_indices}")

    # --- Neighbor selection ---
    neighbor_selection_mode = (
        str(
            os.getenv(
                "NEIGHBOR_SELECTION_MODE",
                getattr(config, "NEIGHBOR_SELECTION_MODE", "adjacent"),
            )
        )
        .strip()
        .lower()
    )
    # local pool pose cache (for neighbor selection)
    local_pose = {}
    for i in available_indices:
        _, p, q = data_loader.get_camera_pose(i)
        if p is not None and q is not None:
            local_pose[i] = {
                "pos": tuple(p),
                "quat": tuple(q),
            }

    # parameters for adjacent (existing behavior)
    neighbor_each_side = int(getattr(config, "NEIGHBOR_KEYFRAMES_EACH_SIDE", 3) or 3)
    neighbor_each_side = max(0, neighbor_each_side)

    # parameters for nearest-neighbor selection (by distance)
    nearest_count = int(getattr(config, "NEIGHBOR_NEAREST_COUNT", 10) or 10)
    nearest_r_min = float(getattr(config, "NEIGHBOR_NEAREST_MIN_RADIUS_M", 0.0) or 0.0)
    nearest_r_max = float(getattr(config, "NEIGHBOR_NEAREST_MAX_RADIUS_M", 1e9) or 1e9)

    # image caches
    logging.info("Pre-loading reference images...")
    loaded_images = {}  # local idx -> RGB image
    loaded_images_by_path = {}  # path -> RGB image (for global neighbors)
    for idx in target_indices:
        left_path, _ = data_loader.get_image_paths(idx)
        img = cv2.imread(left_path)
        if img is not None:
            loaded_images[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _neighbors_for_ref(ref_idx: int):
        # returns list of frame dicts with keys: pos, quat, left_path, id
        if neighbor_selection_mode == "adjacent":
            if neighbor_each_side <= 0:
                return []
            pos = bisect.bisect_left(available_indices, ref_idx)
            out = []
            for k in range(1, neighbor_each_side + 1):
                j = pos - k
                if j >= 0:
                    ni = available_indices[j]
                    if ni != ref_idx and ni in local_pose:
                        fr = dict(local_pose[ni])
                        fr["left_path"] = data_loader.get_image_paths(ni)[0]
                        fr["id"] = ni
                        out.append(fr)
            for k in range(1, neighbor_each_side + 1):
                j = pos + k
                if j < len(available_indices):
                    ni = available_indices[j]
                    if ni != ref_idx and ni in local_pose:
                        fr = dict(local_pose[ni])
                        fr["left_path"] = data_loader.get_image_paths(ni)[0]
                        fr["id"] = ni
                        out.append(fr)
            return out

        # nearest-by-distance
        ref = local_pose.get(ref_idx, None)
        if ref is None:
            return []
        candidates = [
            {
                "pos": local_pose[i]["pos"],
                "quat": local_pose[i]["quat"],
                "left_path": data_loader.get_image_paths(i)[0],
                "id": i,
            }
            for i in available_indices
            if i != ref_idx and i in local_pose
        ]
        picked = _select_nearest_neighbors(
            ref_pos=tuple(ref["pos"]),
            candidates=candidates,
            count=nearest_count,
            r_min=nearest_r_min,
            r_max=nearest_r_max,
        )
        return picked

    def _get_neighbor_view(ref_idx: int, fr: dict):
        # load neighbor image & pose (R,T)
        left_path = str(fr.get("left_path", "")).strip()
        if not left_path:
            return None

        # prefer already loaded local images
        ni = int(fr.get("id", -1))
        img = loaded_images.get(ni, None)
        if img is None:
            bgr = cv2.imread(left_path)
            if bgr is None:
                return None
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            loaded_images[ni] = img

        pos = fr.get("pos", None)
        quat = fr.get("quat", None)
        if pos is None or quat is None:
            return None
        R_n, T_n = _pose_unity_to_cv_RT(pos, quat)
        # all_optimized_depthsのキーと一致する
        image_idx = int(fr.get("id", -1))
        return {
            "image": img,
            "image_idx": image_idx,
            "R": R_n,
            "T": T_n,
            "K": config.K,
        }

    # --- ステップ1: 各ビューの深度マップを最適化 & 光度フィルタリング ---
    logging.info("")
    logging.info("=" * 80)
    logging.info("ステップ1: 各ビューの深度マップを最適化 & 光度フィルタリング")
    logging.info(f"処理対象: {len(target_indices)}個の画像ペア")
    logging.info("=" * 80)
    logging.info("")
    all_optimized_depths = {}
    all_optimized_normals = {}  # 最適化された法線マップを保存
    # 各ステージの深度マップを保存（エラーマップの統一スケール用）
    all_stage_depths = {}
    # 各画像のポーズ情報を保存（幾何学的一貫性フィルタリング用）
    all_poses = {}
    all_images = {}
    all_gt_depths = {}

    # 各ステップの処理時間を累積するための辞書
    total_times = {
        "disparity_generation": 0.0,  # 視差画像の生成
        "disparity_to_depth": 0.0,  # 視差から深度への変換
        "depth_refinement": 0.0,  # 深度画像の改善
        "photometric_filtering": 0.0,  # Photometric Consistencyフィルタリング
        "geometric_filtering": 0.0,  # Geometric Consistencyフィルタリング
        "pointcloud_generation": 0.0,  # 三次元点群の生成
        "pointcloud_integration": 0.0,  # 三次元点群の統合
        "pointcloud_filtering": 0.0,  # 点群のフィルタリング
    }

    # マルチスレッド並列化の設定
    max_workers = getattr(config, "MAX_WORKERS", 4)
    logging.info(f"Using ThreadPoolExecutor with max_workers={max_workers}")

    # CPU処理を並列実行（データ読み込み、視差推定、深度変換まで）
    cpu_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 各ビューのCPU処理を並列実行
        future_to_idx = {
            executor.submit(
                _process_single_view_cpu,
                idx,
                all_pairs_data,
                loaded_images,
                image_processor,
                depth_estimator,
                depth_optimization,
                _neighbors_for_ref,
                _get_neighbor_view,
                _log_selected_neighbors,
                _compute_normals_from_depth,
            ): idx
            for idx in target_indices
        }

        # 完了したタスクから順に処理
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                cpu_results[idx] = result
            except Exception as e:
                logging.error(
                    f"Error in CPU processing for image pair {idx}: {e}", exc_info=True
                )
                cpu_results[idx] = {
                    "idx": idx,
                    "success": False,
                    "error": str(e),
                }

    # GPU処理（PatchMatch）と光度フィルタリングは順次実行（GPUリソースの競合を避けるため）
    for idx in target_indices:
        if idx not in cpu_results:
            continue

        result = cpu_results[idx]
        if not result.get("success", False):
            if result.get("error"):
                logging.warning(
                    f"Skipping image pair {idx} due to CPU processing error: {result['error']}"
                )
            continue

        filename_stem = result["filename_stem"]
        initial_depth = result["initial_depth"]
        d_cost = result["d_cost"]
        li_rgb = result["li_rgb"]
        ref_pose = result["ref_pose"]
        neighbor_views_data = result["neighbor_views_data"]
        gt_depth = result["gt_depth"]
        save_each_depth_dir = result["save_each_depth_dir"]
        save_each_normal_dir = result["save_each_normal_dir"]
        time_csv_path = result["time_csv_path"]
        view_metrics = result["view_metrics"]

        # 処理開始のログ
        logging.info("=" * 80)
        logging.info(f"処理開始: 画像ペア {idx} (ファイル: {filename_stem})")
        logging.info("=" * 80)

        # 近傍ビューのログ出力（GPU処理部分で一度だけ）
        neighbor_frames = _neighbors_for_ref(idx)
        _log_selected_neighbors(idx, neighbor_frames)

        # 初期深度の有効ピクセル数をログ出力
        valid_pixels_initial = np.sum(np.isfinite(initial_depth))
        logging.info(f"[Initial Depth] Valid pixels: {valid_pixels_initial}")

        try:
            # PatchMatchを実行（GPU処理は順次実行）
            logging.info("-" * 80)
            logging.info(f"[{filename_stem}] ステップ2: PatchMatch MVS深度最適化")
            logging.info("-" * 80)
            refine_start = time.time()
            (
                optimized_depth,
                optimized_normal,
                iter_times_gpu,
            ) = depth_optimization.refine_depth_with_patchmatch(
                initial_depth=initial_depth,
                initial_depth_error=d_cost,
                ref_image=li_rgb,
                ref_pose=ref_pose,
                neighbor_views_data=neighbor_views_data,
                gt_depth=gt_depth,
                ref_idx=idx,
                filename_stem=filename_stem,
            )
            refine_elapsed = time.time() - refine_start
            total_times["disparity_generation"] += result.get("disp_elapsed", 0.0)
            total_times["disparity_to_depth"] += result.get("depth_conv_elapsed", 0.0)
            total_times["depth_refinement"] += refine_elapsed
            logging.info(
                f"[{filename_stem}] PatchMatch最適化完了 (経過時間: {refine_elapsed:.2f}秒)"
            )
            # 各イテレーションの時間をtime.csvに記録
            if iter_times_gpu is not None and len(iter_times_gpu) > 0:
                for iter_num, iter_time in enumerate(iter_times_gpu, 1):
                    append_to_csv(
                        time_csv_path, [f"iter_{iter_num}", f"{iter_time:.6f}"]
                    )
                logging.info(
                    f"[{filename_stem}] {len(iter_times_gpu)}個のイテレーション時間をtime.csvに保存しました: {time_csv_path}"
                )
            else:
                logging.warning(
                    f"[{filename_stem}] iter_times_gpu is None or empty for index {idx}, skipping iteration time recording"
                )

            # 最適化後の深度の有効ピクセル数をログ出力
            valid_pixels_before_photo = np.sum(np.isfinite(optimized_depth))
            logging.info(f"[Optimized Depth] Valid pixels: {valid_pixels_before_photo}")

            # 光度一貫性フィルタリング
            logging.info("-" * 80)
            logging.info(f"[{filename_stem}] ステップ3: 光度一貫性フィルタリング")
            logging.info("-" * 80)
            photo_start = time.time()
            photometrically_filtered_depth = (
                depth_optimization.filter_depth_map_by_photometric_consistency(
                    optimized_depth,
                    li_rgb,
                    ref_pose,
                    neighbor_views_data,
                )
            )
            photo_elapsed = time.time() - photo_start
            total_times["photometric_filtering"] += photo_elapsed
            logging.info(
                f"[{filename_stem}] 光度フィルタリング完了 (経過時間: {photo_elapsed:.2f}秒)"
            )
            append_to_csv(time_csv_path, ["photometric", f"{photo_elapsed:.6f}"])

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

            # 光度フィルタリング後の深度の有効ピクセル数をログ出力
            valid_pixels_after_photo = np.sum(
                np.isfinite(photometrically_filtered_depth)
            )
            pixels_filtered_photo = valid_pixels_before_photo - valid_pixels_after_photo
            logging.info(
                f"  [Photometric Filtered] Valid pixels: {valid_pixels_after_photo} "
                f"({pixels_filtered_photo} filtered, {pixels_filtered_photo/valid_pixels_before_photo*100:.2f}%)"
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
            if optimized_normal is not None:
                all_optimized_normals[idx] = optimized_normal
            all_poses[idx] = ref_pose
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
            logging.info(f"[{filename_stem}] 深度マップを保存しました。")
            logging.info("=" * 80)
            logging.info(f"処理完了: 画像ペア {idx} (ファイル: {filename_stem})")
            logging.info("=" * 80)
            logging.info("")

        except Exception as e:
            logging.error(
                f"Error in GPU processing for image pair {idx}: {e}", exc_info=True
            )

        evaluation_results.append(view_metrics)

    # --- ステップ2: 全画像の深度マップが揃った状態で幾何学的一貫性フィルタリングを実行 ---
    logging.info("")
    logging.info("=" * 80)
    logging.info("ステップ2: 幾何学的一貫性フィルタリング")
    logging.info(f"処理対象: {len(all_optimized_depths)}個の深度マップ")
    logging.info("=" * 80)
    logging.info("")
    all_geometrically_filtered_depths = {}
    for idx in target_indices:
        if idx not in all_optimized_depths:
            continue

        # ファイル名ベースのフォルダ名を取得（all_pairs_dataから）
        if idx in all_pairs_data:
            _, _, left_path, _, _ = all_pairs_data[idx]
            filename_stem = Path(left_path).stem
        else:
            filename_stem = f"{idx:04d}"
        save_each_depth_dir = os.path.join(config.DEPTH_IMAGE_DIR, filename_stem)
        save_each_normal_dir = os.path.join(config.NORMAL_IMAGE_DIR, filename_stem)
        gt_depth = all_gt_depths.get(idx, None)
        photometrically_filtered_depth = all_optimized_depths[idx]
        ref_pose = all_poses[idx]

        # 近傍ビューのデータを準備
        neighbor_views_data = []
        neighbor_frames = _neighbors_for_ref(idx)
        _log_selected_neighbors(idx, neighbor_frames)
        for fr in neighbor_frames:
            # For Step 2, require that neighbor has an optimized depth.
            ni = int(fr.get("id", -1))
            if ni not in all_optimized_depths:
                continue
            nv = _get_neighbor_view(idx, fr)
            if nv is not None:
                neighbor_views_data.append(nv)

        # 幾何学的一貫性フィルタリングを実行
        try:
            valid_pixels_after_photo = np.sum(
                np.isfinite(photometrically_filtered_depth)
            )
            # time.csvのパスを取得
            if idx in all_pairs_data:
                _, _, left_path, _, _ = all_pairs_data[idx]
                filename_stem = Path(left_path).stem
            else:
                filename_stem = f"depth_{idx:04d}"
            csv_subdir = os.path.join(config.CSV_DIR, filename_stem)
            time_csv_path = os.path.join(csv_subdir, "time.csv")
            logging.info(f"[{filename_stem}] 幾何学的一貫性フィルタリングを実行中...")
            geo_start = time.time()
            geometrically_filtered_depth = (
                depth_optimization.filter_depth_map_by_geometric_consistency(
                    ref_depth_map=photometrically_filtered_depth,
                    ref_pose=ref_pose,
                    neighbor_views_data=neighbor_views_data,
                    all_optimized_depths=all_optimized_depths,
                )
            )
            geo_elapsed = time.time() - geo_start
            total_times["geometric_filtering"] += geo_elapsed
            append_to_csv(time_csv_path, ["geometric", f"{geo_elapsed:.6f}"])
            logging.info(
                f"[{filename_stem}] 幾何学的一貫性フィルタリング完了 (経過時間: {geo_elapsed:.2f}秒)"
            )
            logging.debug(
                f"Saved geometric time ({geo_elapsed:.6f}s) to {time_csv_path}"
            )
            all_geometrically_filtered_depths[idx] = geometrically_filtered_depth

            # 幾何学フィルタリング後の深度の有効ピクセル数をログ出力
            valid_pixels_after_geo = np.sum(np.isfinite(geometrically_filtered_depth))
            pixels_filtered_geo = valid_pixels_after_photo - valid_pixels_after_geo
            logging.info(
                f"[{filename_stem}] 幾何学フィルタリング後: 有効ピクセル数={valid_pixels_after_geo} "
                f"(フィルタリング={pixels_filtered_geo}ピクセル, {pixels_filtered_geo/valid_pixels_after_photo*100:.2f}%)"
            )
            # 幾何学フィルタリング後の深度マップも保存
            if idx in all_stage_depths:
                all_stage_depths[idx]["geometric"] = geometrically_filtered_depth.copy()
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

    # --- ステップ2.5: 深度マップをEXR形式で保存（絶対的な深度値が読み取れる形式） ---
    # DEBUG_SAVE_DEPTH_MAPSがFalseの場合は保存しない
    if config.DEBUG_SAVE_DEPTH_MAPS and all_stage_depths:
        logging.info("")
        logging.info("=" * 80)
        logging.info("ステップ2.5: 深度マップをEXR形式で保存")
        logging.info(f"保存対象: {len(all_stage_depths)}個の深度マップ")
        logging.info("=" * 80)
        logging.info("")
        for idx in target_indices:
            if idx not in all_stage_depths:
                continue

            stage_depths = all_stage_depths[idx]
            save_each_depth_dir = stage_depths.get("save_dir")

            if save_each_depth_dir is None:
                continue

            # 各ステージの深度マップをEXR形式で保存
            stage_map = {
                "initial": "depth_initial.exr",
                "optimized": "depth_optimized.exr",
                "photometric": "depth_photometric.exr",
                "geometric": "depth_geometric.exr",
            }
            for stage_name, depth in stage_depths.items():
                if stage_name == "save_dir":
                    continue
                if stage_name in stage_map:
                    save_depth_map_as_exr(
                        depth,
                        os.path.join(save_each_depth_dir, stage_map[stage_name]),
                    )

    # --- ステップ3: 点群への変換と統合 ---
    logging.info("")
    logging.info("=" * 80)
    logging.info("ステップ3: 点群への変換と統合")
    logging.info(f"統合対象: {len(all_optimized_depths)}個の深度マップ")
    logging.info("=" * 80)
    logging.info("")
    merged_pts_list, merged_cols_list = [], []
    merged_normals_list = []  # 法線リストを追加
    last_integ_pts, last_integ_cols = None, None
    last_integ_normals = None

    for idx in target_indices:
        if idx not in all_geometrically_filtered_depths:
            continue

        geometrically_filtered_depth = all_geometrically_filtered_depths[idx]
        ref_pose = all_poses[idx]
        li_rgb = all_images[idx]
        R_mat = ref_pose["R"]
        T_pos = ref_pose["T"]

        # 最適化された法線を取得（存在する場合）
        optimized_normal = all_optimized_normals.get(idx)

        # time.csvのパスを取得
        if idx in all_pairs_data:
            _, _, left_path, _, _ = all_pairs_data[idx]
            filename_stem = Path(left_path).stem
        else:
            filename_stem = f"{idx:04d}"
        csv_subdir = os.path.join(config.CSV_DIR, filename_stem)
        time_csv_path = os.path.join(csv_subdir, "time.csv")

        # 透視投影深度マップから直接ワールド座標の点群に変換（オルソ投影をスキップ）
        logging.info(f"[{filename_stem}] 点群への変換中...")
        pointcloud_start = time.time()
        if optimized_normal is not None:
            world_points, world_colors, world_normals = depth_estimator.depth_to_world(
                geometrically_filtered_depth,
                li_rgb,
                config.K,
                R_mat,
                T_pos,
                normal_map=optimized_normal,
            )
            merged_normals_list.append(world_normals)
        else:
            world_points, world_colors = depth_estimator.depth_to_world(
                geometrically_filtered_depth, li_rgb, config.K, R_mat, T_pos
            )
        merged_pts_list.append(world_points)
        merged_cols_list.append(world_colors)
        pointcloud_elapsed = time.time() - pointcloud_start
        total_times["pointcloud_generation"] += pointcloud_elapsed
        append_to_csv(time_csv_path, ["pointcloud", f"{pointcloud_elapsed:.6f}"])
        logging.info(
            f"[{filename_stem}] 点群変換完了 (経過時間: {pointcloud_elapsed:.2f}秒, 点群数: {len(world_points):,}点)"
        )

    # --- 全点群を一度だけ統合 ---
    if merged_pts_list:
        logging.info("")
        logging.info("-" * 80)
        logging.info("全点群を統合中...")
        logging.info(f"統合対象: {len(merged_pts_list)}個の点群")
        logging.info("-" * 80)
        integ_start = time.time()
        # 法線リストが存在する場合のみ統合に含める
        normals_list_for_integration = (
            merged_normals_list if merged_normals_list else None
        )
        if normals_list_for_integration:
            integ_pts, integ_cols, integ_normals = (
                point_cloud_integrator.integrate_depth_maps_median(
                    merged_pts_list,
                    merged_cols_list,
                    normals_list=normals_list_for_integration,
                    voxel_size=0.1,
                )
            )
            last_integ_normals = integ_normals
        else:
            integ_pts, integ_cols = point_cloud_integrator.integrate_depth_maps_median(
                merged_pts_list, merged_cols_list, voxel_size=0.1
            )
        integ_elapsed = time.time() - integ_start
        total_times["pointcloud_integration"] += integ_elapsed
        logging.info(
            f"点群統合完了 (経過時間: {integ_elapsed:.2f}秒, 統合点数: {len(integ_pts):,}点)"
        )
        last_integ_pts, last_integ_cols = integ_pts, integ_cols

    # --- 最終保存 ---
    logging.info("")
    logging.info("=" * 80)
    logging.info("最終ステップ: 統合された点群を保存")
    logging.info("=" * 80)
    logging.info("")
    if merged_pts_list:
        merged_pts = (
            last_integ_pts if last_integ_pts is not None else np.vstack(merged_pts_list)
        )
        merged_cols = (
            last_integ_cols
            if last_integ_cols is not None
            else np.vstack(merged_cols_list)
        )

        # 複数ビュー可視性フィルタリング（デフォルト: 有効）
        filter_start = time.time()
        if getattr(config, "MULTI_VIEW_VISIBILITY_FILTER_ENABLED", True):
            logging.info("-" * 80)
            logging.info("複数ビュー可視性フィルタリングを適用中...")
            logging.info("-" * 80)
            visibility_threshold = getattr(config, "MULTI_VIEW_VISIBILITY_THRESHOLD", 2)
            geometric_error_threshold = getattr(
                config, "MULTI_VIEW_GEOMETRIC_ERROR_THRESHOLD", 0.05
            )
            # フィルタリング前の点群をバックアップ
            original_pts = merged_pts.copy()
            original_cols = merged_cols.copy()

            (
                merged_pts,
                merged_cols,
            ) = point_cloud_integrator.filter_points_by_multi_view_visibility(
                merged_pts,
                merged_cols,
                all_poses,
                all_geometrically_filtered_depths,
                visibility_threshold=visibility_threshold,
                geometric_error_threshold=geometric_error_threshold,
            )

            # フィルタリング後にポイントが0になった場合、フィルタリング前の点群を使用
            if len(merged_pts) == 0:
                logging.warning(
                    "Multi-view visibility filtering removed all points. Using unfiltered point cloud."
                )
                merged_pts = original_pts
                merged_cols = original_cols

        # 統合された法線がある場合は使用、なければNone
        merged_normals = None
        if last_integ_normals is not None:
            merged_normals = last_integ_normals
        elif merged_normals_list:
            # 統合されていない場合は結合
            merged_normals = np.vstack(merged_normals_list)

        final_pcd = point_cloud_integrator.process_and_save_final_point_cloud(
            merged_pts,
            merged_cols,
            config.POINT_CLOUD_FILE_PATH,
            normals_list=merged_normals,
        )
        filter_elapsed = time.time() - filter_start
        total_times["pointcloud_filtering"] += filter_elapsed
        if final_pcd and len(final_pcd.points) > 0:
            # 点群表示の制御（デフォルトは表示しない）
            show_point_cloud = getattr(config, "SHOW_POINT_CLOUD", False)
            if show_point_cloud:
                logging.info(
                    "Showing final integrated point cloud. Close the window to exit."
                )
                o3d.visualization.draw_geometries([final_pcd])
            # 保存完了のログは write_ply 内で出力されるため、ここでは出力しない
    else:
        logging.warning("No point clouds were generated.")

    end_time = time.time()
    total_elapsed = end_time - start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = total_elapsed % 60

    if hours > 0:
        time_str = f"{hours}時間{minutes}分{seconds:.1f}秒"
    elif minutes > 0:
        time_str = f"{minutes}分{seconds:.1f}秒"
    else:
        time_str = f"{seconds:.1f}秒"

    logging.info("")
    logging.info("=" * 80)
    logging.info(f"全体処理完了 (総経過時間: {time_str} / {total_elapsed:.2f}秒)")
    logging.info("=" * 80)

    # 各ステップの処理時間の集計を表示
    logging.info("")
    logging.info("=" * 80)
    logging.info("各ステップの処理時間集計（全画像合計）")
    logging.info("=" * 80)
    step_names = {
        "disparity_generation": "視差画像の生成",
        "disparity_to_depth": "視差から深度への変換",
        "depth_refinement": "深度画像の改善",
        "photometric_filtering": "Photometric Consistencyフィルタリング",
        "geometric_filtering": "Geometric Consistencyフィルタリング",
        "pointcloud_generation": "三次元点群の生成",
        "pointcloud_integration": "三次元点群の統合",
        "pointcloud_filtering": "点群のフィルタリング",
    }

    total_measured_time = sum(total_times.values())
    for key, name in step_names.items():
        elapsed = total_times[key]
        percentage = (
            (elapsed / total_measured_time * 100) if total_measured_time > 0 else 0.0
        )
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        if hours > 0:
            time_str = f"{hours}時間{minutes}分{seconds:.1f}秒"
        elif minutes > 0:
            time_str = f"{minutes}分{seconds:.1f}秒"
        else:
            time_str = f"{seconds:.2f}秒"
        logging.info(f"  {name}: {time_str} ({elapsed:.2f}秒, {percentage:.1f}%)")

    logging.info("-" * 80)
    total_hours = int(total_measured_time // 3600)
    total_minutes = int((total_measured_time % 3600) // 60)
    total_seconds = total_measured_time % 60
    if total_hours > 0:
        total_time_str = f"{total_hours}時間{total_minutes}分{total_seconds:.1f}秒"
    elif total_minutes > 0:
        total_time_str = f"{total_minutes}分{total_seconds:.1f}秒"
    else:
        total_time_str = f"{total_seconds:.1f}秒"
    logging.info(f"  合計（測定対象）: {total_time_str} ({total_measured_time:.2f}秒)")
    logging.info("=" * 80)

    # DEBUG_SAVE_DEPTH_MAPSがtrueの場合は評価コマンドを表示
    if config.DEBUG_SAVE_DEPTH_MAPS:
        logging.info("")
        logging.info("=" * 80)
        logging.info("深度画像の評価を実行するには、以下のコマンドを実行してください:")
        logging.info("=" * 80)
        pred_dir = config.DEPTH_IMAGE_DIR
        gt_dir = getattr(config, "LABEL_DEPTH_IMAGE_DIR", "")
        if gt_dir:
            eval_cmd = (
                f"python3 evaluation/depth_evaluate.py "
                f"--pred-dir {pred_dir} "
                f"--gt-dir {gt_dir}"
            )
            logging.info(eval_cmd)
        else:
            logging.warning(
                "LABEL_DEPTH_IMAGE_DIRが設定されていないため、評価コマンドを生成できません。"
            )
        logging.info("=" * 80)
        logging.info("")

    # 点群が生成された場合は点群評価コマンドを表示
    if merged_pts_list and os.path.exists(config.POINT_CLOUD_FILE_PATH):
        logging.info("")
        logging.info("=" * 80)
        logging.info("点群の評価を実行するには、以下のコマンドを実行してください:")
        logging.info("=" * 80)
        pred_pointcloud = config.POINT_CLOUD_FILE_PATH
        # 真値メッシュのパスは設定から取得できないため、プレースホルダーとして表示
        eval_cmd = (
            f"python3 evaluation/pointcloud_evaluation.py "
            f"--pred_pointcloud {pred_pointcloud} "
            f"--gt_mesh <真値メッシュのパス>"
        )
        logging.info(eval_cmd)
        logging.info("=" * 80)
        logging.info("")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
