# mvs/main.py

import bisect
import csv
import logging
import os
import time
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
    if not all_pairs_data:
        logging.error(
            "No valid image pairs found. Check images under images/image_0 & image_1 and the txt/drone_image_log.txt."
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
    # 各ステージの深度マップを保存（エラーマップの統一スケール用）
    all_stage_depths = {}
    # 各画像のポーズ情報を保存（幾何学的一貫性フィルタリング用）
    all_poses = {}
    all_images = {}
    all_gt_depths = {}

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

        # ファイル名（拡張子なし）を取得してフォルダ名に使用
        filename_stem = Path(left_path).stem

        # 処理開始のログ（区切り線付き）
        logging.info("=" * 80)
        logging.info(f"処理開始: 画像ペア {idx} (ファイル: {filename_stem})")
        logging.info(f"  左画像: {left_path}")
        logging.info(f"  右画像: {right_path}")
        logging.info("=" * 80)

        view_metrics = {"image_index": idx}
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
        logging.info(f"Initialized time.csv at {time_csv_path}")

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
                # GT depthのPNG保存（clear_folderの後に保存するため、ここで実行）
                if (
                    bool(getattr(config, "DEBUG_SAVE_GT_DEPTH_MAPS", True))
                    and config.DEBUG_SAVE_DEPTH_MAPS
                ):
                    gt_png_path = os.path.join(
                        save_each_depth_dir, f"gt_depth_{idx:04d}.png"
                    )
                    save_depth_map_as_image(gt_depth, gt_png_path)
                    logging.info(f"Saved GT depth map to {gt_png_path}")

        try:
            li_bgr = cv2.imread(left_path)
            ri_bgr = cv2.imread(right_path)
            if li_bgr is None or ri_bgr is None:
                continue

            li_rgb = loaded_images[idx]
            li_gray = cv2.cvtColor(li_bgr, cv2.COLOR_BGR2GRAY)
            ri_gray = cv2.cvtColor(ri_bgr, cv2.COLOR_BGR2GRAY)

            # 初期深度マップと深度誤差コストを計算
            logging.info("-" * 80)
            logging.info(f"[{filename_stem}] ステップ1: 初期深度マップの計算")
            logging.info("-" * 80)
            start_time_initial_depth = time.time()
            disp = image_processor.create_disparity(li_gray, ri_gray)
            initial_depth = depth_estimator.disparity_to_depth(disp)
            end_time_initial_depth = time.time()
            elapsed_initial = end_time_initial_depth - start_time_initial_depth
            logging.info(
                f"[{filename_stem}] 初期深度計算完了 (経過時間: {elapsed_initial:.2f}秒)"
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

            # 初期深度の有効ピクセル数をログ出力
            valid_pixels_initial = np.sum(np.isfinite(initial_depth))
            logging.info(f"[Initial Depth] Valid pixels: {valid_pixels_initial}")

            # PatchMatchによる深度マップの最適化
            neighbor_views_data = []
            neighbor_frames = _neighbors_for_ref(idx)
            _log_selected_neighbors(idx, neighbor_frames)
            for fr in neighbor_frames:
                nv = _get_neighbor_view(idx, fr)
                if nv is not None:
                    neighbor_views_data.append(nv)

            # PatchMatchを実行（全体計測とイテレーション内計測は関数側で行う）
            logging.info("-" * 80)
            logging.info(f"[{filename_stem}] ステップ2: PatchMatch MVS深度最適化")
            logging.info("-" * 80)
            refine_start = time.time()
            optimized_depth, iter_times_gpu = (
                depth_optimization.refine_depth_with_patchmatch(
                    initial_depth=initial_depth,
                    initial_depth_error=d_cost,
                    ref_image=li_rgb,
                    ref_pose={"R": R_mat, "T": T_pos, "K": config.K},
                    neighbor_views_data=neighbor_views_data,
                    gt_depth=gt_depth,
                    ref_idx=idx,
                    filename_stem=filename_stem,
                )
            )
            refine_elapsed = time.time() - refine_start
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
                    {"R": R_mat, "T": T_pos, "K": config.K},
                    neighbor_views_data,
                )
            )
            photo_elapsed = time.time() - photo_start
            logging.info(
                f"[{filename_stem}] 光度フィルタリング完了 (経過時間: {photo_elapsed:.2f}秒)"
            )
            append_to_csv(time_csv_path, ["photometric", f"{photo_elapsed:.6f}"])
            logging.debug(
                f"[{filename_stem}] 光度フィルタリング時間をtime.csvに保存しました: {time_csv_path}"
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
            logging.info(f"[{filename_stem}] 深度マップを保存しました。")
            logging.info("=" * 80)
            logging.info(f"処理完了: 画像ペア {idx} (ファイル: {filename_stem})")
            logging.info("=" * 80)
            logging.info("")

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

            #             os.path.join(
            #                 config.DEPTH_IMAGE_DIR,
            #                 f"depth_{idx:04d}",
            #                 "fused_ortho_running.png",
            #             ),
            #             swap_axes=True,
            #             flip_y=True,
            #             flip_x=True,
            #         )

        except Exception as e:
            logging.error(f"Error in Step 1 for image pair {idx}: {e}", exc_info=True)

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
    last_integ_pts, last_integ_cols = None, None

    for idx in target_indices:
        if idx not in all_geometrically_filtered_depths:
            continue

        geometrically_filtered_depth = all_geometrically_filtered_depths[idx]
        ref_pose = all_poses[idx]
        li_rgb = all_images[idx]
        R_mat = ref_pose["R"]
        T_pos = ref_pose["T"]

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
        world_points, world_colors = depth_estimator.depth_to_world(
            geometrically_filtered_depth, li_rgb, config.K, R_mat, T_pos
        )
        merged_pts_list.append(world_points)
        merged_cols_list.append(world_colors)
        pointcloud_elapsed = time.time() - pointcloud_start
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
        integ_pts, integ_cols = point_cloud_integrator.integrate_depth_maps_median(
            merged_pts_list, merged_cols_list, voxel_size=0.1
        )
        integ_elapsed = time.time() - integ_start
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

            # フィルタリング後にポイントが0になった場合、フィルタリング前の点群を使用
            if len(merged_pts) == 0:
                logging.warning(
                    "Multi-view visibility filtering removed all points. Using unfiltered point cloud."
                )
                merged_pts = original_pts
                merged_cols = original_cols

        final_pcd = point_cloud_integrator.process_and_save_final_point_cloud(
            merged_pts, merged_cols, config.POINT_CLOUD_FILE_PATH
        )
        if final_pcd and len(final_pcd.points) > 0:
            # 点群表示の制御（デフォルトは表示しない）
            show_point_cloud = getattr(config, "SHOW_POINT_CLOUD", False)
            if show_point_cloud:
                logging.info(
                    "Showing final integrated point cloud. Close the window to exit."
                )
                o3d.visualization.draw_geometries([final_pcd])
            else:
                logging.info(
                    f"Point cloud saved to {config.POINT_CLOUD_FILE_PATH} (display disabled)"
                )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
