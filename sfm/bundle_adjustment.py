"""
バンドル調整モジュール。
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mvs.config as config
except Exception:
    config = None


# =============================================================================
# PyTorch ユーティリティ（投影・回転変換）
# =============================================================================


def _axis_angle_to_rotation_matrix_torch(axis_angle):
    """
    Rodrigues の公式により、軸角表現を回転行列に変換する。

    Args:
        axis_angle: (N, 3) 回転ベクトル（向き=回転軸、ノルム=回転角 [rad]）

    Returns:
        R: (N, 3, 3) 回転行列
    """
    angle = torch.norm(axis_angle, dim=1, keepdim=True)
    angle = torch.clamp(angle, min=1e-8)
    axis = axis_angle / angle

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one_minus_cos = 1.0 - cos

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    R = torch.zeros(
        (axis_angle.shape[0], 3, 3), device=axis_angle.device, dtype=axis_angle.dtype
    )

    R[:, 0, 0] = cos[:, 0] + x * x * one_minus_cos[:, 0]
    R[:, 0, 1] = x * y * one_minus_cos[:, 0] - z * sin[:, 0]
    R[:, 0, 2] = x * z * one_minus_cos[:, 0] + y * sin[:, 0]

    R[:, 1, 0] = y * x * one_minus_cos[:, 0] + z * sin[:, 0]
    R[:, 1, 1] = cos[:, 0] + y * y * one_minus_cos[:, 0]
    R[:, 1, 2] = y * z * one_minus_cos[:, 0] - x * sin[:, 0]

    R[:, 2, 0] = z * x * one_minus_cos[:, 0] - y * sin[:, 0]
    R[:, 2, 1] = z * y * one_minus_cos[:, 0] + x * sin[:, 0]
    R[:, 2, 2] = cos[:, 0] + z * z * one_minus_cos[:, 0]

    return R


def _project_points_torch(points_3d, rvecs, tvecs, K, camera_indices, point_indices):
    """
    3D点を各カメラの2D画像座標へ投影する。
    """
    # 観測に対応するパラメータを抽出
    r_obs = rvecs[camera_indices]  # (N, 3)
    t_obs = tvecs[camera_indices]  # (N, 3)
    X_obs = points_3d[point_indices]  # (N, 3)

    # 回転行列への変換
    R_obs = _axis_angle_to_rotation_matrix_torch(r_obs)  # (N, 3, 3)

    # カメラ座標系へ変換: X_cam = R * X + t
    # (N, 3, 3) @ (N, 3, 1) + (N, 3, 1) -> (N, 3, 1)
    X_obs_unsqueezed = X_obs.unsqueeze(-1)
    t_obs_unsqueezed = t_obs.unsqueeze(-1)

    X_cam = torch.bmm(R_obs, X_obs_unsqueezed) + t_obs_unsqueezed
    X_cam = X_cam.squeeze(-1)  # (N, 3)

    # 深度による正規化
    z = X_cam[:, 2:3]
    # カメラ後方の点は無視できないが、数値安定性のためイプシロン処理
    # (実際にはHuberLossが外れ値として処理してくれることを期待)
    z = torch.where(z < 1e-6, torch.tensor(1e-6, device=z.device), z)

    x_norm = X_cam[:, 0:1] / z
    y_norm = X_cam[:, 1:2] / z

    # 内部パラメータの適用
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * x_norm + cx
    v = fy * y_norm + cy

    return torch.cat([u, v], dim=1)


def _optimize_bundle_adjustment_gpu(
    camera_params: Dict[int, Tuple[np.ndarray, np.ndarray]],
    points_3d: np.ndarray,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    sorted_indices: List[str],
    fixed_cam_idx: int,
    cam_to_idx: Dict[int, int],
    n_iterations: int = 100,
    learning_rate: float = 1e-3,
    huber_delta: float = 2.0,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 10,
    log_every: int = 20,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], np.ndarray, float, float]:
    """
    PyTorch を用いてバンドル調整を実行する。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[SfM] Running Bundle Adjustment on {device} (lr={learning_rate})...")

    # --- データの準備 ---
    n_cams = len(sorted_indices)

    # パラメータの初期値をTensor化
    init_rvecs_np = np.zeros((n_cams, 3), dtype=np.float32)
    init_tvecs_np = np.zeros((n_cams, 3), dtype=np.float32)

    for ds_id in sorted_indices:
        ds_id_int = int(ds_id)
        idx = cam_to_idx[ds_id_int]
        r, t = camera_params[ds_id_int]
        init_rvecs_np[idx] = r
        init_tvecs_np[idx] = t

    # 固定カメラのインデックス
    fixed_idx = cam_to_idx[fixed_cam_idx]

    # --- パラメータ設定 ---

    # 全カメラのパラメータを作成
    t_rvecs = torch.tensor(init_rvecs_np, device=device, dtype=torch.float32)
    t_tvecs = torch.tensor(init_tvecs_np, device=device, dtype=torch.float32)

    t_rvecs.requires_grad = True
    t_tvecs.requires_grad = True
    t_points_3d = torch.tensor(
        points_3d, device=device, dtype=torch.float32, requires_grad=True
    )

    t_K = torch.tensor(K, device=device, dtype=torch.float32)

    mapped_camera_indices = np.array([cam_to_idx[int(c)] for c in camera_indices])
    t_camera_indices = torch.tensor(
        mapped_camera_indices, device=device, dtype=torch.long
    )
    t_point_indices = torch.tensor(point_indices, device=device, dtype=torch.long)
    t_observations = torch.tensor(points_2d, device=device, dtype=torch.float32)

    # --- 最適化 ---
    optimizer = torch.optim.Adam([t_rvecs, t_tvecs, t_points_3d], lr=learning_rate)

    # 学習率スケジューラ (停滞したら下げる)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

    initial_loss = 0.0
    best_loss = float("inf")

    # 最良の結果を保持する変数
    best_rvecs = init_rvecs_np.copy()
    best_tvecs = init_tvecs_np.copy()

    # 固定値を保存（念のため毎ステップ書き戻す用）
    fixed_rvec = t_rvecs[fixed_idx].clone().detach()
    fixed_tvec = t_tvecs[fixed_idx].clone().detach()

    for i in range(n_iterations):
        optimizer.zero_grad()

        # 投影
        projections = _project_points_torch(
            t_points_3d, t_rvecs, t_tvecs, t_K, t_camera_indices, t_point_indices
        )

        # 損失計算 (Huber Loss = Robust Loss)
        # deltaを小さくすることで、外れ値（大きくズレた点）の影響をより抑える
        loss = F.huber_loss(
            projections, t_observations, delta=huber_delta, reduction="mean"
        )

        # RMSE計算 (評価用)
        with torch.no_grad():
            rmse = torch.sqrt(
                torch.mean(torch.sum((projections - t_observations) ** 2, dim=1))
            )
            rmse_val = rmse.item()
            if i == 0:
                initial_loss = rmse_val
                best_loss = rmse_val
                # 初期状態が悪化しないよう、初期値をbestとして保存
                best_rvecs = t_rvecs.detach().cpu().numpy()
                best_tvecs = t_tvecs.detach().cpu().numpy()

            if rmse_val < best_loss:
                best_loss = rmse_val
                best_rvecs = t_rvecs.detach().cpu().numpy()
                best_tvecs = t_tvecs.detach().cpu().numpy()

        loss.backward()

        t_rvecs.grad[fixed_idx] = 0.0
        t_tvecs.grad[fixed_idx] = 0.0

        optimizer.step()

        with torch.no_grad():
            t_rvecs[fixed_idx] = fixed_rvec
            t_tvecs[fixed_idx] = fixed_tvec

        scheduler.step(rmse_val)

        if log_every > 0 and (i % log_every == 0 or i == n_iterations - 1):
            logging.info(
                f"[SfM-GPU] Iter {i+1}/{n_iterations}, RMSE: {rmse_val:.4f} (Best: {best_loss:.4f})"
            )

    # --- 結果の判定と書き出し ---

    if best_loss > initial_loss * 1.1:  # 10%以上悪化した場合は警告
        logging.warning(
            f"[SfM] Optimization diverged! (Init: {initial_loss:.4f} -> Best: {best_loss:.4f}). Reverting to initial guess."
        )
        refined_rvecs = init_rvecs_np
        refined_tvecs = init_tvecs_np
        final_loss_ret = initial_loss
    else:
        refined_rvecs = best_rvecs
        refined_tvecs = best_tvecs
        final_loss_ret = best_loss

    refined_camera_params = {}
    for ds_id in sorted_indices:
        ds_id_int = int(ds_id)
        idx = cam_to_idx[ds_id_int]
        refined_camera_params[ds_id_int] = (refined_rvecs[idx], refined_tvecs[idx])

    return refined_camera_params, None, initial_loss, final_loss_ret


# =============================================================================
# 前処理（特徴点抽出・マッチング・三角測量）
# =============================================================================


def _get_feature_detector():
    """SIFT が利用可能なら SIFT、そうでなければ ORB の検出器を返す。"""
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(), "SIFT"
    return cv2.ORB_create(nfeatures=5000), "ORB"


def _create_matcher(det_name: str) -> cv2.BFMatcher:
    """検出器名に応じた BFMatcher（SIFT: L2, ORB: Hamming）を返す。"""
    if det_name == "SIFT":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _extract_features(
    detector, img: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
        kpts_xy: (N, 2) float64
        desc: (N, D) or None
    """
    kps, desc = detector.detectAndCompute(img, None)
    if not kps or desc is None:
        return np.empty((0, 2), dtype=np.float64), None
    kpts_xy = np.array([kp.pt for kp in kps], dtype=np.float64)
    return kpts_xy, desc


def _match_descriptors_mutual_ransac(
    matcher: cv2.BFMatcher,
    kpts1_xy: np.ndarray,
    desc1: Optional[np.ndarray],
    kpts2_xy: np.ndarray,
    desc2: Optional[np.ndarray],
    max_matches: int,
    ratio: float,
    ransac_reproj_threshold: float = 1.0,
    ransac_confidence: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2画像間で特徴点マッチングを行い、幾何的に妥当な対応点のインデックスを返す。
    """
    if desc1 is None or desc2 is None or kpts1_xy.shape[0] < 8 or kpts2_xy.shape[0] < 8:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    raw_f = matcher.knnMatch(desc1, desc2, k=2)
    good_f = []
    for m_n in raw_f:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good_f.append(m)
    if not good_f:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    good_f.sort(key=lambda m: m.distance)
    if max_matches > 0:
        good_f = good_f[:max_matches]

    # reverse for mutual check
    raw_r = matcher.knnMatch(desc2, desc1, k=2)
    best_r: Dict[int, Tuple[float, int]] = {}
    for m_n in raw_r:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            prev = best_r.get(m.queryIdx)
            if prev is None or m.distance < prev[0]:
                best_r[m.queryIdx] = (float(m.distance), int(m.trainIdx))

    mutual = []
    for m in good_f:
        rev = best_r.get(int(m.trainIdx))
        if rev is None:
            continue
        _, rev_train = rev
        if int(rev_train) == int(m.queryIdx):
            mutual.append(m)

    if len(mutual) < 8:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    idx1 = np.array([m.queryIdx for m in mutual], dtype=np.int32)
    idx2 = np.array([m.trainIdx for m in mutual], dtype=np.int32)
    pts1 = kpts1_xy[idx1]
    pts2 = kpts2_xy[idx2]

    F, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, ransac_reproj_threshold, ransac_confidence
    )
    if F is None or mask is None:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    mask = mask.ravel().astype(bool)
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    return idx1, idx2


def _match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    max_matches: int,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2画像間でマッチした特徴点の座標を返す。
    """
    detector, det_name = _get_feature_detector()
    matcher = _create_matcher(det_name)
    kpts1_xy, d1 = _extract_features(detector, img1)
    kpts2_xy, d2 = _extract_features(detector, img2)
    idx1, idx2 = _match_descriptors_mutual_ransac(
        matcher=matcher,
        kpts1_xy=kpts1_xy,
        desc1=d1,
        kpts2_xy=kpts2_xy,
        desc2=d2,
        max_matches=max_matches,
        ratio=ratio,
    )
    if idx1.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return kpts1_xy[idx1], kpts2_xy[idx2]


def _triangulate_points(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2視点の対応点から三角測量により3D座標を計算する。
    両カメラの前方（z > 0）にあり、有限値である点のみを返す。
    """
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    pts1_h = pts1.T
    pts2_h = pts2.T
    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    X = (X_h[:3] / X_h[3:4]).T

    if X.size == 0:
        return np.empty((0, 3), dtype=np.float64), pts1, pts2

    z1 = (R1 @ X.T + t1.reshape(3, 1))[2]
    z2 = (R2 @ X.T + t2.reshape(3, 1))[2]
    mask = np.isfinite(X).all(axis=1) & (z1 > 0) & (z2 > 0)
    return X[mask], pts1[mask], pts2[mask]


def run_bundle_adjustment(
    all_pairs_data: dict,
    K: np.ndarray,
) -> dict:
    """
    GPU (PyTorch) Accelerated Bundle Adjustment
    """
    if not all_pairs_data or len(all_pairs_data) < 2:
        logging.info("[SfM] Not enough camera pairs for bundle adjustment.")
        return all_pairs_data

    # PyTorchが使えない場合は警告を出してそのまま返す（またはScipy版へフォールバック）
    if not TORCH_AVAILABLE:
        logging.error(
            "[SfM] PyTorch is not available. Please install torch to use GPU Bundle Adjustment."
        )
        return all_pairs_data

    # ノイズスケールが0の場合は BA をスキップ（初期位置が既に正確な場合）
    if config is not None:
        pos_scale = float(getattr(config, "POSITION_ERROR_SCALE", 0.0) or 0.0)
        rot_scale = float(getattr(config, "ROTATION_ERROR_SCALE", 0.0) or 0.0)
        if pos_scale <= 0.0 and rot_scale <= 0.0:
            logging.info("[SfM] Bundle adjustment skipped (noise scale is 0).")
            return all_pairs_data

    def _cfg(name: str, default):
        if config is None:
            return default
        return getattr(config, name, default)

    max_matches = int(_cfg("SFM_MAX_MATCHES", 2000) or 2000)
    max_points_per_pair = int(_cfg("SFM_MAX_POINTS_PER_PAIR", 500) or 500)
    ratio = float(_cfg("SFM_MATCH_RATIO", 0.75) or 0.75)

    sorted_indices = sorted(all_pairs_data.keys())
    # マッピング作成 (DatasetID -> 0..N)
    cam_to_idx = {int(ds_id): i for i, ds_id in enumerate(sorted_indices)}

    image_cache = {}

    def _read_gray(path: str) -> Optional[np.ndarray]:
        if path in image_cache:
            return image_cache[path]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        image_cache[path] = img
        return img

    # 各フレームのカメラパラメータ（rvec, tvec）を all_pairs_data から取得
    camera_params = {}
    for idx in sorted_indices:
        _, t_cv, _, _, R_cv = all_pairs_data[idx]
        rvec, _ = cv2.Rodrigues(np.asarray(R_cv, dtype=np.float64))
        camera_params[int(idx)] = (
            rvec.reshape(3),
            np.asarray(t_cv, dtype=np.float64).reshape(3),
        )

    # 隣接フレームペアごとに特徴点マッチング→三角測量→観測リスト構築
    points_3d_list: List[np.ndarray] = []
    points_2d_list: List[np.ndarray] = []
    camera_indices: List[int] = []
    point_indices: List[int] = []

    point_offset = 0
    t_start = time.time()
    logging.info(
        "[SfM] Building observations from %d adjacent pairs...",
        max(0, len(sorted_indices) - 1),
    )

    for i in range(len(sorted_indices) - 1):
        idx1 = int(sorted_indices[i])
        idx2 = int(sorted_indices[i + 1])
        _, t1, left1, _, R1 = all_pairs_data[idx1]
        _, t2, left2, _, R2 = all_pairs_data[idx2]
        img1 = _read_gray(left1)
        img2 = _read_gray(left2)
        if img1 is None or img2 is None:
            continue

        pts1, pts2 = _match_features(img1, img2, max_matches=max_matches, ratio=ratio)
        if len(pts1) < 8:
            continue

        X, pts1_f, pts2_f = _triangulate_points(
            K.astype(np.float64),
            np.asarray(R1, dtype=np.float64),
            np.asarray(t1, dtype=np.float64),
            np.asarray(R2, dtype=np.float64),
            np.asarray(t2, dtype=np.float64),
            pts1,
            pts2,
        )
        if X.shape[0] == 0:
            continue

        if max_points_per_pair > 0 and X.shape[0] > max_points_per_pair:
            # ランダムに間引く
            choice = np.random.choice(X.shape[0], max_points_per_pair, replace=False)
            X = X[choice]
            pts1_f = pts1_f[choice]
            pts2_f = pts2_f[choice]

        points_3d_list.append(X)
        points_2d_list.append(pts1_f)
        points_2d_list.append(pts2_f)

        # ここでの camera_indices は Dataset ID
        camera_indices.extend([idx1] * len(X))
        camera_indices.extend([idx2] * len(X))
        point_indices.extend(list(range(point_offset, point_offset + len(X))))
        point_indices.extend(list(range(point_offset, point_offset + len(X))))
        point_offset += len(X)

    if not points_3d_list:
        logging.warning("[SfM] No valid 3D points for bundle adjustment.")
        return all_pairs_data

    points_3d = np.vstack(points_3d_list)
    points_2d = np.vstack(points_2d_list)
    camera_indices_np = np.asarray(camera_indices, dtype=np.int32)
    point_indices_np = np.asarray(point_indices, dtype=np.int32)

    logging.info(
        "[SfM] Observations built: points=%d, observations=%d (%.2fs)",
        points_3d.shape[0],
        points_2d.shape[0],
        time.time() - t_start,
    )
    logging.info("")

    # 最初のフレームを固定し、PyTorch で最適化を実行
    fixed_cam_idx = int(sorted_indices[0])

    # 最適化パラメータを設定ファイルから読み込み
    n_iterations = int(_cfg("SFM_BA_ITERATIONS", 100) or 100)
    learning_rate = float(_cfg("SFM_BA_LEARNING_RATE", 0.001) or 0.001)
    huber_delta = float(_cfg("SFM_BA_HUBER_DELTA", 2.0) or 2.0)
    scheduler_factor = float(_cfg("SFM_BA_SCHEDULER_FACTOR", 0.5) or 0.5)
    scheduler_patience = int(_cfg("SFM_BA_SCHEDULER_PATIENCE", 10) or 10)
    log_every = int(_cfg("SFM_BA_LOG_EVERY", 20) or 20)

    logging.info(
        "[SfM] Bundle Adjustment parameters: iterations=%d, lr=%.4f, huber_delta=%.2f, "
        "scheduler_factor=%.2f, scheduler_patience=%d",
        n_iterations,
        learning_rate,
        huber_delta,
        scheduler_factor,
        scheduler_patience,
    )

    refined_params, _, init_rmse, final_rmse = _optimize_bundle_adjustment_gpu(
        camera_params=camera_params,
        points_3d=points_3d,
        camera_indices=camera_indices_np,
        point_indices=point_indices_np,
        points_2d=points_2d,
        K=K,
        sorted_indices=sorted_indices,
        fixed_cam_idx=fixed_cam_idx,
        cam_to_idx=cam_to_idx,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        huber_delta=huber_delta,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        log_every=log_every,
    )

    # 最適化前後のカメラ中心の移動量をログ出力用に計算
    corrections = []
    for idx in sorted_indices:
        rvec0, tvec0 = camera_params[int(idx)]
        rvec1, tvec1 = refined_params[int(idx)]
        R0, _ = cv2.Rodrigues(rvec0)
        R1, _ = cv2.Rodrigues(rvec1)
        C0 = -R0.T @ tvec0.reshape(3, 1)
        C1 = -R1.T @ tvec1.reshape(3, 1)
        corrections.append(float(np.linalg.norm(C1 - C0)))
    avg_correction = float(np.mean(corrections)) if corrections else 0.0

    logging.info("")
    logging.info("=" * 80)
    logging.info("[SfM] Bundle Adjustment Report:")
    logging.info(f"  - Initial RMSE: {init_rmse:.4f} pixels")
    logging.info(
        f"  - Final RMSE:   {final_rmse:.4f} pixels "
        f"(Improved by {max(0.0, init_rmse - final_rmse):.4f} pixels)"
    )
    logging.info(f"  - Average Camera Correction: {avg_correction:.4f} meters")
    logging.info("=" * 80)

    # 最適化後のカメラパラメータで all_pairs_data を更新して返す
    refined_pairs_data = {}
    for idx in sorted_indices:
        idx_int = int(idx)
        _, _, left_path, right_path, _ = all_pairs_data[idx_int]
        rvec, tvec = refined_params[idx_int]
        R_cv, _ = cv2.Rodrigues(rvec)
        refined_pairs_data[idx_int] = (
            idx_int,
            tvec.astype(np.float32),
            left_path,
            right_path,
            R_cv.astype(np.float32),
        )

    return refined_pairs_data
