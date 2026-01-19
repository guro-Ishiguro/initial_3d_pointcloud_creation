import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import mvs.config as config
except Exception:  # pragma: no cover - config is optional for standalone use
    config = None


def _get_feature_detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(), "SIFT"
    return cv2.ORB_create(nfeatures=5000), "ORB"


def _match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    max_matches: int,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    detector, det_name = _get_feature_detector()
    k1, d1 = detector.detectAndCompute(img1, None)
    k2, d2 = detector.detectAndCompute(img2, None)

    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    if det_name == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    raw = matcher.knnMatch(d1, d2, k=2)
    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            good.append(m)

    if not good:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    good.sort(key=lambda m: m.distance)
    if max_matches > 0:
        good = good[:max_matches]

    pts1 = np.array([k1[m.queryIdx].pt for m in good], dtype=np.float64)
    pts2 = np.array([k2[m.trainIdx].pt for m in good], dtype=np.float64)

    if len(pts1) < 8:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    # RANSAC-based filtering with fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None or mask is None:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    mask = mask.ravel().astype(bool)
    pts1 = pts1[mask]
    pts2 = pts2[mask]
    return pts1, pts2


def _triangulate_points(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    pts1_h = pts1.T
    pts2_h = pts2.T
    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    X = (X_h[:3] / X_h[3:4]).T

    if X.size == 0:
        return np.empty((0, 3), dtype=np.float64), pts1, pts2

    # Cheirality check
    z1 = (R1 @ X.T + t1.reshape(3, 1))[2]
    z2 = (R2 @ X.T + t2.reshape(3, 1))[2]
    mask = np.isfinite(X).all(axis=1) & (z1 > 0) & (z2 > 0)
    return X[mask], pts1[mask], pts2[mask]


def _project_points(
    X: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    X_cam = (R @ X.T + tvec.reshape(3, 1)).T
    z = X_cam[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x = X_cam[:, 0:1] / z
    y = X_cam[:, 1:2] / z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.hstack([u, v])


def _rodrigues_torch(rvec: torch.Tensor) -> torch.Tensor:
    """Batched Rodrigues: rvec (N,3) -> R (N,3,3)."""
    eps = 1e-9
    theta = torch.linalg.norm(rvec, dim=1, keepdim=True)  # (N,1)
    theta2 = theta * theta
    a = torch.where(theta > eps, torch.sin(theta) / theta, 1.0 - theta2 / 6.0)
    b = torch.where(
        theta > eps, (1.0 - torch.cos(theta)) / (theta2 + eps), 0.5 - theta2 / 24.0
    )

    rx, ry, rz = rvec[:, 0], rvec[:, 1], rvec[:, 2]
    zero = torch.zeros_like(rx)
    K = torch.stack(
        [
            torch.stack([zero, -rz, ry], dim=1),
            torch.stack([rz, zero, -rx], dim=1),
            torch.stack([-ry, rx, zero], dim=1),
        ],
        dim=1,
    )  # (N,3,3)
    K2 = torch.bmm(K, K)
    eye_matrix = torch.eye(3, device=rvec.device, dtype=rvec.dtype).unsqueeze(0)
    a = a.view(-1, 1, 1)
    b = b.view(-1, 1, 1)
    return eye_matrix + a * K + b * K2


def _project_points_torch(
    X: torch.Tensor,
    rvec: torch.Tensor,
    tvec: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Batched projection: X (N,3), rvec/tvec (N,3) -> uv (N,2)."""
    R = _rodrigues_torch(rvec)
    X_cam = torch.bmm(R, X.unsqueeze(2)).squeeze(2) + tvec
    z = X_cam[:, 2:3]
    z = torch.where(torch.abs(z) < 1e-9, torch.full_like(z, 1e-9), z)
    x = X_cam[:, 0:1] / z
    y = X_cam[:, 1:2] / z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return torch.cat([u, v], dim=1)


def _compute_rmse_torch(
    rvecs: torch.Tensor,
    tvecs: torch.Tensor,
    points_3d: torch.Tensor,
    camera_indices: torch.Tensor,
    point_indices: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
) -> float:
    if points_2d.numel() == 0:
        return 0.0
    with torch.no_grad():
        rvec_obs = rvecs[camera_indices]
        tvec_obs = tvecs[camera_indices]
        X_obs = points_3d[point_indices]
        proj = _project_points_torch(X_obs, rvec_obs, tvec_obs, K)
        residuals = proj - points_2d
        rmse = torch.sqrt(torch.mean(residuals**2))
    return float(rmse.item())


def run_bundle_adjustment(
    all_pairs_data: dict,
    K: np.ndarray,
) -> dict:
    """
    Args:
        all_pairs_data: ノイズが付加された初期ポーズを含む辞書
        K: カメラ内部パラメータ
    Returns:
        refined_pairs_data: バンドル調整により補正されたポーズを含む同形式の辞書
    """
    if not all_pairs_data or len(all_pairs_data) < 2:
        logging.info("[SfM] Not enough camera pairs for bundle adjustment.")
        return all_pairs_data

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

    max_matches = int(_cfg("SFM_MAX_MATCHES", 1500) or 1500)
    max_points_per_pair = int(_cfg("SFM_MAX_POINTS_PER_PAIR", 500) or 500)
    ratio = float(_cfg("SFM_MATCH_RATIO", 0.75) or 0.75)

    sorted_indices = sorted(all_pairs_data.keys())
    image_cache = {}

    def _read_gray(path: str) -> Optional[np.ndarray]:
        if path in image_cache:
            return image_cache[path]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        image_cache[path] = img
        return img

    # Build initial camera params (rvec, tvec) from all_pairs_data
    camera_params = {}
    for idx in sorted_indices:
        _, t_cv, _, _, R_cv = all_pairs_data[idx]
        rvec, _ = cv2.Rodrigues(np.asarray(R_cv, dtype=np.float64))
        camera_params[int(idx)] = (
            rvec.reshape(3),
            np.asarray(t_cv, dtype=np.float64).reshape(3),
        )

    points_3d_list: List[np.ndarray] = []
    points_2d_list: List[np.ndarray] = []
    camera_indices: List[int] = []
    point_indices: List[int] = []

    point_offset = 0
    t_start = time.time()
    logging.info(
        "[SfM] Building observations from %d adjacent pairs (max_matches=%d, max_points_per_pair=%d)",
        max(0, len(sorted_indices) - 1),
        max_matches,
        max_points_per_pair,
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
            X = X[:max_points_per_pair]
            pts1_f = pts1_f[:max_points_per_pair]
            pts2_f = pts2_f[:max_points_per_pair]

        start_idx = point_offset
        points_3d_list.append(X)
        points_2d_list.append(pts1_f)
        points_2d_list.append(pts2_f)
        camera_indices.extend([idx1] * len(X))
        camera_indices.extend([idx2] * len(X))
        point_indices.extend(list(range(start_idx, start_idx + len(X))))
        point_indices.extend(list(range(start_idx, start_idx + len(X))))
        point_offset += len(X)

        if (i + 1) % 5 == 0 or (i + 1) == (len(sorted_indices) - 1):
            logging.info(
                "[SfM] Pair %d/%d -> triangulated points=%d (total=%d)",
                i + 1,
                len(sorted_indices) - 1,
                len(X),
                point_offset,
            )

    if not points_3d_list:
        logging.warning("[SfM] No valid 3D points for bundle adjustment.")
        return all_pairs_data

    points_3d = np.vstack(points_3d_list)
    points_2d = np.vstack(points_2d_list)
    camera_indices = np.asarray(camera_indices, dtype=np.int32)
    point_indices = np.asarray(point_indices, dtype=np.int32)

    fixed_cam_idx = int(sorted_indices[0])
    idx_to_pos = {int(idx): i for i, idx in enumerate(sorted_indices)}
    var_cam_indices = [int(idx) for idx in sorted_indices if int(idx) != fixed_cam_idx]
    var_cam_positions = [idx_to_pos[idx] for idx in var_cam_indices]

    # Prepare torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    K_t = torch.tensor(K.astype(np.float32), device=device, dtype=dtype)

    n_cams = len(sorted_indices)
    base_rvecs = np.zeros((n_cams, 3), dtype=np.float32)
    base_tvecs = np.zeros((n_cams, 3), dtype=np.float32)
    for idx in sorted_indices:
        pos = idx_to_pos[int(idx)]
        rvec, tvec = camera_params[int(idx)]
        base_rvecs[pos] = rvec.astype(np.float32)
        base_tvecs[pos] = tvec.astype(np.float32)

    base_rvecs_t = torch.tensor(base_rvecs, device=device, dtype=dtype)
    base_tvecs_t = torch.tensor(base_tvecs, device=device, dtype=dtype)

    var_rvecs = (
        torch.tensor(base_rvecs[var_cam_positions], device=device, dtype=dtype)
        if var_cam_positions
        else torch.empty((0, 3), device=device, dtype=dtype)
    )
    var_tvecs = (
        torch.tensor(base_tvecs[var_cam_positions], device=device, dtype=dtype)
        if var_cam_positions
        else torch.empty((0, 3), device=device, dtype=dtype)
    )
    var_rvecs = var_rvecs.requires_grad_(True)
    var_tvecs = var_tvecs.requires_grad_(True)

    points_3d_t = torch.tensor(points_3d.astype(np.float32), device=device, dtype=dtype)
    points_3d_t = points_3d_t.requires_grad_(True)

    camera_indices_pos = torch.tensor(
        [idx_to_pos[int(i)] for i in camera_indices],
        device=device,
        dtype=torch.long,
    )
    point_indices_t = torch.tensor(point_indices, device=device, dtype=torch.long)
    points_2d_t = torch.tensor(points_2d.astype(np.float32), device=device, dtype=dtype)
    var_cam_positions_t = torch.tensor(
        var_cam_positions, device=device, dtype=torch.long
    )

    logging.info(
        "[SfM] Observations built: points=%d, observations=%d (%.2fs)",
        points_3d.shape[0],
        points_2d.shape[0],
        time.time() - t_start,
    )

    # Assemble full camera params for initial RMSE
    full_rvecs_init = base_rvecs_t.clone()
    full_tvecs_init = base_tvecs_t.clone()
    if var_cam_positions:
        full_rvecs_init[var_cam_positions_t] = var_rvecs.detach()
        full_tvecs_init[var_cam_positions_t] = var_tvecs.detach()

    initial_rmse = _compute_rmse_torch(
        full_rvecs_init,
        full_tvecs_init,
        points_3d_t.detach(),
        camera_indices_pos,
        point_indices_t,
        points_2d_t,
        K_t,
    )

    iters = int(_cfg("SFM_BA_ITERS", 50) or 50)
    lr = float(_cfg("SFM_BA_LR", 0.02) or 0.02)
    huber_delta = float(_cfg("SFM_BA_HUBER_DELTA", 1.0) or 1.0)
    log_every = int(_cfg("SFM_BA_LOG_EVERY", 25) or 25)

    params = [points_3d_t]
    if var_cam_positions:
        params.extend([var_rvecs, var_tvecs])

    optimizer = torch.optim.Adam(params, lr=lr)

    logging.info(
        "[SfM] Starting bundle adjustment (device=%s, iters=%d, lr=%.3f)...",
        device.type,
        iters,
        lr,
    )

    for i in range(iters):
        optimizer.zero_grad(set_to_none=True)
        full_rvecs = base_rvecs_t.clone()
        full_tvecs = base_tvecs_t.clone()
        if var_cam_positions:
            full_rvecs[var_cam_positions_t] = var_rvecs
            full_tvecs[var_cam_positions_t] = var_tvecs

        rvec_obs = full_rvecs[camera_indices_pos]
        tvec_obs = full_tvecs[camera_indices_pos]
        X_obs = points_3d_t[point_indices_t]

        proj = _project_points_torch(X_obs, rvec_obs, tvec_obs, K_t)
        loss = F.huber_loss(proj, points_2d_t, delta=huber_delta, reduction="mean")
        loss.backward()
        optimizer.step()

        if log_every > 0 and ((i + 1) % log_every == 0 or (i + 1) == iters):
            logging.info("[SfM] BA iter %d/%d loss=%.6f", i + 1, iters, loss.item())

    # Final parameters
    full_rvecs_final = base_rvecs_t.clone()
    full_tvecs_final = base_tvecs_t.clone()
    if var_cam_positions:
        full_rvecs_final[var_cam_positions_t] = var_rvecs.detach()
        full_tvecs_final[var_cam_positions_t] = var_tvecs.detach()

    refined_camera_params = {}
    full_rvecs_np = full_rvecs_final.detach().cpu().numpy()
    full_tvecs_np = full_tvecs_final.detach().cpu().numpy()
    for idx in sorted_indices:
        pos = idx_to_pos[int(idx)]
        refined_camera_params[int(idx)] = (
            full_rvecs_np[pos].astype(np.float64),
            full_tvecs_np[pos].astype(np.float64),
        )

    final_rmse = _compute_rmse_torch(
        full_rvecs_final,
        full_tvecs_final,
        points_3d_t.detach(),
        camera_indices_pos,
        point_indices_t,
        points_2d_t,
        K_t,
    )

    # Average camera correction in world coordinates
    corrections = []
    for idx in sorted_indices:
        rvec0, tvec0 = camera_params[int(idx)]
        rvec1, tvec1 = refined_camera_params[int(idx)]
        R0, _ = cv2.Rodrigues(rvec0)
        R1, _ = cv2.Rodrigues(rvec1)
        C0 = -R0.T @ tvec0.reshape(3, 1)
        C1 = -R1.T @ tvec1.reshape(3, 1)
        corrections.append(float(np.linalg.norm(C1 - C0)))
    avg_correction = float(np.mean(corrections)) if corrections else 0.0

    logging.info("[SfM] Bundle Adjustment Report:")
    logging.info(f"  - Initial RMSE: {initial_rmse:.2f} pixels")
    logging.info(
        f"  - Final RMSE:   {final_rmse:.2f} pixels "
        f"(Improved by {max(0.0, initial_rmse - final_rmse):.2f} pixels)"
    )
    logging.info(f"  - Average Camera Correction: {avg_correction:.3f} meters")

    refined_pairs_data = {}
    for idx in sorted_indices:
        idx_int = int(idx)
        _, _, left_path, right_path, _ = all_pairs_data[idx_int]
        rvec, tvec = refined_camera_params[idx_int]
        R_cv, _ = cv2.Rodrigues(rvec)
        refined_pairs_data[idx_int] = (
            idx_int,
            tvec.astype(np.float32),
            left_path,
            right_path,
            R_cv.astype(np.float32),
        )

    return refined_pairs_data
