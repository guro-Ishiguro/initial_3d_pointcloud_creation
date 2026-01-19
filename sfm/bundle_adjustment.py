import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

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


def _compute_rmse(
    camera_params: Dict[int, Tuple[np.ndarray, np.ndarray]],
    points_3d: np.ndarray,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
) -> float:
    if len(points_2d) == 0:
        return 0.0
    residuals = []
    for cam_idx, pt_idx, obs in zip(camera_indices, point_indices, points_2d):
        rvec, tvec = camera_params[int(cam_idx)]
        proj = _project_points(points_3d[pt_idx : pt_idx + 1], rvec, tvec, K)[0]
        residuals.append(proj - obs)
    residuals = np.vstack(residuals)
    return float(np.sqrt(np.mean(residuals**2)))


def _build_sparsity(
    n_cams_var: int,
    n_points: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    cam_to_var: Dict[int, int],
) -> lil_matrix:
    n_obs = camera_indices.size
    m = n_obs * 2
    n = n_cams_var * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    for i in range(n_obs):
        cam = int(camera_indices[i])
        if cam in cam_to_var:
            c = cam_to_var[cam]
            A[2 * i : 2 * i + 2, c * 6 : c * 6 + 6] = 1
        p = int(point_indices[i])
        A[2 * i : 2 * i + 2, n_cams_var * 6 + p * 3 : n_cams_var * 6 + p * 3 + 3] = 1
    return A


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
    cam_to_var = {
        int(idx): k for k, idx in enumerate(sorted_indices) if int(idx) != fixed_cam_idx
    }

    n_cams_var = len(sorted_indices) - 1
    x0_cams = []
    for idx in sorted_indices:
        if int(idx) == fixed_cam_idx:
            continue
        rvec, tvec = camera_params[int(idx)]
        x0_cams.append(rvec)
        x0_cams.append(tvec)
    x0_cams = np.hstack(x0_cams) if x0_cams else np.empty((0,), dtype=np.float64)
    x0_pts = points_3d.ravel()
    x0 = np.hstack([x0_cams, x0_pts])

    def residuals(params: np.ndarray) -> np.ndarray:
        cam_params_var = params[: n_cams_var * 6].reshape(-1, 6)
        pts = params[n_cams_var * 6 :].reshape(-1, 3)
        res = np.zeros((points_2d.shape[0], 2), dtype=np.float64)

        for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
            cam_idx = int(cam_idx)
            if cam_idx == fixed_cam_idx:
                rvec, tvec = camera_params[cam_idx]
            else:
                c = cam_to_var[cam_idx]
                rvec = cam_params_var[c, :3]
                tvec = cam_params_var[c, 3:6]
            proj = _project_points(pts[pt_idx : pt_idx + 1], rvec, tvec, K)[0]
            res[i] = proj - points_2d[i]
        return res.ravel()

    sparsity = _build_sparsity(
        n_cams_var=n_cams_var,
        n_points=points_3d.shape[0],
        camera_indices=camera_indices,
        point_indices=point_indices,
        cam_to_var=cam_to_var,
    )

    logging.info(
        "[SfM] Observations built: points=%d, observations=%d (%.2fs)",
        points_3d.shape[0],
        points_2d.shape[0],
        time.time() - t_start,
    )

    initial_rmse = _compute_rmse(
        camera_params,
        points_3d,
        camera_indices,
        point_indices,
        points_2d,
        K,
    )

    logging.info(
        "[SfM] Starting bundle adjustment (max_nfev=%d)...",
        int(_cfg("SFM_MAX_NFEV", 100) or 100),
    )
    result = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity,
        x_scale="jac",
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        verbose=2,
        max_nfev=int(_cfg("SFM_MAX_NFEV", 100) or 100),
    )

    refined_params = result.x
    cam_params_var = refined_params[: n_cams_var * 6].reshape(-1, 6)
    refined_points = refined_params[n_cams_var * 6 :].reshape(-1, 3)

    refined_camera_params = dict(camera_params)
    for idx in sorted_indices:
        if int(idx) == fixed_cam_idx:
            continue
        c = cam_to_var[int(idx)]
        rvec = cam_params_var[c, :3]
        tvec = cam_params_var[c, 3:6]
        refined_camera_params[int(idx)] = (rvec, tvec)

    final_rmse = _compute_rmse(
        refined_camera_params,
        refined_points,
        camera_indices,
        point_indices,
        points_2d,
        K,
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
