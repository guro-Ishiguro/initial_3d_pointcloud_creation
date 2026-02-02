"""
点群統合モジュール。
"""

import logging
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
from numba import njit, prange


class PointCloudIntegrator:
    """
    複数深度マップ由来の点群を統合し、外れ値除去・多視点フィルタ・PLY 出力を行うクラス。
    """

    def __init__(self, config):
        self.config = config

    def integrate_depth_maps_median(
        self, points_list, colors_list, normals_list=None, voxel_size=0.1
    ):
        """
        複数の深度マップから生成された点群を統合し、ボクセルグリッド内で位置・色（および法線）のメディアンを計算する。
        """
        if not points_list:
            logging.warning("No points to integrate.")
            if normals_list is not None:
                return np.array([]), np.array([]), np.array([])
            return np.array([]), np.array([])

        all_pts = np.vstack(points_list)
        all_cols = np.vstack(colors_list)
        has_normals = normals_list is not None and len(normals_list) > 0
        if has_normals:
            all_normals = np.vstack(normals_list)

        valid = np.isfinite(all_pts).all(axis=1)
        if has_normals:
            valid = valid & np.isfinite(all_normals).all(axis=1)
        pts = all_pts[valid]
        cols = all_cols[valid]
        if has_normals:
            normals = all_normals[valid]

        if pts.shape[0] == 0:
            logging.warning("No valid points after filtering for integration.")
            if has_normals:
                return np.array([]), np.array([]), np.array([])
            return np.array([]), np.array([])

        # 各点が属するボクセルID（整数グリッド座標）を計算
        vids = np.floor(pts / voxel_size).astype(np.int64)

        # ボクセルIDで辞書順ソートし、同一ボクセル内の点を連続させる
        sort_keys = vids.T
        sort_indices = np.lexsort(sort_keys)
        vids_sorted = vids[sort_indices]
        pts_sorted = pts[sort_indices]
        cols_sorted = cols[sort_indices]
        if has_normals:
            normals_sorted = normals[sort_indices]

        # ボクセル境界のインデックス（各ボクセルの先頭・末尾）を取得
        voxel_diff = np.any(vids_sorted[1:] != vids_sorted[:-1], axis=1)
        unique_voxel_indices = np.concatenate(
            ([0], np.where(voxel_diff)[0] + 1, [len(vids_sorted)])
        )

        num_voxels = len(unique_voxel_indices) - 1
        if num_voxels == 0:
            logging.warning("No voxels found after integration.")
            if has_normals:
                return np.array([]), np.array([]), np.array([])
            return np.array([]), np.array([])

        # ボクセルごとにメディアン（法線は平均して正規化）を Numba JIT で計算
        if has_normals:
            med_pts, med_cols, med_normals = _compute_voxel_medians_with_normals_jit(
                pts_sorted,
                cols_sorted,
                normals_sorted,
                unique_voxel_indices,
                num_voxels,
            )
        else:
            med_pts, med_cols = _compute_voxel_medians_jit(
                pts_sorted,
                cols_sorted,
                unique_voxel_indices,
                num_voxels,
            )

        logging.info(f"Integrated {num_voxels} points from multiple depth maps.")
        if has_normals:
            return med_pts, med_cols, med_normals
        return med_pts, med_cols

    def filter_points_by_multi_view_visibility(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        all_poses: Dict[int, Dict],
        all_depth_maps: Dict[int, np.ndarray],
        visibility_threshold: int = 2,
        geometric_error_threshold: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        各点を全カメラ視点で再投影し、深度一貫性を満たす視点数が閾値以上の点のみ残す。
        """
        if points.shape[0] == 0:
            return points, colors

        logging.info(
            f"Filtering {points.shape[0]} points by multi-view visibility "
            f"(threshold: {visibility_threshold} views, error: {geometric_error_threshold})"
        )

        view_indices = list(all_poses.keys())
        K_list = [all_poses[idx]["K"].astype(np.float32) for idx in view_indices]
        R_list = [all_poses[idx]["R"].astype(np.float32) for idx in view_indices]
        T_list = [all_poses[idx]["T"].astype(np.float32) for idx in view_indices]
        depth_maps_list = [
            all_depth_maps[idx].astype(np.float32) for idx in view_indices
        ]

        K_array = np.stack(K_list)
        R_array = np.stack(R_list)
        T_array = np.stack(T_list)
        depth_maps_array = np.stack(depth_maps_list)

        # 各点について、深度一貫性を満たす視点数を Numba JIT でカウント
        visibility_counts = _check_point_visibility_jit(
            points.astype(np.float32),
            K_array,
            R_array,
            T_array,
            depth_maps_array,
            geometric_error_threshold,
        )

        valid_mask = visibility_counts >= visibility_threshold
        filtered_points = points[valid_mask]
        filtered_colors = colors[valid_mask]

        filtered_count = np.sum(valid_mask)
        logging.info(
            f"Multi-view visibility filtering: {filtered_count}/{points.shape[0]} points "
            f"({filtered_count/points.shape[0]*100:.2f}%) passed the filter"
        )

        return filtered_points, filtered_colors

    def process_and_save_final_point_cloud(
        self, points_list, colors_list, file_path, normals_list=None
    ):
        """
        統合済み点群を Open3D で統計的外れ値除去し、PLY ファイルとして保存する。
        """
        if points_list is None or points_list.size == 0:
            logging.warning("No point clouds to process.")
            return None

        points = np.vstack(points_list)
        colors = np.vstack(colors_list)
        has_normals = normals_list is not None
        if has_normals:
            if isinstance(normals_list, np.ndarray):
                normals = normals_list
            else:
                normals = np.vstack(normals_list)

        if points.shape[0] == 0:
            logging.warning("No points to process for final point cloud.")
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if has_normals:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        logging.info(f"Initial merged point cloud size: {len(pcd.points)}")

        # 統計的外れ値除去（近傍50点・標準偏差4倍以上を除外）
        pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=50, std_ratio=4.0
        )
        logging.info(
            f"Point cloud size after outlier removal: {len(pcd_filtered.points)} "
            f"(removed {len(pcd.points) - len(pcd_filtered.points)} points)"
        )

        has_normals_after = False
        if has_normals:
            inlier_indices_np = np.asarray(inlier_indices)
            normals_filtered = normals[inlier_indices_np]
            pcd_filtered.normals = o3d.utility.Vector3dVector(normals_filtered)
            has_normals_after = True

        if has_normals_after:
            self.write_ply(
                file_path,
                np.asarray(pcd_filtered.points),
                np.asarray(pcd_filtered.colors),
                np.asarray(pcd_filtered.normals),
            )
        else:
            self.write_ply(
                file_path,
                np.asarray(pcd_filtered.points),
                np.asarray(pcd_filtered.colors),
            )

        return pcd

    @staticmethod
    def write_ply(filename, vertices, colors, normals=None):
        """
        点群（頂点・色・任意で法線）を ASCII PLY 形式でファイルに書き込む。
        """
        assert (
            vertices.shape[0] == colors.shape[0]
        ), "Vertices and colors must have the same number of points."
        if normals is not None:
            assert (
                vertices.shape[0] == normals.shape[0]
            ), "Vertices and normals must have the same number of points."

        colors_uchar = (colors * 255).astype(np.uint8)

        # ASCII PLY ヘッダーとデータ行を生成
        header_lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        if normals is not None:
            header_lines.extend(
                ["property float nx", "property float ny", "property float nz"]
            )
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        if normals is not None:
            data = np.hstack((vertices, colors_uchar, normals))
            fmt = "%f %f %f %d %d %d %f %f %f"
        else:
            data = np.hstack((vertices, colors_uchar))
            fmt = "%f %f %f %d %d %d"

        with open(filename, "w") as f:
            f.write(header)
            np.savetxt(f, data, fmt=fmt)
        logging.info(f"Final point cloud saved to {filename}")


@njit(fastmath=True, parallel=True)
def _compute_voxel_medians_jit(
    pts_sorted: np.ndarray,
    cols_sorted: np.ndarray,
    unique_voxel_indices: np.ndarray,
    num_voxels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ボクセルIDでソート済みの点・色から、各ボクセル内の位置・色のメディアンを計算する。
    """
    med_pts = np.zeros((num_voxels, 3), dtype=np.float32)
    med_cols = np.zeros((num_voxels, 3), dtype=np.float32)

    for i in prange(num_voxels):
        start_idx = unique_voxel_indices[i]
        end_idx = unique_voxel_indices[i + 1]
        voxel_size_i = end_idx - start_idx

        # ボクセル内の中央要素（奇数ならその値、偶数なら中央2要素の平均）
        mid_idx = start_idx + voxel_size_i // 2
        if voxel_size_i % 2 == 1:
            med_pts[i, 0] = pts_sorted[mid_idx, 0]
            med_pts[i, 1] = pts_sorted[mid_idx, 1]
            med_pts[i, 2] = pts_sorted[mid_idx, 2]
            med_cols[i, 0] = cols_sorted[mid_idx, 0]
            med_cols[i, 1] = cols_sorted[mid_idx, 1]
            med_cols[i, 2] = cols_sorted[mid_idx, 2]
        else:
            med_pts[i, 0] = (pts_sorted[mid_idx - 1, 0] + pts_sorted[mid_idx, 0]) * 0.5
            med_pts[i, 1] = (pts_sorted[mid_idx - 1, 1] + pts_sorted[mid_idx, 1]) * 0.5
            med_pts[i, 2] = (pts_sorted[mid_idx - 1, 2] + pts_sorted[mid_idx, 2]) * 0.5
            med_cols[i, 0] = (
                cols_sorted[mid_idx - 1, 0] + cols_sorted[mid_idx, 0]
            ) * 0.5
            med_cols[i, 1] = (
                cols_sorted[mid_idx - 1, 1] + cols_sorted[mid_idx, 1]
            ) * 0.5
            med_cols[i, 2] = (
                cols_sorted[mid_idx - 1, 2] + cols_sorted[mid_idx, 2]
            ) * 0.5

    return med_pts, med_cols


@njit(fastmath=True, parallel=True)
def _compute_voxel_medians_with_normals_jit(
    pts_sorted: np.ndarray,
    cols_sorted: np.ndarray,
    normals_sorted: np.ndarray,
    unique_voxel_indices: np.ndarray,
    num_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ボクセルごとに点・色はメディアン、法線は平均をとり正規化して返す。
    """
    med_pts = np.zeros((num_voxels, 3), dtype=np.float32)
    med_cols = np.zeros((num_voxels, 3), dtype=np.float32)
    med_normals = np.zeros((num_voxels, 3), dtype=np.float32)

    for i in prange(num_voxels):
        start_idx = unique_voxel_indices[i]
        end_idx = unique_voxel_indices[i + 1]
        voxel_size_i = end_idx - start_idx

        # ボクセル内の中央要素（奇数ならその値、偶数なら中央2要素の平均）
        mid_idx = start_idx + voxel_size_i // 2
        if voxel_size_i % 2 == 1:
            med_pts[i, 0] = pts_sorted[mid_idx, 0]
            med_pts[i, 1] = pts_sorted[mid_idx, 1]
            med_pts[i, 2] = pts_sorted[mid_idx, 2]
            med_cols[i, 0] = cols_sorted[mid_idx, 0]
            med_cols[i, 1] = cols_sorted[mid_idx, 1]
            med_cols[i, 2] = cols_sorted[mid_idx, 2]
        else:
            med_pts[i, 0] = (pts_sorted[mid_idx - 1, 0] + pts_sorted[mid_idx, 0]) * 0.5
            med_pts[i, 1] = (pts_sorted[mid_idx - 1, 1] + pts_sorted[mid_idx, 1]) * 0.5
            med_pts[i, 2] = (pts_sorted[mid_idx - 1, 2] + pts_sorted[mid_idx, 2]) * 0.5
            med_cols[i, 0] = (
                cols_sorted[mid_idx - 1, 0] + cols_sorted[mid_idx, 0]
            ) * 0.5
            med_cols[i, 1] = (
                cols_sorted[mid_idx - 1, 1] + cols_sorted[mid_idx, 1]
            ) * 0.5
            med_cols[i, 2] = (
                cols_sorted[mid_idx - 1, 2] + cols_sorted[mid_idx, 2]
            ) * 0.5

        # 法線はボクセル内で平均をとり、正規化（零ベクトルなら (0,0,1)）
        avg_nx = 0.0
        avg_ny = 0.0
        avg_nz = 0.0
        for j in range(start_idx, end_idx):
            avg_nx += normals_sorted[j, 0]
            avg_ny += normals_sorted[j, 1]
            avg_nz += normals_sorted[j, 2]
        inv_size = 1.0 / voxel_size_i
        avg_nx *= inv_size
        avg_ny *= inv_size
        avg_nz *= inv_size

        norm_sq = avg_nx * avg_nx + avg_ny * avg_ny + avg_nz * avg_nz
        if norm_sq > 1e-12:
            norm_inv = 1.0 / np.sqrt(norm_sq)
            med_normals[i, 0] = avg_nx * norm_inv
            med_normals[i, 1] = avg_ny * norm_inv
            med_normals[i, 2] = avg_nz * norm_inv
        else:
            med_normals[i, 0] = 0.0
            med_normals[i, 1] = 0.0
            med_normals[i, 2] = 1.0

    return med_pts, med_cols, med_normals


@njit(fastmath=True)
def _check_point_visibility_jit(
    points: np.ndarray,
    K_array: np.ndarray,
    R_array: np.ndarray,
    T_array: np.ndarray,
    depth_maps_array: np.ndarray,
    error_threshold: float,
) -> np.ndarray:
    """
    各点を全カメラで再投影し、画像内かつ深度相対誤差が error_threshold 未満の視点数を返す。
    """
    N = points.shape[0]
    M = K_array.shape[0]
    visibility_counts = np.zeros(N, dtype=np.int32)

    for i in range(N):
        point_world = points[i]
        consistent_views = 0

        for j in range(M):
            K = K_array[j]
            R = R_array[j]
            T = T_array[j]
            depth_map = depth_maps_array[j]
            h, w = depth_map.shape

            # ワールド座標をカメラ座標に変換（Z が正でない場合はスキップ）
            point_cam = R @ point_world + T

            if point_cam[2] < 1e-6:
                continue

            # 画像座標 (u, v) と深度 d_proj を計算
            point_img_h = K @ point_cam
            u = point_img_h[0] / point_img_h[2]
            v = point_img_h[1] / point_img_h[2]
            d_proj = point_cam[2]

            if not (0 <= u < w and 0 <= v < h):
                continue

            r = int(round(v))
            c = int(round(u))

            if not (0 <= c < w and 0 <= r < h):
                continue

            d_actual = depth_map[r, c]

            if not np.isfinite(d_actual) or d_actual < 1e-6:
                continue

            # 再投影深度と実深度の相対誤差が閾値未満なら「一貫」とカウント
            relative_error = np.abs(d_proj - d_actual) / d_actual

            if relative_error < error_threshold:
                consistent_views += 1

        visibility_counts[i] = consistent_views

    return visibility_counts
