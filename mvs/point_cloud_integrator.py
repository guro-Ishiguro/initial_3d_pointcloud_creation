import logging
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
from numba import njit


class PointCloudIntegrator:
    def __init__(self, config):
        self.config = config

    def integrate_depth_maps_median(
        self, points_list, colors_list, normals_list=None, voxel_size=0.1
    ):
        """
        複数の深度マップから生成された点群を統合し、ボクセルグリッド内でメディアンを計算する。

        Args:
            points_list: 点群のリスト
            colors_list: 色のリスト
            normals_list: 法線のリスト（オプション）
            voxel_size: ボクセルサイズ

        Returns:
            med_pts: 統合された点群
            med_cols: 統合された色
            med_normals: 統合された法線（normals_listが渡された場合のみ）
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

        # NaNや無限大の点を除去
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

        # 各点を対応するボクセルIDに割り当てる
        vids = np.floor(pts / voxel_size).astype(int)

        voxel_dict = {}
        for i, vid in enumerate(map(tuple, vids)):
            voxel_dict.setdefault(vid, []).append(i)

        med_pts, med_cols = [], []
        med_normals = [] if has_normals else None
        for idxs in voxel_dict.values():
            voxel_pts = pts[idxs]
            voxel_cols = cols[idxs]
            # 各ボクセル内の点のメディアンを計算
            med_pts.append(np.median(voxel_pts, axis=0))
            med_cols.append(np.median(voxel_cols, axis=0))
            if has_normals:
                # 法線は平均化して正規化
                voxel_normals = normals[idxs]
                avg_normal = np.mean(voxel_normals, axis=0)
                norm = np.linalg.norm(avg_normal)
                if norm > 1e-6:
                    avg_normal = avg_normal / norm
                else:
                    avg_normal = np.array([0.0, 0.0, 1.0])
                med_normals.append(avg_normal)

        logging.info(f"Integrated {len(med_pts)} points from multiple depth maps.")
        if has_normals:
            return (
                np.array(med_pts),
                np.array(med_cols),
                np.array(med_normals),
            )
        return np.array(med_pts), np.array(med_cols)

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
        複数のカメラ視点から生成された点群を使って、信頼性のある点群だけを残す。

        各点について：
        1. 他のビューの深度マップに投影
        2. 投影位置の深度値と実際の深度値の一貫性をチェック
        3. 閾値以上のビューから一貫性が確認された点だけを残す

        Args:
            points: 点群の座標 (N, 3)
            colors: 点群の色 (N, 3)
            all_poses: 各ビューのカメラポーズ辞書 {idx: {"R": R, "T": T, "K": K}}
            all_depth_maps: 各ビューの深度マップ辞書 {idx: depth_map}
            visibility_threshold: 可視性の閾値（この数以上のビューから見える点だけを残す）
            geometric_error_threshold: 幾何学的エラーの閾値（相対深度差）

        Returns:
            フィルタリングされた点群と色
        """
        if points.shape[0] == 0:
            return points, colors

        logging.info(
            f"Filtering {points.shape[0]} points by multi-view visibility "
            f"(threshold: {visibility_threshold} views, error: {geometric_error_threshold})"
        )

        # カメラパラメータをリストに変換
        view_indices = list(all_poses.keys())
        K_list = [all_poses[idx]["K"].astype(np.float32) for idx in view_indices]
        R_list = [all_poses[idx]["R"].astype(np.float32) for idx in view_indices]
        T_list = [all_poses[idx]["T"].astype(np.float32) for idx in view_indices]
        depth_maps_list = [
            all_depth_maps[idx].astype(np.float32) for idx in view_indices
        ]

        # NumPy配列に変換
        K_array = np.stack(K_list)
        R_array = np.stack(R_list)
        T_array = np.stack(T_list)
        depth_maps_array = np.stack(depth_maps_list)

        # 各点の可視性をチェック
        visibility_counts = _check_point_visibility_jit(
            points.astype(np.float32),
            K_array,
            R_array,
            T_array,
            depth_maps_array,
            geometric_error_threshold,
        )

        # 閾値以上の可視性を持つ点だけを残す
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
        """最終的な点群を処理し、PLYファイルとして保存する"""
        if points_list is None or points_list.size == 0:
            logging.warning("No point clouds to process.")
            return None

        # リストを結合して単一の配列にする
        points = np.vstack(points_list)
        colors = np.vstack(colors_list)
        has_normals = normals_list is not None
        if has_normals:
            # normals_listが配列の場合はそのまま使用、リストの場合は結合
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

        # 外れ値除去（インデックスを取得）
        pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=50, std_ratio=4.0
        )
        logging.info(
            f"Point cloud size after outlier removal: {len(pcd_filtered.points)} "
            f"(removed {len(pcd.points) - len(pcd_filtered.points)} points)"
        )

        # 外れ値除去後の法線を取得
        has_normals_after = False
        if has_normals:
            # インデックスを使って法線もフィルタリング
            inlier_indices_np = np.asarray(inlier_indices)
            normals_filtered = normals[inlier_indices_np]
            pcd_filtered.normals = o3d.utility.Vector3dVector(normals_filtered)
            has_normals_after = True

        # PLYファイルとして保存
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
        """点群データをPLYファイルとして書き込む"""
        assert (
            vertices.shape[0] == colors.shape[0]
        ), "Vertices and colors must have the same number of points."
        if normals is not None:
            assert (
                vertices.shape[0] == normals.shape[0]
            ), "Vertices and normals must have the same number of points."

        colors_uchar = (colors * 255).astype(np.uint8)

        # ヘッダーを構築
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

        # データを構築
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
    各点が何個のビューから見えるかをカウントする。

    Args:
        points: 点群の座標 (N, 3) - ワールド座標
        K_array: カメラ内部パラメータ (M, 3, 3)
        R_array: 回転行列 (M, 3, 3) - world->camera
        T_array: 並進ベクトル (M, 3) - world->camera
        depth_maps_array: 深度マップ (M, H, W)
        error_threshold: 幾何学的エラーの閾値

    Returns:
        各点の可視性カウント (N,)
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

            # ワールド座標からカメラ座標へ変換
            point_cam = R @ point_world + T

            # カメラの後ろにある点は無視
            if point_cam[2] < 1e-6:
                continue

            # 画像座標へ投影
            point_img_h = K @ point_cam
            u = point_img_h[0] / point_img_h[2]
            v = point_img_h[1] / point_img_h[2]
            d_proj = point_cam[2]

            # 画像範囲外かチェック
            if not (0 <= u < w and 0 <= v < h):
                continue

            # 最も近いピクセルの深度値を取得
            r = int(round(v))
            c = int(round(u))

            if not (0 <= c < w and 0 <= r < h):
                continue

            d_actual = depth_map[r, c]

            # 深度が有効かチェック
            if not np.isfinite(d_actual) or d_actual < 1e-6:
                continue

            # 幾何学的なエラーを計算（相対深度差）
            relative_error = np.abs(d_proj - d_actual) / d_actual

            if relative_error < error_threshold:
                consistent_views += 1

        visibility_counts[i] = consistent_views

    return visibility_counts
