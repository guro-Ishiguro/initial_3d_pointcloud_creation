import os
import numpy as np
import cv2
import logging
from numba import njit

from utils import quaternion_to_rotation_matrix, clear_folder


def plot_reprojection_for_debug(
    neighbors_info,
    cur_idx,
    target_p_idx,
    camera_poses,
    neighbor_images,
    candidate_point,
    z_cand_idx,
    mid_z_cand_idx,
    K,
):
    # 各近隣画像に投影点を描画
    for neighbor_idx, _, _, _ in neighbors_info:
        neighbor_reprojected_image_file = os.path.join(
            "plotted_z_candidates",
            f"image_{cur_idx}_point_{target_p_idx}_z_cand_neighbor_{neighbor_idx}.jpg",
        )
        neighbor_R, neighbor_T = camera_poses[neighbor_idx]
        neighbor_image = np.copy(neighbor_images.get(neighbor_idx))
        if neighbor_image is None:
            continue

        # 点を投影
        p_cam_debug = (neighbor_R.T @ (candidate_point - neighbor_T).T).T
        # 描画対象画像の準備
        if os.path.exists(neighbor_reprojected_image_file):
            neighbor_image_to_draw = cv2.imread(neighbor_reprojected_image_file)
            neighbor_image_to_draw = cv2.cvtColor(
                neighbor_image_to_draw, cv2.COLOR_BGR2RGB
            )
        else:
            neighbor_image_to_draw = neighbor_image.copy()

        # 点の投影と描画
        if p_cam_debug[2] > 0.1:
            uv_hom_debug = K @ p_cam_debug
            u_debug, v_debug = int(round(uv_hom_debug[0] / uv_hom_debug[2])), int(
                round(uv_hom_debug[1] / uv_hom_debug[2])
            )
            v_debug = neighbor_image_to_draw.shape[0] - 1 - v_debug  # Y軸反転

            if (
                0 <= u_debug < neighbor_image_to_draw.shape[1]
                and 0 <= v_debug < neighbor_image_to_draw.shape[0]
            ):
                plot_color = (
                    (255, 0, 0) if z_cand_idx == mid_z_cand_idx else (0, 0, 255)
                )
                cv2.circle(
                    neighbor_image_to_draw,
                    (u_debug, v_debug),
                    3,
                    plot_color,
                    -1,
                )

        # 保存
        cv2.imwrite(
            neighbor_reprojected_image_file,
            cv2.cvtColor(neighbor_image_to_draw, cv2.COLOR_RGB2BGR),
        )


class PointCloudFilter:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def grid_sampling(points, colors, grid_size):
        """グリッドサンプリングによって点群をダウンサンプリングする"""
        # NaNまたは無限大の値をポイントから除去
        valid_mask = np.isfinite(points).all(axis=1)

        filtered_points = points[valid_mask]
        filtered_colors = colors[valid_mask]

        if filtered_points.shape[0] == 0:
            logging.warning("No valid points after filtering for grid sampling.")
            return np.array([]), np.array([])  # 点がない場合は空の配列を返す

        ids = np.floor(filtered_points / grid_size).astype(int)
        # ユニークなグリッドセルIDごとに最初の点を保持
        _, idxs = np.unique(ids, axis=0, return_index=True)

        return filtered_points[idxs], filtered_colors[idxs]

    def filter_points_by_depth(self, world_coords, colors, depth_cost, census_cost):
        """深度コストとCensusコストに基づいて点群をフィルタリングする"""
        before = world_coords.shape[0]
        km = depth_cost < self.config.DEPTH_ERROR_THRESHOLD
        kept = world_coords[km]
        kcols = colors[km]

        dropped = np.where(~km)[0]
        revive_mask = census_cost[dropped] < self.config.DISP_COST_THRESHOLD
        rev = world_coords[dropped][revive_mask]
        rev_c = colors[dropped][revive_mask]

        merged = np.vstack([kept, rev])
        return merged, np.vstack([kcols, rev_c])

    @staticmethod
    @njit
    def compute_multiview_cost(
        point_3d,
        ref_color,
        camera_K,
        neighbor_R,
        neighbor_T,
        neighbor_image,
        cost_type="SSD",
        color_threshold=100,
    ):
        """
        単一の3D点に対する複数視点コストを計算する。
        Args:
            point_3d (np.ndarray): 3D点のワールド座標 (x, y, z).
            ref_color (np.ndarray): 参照画像での3D点の色 (R, G, B) / 255.0.
            camera_K (np.ndarray): カメラの内部パラメータ行列 K.
            neighbor_R (np.ndarray): 隣接カメラの回転行列 R.
            neighbor_T (np.ndarray): 隣接カメラの並進ベクトル T.
            neighbor_image (np.ndarray): 隣接カメラの画像 (H, W, 3).
            cost_type (str): 使用する光度コストのタイプ ("SSD", "SAD").
            color_threshold (int): カラー差の閾値 (0-255).

        Returns:
            float: 計算されたコスト (小さいほど良い).
        """
        # ワールド座標から隣接カメラ座標へ
        p_cam = (neighbor_R.T @ (point_3d - neighbor_T).T).T
        if p_cam[2] <= 0.1:  # 非常に小さい正の値でZ > 0を保証
            return 1e9  # 高いコストと誤差

        # カメラ座標から画像座標へ
        uv_hom = camera_K @ p_cam
        u, v = uv_hom[0] / uv_hom[2], uv_hom[1] / uv_hom[2]
        v = neighbor_image.shape[0] - 1 - v

        # 画像の境界チェック
        h, w, _ = neighbor_image.shape
        if not (0 <= u < w and 0 <= v < h):
            return 1e9  # 範囲外なら高コスト

        # 画像のピクセル値取得
        v_int, u_int = int(round(v)), int(round(u))
        neighbor_color = neighbor_image[v_int, u_int].astype(np.float32) / 255.0

        # 光度コスト計算
        if cost_type == "SSD":
            photometric_cost = np.sum((ref_color - neighbor_color) ** 2)
        elif cost_type == "SAD":
            photometric_cost = np.sum(np.abs(ref_color - neighbor_color))
        else:
            photometric_cost = np.sum((ref_color - neighbor_color) ** 2)

        # カラー閾値による追加フィルタリング
        diff_sq = np.sum(((ref_color - neighbor_color) * 255) ** 2)
        color_diff_norm = np.sqrt(diff_sq)
        if color_diff_norm > color_threshold:
            photometric_cost += 1000
        else:
            photometric_cost += color_diff_norm

        return photometric_cost

    def refine_points_with_multiview_cost(
        self,
        cur_idx,
        points,
        colors,
        camera_data,
        K,
        image_dir,
        max_search_range=0.5,
        num_steps=10,
    ):
        """
        複数視点コストに基づいて点群の深度を最適化する。
        各点について、カメラの視線ベクトル方向に深度を変化させ、最もコストの低い深度を見つけ出す。
        """
        clear_folder("plotted_z_candidates")
        os.makedirs("plotted_z_candidates", exist_ok=True)

        optimized_points = np.copy(points)
        optimized_colors = np.copy(colors)

        num_points = points.shape[0]
        points_updated_count = 0

        if num_points == 0:
            logging.warning(
                f"No valid points for multiview refinement at index {cur_idx}."
            )
            return np.array([]), np.array([])

        try:
            fname_ref, pos_ref, quat_ref = camera_data[cur_idx]
            R_ref = quaternion_to_rotation_matrix(*quat_ref).astype(np.float32)
            T_ref = np.array(pos_ref, dtype=np.float32)
        except IndexError:
            logging.error(f"Could not retrieve camera data for reference index {cur_idx}.")
            return points, colors  # 処理をスキップして元の点を返す

        # 近隣カメラの情報をキャッシュ
        camera_poses = {}
        neighbor_images = {}
        n = len(camera_data)
        neighbors_info = []

        for off in (-2, -1, 1, 2):
            idx = cur_idx + off
            if 0 <= idx < n:
                fname, pos, quat = camera_data[idx]
                R = quaternion_to_rotation_matrix(*quat).astype(np.float32)
                T = np.array(pos, dtype=np.float32)
                camera_poses[idx] = (R, T)
                neighbor_image_path = os.path.join(image_dir, fname)
                try:
                    neighbor_images[idx] = cv2.cvtColor(
                        cv2.imread(neighbor_image_path), cv2.COLOR_BGR2RGB
                    )
                    if neighbor_images[idx] is None:
                        logging.warning(f"Failed to load image: {neighbor_image_path}")
                        continue
                except Exception as e:
                    logging.error(
                        f"Error loading neighbor image {neighbor_image_path}: {e}"
                    )
                    continue
                neighbors_info.append((idx, fname, pos, quat))

        if not neighbors_info:
            logging.warning(
                f"No valid neighbors found for image {cur_idx}. Skipping multiview refinement."
            )
            return optimized_points, optimized_colors

        logging.info(f"{num_points}点ある{cur_idx}番目から生成される点群を複数視点コストに基づいて深度を最適化中...")
        
        target_p_idx = 17000 # デバッグ用

        # 複数視点最適化のコアロジック
        for p_idx in range(num_points):
            current_point = points[p_idx]
            current_color = colors[p_idx]
            
            p_cam_ref = (R_ref.T @ (current_point - T_ref).T).T
            initial_cam_z = p_cam_ref[2]

            search_range = self.config.DEPTH_SEARCH_RANGE
            best_cost = float("inf")
            best_point = current_point.copy()

            z_start = initial_cam_z - search_range
            z_end = initial_cam_z + search_range

            z_candidates = np.linspace(z_start, z_end, num_steps, dtype=np.float32)
            for z_cand_idx, z_cand in enumerate(z_candidates):
                
                # カメラ座標で新しい候補点を作成
                candidate_point_cam = np.array(
                    [p_cam_ref[0], p_cam_ref[1], z_cand], dtype=np.float32
                )
                # ワールド座標に戻す
                candidate_point = (R_ref @ candidate_point_cam.T).T + T_ref

                total_current_cost = 0.0
                valid_neighbor_views = 0
                
                # デバッグ用のプロット（この部分は修正不要）
                if p_idx == target_p_idx:
                    mid_z_cand_idx = -1 # 元のコードの変数に合わせるためのダミー
                    plot_reprojection_for_debug(
                        neighbors_info,
                        cur_idx,
                        target_p_idx,
                        camera_poses,
                        neighbor_images,
                        candidate_point,
                        z_cand_idx,
                        mid_z_cand_idx,
                        K,
                    )
                
                # 近隣カメラとのコスト計算（この部分は修正不要）
                for neighbor_idx, _, _, _ in neighbors_info:
                    neighbor_R, neighbor_T = camera_poses[neighbor_idx]
                    neighbor_image = neighbor_images.get(neighbor_idx)

                    if neighbor_image is None:
                        continue

                    photo_cost = self.compute_multiview_cost(
                        candidate_point,
                        current_color,
                        K,
                        neighbor_R,
                        neighbor_T,
                        neighbor_image,
                    )

                    if photo_cost < 1e8:
                        total_current_cost += photo_cost
                        valid_neighbor_views += 1
                
                if valid_neighbor_views > 0:
                    total_current_cost /= valid_neighbor_views
                else:
                    total_current_cost = float("inf")

                if total_current_cost < best_cost:
                    best_cost = total_current_cost
                    best_point = candidate_point

            optimized_points[p_idx] = best_point
            
            if not np.allclose(current_point, optimized_points[p_idx], atol=1e-5):
                points_updated_count += 1

        logging.info(f"{num_points}点ある{cur_idx}番目の画像から生成される点群の複数視点最適化が終了")
        logging.info(
            f"Optimized {points_updated_count} points out of {num_points} for image {cur_idx} ({points_updated_count/num_points:.2%})."
        )
        return optimized_points, optimized_colors
