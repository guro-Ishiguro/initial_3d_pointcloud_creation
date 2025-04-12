import os
import shutil
import cv2
import numpy as np
import open3d as o3d
import logging
import argparse
import matplotlib.pyplot as plt

import config
from create_disparity import DisparityCreator
from create_depth import DepthCreator
from create_3d_pointcloud import PointCloudCreator


class PointCloudIntegrator:
    def __init__(self, args):
        self.args = args
        self.B = config.B
        self.focal_length = config.focal_length
        self.camera_height = config.camera_height
        self.K = config.K
        self.pixel_size = config.pixel_size
        self.window_size = config.window_size
        self.min_disp = config.min_disp
        self.num_disp = config.num_disp

        self.disparity_creator = DisparityCreator(
            self.window_size, self.min_disp, self.num_disp
        )
        self.depth_creator = DepthCreator(self.B, self.focal_length)
        self.pc_creator = PointCloudCreator()

    def compute_support_for_view(self, points, colors, position, quaternion, neighbor_image, color_threshold=30):
        """
        各点を指定のカメラパラメータ（position, quaternion）から再投影し、
        隣接画像（neighbor_image）上での色差がしきい値以下ならば1を返す配列を返す。
        points: (N,3) 、colors: (N,3) [RGB, 0-1]
        """
        N = points.shape[0]
        support = np.zeros(N, dtype=np.int32)
        T = np.array(position).reshape(3, 1)
        R = self.quaternion_to_rotation_matrix(*quaternion)
        # カメラ座標系に変換
        p_cam = (R.T @ (points.T - T)).T  # shape (N,3)
        if np.median(p_cam[:, 2]) < 0:
            p_cam = -p_cam
        valid_mask = p_cam[:, 2] > 0
        if not np.any(valid_mask):
            return support
        p_valid = p_cam[valid_mask]
        u = (self.K[0, 0] * (p_valid[:, 0] / p_valid[:, 2]) + self.K[0, 2]).astype(np.int32)
        v = (self.K[1, 1] * (p_valid[:, 1] / p_valid[:, 2]) + self.K[1, 2]).astype(np.int32)
        height, width = neighbor_image.shape[:2]
        # 画像座標系は上下反転
        v = height - 1 - v

        valid_indices = np.where(valid_mask)[0]
        for idx_local, idx in enumerate(valid_indices):
            if (u[idx_local] < 0 or u[idx_local] >= width or 
                v[idx_local] < 0 or v[idx_local] >= height):
                continue
            # 隣接画像（RGBに変換済み）の該当画素の色
            neighbor_color = neighbor_image[v[idx_local], u[idx_local], :].astype(np.float32)
            # 点の色（0-1の範囲）を255倍して比較
            point_color = colors[idx] * 255.0
            diff = np.linalg.norm(point_color - neighbor_color)
            if diff < color_threshold:
                support[idx] = 1
        return support

    def filter_points_by_support(self, points, colors, confidences, current_index, camera_data, image_dir, support_threshold=3):
        """
        現在の画像ペアから得られた各点について、隣接ビューからの支持数を計算し、
        支持数がしきい値以上の点のみを残す。
        - camera_data: ドローン画像ログで、各要素は (file_name, position, quaternion)
        - current_index: 現在の画像ペアのインデックス
        - image_dir: 画像ディレクトリパス
        ここでは、前後2枚ずつ、計4ビューで再投影支持数を計算する。
        """
        n = len(camera_data)
        neighbor_indices = []
        # 前後2枚ずつ取得（存在する場合）
        if current_index - 2 >= 0:
            neighbor_indices.append(camera_data[current_index - 2])
        if current_index - 1 >= 0:
            neighbor_indices.append(camera_data[current_index - 1])
        if current_index + 1 < n:
            neighbor_indices.append(camera_data[current_index + 1])
        if current_index + 2 < n:
            neighbor_indices.append(camera_data[current_index + 2])

        # もし利用可能な隣接ビューが support_threshold 未満なら、thresholdを利用可能枚数に調整
        if len(neighbor_indices) < support_threshold:
            support_threshold = len(neighbor_indices)

        # 各点の支持数を初期化
        support_counts = np.zeros(points.shape[0], dtype=np.int32)

        for neighbor_cam in neighbor_indices:
            neighbor_position = neighbor_cam[1]
            neighbor_quaternion = neighbor_cam[2]
            neighbor_image_path = os.path.join(image_dir, neighbor_cam[0])
            neighbor_image = cv2.imread(neighbor_image_path)
            if neighbor_image is None:
                logging.warning(f"Neighbor image not found: {neighbor_image_path}")
                continue
            # 色比較のためRGBに変換
            neighbor_image = cv2.cvtColor(neighbor_image, cv2.COLOR_BGR2RGB)
            support_view = self.compute_support_for_view(points, colors, neighbor_position, neighbor_quaternion, neighbor_image)
            support_counts += support_view

        keep_indices = support_counts >= support_threshold
        logging.info(f"Support filtering: {points.shape[0]} -> {np.sum(keep_indices)} points (threshold: {support_threshold}, neighbor count: {len(neighbor_indices)})")
        return points[keep_indices], colors[keep_indices], confidences[keep_indices], support_counts[keep_indices]

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        R = np.array(
            [
                [
                    1 - 2 * (qy ** 2 + qz ** 2),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx ** 2 + qz ** 2),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx ** 2 + qy ** 2),
                ],
            ]
        )
        return R

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="3D Point Cloud Creater")
        parser.add_argument(
            "--show-viewer",
            action="store_true",
            help="Show viewer during point cloud generation.",
        )
        parser.add_argument(
            "--record-video", action="store_true", help="Record viewer output to video."
        )
        return parser.parse_args()

    def clear_folder(self, dir_path):
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    logging.info(f"{dir_path} is cleared.")
                except Exception as e:
                    logging.error(f"Error deleting {file_path}: {e}")
        else:
            logging.info(f"The folder {dir_path} does not exist.")

    def save_histogram(
        self, image, output_filename
    ):
        output_file_dir = os.path.join(config.HISTGRAM_DIR, output_filename)
        plt.figure(figsize=(8, 4))
        max_value = np.max(image)
        plt.hist(
            image.flatten(),
            bins=50,
            range=(0, max_value),
            color="skyblue",
            edgecolor="black",
        )
        plt.xlabel(output_filename)
        plt.ylabel("Frequency")
        plt.title(f"{output_filename} Map Distribution")
        plt.tight_layout()
        plt.savefig(output_file_dir)
        plt.close()
        print(f"Histogram saved to {output_file_dir}")

    def process_image_pair(self, image_data):
        (
            img_id,
            T,
            left_image_path,
            right_image_path,
            B,
            focal_length,
            K,
            R,
            camera_height,
            pixel_size,
            window_size,
            min_disp,
            num_disp,
        ) = image_data

        left_image = cv2.imread(left_image_path)
        if left_image is None:
            logging.error(f"Left image not found for ID {img_id}: {left_image_path}")
            return None
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.imread(right_image_path)
        if right_image is None:
            logging.error(f"Right image not found for ID {img_id}: {right_image_path}")
            return None

        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Disparity計算
        disparity = self.disparity_creator.create_disparity(left_gray, right_gray, img_id)
        if disparity is None:
            logging.error(f"Disparity computation failed for ID {img_id}")
            return None

        # Census変換によるコスト計算
        census_left = self.disparity_creator.census_transform_numba(left_gray, window_size)
        census_right = self.disparity_creator.census_transform_numba(right_gray, window_size)
        census_cost = self.disparity_creator.compute_census_cost(
            left_gray, right_gray, disparity, census_left, census_right, window_size
        )

        # 視差コストのヒストグラムの保存
        self.save_histogram(census_cost, f"cencus_cost_{img_id}.png")

        # 深度計算
        depth = self.depth_creator.compute_depth(disparity)

        # 新たに追加した高速化済みの深度誤差コスト計算を呼び出す
        depth_cost = self.depth_creator.compute_depth_error_cost(disparity, window_size)

        # 深度誤差コストのヒストグラムの保存
        self.save_histogram(depth_cost, f"depth_cost_{img_id}.png")

        # コスト画像と深度誤差コスト画像から信頼度を算出する
        confidence = self.depth_creator.compute_confidence(census_cost, depth_cost)

        # 信頼度のヒストグラムの保存
        self.save_histogram(confidence, f"confidence_{img_id}.png")

        # 境界部分の除去（例）
        valid_area = depth > 0
        boundary_mask = valid_area & (
            ~np.roll(valid_area, 10, axis=0)
            | ~np.roll(valid_area, -10, axis=0)
            | ~np.roll(valid_area, 10, axis=1)
            | ~np.roll(valid_area, -10, axis=1)
        )
        depth[boundary_mask] = 0

        # 正射投影変換とワールド座標への変換
        ortho_depth, ortho_color, ortho_conf = self.depth_creator.to_orthographic_projection(
            depth, left_image, confidence, camera_height
        )
        world_coords, ortho_color, ortho_conf = self.depth_creator.depth_to_world(
            ortho_depth, ortho_color, ortho_conf, K, R, T, pixel_size
        )

        # グリッドサンプリング
        world_coords, ortho_color, ortho_conf = self.pc_creator.grid_sampling(
            world_coords, ortho_color, ortho_conf, 0.05
        )
        return world_coords, ortho_color, ortho_conf


    def weighted_integration(self, points, colors, confidences, voxel_size):
        min_avg_conf = 0.95
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
        counts = np.bincount(inverse_indices)
        sum_conf = np.bincount(inverse_indices, weights=confidences)
        avg_conf = sum_conf / counts
        good_voxel_mask = avg_conf >= min_avg_conf
        weighted_sum_x = np.bincount(
            inverse_indices, weights=points[:, 0] * confidences
        )
        weighted_sum_y = np.bincount(
            inverse_indices, weights=points[:, 1] * confidences
        )
        weighted_sum_z = np.bincount(
            inverse_indices, weights=points[:, 2] * confidences
        )
        integrated_points_all = (
            np.column_stack((weighted_sum_x, weighted_sum_y, weighted_sum_z))
            / sum_conf[:, np.newaxis]
        )
        weighted_sum_r = np.bincount(
            inverse_indices, weights=colors[:, 0] * confidences
        )
        weighted_sum_g = np.bincount(
            inverse_indices, weights=colors[:, 1] * confidences
        )
        weighted_sum_b = np.bincount(
            inverse_indices, weights=colors[:, 2] * confidences
        )
        integrated_colors_all = (
            np.column_stack((weighted_sum_r, weighted_sum_g, weighted_sum_b))
            / sum_conf[:, np.newaxis]
        )
        integrated_points = integrated_points_all[good_voxel_mask]
        integrated_colors = integrated_colors_all[good_voxel_mask]
        return integrated_points, integrated_colors

    def run(self):
        # カメラ情報の読み込み（config.DRONE_IMAGE_LOG から）
        drone_image_list = config.DRONE_IMAGE_LOG
        camera_data = []
        with open(drone_image_list, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                file_name = parts[0]
                position = tuple(map(float, parts[1:4]))
                quaternion = tuple(map(float, parts[4:8]))
                camera_data.append((file_name, position, quaternion))

        logging.info("Starting 3D map creation")
        image_pairs_data = [
            (
                idx,
                # カメラ位置
                np.array(camera_data[idx][1], dtype=np.float32),
                # 左右画像パス（例）
                os.path.join(config.IMAGE_DIR, f"left_{str(idx).zfill(6)}.png"),
                os.path.join(config.IMAGE_DIR, f"right_{str(idx).zfill(6)}.png"),
                self.B,
                self.focal_length,
                self.K,
                self.quaternion_to_rotation_matrix(*camera_data[idx][2]),
                self.camera_height,
                self.pixel_size,
                self.window_size,
                self.min_disp,
                self.num_disp,
            )
            for idx in range(130, 131)
        ]

        merged_points_list = []
        merged_colors_list = []
        merged_confidences_list = []
        last_pair = image_pairs_data[-1][0]
        for data in image_pairs_data:
            current_idx = data[0]
            result = self.process_image_pair(data)
            if result is not None:
                points, colors, conf = result
                # points, colors, conf, support = self.filter_points_by_support(
                #     points,
                #     colors,
                #     conf,
                #     current_idx,
                #     camera_data,
                #     config.IMAGE_DIR,
                #     support_threshold=4,
                # )
                merged_points_list.append(points)
                merged_colors_list.append(colors)
                merged_confidences_list.append(conf)
                logging.info(f"Processed {current_idx} out of {last_pair} image pairs.")
            else:
                logging.warning(f"Result is None for image pair {current_idx}.")

        if merged_points_list:
            all_points = np.vstack(merged_points_list)
            all_colors = np.vstack(merged_colors_list)
            all_confidences = np.concatenate(merged_confidences_list)

            logging.info(
                f"3D map creation completed. Total points before weighted integration: {len(all_points)}"
            )
            integrated_points, integrated_colors = self.weighted_integration(
                all_points, all_colors, all_confidences, voxel_size=0.02
            )
            integrated_points[:, 2] = -integrated_points[:, 2]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(integrated_points)
            pcd.colors = o3d.utility.Vector3dVector(integrated_colors)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
            logging.info(f"After outlier removal: {len(pcd.points)} points.")
            #self.pc_creator.write_ply(config.POINT_CLOUD_FILE_PATH, np.asarray(pcd.points), np.asarray(pcd.colors))
            o3d.visualization.draw_geometries([pcd])
        else:
            logging.warning("No points to merge. The point cloud might be empty.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()

    integrator = PointCloudIntegrator(args)
    integrator.run()
