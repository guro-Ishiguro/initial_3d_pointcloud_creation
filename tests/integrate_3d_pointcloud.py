import os
import shutil
import cv2
import numpy as np
import open3d as o3d
import logging
import time
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

    def save_confidence_histogram(
        self, confidence, output_filename="confidence_histogram.png"
    ):
        plt.figure(figsize=(8, 4))
        plt.hist(
            confidence.flatten(),
            bins=50,
            range=(0, 1),
            color="skyblue",
            edgecolor="black",
        )
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.title("Confidence Map Distribution")
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        print(f"Histogram saved to {output_filename}")

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

        disparity = self.disparity_creator.create_disparity(
            left_gray, right_gray, img_id
        )
        if disparity is None:
            logging.error(f"Disparity computation failed for ID {img_id}")
            return None

        # Census変換によるコスト計算（例）
        census_left = self.disparity_creator.census_transform_numba(
            left_gray, window_size
        )
        census_right = self.disparity_creator.census_transform_numba(
            right_gray, window_size
        )
        census_cost = self.disparity_creator.compute_census_cost(
            left_gray, right_gray, disparity, census_left, census_right, window_size
        )

        # 深度計算
        depth = self.depth_creator.compute_depth(disparity)

        # 信頼度はここでは単純にコストを利用（実際は compute_confidence() などを使用）
        confidence = 1 - (census_cost - np.min(census_cost)) / (
            np.ptp(census_cost) + 1e-6
        )

        # 境界除去（例）
        valid_area = depth > 0
        boundary_mask = valid_area & (
            ~np.roll(valid_area, 10, axis=0)
            | ~np.roll(valid_area, -10, axis=0)
            | ~np.roll(valid_area, 10, axis=1)
            | ~np.roll(valid_area, -10, axis=1)
        )
        depth[boundary_mask] = 0

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
        min_avg_conf = 0.9
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
            for idx in range(len(camera_data))
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
