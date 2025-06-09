import os
import numpy as np
import logging
import sys  # sys をインポート

from utils import quaternion_to_rotation_matrix


class DataLoader:
    def __init__(self, image_dir, drone_image_log):
        self.image_dir = image_dir
        self.drone_image_log = drone_image_log
        self.camera_data = self._load_camera_data()

    def _load_camera_data(self):
        """ドローン画像ログからカメラの姿勢データを読み込む"""
        camera_data = []
        try:
            with open(self.drone_image_log) as f:
                for line in f:
                    fn, x, y, z, qx, qy, qz, qw = line.strip().split(",")
                    pos = (float(x), float(y), float(z))
                    quat = (float(qx), float(qy), float(qz), float(qw))
                    camera_data.append((fn, pos, quat))
        except FileNotFoundError:
            logging.error(f"Camera log file not found: {self.drone_image_log}")
            sys.exit(1)  # ファイルが見つからない場合は終了
        logging.info(f"Loaded {len(camera_data)} camera poses.")
        return camera_data

    def get_image_paths(self, idx):
        """指定されたインデックスの左右画像のパスを返す"""
        left_path = os.path.join(self.image_dir, f"left_{str(idx).zfill(6)}.png")
        right_path = os.path.join(self.image_dir, f"right_{str(idx).zfill(6)}.png")
        return left_path, right_path

    def get_camera_pose(self, idx):
        """指定されたインデックスのカメラの姿勢データを返す"""
        if 0 <= idx < len(self.camera_data):
            return self.camera_data[idx]
        else:
            logging.warning(f"Camera data for index {idx} not found.")
            return None, None, None

    def get_all_camera_pairs(self, K):
        """すべてのカメラペアとその関連データを準備する"""
        pairs = []
        for idx in range(len(self.camera_data)):
            fn, pos, quat = self.camera_data[idx]
            left_path, right_path = self.get_image_paths(idx)
            R = quaternion_to_rotation_matrix(*quat).astype(np.float32)
            pairs.append(
                (
                    idx,
                    np.array(pos, dtype=np.float32),
                    left_path,
                    right_path,
                    R,
                )
            )
        return pairs
