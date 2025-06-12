# tests/data_loader.py

import os
import numpy as np
import logging
import sys

# scipyをインポートしてクォータニオンから回転行列への変換を確実に行う
from scipy.spatial.transform import Rotation

# utils.pyのquaternion_to_rotation_matrixは不要になるため、インポートしない
# from utils import quaternion_to_rotation_matrix


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
                    parts = line.strip().split(",")
                    if len(parts) != 8:
                        logging.warning(f"Skipping malformed line: {line.strip()}")
                        continue
                    fn, x, y, z, qx, qy, qz, qw = parts
                    pos = (float(x), float(y), float(z))
                    quat = (float(qx), float(qy), float(qz), float(qw))
                    camera_data.append((fn, pos, quat))
        except FileNotFoundError:
            logging.error(f"Camera log file not found: {self.drone_image_log}")
            sys.exit(1)
        logging.info(f"Loaded {len(camera_data)} camera poses.")
        return camera_data

    def get_image_paths(self, idx):
        """指定されたインデックスの左右画像のパスを返す"""
        if 0 <= idx < len(self.camera_data):
            base_fn, _, _ = self.camera_data[idx]  # e.g., "left_000000.png"
            right_fn = base_fn.replace("left_", "right_")
            left_path = os.path.join(self.image_dir, base_fn)
            right_path = os.path.join(self.image_dir, right_fn)
            return left_path, right_path
        return None, None

    def get_camera_pose(self, idx):
        """指定されたインデックスのカメラの姿勢データを返す"""
        if 0 <= idx < len(self.camera_data):
            return self.camera_data[idx]
        else:
            logging.warning(f"Camera data for index {idx} not found.")
            return None, None, None

    def get_all_camera_pairs(self, K):
        """
        すべてのカメラペアとその関連データを準備する。
        この中でUnity(左手系)からOpenCV(右手系)への座標変換を行う。
        """
        pairs = []
        for idx in range(len(self.camera_data)):
            fn, pos_unity, quat_unity = self.camera_data[idx]
            if not fn.startswith("left_"):
                continue

            # --- Unity(左手系) -> OpenCV(右手系) 座標変換 ---

            # 1. 位置(Position)の変換
            # 世界座標系での平行移動ベクトル
            pos_cv = np.array(
                [pos_unity[0], -pos_unity[1], pos_unity[2]], dtype=np.float32
            )

            # 2. 回転(Rotation)の変換
            quat_cv = np.array(
                [-quat_unity[0], quat_unity[1], -quat_unity[2], quat_unity[3]],
                dtype=np.float32,
            )

            # 3. 変換後のクォータニオンから回転行列を計算
            # カメラ→世界座標系の回転行列
            r = Rotation.from_quat(quat_cv)
            R_cv = r.as_matrix().astype(np.float32)

            # 4. 世界座標系→カメラ座標系への回転行列とカメラ座標系での平行移動ベクトル
            R_cv = R_cv.T
            T_cv = -R_cv @ pos_cv

            left_path, right_path = self.get_image_paths(idx)
            if not os.path.exists(left_path) or not os.path.exists(right_path):
                logging.warning(f"Image files for index {idx} not found. Skipping.")
                continue

            pairs.append(
                (
                    idx,
                    T_cv,  # 変換後の位置
                    left_path,
                    right_path,
                    R_cv,  # 変換後の回転行列
                )
            )
        return pairs
