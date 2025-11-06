import csv
import logging
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import mvs.config as config


class DataLoader:
    def __init__(self, image_root_dir, drone_image_log):
        # images ディレクトリのルート
        self.image_root_dir = image_root_dir
        self.left_dir = os.path.join(image_root_dir, "image_0")
        self.right_dir = os.path.join(image_root_dir, "image_1")
        self.drone_image_log = drone_image_log
        self.camera_data = self._load_camera_data()

    def _load_camera_data(self):
        camera_data = []
        try:
            with open(self.drone_image_log, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    if len(row) != 8:
                        logging.warning(f"Skipping malformed line: {','.join(row)}")
                        continue
                    fn, x, y, z, qx, qy, qz, qw = [c.strip() for c in row]
                    # ヘッダ行の可能性（pos_x 等）や数値変換失敗はスキップ
                    try:
                        pos = (float(x), float(y), float(z))
                        quat = (float(qx), float(qy), float(qz), float(qw))
                    except ValueError:
                        logging.warning(
                            f"Skipping header/non-numeric line: {','.join(row)}"
                        )
                        continue
                    # パスが含まれている場合はベース名のみ使用
                    base_fn = os.path.basename(fn)
                    camera_data.append((base_fn, pos, quat))
        except FileNotFoundError:
            logging.error(f"Camera log file not found: {self.drone_image_log}")
            sys.exit(1)
        logging.info(f"Loaded {len(camera_data)} camera poses.")
        return camera_data

    def _add_noise_to_pose(self, pos, quat, pos_scale, rot_scale):
        pos_error = np.random.randn(3) * pos_scale
        pos_with_error = (
            pos[0] + pos_error[0],
            pos[1] + pos_error[1],
            pos[2] + pos_error[2],
        )
        rot_vec_error = np.random.randn(3) * rot_scale
        error_rotation = Rotation.from_rotvec(rot_vec_error)
        original_rotation = Rotation.from_quat(quat)
        rotated_orientation = original_rotation * error_rotation
        quat_with_error = rotated_orientation.as_quat()
        return pos_with_error, quat_with_error

    def get_image_paths(self, idx):
        if 0 <= idx < len(self.camera_data):
            base_fn, _, _ = self.camera_data[idx]

            # 左画像候補
            left_path = os.path.join(self.left_dir, base_fn)
            if not os.path.exists(left_path) and base_fn.startswith("left_"):
                # left_ 接頭辞を外した名前でも試す
                left_path_alt = os.path.join(
                    self.left_dir, base_fn.replace("left_", "", 1)
                )
                if os.path.exists(left_path_alt):
                    left_path = left_path_alt

            # 右画像候補
            if base_fn.startswith("left_"):
                right_name = base_fn.replace("left_", "right_", 1)
                right_path = os.path.join(self.right_dir, right_name)
                if not os.path.exists(right_path):
                    # 同じサフィックス名（接頭辞なし）で探す
                    right_suffix = base_fn.replace("left_", "", 1)
                    alt = os.path.join(self.right_dir, right_suffix)
                    right_path = alt if os.path.exists(alt) else right_path
            else:
                # 接頭辞なしの場合は同名で左右を対応付ける前提
                right_path = os.path.join(self.right_dir, base_fn)

            return left_path, right_path
        return None, None

    def get_camera_pose(self, idx):
        if 0 <= idx < len(self.camera_data):
            return self.camera_data[idx]
        else:
            logging.warning(f"Camera data for index {idx} not found.")
            return None, None, None

    def get_all_camera_pairs(self, K):
        pairs = []
        position_error_scale = config.POSITION_ERROR_SCALE
        rotation_error_scale = config.ROTATION_ERROR_SCALE
        for idx in range(len(self.camera_data)):
            fn, pos_unity, quat_unity = self.camera_data[idx]
            # 右画像ログ行はスキップ（ある場合）
            if fn.lower().startswith("right_"):
                continue
            pos_unity, quat_unity = self._add_noise_to_pose(
                pos_unity, quat_unity, position_error_scale, rotation_error_scale
            )
            pos_cv = np.array(
                [pos_unity[0], -pos_unity[1], pos_unity[2]], dtype=np.float32
            )
            quat_cv = np.array(
                [-quat_unity[0], quat_unity[1], -quat_unity[2], quat_unity[3]],
                dtype=np.float32,
            )
            r = Rotation.from_quat(quat_cv)
            R_cv = r.as_matrix().astype(np.float32)
            R_cv = R_cv.T
            T_cv = -R_cv @ pos_cv
            left_path, right_path = self.get_image_paths(idx)
            if not os.path.exists(left_path) or not os.path.exists(right_path):
                logging.warning(f"Image files for index {idx} not found. Skipping.")
                continue
            pairs.append((idx, T_cv, left_path, right_path, R_cv))
        return pairs
