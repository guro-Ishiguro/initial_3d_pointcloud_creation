"""
ステレオ画像とカメラポーズを読み込み、MVS処理用のデータを提供するモジュール。

左カメラのポーズCSVから位置・姿勢を読み込み、左右画像パスとOpenCV座標系の
カメラ行列（R, T）を取得する機能を提供する。
"""

import csv
import logging
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import mvs.config as config


class DataLoader:
    """
    ステレオ画像データとカメラポーズを管理するローダー。
    """

    def __init__(self, image_root_dir, left_camera_poses_csv):
        self.image_root_dir = image_root_dir
        self.left_dir = os.path.join(image_root_dir, "image_0")
        self.right_dir = os.path.join(image_root_dir, "image_1")
        self.left_camera_poses_csv = left_camera_poses_csv
        self.camera_data = self._load_camera_data()

    def _load_camera_data(self):
        """
        左カメラポーズCSVを読み込み、(ファイル名, 位置タプル, 四元数タプル) のリストを返す。
        """
        camera_data = []
        try:
            if not self.left_camera_poses_csv or not os.path.exists(
                self.left_camera_poses_csv
            ):
                logging.error(
                    f"Camera poses CSV not found: {self.left_camera_poses_csv}"
                )
                sys.exit(1)
            with open(self.left_camera_poses_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        fn = str(row.get("filename")).strip()
                        x = float(row.get("pos_x"))
                        y = float(row.get("pos_y"))
                        z = float(row.get("pos_z"))
                        qx = float(row.get("rot_x"))
                        qy = float(row.get("rot_y"))
                        qz = float(row.get("rot_z"))
                        qw = float(row.get("rot_w"))
                    except Exception:
                        logging.warning(f"Skipping malformed line: {row}")
                        continue
                    pos = (x, y, z)
                    quat = (qx, qy, qz, qw)
                    base_fn = os.path.basename(fn)
                    camera_data.append((base_fn, pos, quat))
        except FileNotFoundError:
            logging.error(f"Camera poses CSV not found: {self.left_camera_poses_csv}")
            sys.exit(1)
        logging.info(f"Loaded {len(camera_data)} camera poses.")
        return camera_data

    def _add_noise_to_pose(self, pos, quat, pos_scale, rot_scale):
        """
        位置・姿勢にガウスノイズを付加する。
        """
        if pos_scale > 0:
            pos_error = np.random.randn(3) * pos_scale
            pos_with_error = (
                pos[0] + pos_error[0],
                pos[1] + pos_error[1],
                pos[2] + pos_error[2],
            )
        else:
            pos_with_error = pos
        if rot_scale > 0:
            rot_vec_error = np.random.randn(3) * rot_scale
            error_rotation = Rotation.from_rotvec(rot_vec_error)
            original_rotation = Rotation.from_quat(quat)
            rotated_orientation = original_rotation * error_rotation
            quat_with_error = rotated_orientation.as_quat()
        else:
            quat_with_error = quat

        return pos_with_error, quat_with_error

    def _select_frame_indices(self):
        """
        config の KEYFRAME_START_INDEX / KEYFRAME_END_INDEX / FRAME_STRIDE に従い、
        左カメラ画像のみを対象にキーフレームインデックスを選択する。
        """
        n = len(self.camera_data)
        if n <= 0:
            return []

        start = int(getattr(config, "KEYFRAME_START_INDEX", 0) or 0)
        end_cfg = getattr(config, "KEYFRAME_END_INDEX", None)
        end = int(end_cfg) if end_cfg is not None else (n - 1)
        start = max(0, min(start, n - 1))
        end = max(start, min(end, n - 1))
        candidates = list(range(start, end + 1))

        # 右カメラ用の行は除外（左カメラのみ使用）
        filtered = []
        for i in candidates:
            fn, _, _ = self.camera_data[i]
            if str(fn).lower().startswith("right_"):
                continue
            filtered.append(i)

        stride = int(getattr(config, "FRAME_STRIDE", 1) or 1)
        stride = max(1, stride)
        selected = filtered[::stride]

        logging.info(
            f"Frame selection (stride={stride}): selected {len(selected)}/{len(filtered)} frames "
            f"(range {start}-{end})"
        )
        return selected

    def get_image_paths(self, idx):
        """
        指定インデックスに対応する左右画像の絶対パスを返す。
        ファイル名が left_ で始まる場合は right_ に置換して右画像を探す。
        """
        if 0 <= idx < len(self.camera_data):
            base_fn, _, _ = self.camera_data[idx]

            left_path = os.path.join(self.left_dir, base_fn)
            if not os.path.exists(left_path) and base_fn.startswith("left_"):
                left_path_alt = os.path.join(
                    self.left_dir, base_fn.replace("left_", "", 1)
                )
                if os.path.exists(left_path_alt):
                    left_path = left_path_alt

            if base_fn.startswith("left_"):
                right_name = base_fn.replace("left_", "right_", 1)
                right_path = os.path.join(self.right_dir, right_name)
                if not os.path.exists(right_path):
                    right_suffix = base_fn.replace("left_", "", 1)
                    alt = os.path.join(self.right_dir, right_suffix)
                    right_path = alt if os.path.exists(alt) else right_path
            else:
                right_path = os.path.join(self.right_dir, base_fn)

            return left_path, right_path
        return None, None

    def get_camera_pose(self, idx):
        """
        指定インデックスのカメラポーズ（ファイル名・位置・四元数）を返す。
        """
        if 0 <= idx < len(self.camera_data):
            return self.camera_data[idx]
        else:
            logging.warning(f"Camera data for index {idx} not found.")
            return None, None, None

    def get_all_camera_pairs(self, K):
        """
        選択されたキーフレームについて、OpenCV座標系の R, T と左右画像パスをまとめた辞書を返す。
        ポーズに config で指定されたノイズを付加し、Unity系からOpenCV系へ変換する。
        """
        pairs = {}
        position_error_scale = config.POSITION_ERROR_SCALE
        rotation_error_scale = config.ROTATION_ERROR_SCALE
        selected_indices = self._select_frame_indices()
        for idx in selected_indices:
            fn, pos_unity, quat_unity = self.camera_data[idx]
            if fn.lower().startswith("right_"):
                continue
            pos_unity, quat_unity = self._add_noise_to_pose(
                pos_unity, quat_unity, position_error_scale, rotation_error_scale
            )
            # Unity座標系 → OpenCV座標系（Y軸反転、四元数成分の符号調整）
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
            pairs[idx] = (idx, T_cv, left_path, right_path, R_cv)
        return pairs
