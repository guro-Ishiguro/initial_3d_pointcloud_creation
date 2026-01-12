import csv
import logging
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import mvs.config as config


class DataLoader:
    def __init__(self, image_root_dir, left_camera_poses_csv):
        # images ディレクトリのルート
        self.image_root_dir = image_root_dir
        self.left_dir = os.path.join(image_root_dir, "image_0")
        self.right_dir = os.path.join(image_root_dir, "image_1")
        self.left_camera_poses_csv = left_camera_poses_csv
        self.camera_data = self._load_camera_data()

    def _load_camera_data(self):
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
        カメラポーズに誤差を追加する。

        Args:
            pos: カメラ位置 (x, y, z) [メートル単位]
            quat: カメラ回転 (qx, qy, qz, qw) [クォータニオン]
            pos_scale: 位置誤差の標準偏差 [メートル単位]
            rot_scale: 回転誤差の標準偏差 [ラジアン単位]

        Returns:
            pos_with_error: 誤差が追加された位置
            quat_with_error: 誤差が追加された回転
        """
        # 位置誤差: 各軸に独立した正規分布の誤差を追加
        # pos_scaleは標準偏差（メートル単位）
        if pos_scale > 0:
            pos_error = np.random.randn(3) * pos_scale
            pos_with_error = (
                pos[0] + pos_error[0],
                pos[1] + pos_error[1],
                pos[2] + pos_error[2],
            )
        else:
            pos_with_error = pos

        # 回転誤差: 回転ベクトル（axis-angle表現）に正規分布の誤差を追加
        # rot_scaleは標準偏差（ラジアン単位）
        # 注意: 1.0ラジアン ≈ 57度は非常に大きな誤差
        # 実際のSfM誤差は通常0.01-0.1ラジアン（約0.6-6度）程度
        if rot_scale > 0:
            rot_vec_error = np.random.randn(3) * rot_scale
            error_rotation = Rotation.from_rotvec(rot_vec_error)
            original_rotation = Rotation.from_quat(quat)
            # 誤差回転を元の回転に合成（右から掛ける）
            rotated_orientation = original_rotation * error_rotation
            quat_with_error = rotated_orientation.as_quat()
        else:
            quat_with_error = quat

        return pos_with_error, quat_with_error

    def _select_frame_indices(self):
        """
        Select a subset of frame indices (indices into self.camera_data) using stride-based selection.

        Parameters (via app/mvs.yaml overrides read in mvs/config.py):
          - FRAME_STRIDE: int (stride value for frame selection)
          - KEYFRAME_START_INDEX / KEYFRAME_END_INDEX: optional inclusive range clamp
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

        # Filter: ignore rows that look like right camera entries (some logs may include them)
        filtered = []
        for i in candidates:
            fn, _, _ = self.camera_data[i]
            if str(fn).lower().startswith("right_"):
                continue
            filtered.append(i)

        # Stride-based selection
        stride = int(getattr(config, "FRAME_STRIDE", 1) or 1)
        stride = max(1, stride)
        selected = filtered[::stride]

        logging.info(
            f"Frame selection (stride={stride}): selected {len(selected)}/{len(filtered)} frames "
            f"(range {start}-{end})"
        )
        return selected

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
        """
        Build mapping from original frame index -> (idx, T_cv, left_path, right_path, R_cv).
        Keeps idx aligned to self.camera_data indices, so downstream code can use idx consistently
        even when frames are subsampled.
        """
        pairs = {}
        position_error_scale = config.POSITION_ERROR_SCALE
        rotation_error_scale = config.ROTATION_ERROR_SCALE
        selected_indices = self._select_frame_indices()
        for idx in selected_indices:
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
            pairs[idx] = (idx, T_cv, left_path, right_path, R_cv)
        return pairs
