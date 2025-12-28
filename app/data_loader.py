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

    def _select_frame_indices(self):
        """
        Select a subset of frame indices (indices into self.camera_data) to reduce redundant views.

        Selection modes (via app/mvs.yaml overrides read in mvs/config.py):
          - FRAME_SELECTION_MODE: "none" | "stride" | "pose"
          - FRAME_STRIDE: int (used when mode == "stride")
          - KEYFRAME_MIN_TRANSLATION_M: float (used when mode == "pose")
          - KEYFRAME_MIN_ROTATION_DEG: float (used when mode == "pose")
          - KEYFRAME_MAX_FRAME_GAP: int (force-select at least every N frames; used when mode == "pose")
          - KEYFRAME_MIN_FRAME_GAP: int (do not select frames closer than this gap unless forced; used when mode == "pose")
          - KEYFRAME_START_INDEX / KEYFRAME_END_INDEX: optional inclusive range clamp

        Rationale (photogrammetry/CV):
          - Too-dense frames add compute but little parallax; translation/rotation thresholds keep useful baselines.
          - A max-gap guard prevents long dead zones that can hurt multi-view fusion/neighbor selection.
        """
        mode = str(getattr(config, "FRAME_SELECTION_MODE", "none")).strip().lower()

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

        if mode in ("none", "", "off", "false"):
            return filtered

        if mode == "stride":
            stride = int(getattr(config, "FRAME_STRIDE", 1) or 1)
            stride = max(1, stride)
            return filtered[::stride]

        if mode != "pose":
            logging.warning(f"Unknown FRAME_SELECTION_MODE={mode!r}; falling back to none.")
            return filtered

        # pose-based keyframe selection
        min_t = float(getattr(config, "KEYFRAME_MIN_TRANSLATION_M", 0.0) or 0.0)
        min_r_deg = float(getattr(config, "KEYFRAME_MIN_ROTATION_DEG", 0.0) or 0.0)
        max_gap = int(getattr(config, "KEYFRAME_MAX_FRAME_GAP", 0) or 0)
        min_gap = int(getattr(config, "KEYFRAME_MIN_FRAME_GAP", 0) or 0)
        max_gap = max(0, max_gap)
        min_gap = max(0, min_gap)

        if min_t <= 0.0 and min_r_deg <= 0.0 and max_gap <= 0:
            logging.warning(
                "pose mode selected but KEYFRAME_MIN_TRANSLATION_M/KEYFRAME_MIN_ROTATION_DEG/KEYFRAME_MAX_FRAME_GAP are all disabled; falling back to using all candidate frames."
            )
            return filtered

        selected = []
        last_sel = None
        last_pos = None
        last_rot = None

        for i in filtered:
            fn, pos, quat = self.camera_data[i]
            pos_np = np.array(pos, dtype=np.float64)
            try:
                rot = Rotation.from_quat(np.array(quat, dtype=np.float64))
            except Exception:
                # If quaternion is malformed, keep the frame (safer than dropping)
                rot = None

            if last_sel is None:
                selected.append(i)
                last_sel = i
                last_pos = pos_np
                last_rot = rot
                continue

            gap = i - int(last_sel)
            force_by_gap = (max_gap > 0) and (gap >= max_gap)
            if not force_by_gap and (gap < min_gap):
                continue

            # translation
            t_ok = False
            if last_pos is not None and min_t > 0.0:
                t = float(np.linalg.norm(pos_np - last_pos))
                t_ok = t >= min_t

            # rotation
            r_ok = False
            if min_r_deg > 0.0 and (rot is not None) and (last_rot is not None):
                try:
                    delta = (last_rot.inv() * rot)
                    r_deg = float(np.degrees(delta.magnitude()))
                    r_ok = r_deg >= min_r_deg
                except Exception:
                    r_ok = True

            # Select if either translation OR rotation threshold is exceeded (typical keyframe heuristic),
            # or if forced by max-gap.
            if force_by_gap or t_ok or r_ok:
                selected.append(i)
                last_sel = i
                last_pos = pos_np
                last_rot = rot

        logging.info(
            f"Frame selection mode={mode}: selected {len(selected)}/{len(filtered)} frames "
            f"(range {start}-{end}, thresholds: t>={min_t}m r>={min_r_deg}deg, gap min={min_gap} max={max_gap})"
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
