import csv
import os

import numpy as np
from dotenv import load_dotenv

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# .envファイルの読み込み
load_dotenv()

"""
データセット選択の方針:
- 環境変数DATA_TYPEが設定されていればそれを使用
- 設定されていなければ、最初のデータセットを自動選択
- 対話的な選択は行わない（app/cli.pyで行う）
"""

# プロジェクトルート推定と環境変数の取得
DEFAULT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOME_DIR = os.getenv("HOME_DIR", DEFAULT_HOME)

# dataディレクトリ配下のディレクトリを取得
DATA_DIR = os.path.join(HOME_DIR, "data")
directories = [
    d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
]
if not directories:
    raise FileNotFoundError(f"No directories found in {DATA_DIR}")
directories.sort()

_env_data_type = os.getenv("DATA_TYPE", "").strip()


def _is_valid_dataset_name(name: str) -> bool:
    return bool(name) and os.path.isdir(os.path.join(DATA_DIR, name))


# 環境変数DATA_TYPEが設定されていればそれを使用、なければ最初のデータセットを自動選択
if _is_valid_dataset_name(_env_data_type):
    DATA_TYPE = _env_data_type
else:
    # 環境変数が設定されていない場合は、最初のデータセットを自動選択
    DATA_TYPE = directories[0]

# パスの設定
DATA_TYPE_DIR = os.path.join(DATA_DIR, DATA_TYPE)
IMAGE_ROOT_DIR = os.path.join(DATA_TYPE_DIR, "images")
LEFT_IMAGE_DIR = os.path.join(IMAGE_ROOT_DIR, "image_0")
RIGHT_IMAGE_DIR = os.path.join(IMAGE_ROOT_DIR, "image_1")
LABEL_DEPTH_IMAGE_DIR = os.path.join(IMAGE_ROOT_DIR, "depth")
TXT_DIR = os.path.join(DATA_TYPE_DIR, "txt")
LEFT_CAMERA_POSES = os.path.join(TXT_DIR, "left_camera_poses.csv")
CAMERA_PARAMS_CSV = os.path.join(TXT_DIR, "camera_params.csv")
ORB_SLAM_LOG = os.path.join(TXT_DIR, "KeyFrameTrajectory.txt")

OUTPUT_DIR = os.path.join(HOME_DIR, "output")
OUTPUT_TYPE_DIR = os.path.join(OUTPUT_DIR, DATA_TYPE)
POINT_CLOUD_DIR = os.path.join(OUTPUT_TYPE_DIR, "point_cloud")
POINT_CLOUD_FILE_PATH = os.path.join(POINT_CLOUD_DIR, "output.ply")
OLD_POINT_CLOUD_FILE_PATH = os.path.join(POINT_CLOUD_DIR, "old_output.ply")
MESH_DIR = os.path.join(OUTPUT_TYPE_DIR, "mesh")
MESH_FILE_PATH = os.path.join(MESH_DIR, "mesh.ply")
VIDEO_DIR = os.path.join(OUTPUT_TYPE_DIR, "video")
DISPARITY_IMAGE_DIR = os.path.join(OUTPUT_TYPE_DIR, "disparity")
DEPTH_IMAGE_DIR = os.path.join(OUTPUT_TYPE_DIR, "depth")
NORMAL_IMAGE_DIR = os.path.join(OUTPUT_TYPE_DIR, "normal")
HISTGRAM_DIR = os.path.join(OUTPUT_TYPE_DIR, "histgram")
CSV_DIR = os.path.join(OUTPUT_TYPE_DIR, "csv")

"""
カメラ設定の読み込み
必須: データセット内の txt/camera_params.csv
   - baseline,width,height,camera_height,fov_v_deg,fov_h_deg,fx_pixels,fy_pixels,cx_pixels,cy_pixels
"""


def _load_camera_params_from_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if not row:
                return None
            params = {
                "B": float(row.get("baseline")),
                "width": int(row.get("width")),
                "height": int(row.get("height")),
                "camera_height": float(row.get("camera_height")),
                "fov_v": float(row.get("fov_v_deg")),
                "fov_h": float(row.get("fov_h_deg")),
                "fx": float(row.get("fx_pixels")) if row.get("fx_pixels") else None,
                "fy": float(row.get("fy_pixels")) if row.get("fy_pixels") else None,
                "cx": float(row.get("cx_pixels")) if row.get("cx_pixels") else None,
                "cy": float(row.get("cy_pixels")) if row.get("cy_pixels") else None,
            }
            return params
    except Exception:
        return None


_cam_cfg = _load_camera_params_from_csv(CAMERA_PARAMS_CSV)
if _cam_cfg is None:
    raise FileNotFoundError(
        f"camera_params.csv not found or invalid: {CAMERA_PARAMS_CSV}. This file is required."
    )

_required_keys = ["width", "height", "camera_height", "fov_h", "fov_v", "B"]
_missing = [k for k in _required_keys if _cam_cfg.get(k) is None]
if _missing:
    raise ValueError(
        f"Missing required camera parameters: {_missing} in {CAMERA_PARAMS_CSV}"
    )

width = int(_cam_cfg["width"])  # pixels
height = int(_cam_cfg["height"])  # pixels
B = float(_cam_cfg["B"])  # meters
camera_height = float(_cam_cfg["camera_height"])  # meters
fov_h = float(_cam_cfg["fov_h"])  # degrees
fov_v = float(_cam_cfg["fov_v"])  # degrees

# intrinsics
fx = (
    float(_cam_cfg.get("fx"))
    if _cam_cfg.get("fx") is not None
    else width / (2.0 * np.tan(np.deg2rad(fov_h) / 2.0))
)
_fy = _cam_cfg.get("fy")
fy = float(_fy) if _fy is not None else fx
_cx = _cam_cfg.get("cx")
cx = float(_cx) if _cx is not None else (width / 2.0)
_cy = _cam_cfg.get("cy")
cy = float(_cy) if _cy is not None else (height / 2.0)
focal_length = float(fx)
K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
# for orthographic plane sizing
scene_width = 2.0 * camera_height * np.tan(np.deg2rad(fov_h) / 2.0)
scene_height = 2.0 * camera_height * np.tan(np.deg2rad(fov_v) / 2.0)
pixel_size = scene_width / float(width)

# YAML(app/mvs.yaml もしくは APP_MVS_CONFIG) による MVS パラメータの読み込み（必須）
# 視差推定パラメータと可視化パラメータはYAMLから読み込まれる
mvs_yaml_path = os.getenv(
    "APP_MVS_CONFIG", os.path.join(DEFAULT_HOME, "app", "mvs.yaml")
)
if not yaml:
    raise ImportError(
        "PyYAML is required to load configuration. Please install: pip install PyYAML"
    )
if not os.path.exists(mvs_yaml_path):
    raise FileNotFoundError(
        f"Configuration file not found: {mvs_yaml_path}. "
        "Please ensure app/mvs.yaml exists or set APP_MVS_CONFIG environment variable."
    )

with open(mvs_yaml_path, "r") as f:
    _cfg = yaml.safe_load(f) or {}

if not isinstance(_cfg, dict):
    raise ValueError(f"Invalid YAML configuration format in {mvs_yaml_path}")

# 必須パラメータのチェック
_required_params = [
    "WINDOW_SIZE",
    "MIN_DISP",
    "NUM_DISP",
    "VIZ_CMAP",
]
_missing_params = [p for p in _required_params if p not in _cfg or _cfg[p] is None]
if _missing_params:
    raise ValueError(
        f"Missing required parameters in {mvs_yaml_path}: {', '.join(_missing_params)}"
    )

# グローバル変数に設定を反映
g = globals()
for k, v in _cfg.items():
    if v is not None:
        g[k] = v

if "VIZ_DEPTH_MIN" not in _cfg or _cfg["VIZ_DEPTH_MIN"] is None:
    VIZ_DEPTH_MIN = 0
    g["VIZ_DEPTH_MIN"] = VIZ_DEPTH_MIN
    import logging

    logging.info(
        f"VIZ_DEPTH_MIN auto-calculated from camera_height: {VIZ_DEPTH_MIN:.2f} (camera_height={camera_height:.2f} * 0.5)"
    )

if "VIZ_DEPTH_MAX" not in _cfg or _cfg["VIZ_DEPTH_MAX"] is None:
    # カメラ高度の1.0倍を最大値とする
    VIZ_DEPTH_MAX = camera_height * 1.0
    g["VIZ_DEPTH_MAX"] = VIZ_DEPTH_MAX
    import logging

    logging.info(
        f"VIZ_DEPTH_MAX auto-calculated from camera_height: {VIZ_DEPTH_MAX:.2f} (camera_height={camera_height:.2f} * 1.0)"
    )

# VIZ_CMAPが設定された場合、ログに出力
if "VIZ_CMAP" in _cfg:
    import logging

    logging.info(
        f"VIZ_CMAP loaded from YAML: {_cfg['VIZ_CMAP']} (file: {mvs_yaml_path})"
    )


# 環境変数からの設定読み込み（GUIから渡された設定を反映）
# ブール値の文字列を適切に変換
def _str_to_bool(s):
    """文字列をブール値に変換"""
    if isinstance(s, bool):
        return s
    if isinstance(s, str):
        return s.lower() in ("true", "1", "yes", "on")
    return bool(s)


# 環境変数から設定を読み込む
g = globals()
# 主要な設定項目のリスト
_config_keys = [
    "SHOW_POINT_CLOUD",
    "POSITION_ERROR_SCALE",
    "ROTATION_ERROR_SCALE",
    "DEBUG_SAVE_DEPTH_MAPS",
    "DEBUG_SAVE_NORMAL_MAPS",
    "DEBUG_SAVE_GT_DEPTH_MAPS",
    "PATCHMATCH_ITERATIONS",
    "PATCHMATCH_PATCH_SIZE",
    "ZNCC_EPSILON",
    "TOP_K_COSTS",
    "PATCHMATCH_DECAY_RATE",
    "PATCHMATCH_NORMAL_SEARCH_ANGLE",
    "ADAPTIVE_WEIGHT_SIGMA_COLOR",
    "FILTERING_COLOR_DIFFERENCE_THRESHOLD",
    "FILTERING_MIN_CONSISTENT_VIEWS",
    "GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD",
    "GEOMETRIC_MIN_CONSISTENT_VIEWS",
    "FRAME_STRIDE",
    "MAX_NEIGHBORS",
    "NEIGHBOR_SELECTION_MODE",
    "NEIGHBOR_NEAREST_COUNT",
    "VIZ_DEPTH_MIN",
    "VIZ_DEPTH_MAX",
    "VIZ_CMAP",
    "MULTI_VIEW_VISIBILITY_FILTER_ENABLED",
    "MULTI_VIEW_VISIBILITY_THRESHOLD",
    "MULTI_VIEW_GEOMETRIC_ERROR_THRESHOLD",
]

for key in _config_keys:
    env_value = os.getenv(key)
    if env_value is not None:
        # 既にYAMLで設定されている場合はスキップ（YAMLが優先）
        if key not in g or g[key] == getattr(__builtins__, key, None):
            # ブール値の場合は文字列を変換
            if key in (
                "SHOW_POINT_CLOUD",
                "DEBUG_SAVE_DEPTH_MAPS",
                "DEBUG_SAVE_NORMAL_MAPS",
                "DEBUG_SAVE_GT_DEPTH_MAPS",
                "MULTI_VIEW_VISIBILITY_FILTER_ENABLED",
            ):
                g[key] = _str_to_bool(env_value)
            # 数値の場合は型変換を試みる
            elif key in (
                "POSITION_ERROR_SCALE",
                "ROTATION_ERROR_SCALE",
                "ZNCC_EPSILON",
                "PATCHMATCH_DECAY_RATE",
                "PATCHMATCH_NORMAL_SEARCH_ANGLE",
                "ADAPTIVE_WEIGHT_SIGMA_COLOR",
                "FILTERING_COLOR_DIFFERENCE_THRESHOLD",
                "GEOMETRIC_CONSISTENCY_ERROR_THRESHOLD",
                "VIZ_DEPTH_MIN",
                "VIZ_DEPTH_MAX",
                "MULTI_VIEW_GEOMETRIC_ERROR_THRESHOLD",
            ):
                try:
                    g[key] = float(env_value)
                except (ValueError, TypeError):
                    pass
            # 整数の場合は型変換を試みる
            elif key in (
                "PATCHMATCH_ITERATIONS",
                "PATCHMATCH_PATCH_SIZE",
                "TOP_K_COSTS",
                "FILTERING_MIN_CONSISTENT_VIEWS",
                "GEOMETRIC_MIN_CONSISTENT_VIEWS",
                "FRAME_STRIDE",
                "MAX_NEIGHBORS",
                "NEIGHBOR_NEAREST_COUNT",
                "MULTI_VIEW_VISIBILITY_THRESHOLD",
            ):
                try:
                    g[key] = int(env_value)
                except (ValueError, TypeError):
                    pass
            # 文字列の場合はそのまま
            else:
                g[key] = env_value
