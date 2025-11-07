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
データタイプ選択の方針
1. 環境変数 DATA_TYPE が存在し、data配下に一致するディレクトリがあればそれを採用
2. 環境変数 DATA_TYPE_INDEX が存在すれば 1-indexed で採用
3. 環境変数 PM_INTERACTIVE=="1" のときのみ、従来の対話選択を有効化
4. 上記がなければ、data配下の最初のディレクトリを採用
HOME_DIR が未設定の場合は、mvsの親ディレクトリをプロジェクトルートとして扱う
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

# 非対話優先の選択
env_data_type = os.getenv("DATA_TYPE")
env_data_type_index = os.getenv("DATA_TYPE_INDEX")
PM_INTERACTIVE = os.getenv("PM_INTERACTIVE") == "1"

if env_data_type and env_data_type in directories:
    DATA_TYPE = env_data_type
elif env_data_type_index is not None:
    try:
        idx = int(env_data_type_index)
        if 1 <= idx <= len(directories):
            DATA_TYPE = directories[idx - 1]
        else:
            raise ValueError
    except Exception:
        raise ValueError(
            f"Invalid DATA_TYPE_INDEX: {env_data_type_index}. Valid range is [1-{len(directories)}]"
        )
elif PM_INTERACTIVE:
    print("Select data type:")
    for i, directory in enumerate(directories):
        print(f"{i+1}) {directory}")
    choice = input(f"Enter choice [1-{len(directories)}]: ")
    try:
        choice_index = int(choice) - 1
        if choice_index < 0 or choice_index >= len(directories):
            raise ValueError
        DATA_TYPE = directories[choice_index]
    except ValueError:
        print("Invalid choice. Exiting.")
        exit(1)
else:
    # 非対話デフォルト: 先頭を採用
    DATA_TYPE = directories[0]

print(f"Selected data type: {DATA_TYPE}")

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

window_size, min_disp, num_disp = 7, 0, 216

# YAML(app/mvs.yaml もしくは APP_MVS_CONFIG) による MVS パラメータの上書き
try:
    mvs_yaml_path = os.getenv(
        "APP_MVS_CONFIG", os.path.join(DEFAULT_HOME, "app", "mvs.yaml")
    )
    if yaml and os.path.exists(mvs_yaml_path):
        with open(mvs_yaml_path, "r") as f:
            _cfg = yaml.safe_load(f) or {}
        if isinstance(_cfg, dict):
            g = globals()
            for k, v in _cfg.items():
                if v is not None:
                    g[k] = v
except Exception:
    # YAML 読み込みに失敗した場合は、上位でのエラーハンドリングに委ねる
    pass
