from dotenv import load_dotenv
import numpy as np
import os

# .envファイルの読み込み
load_dotenv()

# 環境変数の取得
HOME_DIR = os.getenv("HOME_DIR")

# dataディレクトリ配下のディレクトリを取得
DATA_DIR = os.path.join(HOME_DIR, "data")
directories = [
    d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
]

if not directories:
    raise FileNotFoundError(f"No directories found in {DATA_DIR}")

# ユーザーに選択肢を提示
print("Select data type:")
for i, directory in enumerate(directories):
    print(f"{i+1}) {directory}")

choice = input(f"Enter choice [1-{len(directories)}]: ")

# 入力を検証
try:
    choice_index = int(choice) - 1
    if choice_index < 0 or choice_index >= len(directories):
        raise ValueError
    DATA_TYPE = directories[choice_index]
except ValueError:
    print("Invalid choice. Exiting.")
    exit(1)

print(f"Selected data type: {DATA_TYPE}")

# パスの設定
DATA_TYPE_DIR = os.path.join(DATA_DIR, DATA_TYPE)
IMAGE_DIR = os.path.join(DATA_TYPE_DIR, "images")
TXT_DIR = os.path.join(DATA_TYPE_DIR, "txt")
DRONE_IMAGE_LOG = os.path.join(TXT_DIR, "drone_image_log.txt")
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
HISTGRAM_DIR = os.path.join(OUTPUT_TYPE_DIR, "histgram")

# スケールの設定
B, fov_h, fov_v, width, height = (
    float(DATA_TYPE.split("_")[5]),
    float(DATA_TYPE.split("_")[4]),
    float(DATA_TYPE.split("_")[3]),
    int(DATA_TYPE.split("_")[0]),
    int(DATA_TYPE.split("_")[1]),
)
focal_length = width / (2 * np.tan(fov_h * np.pi / 180 / 2))
camera_height = int(DATA_TYPE.split("_")[2])
cx, cy = int(DATA_TYPE.split("_")[0]) / 2, int(DATA_TYPE.split("_")[1]) / 2
K = np.array(
    [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32
)
scene_width = 2 * camera_height * np.tan(np.radians(fov_h) / 2)
scene_height = 2 * camera_height * np.tan(np.radians(fov_v) / 2)
pixel_size = scene_width / width

window_size, min_disp, num_disp = 5, 0, 216

DEBUG_PATCH_MATCH_VISUALIZATION = False
# rows, colums
DEBUG_PIXEL_COORDS = (230, 490)

# PatchMatchの反復回数
PATCHMATCH_ITERATIONS = 5
# パッチサイズ (奇数)
PATCHMATCH_PATCH_SIZE = 9
# 法線推定に使う近傍のサイズ
NORMAL_ESTIMATION_NEIGHBORHOOD = 7
# ZNCCコスト計算時の小さな値
ZNCC_EPSILON = 1e-6
# 複数視点コストを集計する際の上位何個を考慮するか
TOP_K_COSTS = 3

# --- 適応的ランダム探索のパラメータ ---
# 探索範囲のイテレーションごとの減衰率
PATCHMATCH_DECAY_RATE = 0.9
# 法線探索の角度範囲(degree)の初期値
PATCHMATCH_NORMAL_SEARCH_ANGLE = 20.0

ADAPTIVE_WEIGHT_SIGMA_COLOR = 10

# 処理中の点群などをウィンドウで表示するか
DEBUG_VISUALIZATION = True
# 最適化前後のデプスマップを画像として保存するか
DEBUG_SAVE_DEPTH_MAPS = True

# 深度をランダムに初期化する際の範囲
PATCHMATCH_VANILLA_MIN_DEPTH = 5.0  # meters
PATCHMATCH_VANILLA_MAX_DEPTH = 50.0  # meters
# ランダム探索の初期探索幅
PATCHMATCH_VANILLA_INITIAL_SEARCH_RANGE = 50.0  # meters
