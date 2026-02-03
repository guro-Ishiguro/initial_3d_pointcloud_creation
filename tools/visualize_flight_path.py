"""
カメラポーズ（フライトパス）を3Dで可視化するスクリプト。

output/<dataset>/plots/selected_poses.csv と data/<dataset>/txt/camera_params.csv を読み、
カメラ中心の軌跡と各フレームの視錐台（フラスタム）を Matplotlib の 3D プロットで表示する。
実行するとデータセットの対話選択が行われる。
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def list_datasets(data_dir: str) -> list:
    """
    data ディレクトリ内で、images サブディレクトリを持つデータセット名の一覧を返す。
    """
    if not os.path.isdir(data_dir):
        return []
    datasets = [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and os.path.isdir(os.path.join(data_dir, d, "images"))
    ]
    datasets.sort()
    return datasets


def select_dataset_interactively(data_dir: str) -> str:
    """
    利用可能なデータセットを番号付きで表示し、ユーザー入力で1つ選択してその名前を返す。
    """
    available_datasets = list_datasets(data_dir)
    if not available_datasets:
        print(f"エラー: {data_dir} にデータセットが見つかりません。")
        sys.exit(1)

    print("利用可能なデータセット:")
    for i, ds in enumerate(available_datasets, 1):
        print(f"  {i}) {ds}")

    print("\n可視化するデータセットを選択してください:")
    choice = input("> ").strip()
    if not choice:
        print("データセットが選択されませんでした。")
        sys.exit(1)

    # 選択をパース
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(available_datasets):
            return available_datasets[idx]
    elif choice in available_datasets:
        return choice

    print(f"エラー: 無効な選択です: {choice}")
    sys.exit(1)


# =============================================================================
# データセットの選択とパス設定
# =============================================================================
data_dir = "data"
if not os.path.isdir(data_dir):
    print(f"エラー: dataディレクトリが見つかりません: {data_dir}")
    sys.exit(1)

dataset_name = select_dataset_interactively(data_dir)
print(f"\n選択されたデータセット: {dataset_name}\n")

# カメラパラメータと選択済みポーズCSVのパス
camera_params_path = os.path.join(data_dir, dataset_name, "txt", "camera_params.csv")
selected_poses_path = os.path.join(
    "output", dataset_name, "plots", "selected_poses.csv"
)

# 必須ファイルの存在確認
if not os.path.exists(camera_params_path):
    print(f"エラー: camera_params.csv が見つかりません: {camera_params_path}")
    sys.exit(1)

if not os.path.exists(selected_poses_path):
    print(f"警告: selected_poses.csv が見つかりません: {selected_poses_path}")
    print("このファイルがない場合、可視化を続行できません。")
    sys.exit(1)

# ポーズ（位置・四元数）とカメラFOVの読み込み
poses_df = pd.read_csv(selected_poses_path)
params_df = pd.read_csv(camera_params_path)

# 位置ベクトルと四元数、FOVを配列として取得
poses = poses_df[["pos_x", "pos_y", "pos_z"]].values
quats = poses_df[["rot_x", "rot_y", "rot_z", "rot_w"]].values
fov_h = params_df["fov_h_deg"][0]
fov_v = params_df["fov_v_deg"][0]


def get_frustum_local(fov_h, fov_v, scale=1.0):
    """
    カメラローカル座標系で、視錐台（フラスタム）の頂点を計算する。

    カメラ中心 (0,0,0) と、前方 scale 距離のイメージプレーン四隅の5点を返す。
    FOV は水平・垂直の半角（度数法）で指定。

    Returns:
        [p_center, p_tr, p_tl, p_bl, p_br]: 中心と右上・左上・左下・右下の5点
    """
    rad_h = np.deg2rad(fov_h)
    rad_v = np.deg2rad(fov_v)

    # カメラの前方（Z軸）にscale距離離れた位置のイメージプレーンを計算
    z = scale
    x = z * np.tan(rad_h / 2)
    y = z * np.tan(rad_v / 2)

    # 頂点：中心(0,0,0) と イメージプレーンの4隅
    p_center = np.array([0, 0, 0])
    p_tr = np.array([x, -y, z])  # 右上
    p_tl = np.array([-x, -y, z])  # 左上
    p_bl = np.array([-x, y, z])  # 左下
    p_br = np.array([x, y, z])  # 右下

    return [p_center, p_tr, p_tl, p_bl, p_br]


# =============================================================================
# 3D プロットの作成と描画
# =============================================================================
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection="3d")

# 表示用座標変換: (x, y, z) → (Z, X, Y) で Matplotlib 3D に合わせる
poses_transformed = np.column_stack([poses[:, 2], poses[:, 0], poses[:, 1]])

# カメラ中心の軌跡を黒線で描画
ax.plot(
    poses_transformed[:, 0],
    poses_transformed[:, 1],
    poses_transformed[:, 2],
    "-k",
    label="Flight Path",
    linewidth=1,
    alpha=0.5,
)

# 視錐台のサイズ（カメラ前方にこの距離でイメージプレーンを描く）
frustum_scale = 3.0
local_frustum = get_frustum_local(fov_h, fov_v, scale=frustum_scale)
frustum_color = "blue"

# 各フレームのカメラポーズでフラスタムをワールド座標に変換して描画
for i in range(len(poses)):
    pos = poses[i]
    quat = quats[i]  # ローテーション (x, y, z, w)

    # クォータニオンから回転行列を作成
    r = R.from_quat(quat)
    mat = r.as_matrix()

    # フラスタムの頂点をグローバル座標に変換
    global_frustum = []
    for p in local_frustum:
        p_rot = mat.dot(p)  # 回転
        p_glob = p_rot + pos  # 平行移動
        # 座標変換（poses_transformed と同じ割当）
        # (X, Y, Z) = (Z, X, Y)
        p_transformed = np.array([p_glob[2], p_glob[0], p_glob[1]])
        global_frustum.append(p_transformed)

    c, tr, tl, bl, br = global_frustum

    # 中心から四隅への線を描画
    for corner in [tr, tl, bl, br]:
        ax.plot(
            [c[0], corner[0]],
            [c[1], corner[1]],
            [c[2], corner[2]],
            color=frustum_color,
            alpha=0.8,
            linewidth=0.8,
        )

    # イメージプレーン（底面）の矩形を描画
    rect = [tr, tl, bl, br, tr]
    xs = [p[0] for p in rect]
    ys = [p[1] for p in rect]
    zs = [p[2] for p in rect]
    ax.plot(xs, ys, zs, color=frustum_color, alpha=0.8, linewidth=0.8)

# カメラ中心を赤い点で描画
ax.scatter(
    poses_transformed[:, 0],
    poses_transformed[:, 1],
    poses_transformed[:, 2],
    c="red",
    s=10,
    label="Camera Center",
)

# 軸ラベルとタイトル（座標変換 (Z,X,Y) に合わせて表示）
ax.set_xlabel("Z")
ax.set_ylabel("X")
ax.set_zlabel("Y")
ax.set_title("Camera Poses and Flight Path")

# アスペクト比を揃えて表示範囲を設定
x_limits = ax.get_xlim3d()
y_limits = ax.get_ylim3d()
z_limits = ax.get_zlim3d()
x_range = abs(x_limits[1] - x_limits[0])
y_range = abs(y_limits[1] - y_limits[0])
z_range = abs(z_limits[1] - z_limits[0])
max_range = max([x_range, y_range, z_range])

mid_x = np.mean(x_limits)
mid_y = np.mean(y_limits)
mid_z = np.mean(z_limits)

# ズーム率で表示範囲を調整（1.0=全体表示、0.6=やや拡大）
zoom_factor = 0.6
zoom_range = max_range * zoom_factor
ax.set_xlim(mid_x - zoom_range / 2, mid_x + zoom_range / 2)
ax.set_ylim(mid_y - zoom_range / 2, mid_y + zoom_range / 2)
ax.set_zlim(mid_z - zoom_range / 2, mid_z + zoom_range / 2)

# 表示上の Z 軸を反転（目盛り方向の調整）
ax.invert_xaxis()

# 初期視点: elev=仰角(度), azim=方位角(度)
ax.view_init(elev=30, azim=-90)

plt.show()
