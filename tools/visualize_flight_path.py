import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import os
import sys


def list_datasets(data_dir: str) -> list:
    """dataディレクトリ内のデータセット一覧を取得"""
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
    """対話的にデータセットを選択"""
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


# データセットの選択
data_dir = "data"
if not os.path.isdir(data_dir):
    print(f"エラー: dataディレクトリが見つかりません: {data_dir}")
    sys.exit(1)

dataset_name = select_dataset_interactively(data_dir)
print(f"\n選択されたデータセット: {dataset_name}\n")

# ファイルパスの設定
camera_params_path = os.path.join(data_dir, dataset_name, "txt", "camera_params.csv")
global_selected_poses_path = os.path.join("output", dataset_name, "plots", "csv", "global_selected_poses.csv")

# ファイルの存在確認
if not os.path.exists(camera_params_path):
    print(f"エラー: camera_params.csv が見つかりません: {camera_params_path}")
    sys.exit(1)

if not os.path.exists(global_selected_poses_path):
    print(f"警告: global_selected_poses.csv が見つかりません: {global_selected_poses_path}")
    print("このファイルがない場合、可視化を続行できません。")
    sys.exit(1)

# データの読み込み
poses_df = pd.read_csv(global_selected_poses_path)
params_df = pd.read_csv(camera_params_path)

# データの抽出
poses = poses_df[['pos_x', 'pos_y', 'pos_z']].values
quats = poses_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
fov_h = params_df['fov_h_deg'][0]
fov_v = params_df['fov_v_deg'][0]

# ローカル座標系でのフラスタム（視錐台）の頂点を計算する関数
def get_frustum_local(fov_h, fov_v, scale=1.0):
    # 度数法をラジアンに変換
    rad_h = np.deg2rad(fov_h)
    rad_v = np.deg2rad(fov_v)
    
    # カメラの前方（Z軸）にscale距離離れた位置のイメージプレーンを計算
    z = scale
    x = z * np.tan(rad_h / 2)
    y = z * np.tan(rad_v / 2)
    
    # 頂点：中心(0,0,0) と イメージプレーンの4隅
    p_center = np.array([0, 0, 0])
    p_tr = np.array([x, -y, z])  # 右上
    p_tl = np.array([-x, -y, z]) # 左上
    p_bl = np.array([-x, y, z])  # 左下
    p_br = np.array([x, y, z])   # 右下
    
    return [p_center, p_tr, p_tl, p_bl, p_br]

# 3Dプロットの作成
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# フライトパスの描画（黒線）
ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], '-k', label='Flight Path', linewidth=1, alpha=0.5)

# フラスタムのスケール設定（シーンに合わせて調整）
frustum_scale = 3.0
local_frustum = get_frustum_local(fov_h, fov_v, scale=frustum_scale)
colors = plt.cm.viridis(np.linspace(0, 1, len(poses)))

# 各カメラポーズごとの描画ループ
for i in range(len(poses)):
    pos = poses[i]
    quat = quats[i] # ローテーション (x, y, z, w)
    
    # クォータニオンから回転行列を作成
    r = R.from_quat(quat)
    mat = r.as_matrix()
    
    # フラスタムの頂点をグローバル座標に変換
    global_frustum = []
    for p in local_frustum:
        p_rot = mat.dot(p)     # 回転
        p_glob = p_rot + pos   # 平行移動
        global_frustum.append(p_glob)
        
    c, tr, tl, bl, br = global_frustum
    
    # 中心から四隅への線を描画
    for corner in [tr, tl, bl, br]:
        ax.plot([c[0], corner[0]], [c[1], corner[1]], [c[2], corner[2]], color=colors[i], alpha=0.8, linewidth=0.8)
    
    # イメージプレーン（底面）の矩形を描画
    rect = [tr, tl, bl, br, tr]
    xs = [p[0] for p in rect]
    ys = [p[1] for p in rect]
    zs = [p[2] for p in rect]
    ax.plot(xs, ys, zs, color=colors[i], alpha=0.8, linewidth=0.8)

# カメラ中心位置を点で描画（赤点）
ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], c='red', s=10, label='Camera Center')

# 軸ラベルの設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Poses and Flight Path')

# 軸のスケールを揃える（アスペクト比を等しくする）
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

ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

plt.show()