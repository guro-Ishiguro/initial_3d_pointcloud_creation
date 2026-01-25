import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from scipy.spatial.transform import Rotation as R


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

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(available_datasets):
            return available_datasets[idx]
    elif choice in available_datasets:
        return choice

    print(f"エラー: 無効な選択です: {choice}")
    sys.exit(1)


def get_frustum_local(fov_h_deg: float, fov_v_deg: float, scale: float = 1.0):
    """ローカル座標系でのフラスタム（視錐台）頂点を返す"""
    rad_h = np.deg2rad(float(fov_h_deg))
    rad_v = np.deg2rad(float(fov_v_deg))

    z = float(scale)
    x = z * np.tan(rad_h / 2)
    y = z * np.tan(rad_v / 2)

    p_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    p_tr = np.array([x, -y, z], dtype=np.float64)  # 右上
    p_tl = np.array([-x, -y, z], dtype=np.float64)  # 左上
    p_bl = np.array([-x, y, z], dtype=np.float64)  # 左下
    p_br = np.array([x, y, z], dtype=np.float64)  # 右下
    return [p_center, p_tr, p_tl, p_bl, p_br]


def to_display_coords_xyz(p_xyz: np.ndarray) -> np.ndarray:
    """
    表示用座標変換:
      元の (x, y, z) -> Matplotlib の (X, Y, Z) = (Z, X, Y)
    """
    return np.array([p_xyz[2], p_xyz[0], p_xyz[1]], dtype=np.float64)


def _parse_frame_index_from_filename(name: str) -> int:
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    try:
        return int(stem)
    except Exception:
        return -1


def load_poses_with_indices(data_dir: str, dataset_name: str) -> pd.DataFrame:
    # `left_camera_poses.csv` から index を復元する（filenameのstemがindex想定）
    poses_path = os.path.join(data_dir, dataset_name, "txt", "left_camera_poses.csv")
    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"left_camera_poses.csv が見つかりません: {poses_path}")
    df = pd.read_csv(poses_path)
    if "filename" not in df.columns:
        raise ValueError(f"{poses_path} に filename 列がありません。")
    df = df.copy()
    df["index"] = df["filename"].apply(_parse_frame_index_from_filename).astype(int)
    df = df[df["index"] >= 0].sort_values("index").reset_index(drop=True)
    return df


def load_camera_params(data_dir: str, dataset_name: str) -> Tuple[float, float]:
    camera_params_path = os.path.join(data_dir, dataset_name, "txt", "camera_params.csv")
    if not os.path.exists(camera_params_path):
        raise FileNotFoundError(f"camera_params.csv が見つかりません: {camera_params_path}")
    params_df = pd.read_csv(camera_params_path)
    fov_h = float(params_df["fov_h_deg"][0])
    fov_v = float(params_df["fov_v_deg"][0])
    return fov_h, fov_v


def load_selected_indices_from_output(dataset_name: str) -> List[int]:
    """
    可能なら output/<dataset>/csv のサブフォルダ名（000015など）から選択済みindexを復元する。
    見つからなければ空リスト。
    """
    out_csv_dir = os.path.join("output", dataset_name, "csv")
    if not os.path.isdir(out_csv_dir):
        return []
    out = []
    for d in os.listdir(out_csv_dir):
        p = os.path.join(out_csv_dir, d)
        if not os.path.isdir(p):
            continue
        try:
            out.append(int(d))
        except Exception:
            continue
    out.sort()
    return out


def select_neighbors(
    *,
    ref_pos_xyz: np.ndarray,
    ref_order_pos: int,
    all_indices: List[int],
    all_positions_xyz: np.ndarray,
    mode: str,
    nearest_count: int,
    r_min: float,
    r_max: float,
    adjacent_each_side: int,
) -> List[int]:
    mode = (mode or "nearest").strip().lower()
    if mode == "adjacent":
        k = max(0, int(adjacent_each_side))
        out = []
        for t in range(1, k + 1):
            j = ref_order_pos - t
            if j >= 0:
                out.append(all_indices[j])
        for t in range(1, k + 1):
            j = ref_order_pos + t
            if j < len(all_indices):
                out.append(all_indices[j])
        return out

    # nearest-by-distance (default)
    ref = np.array(ref_pos_xyz, dtype=np.float64)
    diffs = all_positions_xyz - ref[None, :]
    dists = np.linalg.norm(diffs, axis=1)

    # 自分自身を除外
    dists[ref_order_pos] = np.inf

    r_min = max(0.0, float(r_min))
    r_max = max(r_min, float(r_max))
    mask = (dists >= r_min) & (dists <= r_max)
    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        return []

    order = valid_idx[np.argsort(dists[valid_idx])]
    cnt = max(0, int(nearest_count))
    if cnt <= 0:
        return []
    order = order[:cnt]
    return [all_indices[i] for i in order.tolist()]


def build_frustum_segments_world(
    *,
    pos_xyz: np.ndarray,
    quat_xyzw: np.ndarray,
    local_frustum: List[np.ndarray],
) -> List[np.ndarray]:
    """フラスタムの線分（5頂点からの4本 + 底面矩形）を表示座標系で返す"""
    mat = R.from_quat(quat_xyzw).as_matrix()
    world = []
    for p in local_frustum:
        p_glob = mat.dot(p) + pos_xyz
        world.append(to_display_coords_xyz(p_glob))
    c, tr, tl, bl, br = world

    segments = []
    # rays
    for corner in [tr, tl, bl, br]:
        segments.append(np.vstack([c, corner]))
    # rectangle
    segments.append(np.vstack([tr, tl, bl, br, tr]))
    return segments


def plot_segments(ax, segments: List[np.ndarray], *, color: str, alpha: float, linewidth: float):
    artists = []
    for seg in segments:
        artists.append(
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                seg[:, 2],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )[0]
        )
    return artists


def apply_equal_axis_with_zoom(ax, *, zoom_factor: float):
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

    zoom_factor = float(zoom_factor)
    zoom_range = max_range * zoom_factor
    ax.set_xlim(mid_x - zoom_range / 2, mid_x + zoom_range / 2)
    ax.set_ylim(mid_y - zoom_range / 2, mid_y + zoom_range / 2)
    ax.set_zlim(mid_z - zoom_range / 2, mid_z + zoom_range / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="", help="data/<dataset> を指定（未指定なら対話選択）")
    parser.add_argument("--video", action="store_true", help="参照/近傍を色分けした動画を保存する")
    parser.add_argument("--out", default="", help="出力動画パス（例: output/<dataset>/plots/ref_neighbors.mp4）")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--frame-step", type=int, default=1, help="参照indexを何個おきに進めるか")

    parser.add_argument("--neighbor-mode", default="nearest", choices=["nearest", "adjacent"])
    parser.add_argument("--neighbor-count", type=int, default=10)
    parser.add_argument("--neighbor-each-side", type=int, default=5)
    parser.add_argument("--neighbor-r-min", type=float, default=0.0)
    parser.add_argument("--neighbor-r-max", type=float, default=1e9)

    parser.add_argument("--frustum-scale", type=float, default=3.0)
    parser.add_argument("--zoom", type=float, default=0.6)
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim", type=float, default=-90.0)
    parser.add_argument("--show", action="store_true", help="動画保存後にウィンドウ表示も行う")
    args = parser.parse_args()

    data_dir = "data"
    if not os.path.isdir(data_dir):
        print(f"エラー: dataディレクトリが見つかりません: {data_dir}")
        raise SystemExit(1)

    dataset_name = (args.dataset or "").strip()
    if not dataset_name:
        dataset_name = select_dataset_interactively(data_dir)
    print(f"\n選択されたデータセット: {dataset_name}\n")

    # poses + indices
    poses_df = load_poses_with_indices(data_dir, dataset_name)
    selected_indices = load_selected_indices_from_output(dataset_name)
    if selected_indices:
        poses_df = poses_df[poses_df["index"].isin(selected_indices)].copy()
        poses_df = poses_df.sort_values("index").reset_index(drop=True)

    indices = poses_df["index"].astype(int).tolist()
    poses_xyz = poses_df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
    quats_xyzw = poses_df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy(dtype=np.float64)

    if len(indices) == 0:
        print("エラー: 可視化できるポーズが0件です。")
        raise SystemExit(1)

    fov_h, fov_v = load_camera_params(data_dir, dataset_name)
    local_frustum = get_frustum_local(fov_h, fov_v, scale=float(args.frustum_scale))

    # 表示用の中心点（軌跡）
    centers_disp = np.column_stack([poses_xyz[:, 2], poses_xyz[:, 0], poses_xyz[:, 1]])

    # 各カメラのフラスタム線分を事前計算
    frustum_segments_by_idx: Dict[int, List[np.ndarray]] = {}
    for i, idx in enumerate(indices):
        frustum_segments_by_idx[idx] = build_frustum_segments_world(
            pos_xyz=poses_xyz[i],
            quat_xyzw=quats_xyzw[i],
            local_frustum=local_frustum,
        )

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # フライトパス（黒）
    ax.plot(
        centers_disp[:, 0],
        centers_disp[:, 1],
        centers_disp[:, 2],
        "-k",
        linewidth=1,
        alpha=0.4,
        label="Flight Path",
    )

    # ベース: 全フラスタムを黒で描画（動画でも静止画でも）
    base_frustum_artists_by_idx: Dict[int, List] = {}
    base_lw = 0.8
    base_alpha = 0.6
    for idx in indices:
        base_frustum_artists_by_idx[idx] = plot_segments(
            ax,
            frustum_segments_by_idx[idx],
            color="black",
            alpha=base_alpha,
            linewidth=base_lw,
        )

    # ベース: 全カメラ中心（薄黒）
    base_scatter = ax.scatter(
        centers_disp[:, 0],
        centers_disp[:, 1],
        centers_disp[:, 2],
        c="black",
        s=8,
        alpha=0.35,
        label="Camera Center",
    )

    # ハイライト（参照=赤、近傍=青）
    ref_scatter = ax.scatter([], [], [], c="red", s=28, alpha=0.95, label="Reference")
    nbr_scatter = ax.scatter([], [], [], c="blue", s=22, alpha=0.90, label="Neighbors")

    # 軸ラベル
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")

    # 軸範囲（ズーム）
    apply_equal_axis_with_zoom(ax, zoom_factor=float(args.zoom))

    # 軸の反転: Z軸正が左になるように
    ax.invert_xaxis()

    # 視点
    ax.view_init(elev=float(args.elev), azim=float(args.azim))

    # --- Video mode: recolor per frame ---
    if args.video:
        out_path = (args.out or "").strip()
        if not out_path:
            out_path = os.path.join("output", dataset_name, "plots", "ref_neighbors.mp4")
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        step = max(1, int(args.frame_step))
        frame_order_positions = list(range(0, len(indices), step))
        frame_indices = [indices[p] for p in frame_order_positions]
        index_to_order_pos = {idx: i for i, idx in enumerate(indices)}

        last_ref = None
        last_neighbors: List[int] = []

        def _set_frustum_style(cam_idx: int, *, color: str, lw: float, alpha: float):
            for ln in base_frustum_artists_by_idx.get(cam_idx, []):
                ln.set_color(color)
                ln.set_linewidth(lw)
                ln.set_alpha(alpha)

        def _update(frame_k: int):
            nonlocal last_ref, last_neighbors

            ref_idx = frame_indices[frame_k]
            ref_pos = poses_xyz[index_to_order_pos[ref_idx]]

            neighbors = select_neighbors(
                ref_pos_xyz=ref_pos,
                ref_order_pos=index_to_order_pos[ref_idx],
                all_indices=indices,
                all_positions_xyz=poses_xyz,
                mode=args.neighbor_mode,
                nearest_count=args.neighbor_count,
                r_min=args.neighbor_r_min,
                r_max=args.neighbor_r_max,
                adjacent_each_side=args.neighbor_each_side,
            )

            # revert previous highlights
            if last_ref is not None:
                _set_frustum_style(last_ref, color="black", lw=base_lw, alpha=base_alpha)
            for ni in last_neighbors:
                _set_frustum_style(ni, color="black", lw=base_lw, alpha=base_alpha)

            # apply current highlights
            _set_frustum_style(ref_idx, color="red", lw=1.6, alpha=0.95)
            for ni in neighbors:
                _set_frustum_style(ni, color="blue", lw=1.2, alpha=0.90)

            # update scatters
            ref_disp = centers_disp[index_to_order_pos[ref_idx]]
            ref_scatter._offsets3d = ([ref_disp[0]], [ref_disp[1]], [ref_disp[2]])

            if neighbors:
                nbr_disp = np.array([centers_disp[index_to_order_pos[i]] for i in neighbors])
                nbr_scatter._offsets3d = (nbr_disp[:, 0], nbr_disp[:, 1], nbr_disp[:, 2])
            else:
                nbr_scatter._offsets3d = ([], [], [])

            ax.set_title(f"Reference index: {ref_idx} / Neighbors: {neighbors}")

            last_ref = ref_idx
            last_neighbors = neighbors
            return []

        # init with first frame
        _update(0)

        anim = animation.FuncAnimation(
            fig,
            _update,
            frames=len(frame_indices),
            interval=1000.0 / max(1, int(args.fps)),
            blit=False,
        )

        saved = False
        try:
            if out_path.lower().endswith(".gif"):
                anim.save(
                    out_path,
                    writer=animation.PillowWriter(fps=int(args.fps)),
                    dpi=int(args.dpi),
                )
            else:
                anim.save(
                    out_path,
                    writer=animation.FFMpegWriter(fps=int(args.fps)),
                    dpi=int(args.dpi),
                )
            saved = True
        except Exception as e:
            print(f"警告: 動画の保存に失敗しました: {e}")
            print("ヒント: ffmpeg が無い場合は --out xxx.gif を指定してください。")

        if saved:
            print(f"動画を保存しました: {out_path}")

        if args.show:
            ax.legend(loc="best")
            plt.show()
        else:
            plt.close(fig)
        return

    # --- still image mode ---
    ax.set_title("Camera Poses and Flight Path")
    ax.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
