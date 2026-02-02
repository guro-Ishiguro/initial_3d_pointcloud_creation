"""
ユーティリティモジュール。
コマンド引数解析、四元数から回転行列への変換、フォルダ削除、深度マップの読み書き、
評価指標計算・CSV出力、法線・視差マップの可視化保存などを提供する。
"""

import argparse
import csv
import logging
import os
import shutil

import cv2
import numpy as np

# Imath と OpenEXR のインポート（未インストール時は EXR 読み書きを無効化）
try:
    import Imath
    import OpenEXR

    EXR_AVAILABLE = True
except ImportError as e:
    EXR_AVAILABLE = False
    logging.warning(
        f"OpenEXR/Imath not available: {e}. "
        "EXR depth reading/writing functions will not work. "
        "Please install: pip install OpenEXR"
    )

# mvs.config のインポート（CLI単体実行時などで import できない場合は DummyConfig を使用）
try:
    import mvs.config as config

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

    # ダミーの config オブジェクト（可視化用のデフォルト値のみ保持）
    class DummyConfig:
        VIZ_DEPTH_MIN = 0.0
        VIZ_DEPTH_MAX = 50.0
        VIZ_CMAP = "jet"
        camera_height = 50.0

    config = DummyConfig()



def parse_arguments():
    """
    コマンドライン引数を解析し、--show-viewer と --record-video の有無を返す。
    """
    parser = argparse.ArgumentParser(description="3D Point Cloud Creater")
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Show viewer during point cloud generation.",
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Record viewer output to video."
    )
    return parser.parse_args()


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    四元数 (qx, qy, qz, qw) を 3x3 回転行列に変換する。
    """
    R = np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
            ],
        ]
    )
    return R


def clear_folder(dir_path):
    """
    指定ディレクトリ内のファイル・サブディレクトリを再帰的に削除する。
    ディレクトリ自体は残す。存在しない場合はログのみ出力する。
    """
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f"Error deleting {file_path}: {e}")
    else:
        logging.info(f"The folder {dir_path} does not exist.")


def _resolve_cmap_code(cmap_name: str) -> int:
    """
    カラーマップ名（jet, viridis 等）を OpenCV の COLORMAP_* 定数に変換する。
    """
    name = (cmap_name or "").strip().lower()
    table = {
        "jet": cv2.COLORMAP_JET,
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
        "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
        "plasma": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
    }
    return table.get(name, cv2.COLORMAP_JET)


def save_depth_map_as_image(
    depth_map, file_path, viz_min=None, viz_max=None, viz_cmap=None
):
    """
    深度マップを可視化用のカラー画像（PNG等）として保存する。
    有効値の範囲は viz_min/viz_max（未指定時は config）で正規化し、viz_cmap で色付けする。
    """
    try:
        h, w = depth_map.shape
        valid_mask = np.isfinite(depth_map)

        if not valid_mask.any():
            black_image = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                black_image,
                "No valid depth",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imwrite(file_path, black_image)
            return

        min_val = (
            float(viz_min)
            if viz_min is not None
            else float(getattr(config, "VIZ_DEPTH_MIN", 0.0))
        )
        default_max = float(getattr(config, "camera_height", 50.0))
        max_val = (
            float(viz_max)
            if viz_max is not None
            else float(getattr(config, "VIZ_DEPTH_MAX", default_max))
        )

        if max_val - min_val > 1e-6:
            normalized_map = 255.0 * (depth_map - min_val) / (max_val - min_val)
        else:
            normalized_map = np.full(depth_map.shape, 128, dtype=np.float32)

        vis_map = np.nan_to_num(normalized_map).astype(np.uint8)
        # カラーマップの解決（引数優先→config→従来JET）
        cmap_name = (
            str(viz_cmap)
            if viz_cmap is not None
            else str(getattr(config, "VIZ_CMAP", "jet"))
        )
        cmap_code = _resolve_cmap_code(cmap_name)
        logging.debug(
            f"Using colormap: {cmap_name} (code: {cmap_code}) for {file_path}"
        )
        colored_map = cv2.applyColorMap(vis_map, cmap_code)
        colored_map[~valid_mask] = [0, 0, 0]

        # サイドバー無しでそのまま保存
        cv2.imwrite(file_path, colored_map)
    except Exception as e:
        logging.error(f"Failed to save depth map to {file_path}: {e}")


def read_exr_depth(file_path):
    """
    OpenEXR 形式の深度ファイルを読み込み、float32 の深度マップとして返す。
    """
    if not EXR_AVAILABLE:
        logging.error(
            "OpenEXR/Imath is not available. Cannot read EXR files. "
            "Please install: pip install OpenEXR"
        )
        return None
    try:
        exr_file = OpenEXR.InputFile(file_path)
        header = exr_file.header()

        available_channels = list(header["channels"].keys())

        target_channel = ""
        if "R" in available_channels:
            target_channel = "R"
        elif "Y" in available_channels:
            target_channel = "Y"
        else:
            logging.error(
                f"Error: Could not find 'R' or 'Y' channel for depth information."
            )
            logging.error(f"Available channels: {available_channels}")
            return None

        logging.debug(f"Detected '{target_channel}' channel in EXR; using it as depth.")

        dw = header["dataWindow"]
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_bytes = exr_file.channel(target_channel, pt)

        depth_map = np.frombuffer(channel_bytes, dtype=np.float32).copy()
        depth_map = depth_map.reshape(size)
        depth_map[depth_map <= 0] = np.nan

        return depth_map
    except Exception as e:
        logging.error(f"An error occurred while reading the EXR file: {e}")
        return None


def save_depth_map_as_exr(depth_map, file_path):
    """
    深度マップを OpenEXR 形式で保存する。
    """
    if not EXR_AVAILABLE:
        logging.error(
            "OpenEXR/Imath is not available. Cannot save EXR files. "
            "Please install: pip install OpenEXR"
        )
        return
    try:
        h, w = depth_map.shape
        # NaNや無限大を0に変換（EXRではNaNを直接保存できないため）
        depth_clean = depth_map.copy()
        depth_clean[~np.isfinite(depth_clean)] = 0.0

        # float32に変換
        depth_float = depth_clean.astype(np.float32)

        # EXRヘッダーを設定
        header = OpenEXR.Header(w, h)
        header["channels"] = {
            "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }

        # データをバイト列に変換
        depth_bytes = depth_float.tobytes()

        # EXRファイルを書き込み
        exr_file = OpenEXR.OutputFile(file_path, header)
        exr_file.writePixels({"R": depth_bytes})
        exr_file.close()

        # ログは呼び出し元で出力されるため、ここでは出力しない
    except Exception as e:
        logging.error(f"Failed to save depth map as EXR to {file_path}: {e}")


def compute_depth_metrics(pred_depth, gt_depth):
    """
    予測深度と正解深度を比較し、RMSE/MAE/abs_rel/sq_rel/rmse_log/delta1,2,3 を計算する。
    """
    # 有効なピクセルのマスク（予測・真値とも有限かつ真値 > 0）
    valid_mask = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (gt_depth > 0)

    # 有効なピクセルが存在しない場合はNaNを返す
    if np.sum(valid_mask) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "abs_rel": np.nan,
            "rmse_log": np.nan,
            "delta1": np.nan,
            "delta2": np.nan,
            "delta3": np.nan,
        }

    # マスクを適用して有効な深度値のみを抽出
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]

    # 基本的な誤差指標を計算
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    mae = np.mean(np.abs(pred_valid - gt_valid))
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    sq_rel = np.mean(((pred_valid - gt_valid) ** 2) / gt_valid)

    # RMSE log の計算
    # 予測値にも0以下の値がないことを確認
    pred_valid_log = pred_valid[pred_valid > 0]
    gt_valid_log = gt_valid[pred_valid > 0]
    if len(pred_valid_log) > 0:
        rmse_log = np.sqrt(
            np.mean((np.log(pred_valid_log) - np.log(gt_valid_log)) ** 2)
        )
    else:
        rmse_log = np.nan

    # Threshold Accuracy (δ) の計算
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25**2).mean()
    delta3 = (thresh < 1.25**3).mean()

    # 計算した全ての指標を辞書として返す
    return {
        "rmse": rmse,
        "mae": mae,
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse_log": rmse_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


def save_error_map_as_image(pred_depth, gt_depth, file_path, max_error=1.0):
    """
    予測深度と正解深度の絶対誤差を画像化し、max_error でクリップしてカラーマップで保存する。
    """
    valid_mask = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (gt_depth > 0)
    error_map = np.full(pred_depth.shape, np.nan, dtype=np.float32)
    error_map[valid_mask] = np.abs(pred_depth[valid_mask] - gt_depth[valid_mask])

    h, w = error_map.shape
    vis_map = np.nan_to_num(error_map)
    vis_map[vis_map > max_error] = max_error
    vis_map = (vis_map / max_error) * 255.0

    colored_map = cv2.applyColorMap(vis_map.astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored_map[~valid_mask] = [0, 0, 0]

    # サイドバー無しでそのまま保存
    cv2.imwrite(file_path, colored_map)
    logging.info(f"Saved depth error map to {file_path}")


def save_normal_map_as_image(normal_map, file_path):
    """
    法線マップを 0–255 に正規化して RGB 画像として保存する。
    """
    try:
        normalized_normals = normal_map * 0.5 + 0.5
        valid_normals = np.nan_to_num(normalized_normals, nan=0.0)
        normal_image_rgb = (valid_normals * 255).astype(np.uint8)
        normal_image_bgr = cv2.cvtColor(normal_image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, normal_image_bgr)
        # ログは呼び出し元で出力されるため、ここでは出力しない
    except Exception as e:
        logging.error(f"Failed to save normal map to {file_path}: {e}")


def save_disparity_map_with_colorbar(disparity_map, file_path):
    """
    視差マップを可視化用のカラー画像（PNG等）として保存する。
    """
    try:
        h, w = disparity_map.shape
        valid_mask = np.isfinite(disparity_map)

        if not valid_mask.any():
            black_image = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                black_image,
                "No valid disparity",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imwrite(file_path, black_image)
            return

        min_val = disparity_map[valid_mask].min()
        max_val = disparity_map[valid_mask].max()

        if max_val - min_val > 1e-6:
            normalized_map = 255.0 * (disparity_map - min_val) / (max_val - min_val)
        else:
            normalized_map = np.full(disparity_map.shape, 128, dtype=np.float32)

        vis_map = np.nan_to_num(normalized_map).astype(np.uint8)
        colored_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
        colored_map[~valid_mask] = [0, 0, 0]

        cv2.imwrite(file_path, colored_map)
        logging.info(f"Saved disparity map to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save disparity map to {file_path}: {e}")


def initialize_csv(file_path, header):
    """
    CSV ファイルを新規作成し、1行目にヘッダーを書き込む。
    """
    try:
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
    except IOError as e:
        logging.error(f"Could not initialize CSV file {file_path}: {e}")


def append_to_csv(file_path, data_row):
    """
    CSV ファイルの末尾に1行を追記する。親ディレクトリが無い場合は作成する。
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_row)
        logging.debug(f"Appended to CSV: {file_path} - {data_row}")
    except IOError as e:
        logging.error(f"Could not write to CSV file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error writing to CSV file {file_path}: {e}")


def write_stage_metrics_to_csv(
    csv_path: str,
    image_idx: int,
    stage: str,
    valid_pixels: int,
    metrics: dict,
):
    """
    ステージごとの評価指標を1行として CSV に追記する。
    """
    append_to_csv(
        csv_path,
        [
            image_idx,
            stage,
            valid_pixels,
            metrics.get("mae", np.nan),
            metrics.get("abs_rel", np.nan),
            metrics.get("sq_rel", np.nan),
            metrics.get("rmse", np.nan),
            metrics.get("rmse_log", np.nan),
            metrics.get("delta1", np.nan),
            metrics.get("delta2", np.nan),
            metrics.get("delta3", np.nan),
        ],
    )


def write_iteration_metrics_to_csv(
    csv_files: dict,
    image_idx: int,
    iteration: int,
    elapsed_time: float,
    metrics: dict,
    valid_pixels: int = 0,
):
    """
    イテレーションごとに、メトリクス名をキーとする CSV ファイル群に1行ずつ追記する。
    """
    for metric_key, csv_path in csv_files.items():
        if metric_key in metrics:
            append_to_csv(
                csv_path,
                [image_idx, iteration, valid_pixels, metrics[metric_key]],
            )
