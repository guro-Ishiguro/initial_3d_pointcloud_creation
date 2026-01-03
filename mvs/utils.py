import argparse
import csv
import logging
import os
import shutil

import cv2
import numpy as np

# ImathとOpenEXRのインポート（条件付き）
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

# mvs.configのインポート（条件付き）
try:
    import mvs.config as config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # ダミーのconfigオブジェクトを作成
    class DummyConfig:
        VIZ_DEPTH_MIN = 0.0
        VIZ_DEPTH_MAX = 50.0
        VIZ_CMAP = "jet"
        camera_height = 50.0
    config = DummyConfig()

# ここでの basicConfig は削除（共通初期化は mvs.logging_setup.setup_logging 側に統一）


def parse_arguments():
    """コマンド引数の値を受け取る"""
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
    """四元数を回転行列に変換"""
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
    """指定フォルダの中身を削除する"""
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
    デプスマップを保存する。
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

        # 可視化レンジの解決（引数優先→config→従来値）
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
        colored_map = cv2.applyColorMap(vis_map, _resolve_cmap_code(cmap_name))
        colored_map[~valid_mask] = [0, 0, 0]

        # サイドバー無しでそのまま保存
        cv2.imwrite(file_path, colored_map)
    except Exception as e:
        logging.error(f"Failed to save depth map to {file_path}: {e}")


def read_exr_depth(file_path):
    """
    OpenEXRライブラリを使用して、単一チャンネルのEXR深度ファイルを読み込む。
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

        # noisy: channel detection log
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
    深度マップをEXR形式で保存する（絶対的な深度値が読み取れる形式）。
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
        header["channels"] = {"R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        
        # データをバイト列に変換
        depth_bytes = depth_float.tobytes()
        
        # EXRファイルを書き込み
        exr_file = OpenEXR.OutputFile(file_path, header)
        exr_file.writePixels({"R": depth_bytes})
        exr_file.close()
        
        logging.info(f"Saved depth map as EXR to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save depth map as EXR to {file_path}: {e}")


def compute_depth_metrics(pred_depth, gt_depth):
    """
    予測深度と正解深度を比較し、評価指標を計算する。
    """
    # 有効なピクセルのマスクを生成 (予測・真値ともに有限値で、かつ真値が0より大きい)
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
    深度誤差を計算し、カラーマップとして可視化して保存する。
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
    法線マップを画像ファイルとして保存する。
    """
    try:
        normalized_normals = normal_map * 0.5 + 0.5
        valid_normals = np.nan_to_num(normalized_normals, nan=0.0)
        normal_image_rgb = (valid_normals * 255).astype(np.uint8)
        normal_image_bgr = cv2.cvtColor(normal_image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, normal_image_bgr)
        logging.info(f"Saved normal map to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save normal map to {file_path}: {e}")


def save_disparity_map_with_colorbar(disparity_map, file_path):
    """
    視差マップをカラーバー付きの画像として保存する。
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

        # 有効な視差値から最小値と最大値を取得
        min_val = disparity_map[valid_mask].min()
        max_val = disparity_map[valid_mask].max()

        if max_val - min_val > 1e-6:
            # 0-255の範囲に正規化
            normalized_map = 255.0 * (disparity_map - min_val) / (max_val - min_val)
        else:
            normalized_map = np.full(disparity_map.shape, 128, dtype=np.float32)

        # NaNの値を0に変換し、uint8にキャスト
        vis_map = np.nan_to_num(normalized_map).astype(np.uint8)
        # カラーマップを適用
        colored_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
        # 無効な領域を黒で塗りつぶす
        colored_map[~valid_mask] = [0, 0, 0]

        # サイドバー無しでそのまま保存
        cv2.imwrite(file_path, colored_map)
        logging.info(f"Saved disparity map to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save disparity map to {file_path}: {e}")


def initialize_csv(file_path, header):
    """
    CSVファイルを初期化し、ヘッダーを書き込む。
    """
    try:
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
    except IOError as e:
        logging.error(f"Could not initialize CSV file {file_path}: {e}")


def append_to_csv(file_path, data_row):
    """
    CSVファイルに新しい行を追記する。
    """
    try:
        with open(file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_row)
    except IOError as e:
        logging.error(f"Could not write to CSV file {file_path}: {e}")


def write_stage_metrics_to_csv(
    csv_path: str,
    image_idx: int,
    stage: str,
    valid_pixels: int,
    metrics: dict,
):
    """
    ステージごとの評価指標をresults.csvに書き込む。
    
    Args:
        csv_path: results.csvのパス
        image_idx: 画像インデックス
        stage: ステージ名（initial, optimized, photometric, geometric）
        valid_pixels: 有効ピクセル数
        metrics: 評価指標の辞書
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
    イテレーションごとの評価指標を各メトリクスごとのCSVに書き込む。
    時間列は含めない（image_idx, iter, valid_pixels, metric）。
    
    Args:
        csv_files: メトリクスごとのCSVファイルパスの辞書
        image_idx: 画像インデックス
        iteration: イテレーション番号
        elapsed_time: 経過時間（秒）（使用しないが、互換性のため保持）
        metrics: 評価指標の辞書
        valid_pixels: 有効ピクセル数
    """
    for metric_key, csv_path in csv_files.items():
        if metric_key in metrics:
            append_to_csv(
                csv_path,
                [image_idx, iteration, valid_pixels, metrics[metric_key]],
            )
