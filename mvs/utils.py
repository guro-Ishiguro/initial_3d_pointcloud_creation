import argparse
import csv
import logging
import os
import shutil

import config
import cv2
import Imath
import numpy as np
import OpenEXR

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


def save_depth_map_as_image(depth_map, file_path):
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

        min_val = 0
        max_val = config.camera_height

        if max_val - min_val > 1e-6:
            normalized_map = 255.0 * (depth_map - min_val) / (max_val - min_val)
        else:
            normalized_map = np.full(depth_map.shape, 128, dtype=np.float32)

        vis_map = np.nan_to_num(normalized_map).astype(np.uint8)
        colored_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
        colored_map[~valid_mask] = [0, 0, 0]

        # カラーバー用の設定
        colorbar_width = 80
        total_width = w + colorbar_width
        output_image = np.zeros((h, total_width, 3), dtype=np.uint8)
        output_image[:, :w] = colored_map

        # カラーバーの生成
        colorbar = np.linspace(0, 255, h).reshape(h, 1)
        colorbar_img = cv2.applyColorMap(np.uint8(colorbar), cv2.COLORMAP_JET)
        colorbar_img = cv2.flip(colorbar_img, 0)

        output_image[:, w : w + 20] = cv2.resize(
            colorbar_img, (20, h), interpolation=cv2.INTER_LINEAR
        )

        # カラーバーにテキストを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            output_image,
            f"{max_val:.2f}",
            (w + 25, 30),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output_image,
            f"{min_val:.2f}",
            (w + 25, h - 10),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(file_path, output_image)
    except Exception as e:
        logging.error(f"Failed to save depth map to {file_path}: {e}")


def read_exr_depth(file_path):
    """
    OpenEXRライブラリを使用して、単一チャンネルのEXR深度ファイルを読み込む。
    """
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

        logging.info(
            f"Info: Detected '{target_channel}' channel in the file. Reading it as depth data."
        )

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

    # カラーバー用の設定
    colorbar_width = 80
    total_width = w + colorbar_width
    output_image = np.zeros((h, total_width, 3), dtype=np.uint8)
    output_image[:, :w] = colored_map

    # カラーバーの生成
    colorbar = np.linspace(0, 255, h).reshape(h, 1)
    colorbar_img = cv2.applyColorMap(np.uint8(colorbar), cv2.COLORMAP_INFERNO)
    colorbar_img = cv2.flip(colorbar_img, 0)

    output_image[:, w : w + 20] = cv2.resize(
        colorbar_img, (20, h), interpolation=cv2.INTER_LINEAR
    )

    # カラーバーにテキストを追加
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        output_image,
        f"{max_error:.2f}",
        (w + 25, 30),
        font,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output_image,
        f"0.00",
        (w + 25, h - 10),
        font,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(file_path, output_image)
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

        # カラーバー用の設定
        colorbar_width = 80
        total_width = w + colorbar_width
        output_image = np.zeros((h, total_width, 3), dtype=np.uint8)
        output_image[:, :w] = colored_map

        # カラーバーの生成
        colorbar = np.linspace(0, 255, h).reshape(h, 1)
        colorbar_img = cv2.applyColorMap(np.uint8(colorbar), cv2.COLORMAP_JET)
        # カラーバーを上下反転させる（値が小さい方が下になるように）
        colorbar_img = cv2.flip(colorbar_img, 0)

        # 出力画像にカラーバーを配置
        output_image[:, w : w + 20] = cv2.resize(
            colorbar_img, (20, h), interpolation=cv2.INTER_LINEAR
        )

        # カラーバーにテキストを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            output_image,
            f"{max_val:.2f}",  # 最大値を表示
            (w + 25, 30),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output_image,
            f"{min_val:.2f}",  # 最小値を表示
            (w + 25, h - 10),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(file_path, output_image)
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
