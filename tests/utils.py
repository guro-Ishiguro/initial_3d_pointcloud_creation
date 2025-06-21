import os
import shutil
import numpy as np
import logging
import argparse
import cv2

# ログ設定はここで一元的に行う
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    デプスマップ（float配列）を視覚的に確認可能なカラー画像として保存する。
    NaN（無効な深度）のピクセルは黒色で表示される。
    """
    try:
        # 有効な深度値のマスクを作成
        valid_mask = np.isfinite(depth_map)

        # 有効な深度値が存在しない場合は、黒い画像を保存
        if not valid_mask.any():
            h, w = depth_map.shape
            black_image = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.imwrite(file_path, black_image)
            return

        min_val = np.min(depth_map[valid_mask])
        max_val = np.max(depth_map[valid_mask])

        # 0-255の範囲に正規化
        if max_val - min_val > 1e-6:
            # 深度が遠いほど値が大きくなるように正規化 (near=255, far=1)
            normalized_map = 255.0 * (1.0 - (depth_map - min_val) / (max_val - min_val))
        else:
            # 全ての深度が同じ値の場合
            normalized_map = np.full(depth_map.shape, 128, dtype=np.float32)

        # uint8に変換
        vis_map = np.nan_to_num(normalized_map).astype(np.uint8)

        # カラーマップを適用
        colored_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

        # 無効な深度（NaN）を持つピクセルを黒にする
        colored_map[~valid_mask] = [0, 0, 0]

        cv2.imwrite(file_path, colored_map)
    except Exception as e:
        logging.error(f"Failed to save depth map to {file_path}: {e}")
