# tests/stereo_main.py

import os
import config
import logging
import time
import cv2
import numpy as np

from utils import (
    save_depth_map_as_image,
    read_exr_depth,
    compute_depth_metrics,
    save_error_map_as_image,
    save_disparity_map_with_colorbar,
)
from data_loader import DataLoader
from depth_estimation import DepthEstimator
from disparity_estimation import DisparityEstimator

if __name__ == "__main__":
    start_time = time.time()

    # --- 初期化 ---
    data_loader = DataLoader(config.STEREO_IMAGE_DIR, config.DRONE_IMAGE_LOG)
    depth_estimator = DepthEstimator(config)
    stereo_optimizer = DisparityEstimator(config)

    # --- 出力ディレクトリ作成 ---
    output_dir = os.path.join(config.OUTPUT_TYPE_DIR, "stereo_patchmatch")
    os.makedirs(output_dir, exist_ok=True)

    # --- 処理対象の画像インデックスを取得 ---
    if hasattr(config, "TARGET_INDICES") and config.TARGET_INDICES:
        target_indices = config.TARGET_INDICES
    else:
        # ターゲットが指定されていない場合は、最初の1枚だけを処理
        target_indices = [0]
    
    idx = target_indices[0] # 最初のインデックスのみ使用
    logging.info(f"Processing image index: {idx}")

    # --- 画像とGT深度の読み込み ---
    left_path, right_path = data_loader.get_image_paths(idx)
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)

    if left_image is None or right_image is None:
        logging.error(f"Could not load images for index {idx}. Exiting.")
        exit()

    left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

    gt_depth_path = os.path.join(config.LABEL_DEPTH_IMAGE_DIR, f"depth_{idx:06d}.exr")
    gt_depth = None
    if os.path.exists(gt_depth_path):
        gt_depth = read_exr_depth(gt_depth_path)
        if gt_depth is not None:
             h, w, _ = left_image.shape
             if gt_depth.shape != (h, w):
                 gt_depth = cv2.resize(gt_depth, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- PatchMatch Stereoの実行 ---
    stereo_disparity = stereo_optimizer.refine_disparity_with_patchmatch_stereo(
        left_image_rgb, right_image_rgb, ref_idx=idx
    )

    # --- 視差を深度に変換 ---
    stereo_depth = depth_estimator.disparity_to_depth(stereo_disparity)
    
    # --- 結果の保存 ---
    disparity_save_path = os.path.join(output_dir, f"stereo_disparity_{idx:04d}.png")
    depth_save_path = os.path.join(output_dir, f"stereo_depth_{idx:04d}.png")
    save_disparity_map_with_colorbar(stereo_disparity, disparity_save_path)
    save_depth_map_as_image(stereo_depth, depth_save_path)
    logging.info(f"Saved stereo disparity and depth maps to {output_dir}")
    
    # --- 評価 ---
    if gt_depth is not None:
        metrics = compute_depth_metrics(stereo_depth, gt_depth)
        logging.info(
            f"[PatchMatch Stereo] RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, AbsRel: {metrics['abs_rel']:.4f}"
        )
        error_map_path = os.path.join(output_dir, f"stereo_error_map_{idx:04d}.png")
        save_error_map_as_image(stereo_depth, gt_depth, error_map_path)
        logging.info(f"Saved error map to {error_map_path}")

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f}s")