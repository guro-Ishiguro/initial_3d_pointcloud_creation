#!/usr/bin/env python3
"""
深度推定結果の評価スクリプト

推定深度と真値深度を比較し、評価指標を計算してCSVファイルに保存します。
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# mvsモジュールをインポートする前に、非対話モードを設定
# これにより、mvs/config.pyの対話的データセット選択をスキップ
os.environ["SKIP_DATASET_SELECTION"] = "1"

# mvsモジュールをインポートするためのパス設定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mvs_dir = os.path.join(project_root, "mvs")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if mvs_dir not in sys.path:
    sys.path.insert(0, mvs_dir)

from mvs.utils import (  # noqa: E402
    append_to_csv,
    compute_depth_metrics,
    initialize_csv,
    read_exr_depth,
    write_iteration_metrics_to_csv,
)


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def find_gt_depth_file(gt_dir: str, filename_stem: str) -> Optional[str]:
    """
    真値深度ファイルを検索する。

    Args:
        gt_dir: 真値深度ディレクトリ
        filename_stem: 画像ファイル名（拡張子なし、例: "000004"）

    Returns:
        真値深度ファイルのパス（見つからない場合はNone）
    """
    # ファイル名からインデックスを推測（6桁の数値）
    try:
        idx = int(filename_stem)
    except ValueError:
        # 数値でない場合は、ファイル名そのものをインデックスとして扱う
        idx_str = filename_stem
    else:
        idx_str = f"{idx:06d}"

    # depth_######.exr と ######.exr の両方に対応
    path1 = os.path.join(gt_dir, f"depth_{idx_str}.exr")
    if os.path.exists(path1):
        return path1

    path2 = os.path.join(gt_dir, f"{idx_str}.exr")
    if os.path.exists(path2):
        return path2

    return None


def find_pred_depth_files(pred_dir: str, filename_stem: str) -> Dict[str, str]:
    """
    推定深度ファイルを検索する。

    Args:
        pred_dir: 推定深度ディレクトリ
        filename_stem: 画像ファイル名（拡張子なし）

    Returns:
        ステージ名をキー、ファイルパスを値とする辞書
    """
    folder_path = os.path.join(pred_dir, filename_stem)
    if not os.path.isdir(folder_path):
        return {}

    files = {}
    stage_files = {
        "initial": "depth_initial.exr",
        "optimized": "depth_optimized.exr",
        "photometric": "depth_photometric.exr",
        "geometric": "depth_geometric.exr",
    }

    for stage, filename in stage_files.items():
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            files[stage] = filepath

    # イテレーションごとのファイルも検索
    for i in range(1, 20):  # 最大20イテレーションまで対応
        iter_file = os.path.join(folder_path, f"depth_iter_{i:02d}.exr")
        if os.path.exists(iter_file):
            files[f"iter_{i:02d}"] = iter_file
        else:
            break  # 連続していない場合は終了

    return files


def evaluate_depth_pair(
    pred_path: str, gt_path: str, stage: str
) -> Tuple[Optional[Dict], int]:
    """
    推定深度と真値深度のペアを評価する。

    Args:
        pred_path: 推定深度ファイルのパス
        gt_path: 真値深度ファイルのパス
        stage: ステージ名

    Returns:
        (評価指標の辞書, 有効ピクセル数) のタプル
    """
    pred_depth = read_exr_depth(pred_path)
    gt_depth = read_exr_depth(gt_path)

    if pred_depth is None or gt_depth is None:
        return None, 0

    # サイズが異なる場合はリサイズ
    if pred_depth.shape != gt_depth.shape:
        h, w = gt_depth.shape
        pred_depth = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_NEAREST)

    metrics = compute_depth_metrics(pred_depth, gt_depth)
    valid_pixels = int(np.sum(np.isfinite(pred_depth)))

    return metrics, valid_pixels


def determine_output_dir(pred_dir: str) -> str:
    """
    出力ディレクトリを決定する。

    pred-dirが output/<group>/<session>/depth の場合、
    output/<group>/<session>/csv を返す。
    """
    pred_path = Path(pred_dir).resolve()

    # output/<group>/<session>/depth の構造を想定
    if pred_path.name == "depth":
        # 親ディレクトリ（セッションディレクトリ）を取得
        session_dir = pred_path.parent
        csv_dir = session_dir / "csv"
        return str(csv_dir)

    # フォールバック: pred-dirの親ディレクトリにcsvを作成
    csv_dir = pred_path.parent / "csv"
    return str(csv_dir)


def get_propagation_method(pred_dir: str) -> str:
    """
    伝播手法を推測する（YAMLファイルから、またはデフォルト値）。

    Args:
        pred_dir: 推定深度ディレクトリ

    Returns:
        伝播手法名（デフォルト: "checkerboard"）
    """
    # YAMLファイルから読み取る
    try:
        import yaml

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        yaml_path = os.path.join(project_root, "app", "mvs.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            method = cfg.get("CHOICED_PROPAGATION_METHOD", "checkerboard")
            if method:
                return str(method)
    except Exception:
        pass

    # 既存のCSVファイルから推測を試みる（フォールバック）
    csv_dir = determine_output_dir(pred_dir)
    if os.path.isdir(csv_dir):
        for folder in os.listdir(csv_dir):
            folder_path = os.path.join(csv_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for f in os.listdir(folder_path):
                if f.endswith(".csv") and "_" in f:
                    # 例: rmse_checkerboard.csv
                    parts = f.split("_")
                    if len(parts) >= 2:
                        method = "_".join(parts[1:]).replace(".csv", "")
                        return method

    return "checkerboard"  # デフォルト


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted depth maps against ground truth"
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory containing predicted depth maps (e.g., output/<DATA_TYPE>/depth)",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="Directory containing ground truth depth maps (e.g., data/<DATA_TYPE>/images/depth)",
    )
    args = parser.parse_args()

    setup_logging()

    pred_dir = os.path.abspath(args.pred_dir)
    gt_dir = os.path.abspath(args.gt_dir)

    if not os.path.isdir(pred_dir):
        logging.error(f"Prediction directory does not exist: {pred_dir}")
        sys.exit(1)

    if not os.path.isdir(gt_dir):
        logging.error(f"Ground truth directory does not exist: {gt_dir}")
        sys.exit(1)

    # 出力ディレクトリを決定
    output_csv_dir = determine_output_dir(pred_dir)
    os.makedirs(output_csv_dir, exist_ok=True)

    # pred-dir内の各フォルダ（画像ごと）を処理
    pred_folders = [
        d for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))
    ]
    pred_folders.sort()

    if not pred_folders:
        logging.warning(f"No prediction folders found in {pred_dir}")
        sys.exit(1)

    logging.info(f"Found {len(pred_folders)} prediction folders")

    # 各画像ごとに処理
    for filename_stem in pred_folders:
        # 真値深度ファイルを検索
        gt_path = find_gt_depth_file(gt_dir, filename_stem)
        if gt_path is None:
            logging.warning(
                f"Ground truth depth not found for {filename_stem}, skipping"
            )
            continue

        # 推定深度ファイルを検索
        pred_files = find_pred_depth_files(pred_dir, filename_stem)
        if not pred_files:
            logging.warning(
                f"No prediction depth files found for {filename_stem}, skipping"
            )
            continue

        # インデックスを取得（ファイル名から推測）
        try:
            image_idx = int(filename_stem)
        except ValueError:
            image_idx = hash(filename_stem) % 1000000  # フォールバック

        # 各ステージ（initial, optimized, photometric, geometric）を評価（ログ出力のみ）
        stage_order = ["initial", "optimized", "photometric", "geometric"]
        for stage in stage_order:
            if stage not in pred_files:
                continue

            metrics, valid_pixels = evaluate_depth_pair(
                pred_files[stage], gt_path, stage
            )
            if metrics is None:
                continue

            logging.info(
                f"{filename_stem} {stage}: MAE={metrics['mae']:.4f}, "
                f"RMSE={metrics['rmse']:.4f}, AbsRel={metrics['abs_rel']:.4f}"
            )

        # 各イテレーションごとの評価（メトリクスごとのCSV）
        iter_files = {k: v for k, v in pred_files.items() if k.startswith("iter_")}
        # イテレーションファイルがある場合、またはphotometric/geometricがある場合にCSVを作成
        if iter_files or "photometric" in pred_files or "geometric" in pred_files:
            # CSVディレクトリを作成
            csv_subdir = os.path.join(output_csv_dir, filename_stem)
            os.makedirs(csv_subdir, exist_ok=True)

            # 伝播手法を取得
            propagation_method = get_propagation_method(pred_dir)

            # 各メトリクスごとのCSVファイルを初期化（時間列なし）
            csv_files = {
                "rmse": os.path.join(csv_subdir, f"rmse_{propagation_method}.csv"),
                "mae": os.path.join(csv_subdir, f"mae_{propagation_method}.csv"),
                "abs_rel": os.path.join(
                    csv_subdir, f"abs_rel_{propagation_method}.csv"
                ),
                "sq_rel": os.path.join(csv_subdir, f"sq_rel_{propagation_method}.csv"),
                "rmse_log": os.path.join(
                    csv_subdir, f"rmse_log_{propagation_method}.csv"
                ),
                "delta1": os.path.join(csv_subdir, f"delta1_{propagation_method}.csv"),
                "delta2": os.path.join(csv_subdir, f"delta2_{propagation_method}.csv"),
                "delta3": os.path.join(csv_subdir, f"delta3_{propagation_method}.csv"),
            }

            # 既存のCSVファイルがある場合はクリア
            for csv_path in csv_files.values():
                if os.path.exists(csv_path):
                    os.remove(csv_path)

            # 時間列なしで初期化（image_idx, iter, valid_pixels, metric）
            for metric, csv_path in csv_files.items():
                initialize_csv(csv_path, ["image_idx", "iter", "valid_pixels", metric])

            # 初期深度（iter=0）を評価
            if "initial" in pred_files:
                init_metrics, init_valid_pixels = evaluate_depth_pair(
                    pred_files["initial"], gt_path, "initial"
                )
                if init_metrics is not None:
                    write_iteration_metrics_to_csv(
                        csv_files, image_idx, 0, 0.0, init_metrics, init_valid_pixels
                    )

            # 各イテレーションを評価
            sorted_iters = sorted(iter_files.keys())
            for iter_key in sorted_iters:
                iter_num = int(iter_key.split("_")[1])
                metrics, valid_pixels = evaluate_depth_pair(
                    iter_files[iter_key], gt_path, iter_key
                )
                if metrics is not None:
                    # 時間列は使用しない（互換性のため0.0を渡す）
                    write_iteration_metrics_to_csv(
                        csv_files, image_idx, iter_num, 0.0, metrics, valid_pixels
                    )

            # photometricステージを評価（iter="photometric"として記録）
            if "photometric" in pred_files:
                photo_metrics, photo_valid_pixels = evaluate_depth_pair(
                    pred_files["photometric"], gt_path, "photometric"
                )
                if photo_metrics is not None:
                    # iter列に文字列"photometric"を入れるため、特別な処理が必要
                    for metric_key, csv_path in csv_files.items():
                        if metric_key in photo_metrics:
                            append_to_csv(
                                csv_path,
                                [
                                    image_idx,
                                    "photometric",
                                    photo_valid_pixels,
                                    photo_metrics[metric_key],
                                ],
                            )

            # geometricステージを評価（iter="geometric"として記録）
            if "geometric" in pred_files:
                geo_metrics, geo_valid_pixels = evaluate_depth_pair(
                    pred_files["geometric"], gt_path, "geometric"
                )
                if geo_metrics is not None:
                    # iter列に文字列"geometric"を入れるため、特別な処理が必要
                    for metric_key, csv_path in csv_files.items():
                        if metric_key in geo_metrics:
                            append_to_csv(
                                csv_path,
                                [
                                    image_idx,
                                    "geometric",
                                    geo_valid_pixels,
                                    geo_metrics[metric_key],
                                ],
                            )

    logging.info(f"Evaluation complete. Results saved to {output_csv_dir}")


if __name__ == "__main__":
    main()
