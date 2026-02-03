#!/usr/bin/env python3
"""
評価指標CSVファイルを集約して平均値を計算し、result.csv に保存するスクリプト。
"""

import argparse
import csv
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_metric_csv_files(csv_dir: str) -> Dict[str, List[str]]:
    """
    指定されたディレクトリ内の各サブフォルダから評価指標CSVファイルを探す
    """
    metric_files = defaultdict(list)

    # 指定されたディレクトリ内のサブディレクトリを取得
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        logging.error(f"Directory not found: {csv_dir}")
        return metric_files

    # サブディレクトリを取得（ファイル名のフォルダ）
    subdirs = [d for d in csv_path.iterdir() if d.is_dir()]

    if not subdirs:
        logging.warning(f"No subdirectories found in {csv_dir}")
        return metric_files

    # 各サブディレクトリ内の評価指標CSVファイルを探す
    for subdir in subdirs:
        # time.csvは除外
        csv_files = [f for f in subdir.glob("*_checkerboard.csv")]

        for csv_file in csv_files:
            # 評価指標名を抽出（例：abs_rel_checkerboard.csv -> abs_rel）
            metric_name = csv_file.stem.replace("_checkerboard", "")
            metric_files[metric_name].append(str(csv_file))

    return metric_files


def read_metric_csv(csv_path: str) -> Dict[str, float]:
    """
    評価指標CSVファイルを読み込み、各段階（iter）での値を辞書として返す
    """
    values_by_iter = {}

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iter_key = row["iter"]
                # 評価指標の列名を取得（最後の列）
                metric_value = None
                for key, value in row.items():
                    if key not in ["image_idx", "iter", "valid_pixels"]:
                        try:
                            metric_value = float(value)
                            break
                        except (ValueError, TypeError):
                            continue

                if metric_value is not None:
                    values_by_iter[iter_key] = metric_value
    except Exception as e:
        logging.warning(f"Failed to read {csv_path}: {e}")

    return values_by_iter


def aggregate_metrics(csv_dir: str) -> Dict[str, Dict[str, float]]:
    """
    各評価指標ごとに、各段階での平均値を計算する
    """
    metric_files = find_metric_csv_files(csv_dir)

    if not metric_files:
        logging.error("No metric CSV files found")
        return {}

    # 各評価指標ごとに集計
    aggregated = defaultdict(lambda: defaultdict(list))

    for metric_name, file_paths in metric_files.items():
        logging.info(f"Processing {metric_name}: {len(file_paths)} files")

        # 各ファイルから値を読み込む
        for file_path in file_paths:
            values_by_iter = read_metric_csv(file_path)

            for iter_key, value in values_by_iter.items():
                aggregated[metric_name][iter_key].append(value)

    # 平均値を計算
    result = {}
    for metric_name, iter_values in aggregated.items():
        result[metric_name] = {}
        for iter_key, values in iter_values.items():
            if values:
                result[metric_name][iter_key] = sum(values) / len(values)
            else:
                result[metric_name][iter_key] = None

    return result


def write_result_csv(csv_dir: str, aggregated: Dict[str, Dict[str, float]]):
    """
    集約された評価指標をresult.csvに書き込む
    """
    if not aggregated:
        logging.error("No aggregated data to write")
        return

    # 全評価指標に現れる iter を収集
    all_iters = set()
    for metric_data in aggregated.values():
        all_iters.update(metric_data.keys())

    # iter を数値昇順＋特殊ラベル（photometric, geometric）の順でソート
    def sort_key(iter_str):
        if iter_str == "photometric":
            return (1, 999)
        elif iter_str == "geometric":
            return (1, 1000)
        else:
            try:
                return (0, int(iter_str))
            except ValueError:
                return (1, 0)

    sorted_iters = sorted(all_iters, key=sort_key)

    # 評価指標名をソート
    sorted_metrics = sorted(aggregated.keys())

    # result.csvに書き込む
    result_path = os.path.join(csv_dir, "result.csv")

    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)

        # ヘッダー行：iter, metric1, metric2, ...
        header = ["iter"] + sorted_metrics
        writer.writerow(header)

        # 各行：段階ごとの平均値
        for iter_key in sorted_iters:
            row = [iter_key]
            for metric_name in sorted_metrics:
                value = aggregated[metric_name].get(iter_key)
                if value is not None:
                    row.append(f"{value:.6f}")
                else:
                    row.append("")
            writer.writerow(row)

    logging.info(f"Result CSV saved to {result_path}")


def main():
    """
    コマンドライン引数で指定したディレクトリ内の評価指標CSVを集約し、
    result.csv を同じディレクトリに書き出す。
    """
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics from CSV files"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Directory path containing subdirectories with metric CSV files",
    )

    args = parser.parse_args()

    csv_dir = os.path.abspath(args.csv_dir)

    if not os.path.exists(csv_dir):
        logging.error(f"Directory not found: {csv_dir}")
        return 1

    logging.info(f"Processing CSV directory: {csv_dir}")

    # サブディレクトリ内の *_checkerboard.csv を集計して平均を計算
    aggregated = aggregate_metrics(csv_dir)

    if not aggregated:
        logging.error("No data aggregated")
        return 1

    # 集約結果を csv_dir/result.csv に出力
    write_result_csv(csv_dir, aggregated)

    logging.info("Aggregation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
