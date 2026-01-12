#!/usr/bin/env python3
"""
複数のデータセットを1つのデータセットに統合するスクリプト

使用方法:
    python3 merge_datasets.py dataset1 dataset2 dataset3 ...
    または
    python3 merge_datasets.py  # 対話的に選択

統合後のデータセットは data/<merged_name>/ に作成され、
python3 app/cli.py で単一データセットとして実行可能になります。
"""

import argparse
import csv
import os
import shutil
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


def merge_datasets(
    source_datasets: list,
    data_dir: str,
    merged_name: str,
    merge_right_poses: bool = True,
) -> bool:
    """
    複数のデータセットを1つのデータセットに統合

    Args:
        source_datasets: 統合元のデータセット名のリスト
        data_dir: dataディレクトリのパス
        merged_name: 統合後のデータセット名
        merge_right_poses: right_camera_poses.csvも統合するか

    Returns:
        成功した場合True
    """
    merged_dir = os.path.join(data_dir, merged_name)
    merged_images_dir = os.path.join(merged_dir, "images")
    merged_txt_dir = os.path.join(merged_dir, "txt")

    # 統合先ディレクトリが既に存在する場合は確認
    if os.path.exists(merged_dir):
        response = input(f"統合先ディレクトリ {merged_dir} が既に存在します。上書きしますか？ (y/N): ")
        if response.lower() != "y":
            print("統合をキャンセルしました。")
            return False
        shutil.rmtree(merged_dir)

    # ディレクトリを作成
    os.makedirs(merged_images_dir, exist_ok=True)
    os.makedirs(os.path.join(merged_images_dir, "image_0"), exist_ok=True)
    os.makedirs(os.path.join(merged_images_dir, "image_1"), exist_ok=True)
    os.makedirs(os.path.join(merged_images_dir, "depth"), exist_ok=True)
    os.makedirs(merged_txt_dir, exist_ok=True)

    # 統合処理
    total_frames = 0
    left_poses_rows = []
    right_poses_rows = []
    camera_params = None

    # 各データセットのフレーム数を事前にカウント
    dataset_frame_counts = []
    for dataset_name in source_datasets:
        source_dir = os.path.join(data_dir, dataset_name)
        source_images_dir = os.path.join(source_dir, "images")
        source_image0_dir = os.path.join(source_images_dir, "image_0")
        if os.path.isdir(source_image0_dir):
            image0_files = [
                f
                for f in os.listdir(source_image0_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            dataset_frame_counts.append(len(image0_files))
        else:
            dataset_frame_counts.append(0)

    for ds_idx, dataset_name in enumerate(source_datasets):
        source_dir = os.path.join(data_dir, dataset_name)
        if not os.path.isdir(source_dir):
            print(f"警告: データセット {dataset_name} が見つかりません。スキップします。")
            continue

        source_images_dir = os.path.join(source_dir, "images")
        source_txt_dir = os.path.join(source_dir, "txt")

        # 画像ファイルをコピー
        for img_type in ["image_0", "image_1"]:
            source_img_dir = os.path.join(source_images_dir, img_type)
            dest_img_dir = os.path.join(merged_images_dir, img_type)

            if not os.path.isdir(source_img_dir):
                print(f"警告: {dataset_name}/images/{img_type} が見つかりません。スキップします。")
                continue

            # 画像ファイルを連番でコピー（ファイル名の数値でソート）
            image_files = []
            for f in os.listdir(source_img_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    # ファイル名から数値を抽出してソート
                    try:
                        idx = int(os.path.splitext(f)[0])
                        image_files.append((idx, f))
                    except ValueError:
                        # 数値が抽出できない場合はファイル名でソート
                        image_files.append((999999, f))

            image_files.sort(key=lambda x: x[0])

            for img_idx, img_file in enumerate(image_files):
                _, filename = img_file
                new_idx = total_frames + img_idx
                ext = os.path.splitext(filename)[1]
                new_filename = f"{new_idx:06d}{ext}"
                source_path = os.path.join(source_img_dir, filename)
                dest_path = os.path.join(dest_img_dir, new_filename)
                shutil.copy2(source_path, dest_path)

        # GT深度ファイルをコピー
        source_depth_dir = os.path.join(source_images_dir, "depth")
        dest_depth_dir = os.path.join(merged_images_dir, "depth")

        if os.path.isdir(source_depth_dir):
            # 深度ファイルをインデックスでソート
            depth_files = []
            for f in os.listdir(source_depth_dir):
                if f.lower().endswith(".exr"):
                    # 元のインデックスを抽出（depth_000000.exr または 000000.exr）
                    if f.startswith("depth_"):
                        idx_str = f[6:12]  # depth_の後の6桁
                    else:
                        idx_str = f[:6]  # 最初の6桁
                    try:
                        idx = int(idx_str)
                        depth_files.append((idx, f))
                    except ValueError:
                        depth_files.append((999999, f))

            depth_files.sort(key=lambda x: x[0])

            for depth_idx, (_, depth_file) in enumerate(depth_files):
                new_idx = total_frames + depth_idx
                new_filename = f"{new_idx:06d}.exr"
                source_path = os.path.join(source_depth_dir, depth_file)
                dest_path = os.path.join(dest_depth_dir, new_filename)
                shutil.copy2(source_path, dest_path)

        # カメラポーズCSVを読み込み
        left_poses_path = os.path.join(source_txt_dir, "left_camera_poses.csv")
        if os.path.exists(left_poses_path):
            with open(left_poses_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    # filenameを新しいインデックスに更新
                    old_filename = row.get("filename", "").strip()
                    # 元のインデックスを抽出
                    try:
                        old_idx = int(os.path.splitext(old_filename)[0])
                        new_idx = total_frames + old_idx
                        row["filename"] = f"{new_idx:06d}.png"
                    except (ValueError, AttributeError):
                        # インデックスが抽出できない場合はそのまま
                        pass
                    left_poses_rows.append(row)

        # 右カメラポーズCSVを読み込み（オプション）
        if merge_right_poses:
            right_poses_path = os.path.join(source_txt_dir, "right_camera_poses.csv")
            if os.path.exists(right_poses_path):
                with open(right_poses_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row:
                            continue
                        # filenameを新しいインデックスに更新
                        old_filename = row.get("filename", "").strip()
                        try:
                            old_idx = int(os.path.splitext(old_filename)[0])
                            new_idx = total_frames + old_idx
                            row["filename"] = f"{new_idx:06d}.png"
                        except (ValueError, AttributeError):
                            pass
                        right_poses_rows.append(row)

        # カメラパラメータCSVを最初のデータセットからコピー
        if camera_params is None:
            camera_params_path = os.path.join(source_txt_dir, "camera_params.csv")
            if os.path.exists(camera_params_path):
                with open(camera_params_path, "r") as f:
                    camera_params = f.read()

        # フレーム数を更新（事前にカウントした値を使用）
        frame_count = dataset_frame_counts[ds_idx]
        total_frames += frame_count

        print(f"  {dataset_name}: {frame_count} フレームを統合しました。")

    # CSVファイルを書き込み
    if left_poses_rows:
        left_poses_output = os.path.join(merged_txt_dir, "left_camera_poses.csv")
        with open(left_poses_output, "w", newline="") as f:
            if left_poses_rows:
                writer = csv.DictWriter(f, fieldnames=left_poses_rows[0].keys())
                writer.writeheader()
                writer.writerows(left_poses_rows)
        print(f"  left_camera_poses.csv: {len(left_poses_rows)} 行を書き込みました。")

    if merge_right_poses and right_poses_rows:
        right_poses_output = os.path.join(merged_txt_dir, "right_camera_poses.csv")
        with open(right_poses_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=right_poses_rows[0].keys())
            writer.writeheader()
            writer.writerows(right_poses_rows)
        print(f"  right_camera_poses.csv: {len(right_poses_rows)} 行を書き込みました。")

    if camera_params:
        camera_params_output = os.path.join(merged_txt_dir, "camera_params.csv")
        with open(camera_params_output, "w") as f:
            f.write(camera_params)
        print(f"  camera_params.csv をコピーしました。")

    print(f"\n統合完了: {total_frames} フレーム、{len(source_datasets)} データセット")
    print(f"統合先: {merged_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="複数のデータセットを1つのデータセットに統合")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="統合するデータセット名（複数指定可能）。指定がない場合は対話的に選択",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="dataディレクトリのパス（デフォルト: data）",
    )
    parser.add_argument(
        "--name",
        help="統合後のデータセット名（指定がない場合は自動生成）",
    )
    parser.add_argument(
        "--no-right-poses",
        action="store_true",
        help="right_camera_poses.csvを統合しない",
    )

    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"エラー: dataディレクトリが見つかりません: {data_dir}")
        sys.exit(1)

    available_datasets = list_datasets(data_dir)
    if not available_datasets:
        print(f"エラー: {data_dir} にデータセットが見つかりません。")
        sys.exit(1)

    # データセットの選択
    if args.datasets:
        selected = args.datasets
    else:
        # 対話的に選択
        print("利用可能なデータセット:")
        for i, ds in enumerate(available_datasets, 1):
            print(f"  {i}) {ds}")

        print("\n統合するデータセットを選択してください（カンマ区切りまたはスペース区切り）:")
        choice = input("> ").strip()
        if not choice:
            print("データセットが選択されませんでした。")
            sys.exit(1)

        # 選択をパース
        parts = [p.strip() for p in choice.replace(",", " ").split() if p.strip()]
        selected = []
        for part in parts:
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(available_datasets):
                    selected.append(available_datasets[idx])
            elif part in available_datasets:
                selected.append(part)

    if not selected:
        print("エラー: 有効なデータセットが選択されませんでした。")
        sys.exit(1)

    # 存在確認
    invalid = [ds for ds in selected if ds not in available_datasets]
    if invalid:
        print(f"エラー: 以下のデータセットが見つかりません: {', '.join(invalid)}")
        sys.exit(1)

    # 統合後の名前を決定
    if args.name:
        merged_name = args.name
    else:
        merged_name = "_".join(selected)

    print(f"\n統合を開始します:")
    print(f"  統合元: {', '.join(selected)}")
    print(f"  統合先: {merged_name}")
    print()

    success = merge_datasets(
        selected,
        data_dir,
        merged_name,
        merge_right_poses=not args.no_right_poses,
    )

    if success:
        print(f"\n統合が完了しました。以下のコマンドで実行できます:")
        print(f"  python3 app/cli.py --dataset {merged_name}")
        sys.exit(0)
    else:
        print("\n統合に失敗しました。")
        sys.exit(1)


if __name__ == "__main__":
    main()
