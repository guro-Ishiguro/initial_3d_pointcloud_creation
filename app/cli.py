"""
MVSパイプライン用のコマンドラインエントリポイント。
YAML設定パスとデータセット名を引数で受け取り、環境変数と sys.path を設定したうえで、mvs.main.run を呼び出す。
"""

import argparse
import os
import sys


def _list_datasets(project_root: str):
    """
    プロジェクトルート直下の data ディレクトリ内のサブディレクトリ名を
    データセット名としてソート済みリストで返す。
    """
    data_dir = os.path.join(project_root, "data")
    if not os.path.isdir(data_dir):
        return []
    dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    dirs.sort()
    return dirs


def main():
    """
    コマンドライン引数を解析し、設定を適用してから mvs.main.run を実行する。
    --config でYAMLパス、--dataset でデータセット名を指定可能。
    """
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: app/config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset directory name under ./data (skip interactive selection).",
    )
    args = parser.parse_args()

    # mvs モジュールの import のためにプロジェクトルートと mvs を sys.path に追加
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mvs_dir = os.path.join(project_root, "mvs")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if mvs_dir not in sys.path:
        sys.path.insert(0, mvs_dir)

    sys.argv = [sys.argv[0]]

    try:
        from app.settings import apply_env_overrides

        apply_env_overrides(args.config)
    except Exception:
        pass

    datasets = _list_datasets(project_root)
    if not datasets:
        print("No datasets found under ./data")
        sys.exit(2)

    if args.dataset:
        selected_dataset = args.dataset.strip()
        if selected_dataset not in datasets:
            print(f"Error: Dataset '{selected_dataset}' not found.")
            print(f"Available datasets: {', '.join(datasets)}")
            sys.exit(2)
    elif len(datasets) == 1:
        selected_dataset = datasets[0]
    else:
        # 複数データセットがある場合は番号で対話選択
        print("\nSelect dataset:")
        for i, d in enumerate(datasets, 1):
            print(f"{i}) {d}")
        choice = input(f"Enter choice [1-{len(datasets)}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected_dataset = datasets[idx]
            else:
                print("Invalid choice. Aborting.")
                sys.exit(2)
        except ValueError:
            print("Invalid choice. Aborting.")
            sys.exit(2)

    # 選択したデータセットを環境変数に設定し、同一プロセスで mvs.main を実行
    os.environ["DATA_TYPE"] = selected_dataset

    # mvs.main.run に処理を委譲
    try:
        from mvs import main as mvs_main
    except Exception:
        # パッケージ import に失敗した場合は main.py をファイルから動的ロード
        import importlib.util

        main_path = os.path.join(project_root, "mvs", "main.py")
        spec = importlib.util.spec_from_file_location("mvs.main", main_path)
        m = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(m)
        mvs_main = m

    sys.exit(mvs_main.run())


if __name__ == "__main__":
    main()
