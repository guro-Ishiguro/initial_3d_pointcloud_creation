#!/usr/bin/env python3
"""
3回連続でMVS処理を実行するスクリプト
各実行でmvs.yamlの設定を変更し、指定されたデータセットを実行します。
"""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def update_mvs_yaml(mvs_yaml_path, viz_depth_min, viz_depth_max):
    """mvs.yamlのVIZ_DEPTH_MINとVIZ_DEPTH_MAXを更新"""
    with open(mvs_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config["VIZ_DEPTH_MIN"] = viz_depth_min
    config["VIZ_DEPTH_MAX"] = viz_depth_max

    with open(mvs_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config, f, sort_keys=False, default_flow_style=False, allow_unicode=True
        )

    print(
        f"Updated mvs.yaml: VIZ_DEPTH_MIN={viz_depth_min}, VIZ_DEPTH_MAX={viz_depth_max}"
    )


def run_cli(datasets_str):
    """app/cli.pyを実行"""
    project_root = Path(__file__).parent
    cli_path = project_root / "app" / "cli.py"

    cmd = [sys.executable, str(cli_path), "--datasets", datasets_str]
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode


def main():
    project_root = Path(__file__).parent
    mvs_yaml_path = project_root / "app" / "mvs.yaml"

    # 元の設定ファイルをバックアップ
    backup_path = mvs_yaml_path.with_suffix(".yaml.backup")
    if not backup_path.exists():
        shutil.copy2(mvs_yaml_path, backup_path)
        print(f"Backed up mvs.yaml to {backup_path}")

    # 実行設定
    runs = [
        {
            "viz_depth_min": 17.0,
            "viz_depth_max": 36.0,
            "datasets": "4,5,6",
        },
        {
            "viz_depth_min": 22.0,
            "viz_depth_max": 41.0,
            "datasets": "7,8,9",
        },
        {
            "viz_depth_min": 27.0,
            "viz_depth_max": 46.0,
            "datasets": "10,11,12",
        },
    ]

    # 各実行を順番に実行
    for i, run_config in enumerate(runs, 1):
        print(f"\n{'#'*80}")
        print(f"# Run {i}/3")
        print(
            f"# VIZ_DEPTH_MIN={run_config['viz_depth_min']}, VIZ_DEPTH_MAX={run_config['viz_depth_max']}"
        )
        print(f"# Datasets: {run_config['datasets']}")
        print(f"{'#'*80}\n")

        # 設定ファイルを更新
        update_mvs_yaml(
            mvs_yaml_path, run_config["viz_depth_min"], run_config["viz_depth_max"]
        )

        # 実行
        returncode = run_cli(run_config["datasets"])

        if returncode != 0:
            print(f"\n[ERROR] Run {i} failed with return code {returncode}")
            print("Restoring original mvs.yaml...")
            shutil.copy2(backup_path, mvs_yaml_path)
            sys.exit(returncode)

        print(f"\n[SUCCESS] Run {i} completed successfully\n")

    # 元の設定ファイルを復元
    print("Restoring original mvs.yaml...")
    shutil.copy2(backup_path, mvs_yaml_path)
    print("All runs completed successfully!")


if __name__ == "__main__":
    main()
