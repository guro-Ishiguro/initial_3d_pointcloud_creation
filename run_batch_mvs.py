#!/usr/bin/env python3
"""
3回連続でMVS処理を実行するスクリプト
各実行でmvs.yamlとconfig.yamlの設定を変更し、指定されたデータセットを実行します。
"""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml_config(yaml_path):
    """YAML設定ファイルを読み込む"""
    if not yaml_path.exists():
        return {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load {yaml_path}: {e}")
        return {}


def save_yaml_config(yaml_path, config):
    """YAML設定ファイルを保存する"""
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                config, f, sort_keys=False, default_flow_style=False, allow_unicode=True
            )
        return True
    except Exception as e:
        print(f"Error: Failed to save {yaml_path}: {e}")
        return False


def update_mvs_yaml(mvs_yaml_path, viz_depth_min, viz_depth_max):
    """mvs.yamlのVIZ_DEPTH_MINとVIZ_DEPTH_MAXを更新"""
    config = load_yaml_config(mvs_yaml_path)
    config["VIZ_DEPTH_MIN"] = viz_depth_min
    config["VIZ_DEPTH_MAX"] = viz_depth_max

    if save_yaml_config(mvs_yaml_path, config):
        print(
            f"Updated mvs.yaml: VIZ_DEPTH_MIN={viz_depth_min}, VIZ_DEPTH_MAX={viz_depth_max}"
        )
    else:
        raise RuntimeError(f"Failed to update {mvs_yaml_path}")


def update_config_yaml(config_yaml_path, env_updates=None):
    """config.yamlの環境変数設定を更新"""
    if env_updates is None:
        env_updates = {}

    config = load_yaml_config(config_yaml_path)
    if "env" not in config:
        config["env"] = {}

    # 環境変数を更新
    for key, value in env_updates.items():
        config["env"][key] = value

    if save_yaml_config(config_yaml_path, config):
        if env_updates:
            print(f"Updated config.yaml env: {env_updates}")
        return True
    else:
        raise RuntimeError(f"Failed to update {config_yaml_path}")


def print_config_summary(mvs_yaml_path, config_yaml_path):
    """設定ファイルの内容を表示"""
    print("\n--- Configuration Summary ---")
    mvs_config = load_yaml_config(mvs_yaml_path)
    config_yaml = load_yaml_config(config_yaml_path)

    print(f"mvs.yaml: {len(mvs_config)} settings")
    if "VIZ_DEPTH_MIN" in mvs_config and "VIZ_DEPTH_MAX" in mvs_config:
        print(
            f"  VIZ_DEPTH_MIN={mvs_config['VIZ_DEPTH_MIN']}, "
            f"VIZ_DEPTH_MAX={mvs_config['VIZ_DEPTH_MAX']}"
        )

    if "env" in config_yaml:
        env_vars = config_yaml["env"]
        print(f"config.yaml: {len(env_vars)} environment variables")
        for key, value in env_vars.items():
            print(f"  {key}={value}")
    print("---\n")


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
    config_yaml_path = project_root / "app" / "config.yaml"

    # 元の設定ファイルをバックアップ
    mvs_backup_path = mvs_yaml_path.with_suffix(".yaml.backup")
    config_backup_path = config_yaml_path.with_suffix(".yaml.backup")

    if not mvs_backup_path.exists():
        shutil.copy2(mvs_yaml_path, mvs_backup_path)
        print(f"Backed up mvs.yaml to {mvs_backup_path}")

    if config_yaml_path.exists() and not config_backup_path.exists():
        shutil.copy2(config_yaml_path, config_backup_path)
        print(f"Backed up config.yaml to {config_backup_path}")

    # 初期設定を表示
    print_config_summary(mvs_yaml_path, config_yaml_path)

    # 実行設定
    runs = [
        {
            "viz_depth_min": 17.0,
            "viz_depth_max": 36.0,
            "datasets": "4,5,6",
            "env_updates": None,  # 必要に応じて環境変数を更新可能
        },
        {
            "viz_depth_min": 22.0,
            "viz_depth_max": 41.0,
            "datasets": "7,8,9",
            "env_updates": None,
        },
        {
            "viz_depth_min": 27.0,
            "viz_depth_max": 46.0,
            "datasets": "10,11,12",
            "env_updates": None,
        },
    ]

    # 各実行を順番に実行
    for i, run_config in enumerate(runs, 1):
        print(f"\n{'#'*80}")
        print(f"# Run {i}/{len(runs)}")
        print(
            f"# VIZ_DEPTH_MIN={run_config['viz_depth_min']}, VIZ_DEPTH_MAX={run_config['viz_depth_max']}"
        )
        print(f"# Datasets: {run_config['datasets']}")
        if run_config.get("env_updates"):
            print(f"# Config env updates: {run_config['env_updates']}")
        print(f"{'#'*80}\n")

        try:
            # 設定ファイルを更新
            update_mvs_yaml(
                mvs_yaml_path,
                run_config["viz_depth_min"],
                run_config["viz_depth_max"],
            )

            # config.yamlの環境変数を更新（指定されている場合）
            if run_config.get("env_updates") and config_yaml_path.exists():
                update_config_yaml(config_yaml_path, run_config["env_updates"])

            # 実行
            returncode = run_cli(run_config["datasets"])

            if returncode != 0:
                print(f"\n[ERROR] Run {i} failed with return code {returncode}")
                print("Restoring original configuration files...")
                if mvs_backup_path.exists():
                    shutil.copy2(mvs_backup_path, mvs_yaml_path)
                if config_backup_path.exists():
                    shutil.copy2(config_backup_path, config_yaml_path)
                sys.exit(returncode)

            print(f"\n[SUCCESS] Run {i} completed successfully\n")

        except Exception as e:
            print(f"\n[ERROR] Run {i} failed with exception: {e}")
            print("Restoring original configuration files...")
            if mvs_backup_path.exists():
                shutil.copy2(mvs_backup_path, mvs_yaml_path)
            if config_backup_path.exists():
                shutil.copy2(config_backup_path, config_yaml_path)
            raise

    # 元の設定ファイルを復元
    print("Restoring original configuration files...")
    if mvs_backup_path.exists():
        shutil.copy2(mvs_backup_path, mvs_yaml_path)
        print(f"Restored mvs.yaml from {mvs_backup_path}")
    if config_backup_path.exists():
        shutil.copy2(config_backup_path, config_yaml_path)
        print(f"Restored config.yaml from {config_backup_path}")
    print("All runs completed successfully!")


if __name__ == "__main__":
    main()
