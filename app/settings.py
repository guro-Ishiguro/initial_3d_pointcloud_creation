"""
MVSパイプライン用の設定読み込みモジュール。
YAML設定ファイルと env セクションを読み込み、環境変数に反映する。
"""

import os
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # PyYAML は requirements に含まれるが、未インストール時は yaml を None のまま続行
    yaml = None


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    YAML設定ファイルを読み込み、辞書として返す。
    """
    if not yaml or not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def apply_env_overrides(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    YAML設定ファイルの env セクションを環境変数に反映する。
    適用順序: YAML（存在する場合）→ 既存の環境変数 → デフォルト。
    """
    # 1) config_path が未指定なら APP_CONFIG またはデフォルトパスを使用
    if not config_path:
        config_path = os.getenv(
            "APP_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml")
        )
    cfg = _load_yaml_config(config_path)

    # 2) YAML の env セクションを環境変数に一括反映
    env_cfg: Dict[str, Any] = {}
    if isinstance(cfg.get("env"), dict):
        env_cfg = cfg["env"]
        for k, v in env_cfg.items():
            # None や空文字はスキップし、他モジュールのデフォルトに任せる
            if v is None:
                continue
            sv = str(v).strip()
            if sv == "":
                continue
            os.environ[str(k)] = sv

    # 3) 主要なデータセット指定用変数を effective に集約
    effective: Dict[str, Any] = {}
    if "DATA_TYPE" in os.environ:
        effective["DATA_TYPE"] = os.environ["DATA_TYPE"]
    if "DATA_TYPE_INDEX" in os.environ:
        effective["DATA_TYPE_INDEX"] = os.environ["DATA_TYPE_INDEX"]

    # 4) env セクションの内容をマージして返す
    effective.update(env_cfg)
    return effective
