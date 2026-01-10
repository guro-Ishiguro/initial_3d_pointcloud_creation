import os
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # PyYAML is listed in requirements; fail soft if missing
    yaml = None


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not yaml or not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def apply_env_overrides(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Apply environment overrides from a YAML file and sensible defaults.
    Order: YAML (if present) -> existing env -> defaults.
    Returns effective settings applied (for logging/debug).
    """
    # 1) Load from YAML if provided or default
    if not config_path:
        # APP_CONFIG can override the default path
        config_path = os.getenv(
            "APP_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml")
        )
    cfg = _load_yaml_config(config_path)

    # 2) Apply env section generically
    env_cfg: Dict[str, Any] = {}
    if isinstance(cfg.get("env"), dict):
        env_cfg = cfg["env"]
        for k, v in env_cfg.items():
            # skip unset/empty values so code can fall back to defaults/interactive
            if v is None:
                continue
            sv = str(v).strip()
            if sv == "":
                continue
            os.environ[str(k)] = sv

    # 3) Ensure defaults for key variables if still unset
    effective: Dict[str, Any] = {}

    # bubble up some common dataset selectors
    if "DATA_TYPE" in os.environ:
        effective["DATA_TYPE"] = os.environ["DATA_TYPE"]
    if "DATA_TYPE_INDEX" in os.environ:
        effective["DATA_TYPE_INDEX"] = os.environ["DATA_TYPE_INDEX"]

    # 4) Return merged effective (YAML env + defaults)
    effective.update(env_cfg)
    return effective
