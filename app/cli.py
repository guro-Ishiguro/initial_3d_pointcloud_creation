import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud pipeline (YAML-driven)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: app/config.yaml)",
    )
    args = parser.parse_args()

    # Ensure project root and mvs dir on sys.path for module/bare imports inside mvs/*
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mvs_dir = os.path.join(project_root, "mvs")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if mvs_dir not in sys.path:
        sys.path.insert(0, mvs_dir)

    # Remove custom args so downstream parser (mvs.utils.parse_arguments) doesn't see them
    sys.argv = [sys.argv[0]]

    # Apply global settings overrides if present
    try:
        from app.settings import apply_env_overrides

        apply_env_overrides(args.config)
    except Exception:
        pass

    # delegate to original entrypoint
    try:
        from mvs import main as mvs_main
    except Exception:
        # Fallback: import by filename context if package import fails
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
