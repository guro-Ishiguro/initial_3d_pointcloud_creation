import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run 3D point cloud pipeline")
    parser.add_argument(
        "--data_type",
        type=str,
        default=None,
        help="DATA_TYPE name under data/ (overrides env)",
    )
    parser.add_argument(
        "--data_type_index",
        type=int,
        default=None,
        help="DATA_TYPE_INDEX 1-based (overrides env)",
    )
    parser.add_argument("--log_level", type=str, default=None, help="PM_LOG_LEVEL")
    parser.add_argument(
        "--prop_dirs", type=int, default=None, help="PM_PROP_DIRS (4 or 8)"
    )
    parser.add_argument(
        "--priority_sweeps", type=int, default=None, help="PM_PRIORITY_SWEEPS"
    )
    args = parser.parse_args()

    # Ensure project root and mvs dir on sys.path for module/bare imports inside mvs/*
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mvs_dir = os.path.join(project_root, "mvs")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if mvs_dir not in sys.path:
        sys.path.insert(0, mvs_dir)

    if args.data_type:
        os.environ["DATA_TYPE"] = args.data_type
    if args.data_type_index is not None:
        os.environ["DATA_TYPE_INDEX"] = str(args.data_type_index)
    if args.log_level:
        os.environ["PM_LOG_LEVEL"] = args.log_level
    if args.prop_dirs is not None:
        os.environ["PM_PROP_DIRS"] = str(args.prop_dirs)
    if args.priority_sweeps is not None:
        os.environ["PM_PRIORITY_SWEEPS"] = str(args.priority_sweeps)

    # Remove custom args so downstream parser (mvs.utils.parse_arguments) doesn't see them
    sys.argv = [sys.argv[0]]

    # Apply global settings overrides if present
    try:
        from app.settings import apply_env_overrides

        apply_env_overrides()
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
