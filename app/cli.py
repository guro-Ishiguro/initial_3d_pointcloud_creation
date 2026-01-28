import argparse
import os
import sys


def _list_datasets(project_root: str):
    data_dir = os.path.join(project_root, "data")
    if not os.path.isdir(data_dir):
        return []
    dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    dirs.sort()
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud pipeline (YAML-driven)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to MVS YAML config (default: app/mvs.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset directory name under ./data (skip interactive selection).",
    )
    args = parser.parse_args()

    # Ensure project root and mvs dir on sys.path for module/bare imports inside mvs/*
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mvs_dir = os.path.join(project_root, "mvs")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if mvs_dir not in sys.path:
        sys.path.insert(0, mvs_dir)

    # Resolve config path:
    # - If --config is given, use it.
    # - Otherwise, default to app/mvs.yaml under the project root.
    if args.config is None:
        default_config = os.path.join(project_root, "app", "mvs.yaml")
        args.config = default_config

    # Expose the MVS config path so mvs/config.py will prioritise it
    os.environ["APP_MVS_CONFIG"] = os.path.abspath(args.config)

    # Remove custom args so downstream parser (mvs.utils.parse_arguments) doesn't see them
    sys.argv = [sys.argv[0]]

    # Apply global settings overrides (for generic env vars) if present
    try:
        from app.settings import apply_env_overrides

        apply_env_overrides(args.config)
    except Exception:
        pass

    # dataset selection (single dataset only)
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
        # interactive selection
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

    # single dataset: set env and run in-process
    os.environ["DATA_TYPE"] = selected_dataset

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
