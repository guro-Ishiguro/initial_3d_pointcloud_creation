import argparse
import os
import sys
import subprocess


def _list_datasets(project_root: str):
    data_dir = os.path.join(project_root, "data")
    if not os.path.isdir(data_dir):
        return []
    dirs = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    dirs.sort()
    return dirs


def _parse_dataset_selection(inp: str, datasets: list[str]) -> list[str]:
    """
    Accept:
      - "1" / "2,3" / "1 2 3" (1-based indices)
      - dataset names (exact)
      - "all"
    Returns list of dataset names (deduplicated, order preserved).
    """
    s = (inp or "").strip()
    if not s:
        return []
    if s.lower() == "all":
        return datasets[:]

    # split by comma or whitespace
    parts = [p for p in s.replace(",", " ").split() if p]
    out: list[str] = []
    seen = set()
    for p in parts:
        name = None
        if p.isdigit():
            i = int(p) - 1
            if 0 <= i < len(datasets):
                name = datasets[i]
        else:
            if p in datasets:
                name = p
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


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
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset directory name under ./data (skip interactive selection).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help='Multiple datasets (e.g. "1,2" or "SessionA,SessionB" or "all").',
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

    # dataset selection (support batch)
    datasets = _list_datasets(project_root)
    selected: list[str] = []
    if args.dataset:
        selected = [args.dataset.strip()]
    elif args.datasets:
        selected = _parse_dataset_selection(args.datasets, datasets)
        if not selected and args.datasets.strip():
            print(f"Invalid --datasets: {args.datasets!r}")
            sys.exit(2)
    elif len(datasets) > 1:
        # interactive multi-select
        print("\nSelect dataset(s):")
        for i, d in enumerate(datasets, 1):
            print(f"{i}) {d}")
        raw = input(f'Enter choice(s) [1-{len(datasets)}] (e.g. "1" or "1,2" or "all"): ').strip()
        selected = _parse_dataset_selection(raw, datasets)
        if not selected:
            print("No valid selection. Aborting.")
            sys.exit(2)
    elif len(datasets) == 1:
        selected = [datasets[0]]
    else:
        print("No datasets found under ./data")
        sys.exit(2)

    # If multiple datasets selected, run each in a separate subprocess so config paths don't collide.
    if len(selected) > 1:
        print(f"Selected datasets: {selected}")
        rc = 0
        for ds in selected:
            env = os.environ.copy()
            env["DATA_TYPE"] = ds
            cmd = [sys.executable, os.path.join(project_root, "app", "cli.py")]
            if args.config:
                cmd += ["--config", args.config]
            cmd += ["--dataset", ds]
            print(f"\n--- Running dataset: {ds} ---")
            p = subprocess.run(cmd, env=env)
            if p.returncode != 0:
                rc = p.returncode
        sys.exit(rc)

    # single dataset: set env and run in-process
    os.environ["DATA_TYPE"] = selected[0]

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
