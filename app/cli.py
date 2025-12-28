import argparse
import os
import subprocess
import sys
from typing import List


def _list_datasets(project_root: str):
    data_dir = os.path.join(project_root, "data")
    if not os.path.isdir(data_dir):
        return []
    dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    dirs.sort()
    return dirs


def _parse_dataset_selection(inp: str, datasets: List[str]) -> List[str]:
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
    out: List[str] = []
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


def _get_home_dir(project_root: str) -> str:
    # Keep consistent with mvs/config.py default behavior
    return os.getenv("HOME_DIR", project_root)


def _load_mvs_yaml(project_root: str):
    try:
        import yaml  # type: ignore
    except Exception:
        return {}


def _get_mvs_yaml_path(project_root: str) -> str:
    return os.getenv("APP_MVS_CONFIG", os.path.join(project_root, "app", "mvs.yaml"))


def _write_prepass_mvs_yaml(project_root: str, out_path: str) -> str:
    """
    Create a temporary MVS YAML for a fast prepass:
      - PREVIEW_ONLY: true (stop after selection/csv/plots)
      - EXPORT_GT_PER_VIEW_ENABLE: false (avoid expensive GT export)
    Other settings (especially frame selection) are inherited from the current mvs.yaml.
    Returns the written path (or empty string on failure).
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return ""

    base_path = _get_mvs_yaml_path(project_root)
    if not os.path.exists(base_path):
        return ""
    try:
        with open(base_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            cfg = {}
    except Exception:
        cfg = {}

    cfg["PREVIEW_ONLY"] = True
    cfg["EXPORT_GT_PER_VIEW_ENABLE"] = False
    cfg["EXPORT_GT_PER_VIEW_ONLY_TARGET"] = True

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return out_path
    except Exception:
        return ""
    cfg_path = os.getenv("APP_MVS_CONFIG", os.path.join(project_root, "app", "mvs.yaml"))
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_point_clouds(ply_paths: List[str], out_path: str) -> bool:
    try:
        import open3d as o3d
    except Exception:
        print("open3d not available; cannot merge point clouds.")
        return False

    merged = None
    used = 0
    for p in ply_paths:
        if not os.path.exists(p):
            continue
        pc = o3d.io.read_point_cloud(p)
        if pc is None:
            continue
        if merged is None:
            merged = pc
        else:
            merged += pc
        used += 1

    if merged is None or used == 0:
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = o3d.io.write_point_cloud(out_path, merged)
    return bool(ok)


def _merge_selected_frames_csv(session_csvs: List[str], session_names: List[str], out_csv: str) -> bool:
    import csv

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    wrote_any = False
    with open(out_csv, "w", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow(["dataset", "index", "left_path", "right_path"])
        for ds, p in zip(session_names, session_csvs):
            if not os.path.exists(p):
                continue
            with open(p, newline="") as f_in:
                r = csv.DictReader(f_in)
                for row in r:
                    if not row:
                        continue
                    w.writerow(
                        [
                            ds,
                            row.get("index", ""),
                            row.get("left_path", ""),
                            row.get("right_path", ""),
                        ]
                    )
                    wrote_any = True
    return wrote_any


def _load_camera_poses_csv(path: str):
    import csv

    poses = []
    if not os.path.exists(path):
        return poses
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                poses.append(
                    (
                        float(row.get("pos_x")),
                        float(row.get("pos_y")),
                        float(row.get("pos_z")),
                        float(row.get("rot_x")),
                        float(row.get("rot_y")),
                        float(row.get("rot_z")),
                        float(row.get("rot_w")),
                    )
                )
            except Exception:
                continue
    return poses


def _load_selected_indices_from_csv(path: str) -> List[int]:
    import csv

    out: List[int] = []
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            try:
                out.append(int(row.get("index")))
            except Exception:
                continue
    return out


def _save_merged_pose_plot(
    *,
    session_names: List[str],
    session_pose_csvs: List[str],
    session_selected_csvs: List[str],
    out_path: str,
    plane: str = "xz",
    arrow_stride: int = 10,
    arrow_scale: float = 0.25,
    title: str = "",
) -> bool:
    import numpy as np

    try:
        from scipy.spatial.transform import Rotation
    except Exception:
        print("scipy not available; cannot plot pose directions.")
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; cannot plot poses.")
        return False

    plane = (plane or "xz").strip().lower()
    axes_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax_i, ax_j = axes_map.get(plane, (0, 2))
    axis_names = ["x", "y", "z"]

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    any_plotted = False
    cmap = plt.get_cmap("tab10")

    for sidx, (ds, pose_csv, sel_csv) in enumerate(
        zip(session_names, session_pose_csvs, session_selected_csvs)
    ):
        poses = _load_camera_poses_csv(pose_csv)
        sel = _load_selected_indices_from_csv(sel_csv)
        if not poses or not sel:
            continue

        xs, ys = [], []
        qx, qy, qdx, qdy = [], [], [], []
        color = cmap(sidx % 10)

        for k, idx in enumerate(sel):
            if idx < 0 or idx >= len(poses):
                continue
            px, py, pz, rx, ry, rz, rw = poses[idx]
            p = np.array([px, py, pz], dtype=np.float64)
            xs.append(float(p[ax_i]))
            ys.append(float(p[ax_j]))

            if arrow_stride >= 1 and (k % arrow_stride == 0):
                try:
                    rot = Rotation.from_quat(np.array([rx, ry, rz, rw], dtype=np.float64))
                    forward = rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
                    d = np.array([forward[ax_i], forward[ax_j]], dtype=np.float64)
                    n = float(np.linalg.norm(d))
                    if n > 1e-9:
                        d = d / n
                    qx.append(xs[-1])
                    qy.append(ys[-1])
                    qdx.append(float(d[0]))
                    qdy.append(float(d[1]))
                except Exception:
                    pass

        if len(xs) < 2:
            continue

        ax.plot(xs, ys, "-", linewidth=1.2, alpha=0.85, label=ds, color=color)
        ax.scatter(xs, ys, s=6, alpha=0.8, color=color)
        ax.scatter([xs[0]], [ys[0]], s=40, marker="o", color=color)
        ax.scatter([xs[-1]], [ys[-1]], s=40, marker="x", color=color)
        if qx:
            ax.quiver(
                qx,
                qy,
                qdx,
                qdy,
                angles="xy",
                scale_units="xy",
                scale=1.0 / max(1e-6, arrow_scale),
                width=0.003,
                alpha=0.7,
                color=color,
            )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return False

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(axis_names[ax_i])
    ax.set_ylabel(axis_names[ax_j])
    ax.set_title(title or f"Merged selected camera poses ({plane.upper()} plane)")
    ax.legend(loc="best")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _build_global_selected_pose_rows(
    *,
    project_root: str,
    home_dir: str,
    group_output_dir: str,
    session_names: List[str],
) -> List[dict]:
    """
    Build a global, re-indexed list of selected frames across multiple datasets.
    Returns rows with:
      global_idx, dataset, local_idx, left_path, right_path, pos_x,pos_y,pos_z, rot_x,rot_y,rot_z,rot_w
    """
    import csv

    rows: List[dict] = []
    g = 0
    for ds in session_names:
        pose_csv = os.path.join(home_dir, "data", ds, "txt", "left_camera_poses.csv")
        poses = _load_camera_poses_csv(pose_csv)

        sel_csv = os.path.join(group_output_dir, ds, "csv", "selected_frames.csv")
        if not os.path.exists(sel_csv):
            # Fallback: if no selection CSV, assume all frames for that session
            selected_local = list(range(len(poses)))
            left_paths = {}
            right_paths = {}
        else:
            selected_local = []
            left_paths = {}
            right_paths = {}
            with open(sel_csv, newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    if not row:
                        continue
                    try:
                        li = int(row.get("index"))
                    except Exception:
                        continue
                    selected_local.append(li)
                    left_paths[li] = row.get("left_path", "")
                    right_paths[li] = row.get("right_path", "")

        for li in selected_local:
            if li < 0 or li >= len(poses):
                continue
            px, py, pz, rx, ry, rz, rw = poses[li]
            rows.append(
                {
                    "global_idx": g,
                    "dataset": ds,
                    "local_idx": li,
                    "left_path": left_paths.get(li, ""),
                    "right_path": right_paths.get(li, ""),
                    "pos_x": px,
                    "pos_y": py,
                    "pos_z": pz,
                    "rot_x": rx,
                    "rot_y": ry,
                    "rot_z": rz,
                    "rot_w": rw,
                }
            )
            g += 1
    return rows


def _write_global_pose_csv(rows: List[dict], out_csv: str) -> bool:
    import csv

    if not rows:
        return False
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "global_idx",
                "dataset",
                "local_idx",
                "left_path",
                "right_path",
                "pos_x",
                "pos_y",
                "pos_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "rot_w",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return True


def _save_global_pose_plot_from_rows(
    *,
    rows: List[dict],
    out_path: str,
    plane: str = "xz",
    arrow_stride: int = 10,
    arrow_scale: float = 0.25,
    title: str = "",
) -> bool:
    import numpy as np

    if not rows:
        return False
    try:
        from scipy.spatial.transform import Rotation
    except Exception:
        print("scipy not available; cannot plot pose directions.")
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; cannot plot poses.")
        return False

    plane = (plane or "xz").strip().lower()
    axes_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax_i, ax_j = axes_map.get(plane, (0, 2))
    axis_names = ["x", "y", "z"]

    # group by dataset, keep insertion order
    order: List[str] = []
    by_ds = {}
    for r in rows:
        ds = str(r.get("dataset", ""))
        if ds not in by_ds:
            by_ds[ds] = []
            order.append(ds)
        by_ds[ds].append(r)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap("tab10")

    any_plotted = False
    for sidx, ds in enumerate(order):
        rs = by_ds.get(ds, [])
        xs, ys = [], []
        qx, qy, qdx, qdy = [], [], [], []
        color = cmap(sidx % 10)
        for k, r in enumerate(rs):
            px = float(r["pos_x"])
            py = float(r["pos_y"])
            pz = float(r["pos_z"])
            rx = float(r["rot_x"])
            ry = float(r["rot_y"])
            rz = float(r["rot_z"])
            rw = float(r["rot_w"])
            p = np.array([px, py, pz], dtype=np.float64)
            xs.append(float(p[ax_i]))
            ys.append(float(p[ax_j]))
            if arrow_stride >= 1 and (k % arrow_stride == 0):
                try:
                    rot = Rotation.from_quat(np.array([rx, ry, rz, rw], dtype=np.float64))
                    forward = rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
                    d = np.array([forward[ax_i], forward[ax_j]], dtype=np.float64)
                    n = float(np.linalg.norm(d))
                    if n > 1e-9:
                        d = d / n
                    qx.append(xs[-1])
                    qy.append(ys[-1])
                    qdx.append(float(d[0]))
                    qdy.append(float(d[1]))
                except Exception:
                    pass

        if len(xs) < 2:
            continue
        ax.plot(xs, ys, "-", linewidth=1.2, alpha=0.85, label=ds, color=color)
        ax.scatter(xs, ys, s=6, alpha=0.8, color=color)
        ax.scatter([xs[0]], [ys[0]], s=40, marker="o", color=color)
        ax.scatter([xs[-1]], [ys[-1]], s=40, marker="x", color=color)
        if qx:
            ax.quiver(
                qx,
                qy,
                qdx,
                qdy,
                angles="xy",
                scale_units="xy",
                scale=1.0 / max(1e-6, arrow_scale),
                width=0.003,
                alpha=0.7,
                color=color,
            )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return False

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(axis_names[ax_i])
    ax.set_ylabel(axis_names[ax_j])
    ax.set_title(title or f"Merged selected camera poses ({plane.upper()} plane)")
    ax.legend(loc="best")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


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
    selected: List[str] = []
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
        raw = input(
            f'Enter choice(s) [1-{len(datasets)}] (e.g. "1" or "1,2" or "all"): '
        ).strip()
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
        group_name = "_".join(selected)  # keep selection order
        home_dir = _get_home_dir(project_root)
        group_output_dir = os.path.join(home_dir, "output", group_name)
        mvs_cfg = _load_mvs_yaml(project_root)
        need_global_pool = str(mvs_cfg.get("NEIGHBOR_POOL_MODE", "local")).strip().lower() == "global_csv"

        # We keep the user's MVS YAML (if any) for the full run.
        base_mvs_yaml = _get_mvs_yaml_path(project_root)
        prepass_yaml = ""
        global_pose_csv = os.path.join(group_output_dir, "csv", "global_selected_poses.csv")

        def _run_one(ds: str, env_overrides: dict, title: str):
            env = os.environ.copy()
            env.update(env_overrides)
            env["DATA_TYPE"] = ds
            env["OUTPUT_GROUP_NAME"] = group_name
            env["OUTPUT_SESSION_NAME"] = ds
            cmd = [sys.executable, os.path.join(project_root, "app", "cli.py")]
            if args.config:
                cmd += ["--config", args.config]
            cmd += ["--dataset", ds]
            print(f"\n--- {title}: {ds} ---")
            return subprocess.run(cmd, env=env).returncode

        # Stage A: prepass to generate per-session selected_frames.csv quickly,
        # then generate global_selected_poses.csv so the first full MVS run can use it.
        if need_global_pool:
            prepass_yaml = _write_prepass_mvs_yaml(
                project_root,
                os.path.join(group_output_dir, "csv", "mvs_prepass.yaml"),
            )
            if not prepass_yaml:
                print("Failed to create prepass MVS YAML; cannot build global neighbor pool early.")
                need_global_pool = False
            else:
                rc = 0
                for ds in selected:
                    rc = _run_one(
                        ds,
                        {"APP_MVS_CONFIG": prepass_yaml},
                        "Prepass (selection/csv/plots)",
                    )
                    if rc != 0:
                        break
                if rc != 0:
                    print("Prepass failed; abort.")
                    sys.exit(rc)

                # Build global CSV/plot now (before full runs)
                try:
                    rows = _build_global_selected_pose_rows(
                        project_root=project_root,
                        home_dir=home_dir,
                        group_output_dir=group_output_dir,
                        session_names=selected,
                    )
                    if _write_global_pose_csv(rows, global_pose_csv):
                        print(
                            f"Global re-indexed pose CSV saved (pre-run): {global_pose_csv} (count={len(rows)})"
                        )
                    plane = str(mvs_cfg.get("POSE_PLOT_PLANE", "xz") or "xz")
                    arrow_stride = int(mvs_cfg.get("POSE_PLOT_ARROW_STRIDE", 10) or 10)
                    arrow_scale = float(mvs_cfg.get("POSE_PLOT_ARROW_SCALE", 0.25) or 0.25)
                    global_plot = os.path.join(
                        group_output_dir, "plots", "global_selected_camera_poses.png"
                    )
                    _save_global_pose_plot_from_rows(
                        rows=rows,
                        out_path=global_plot,
                        plane=plane,
                        arrow_stride=arrow_stride,
                        arrow_scale=arrow_scale,
                        title=f"Group={group_name} global_idx=0..{max(0, len(rows)-1)}",
                    )
                    print(f"Global re-indexed pose plot saved (pre-run): {global_plot}")
                except Exception as e:
                    print(f"Global re-index outputs (pre-run) failed: {e}")

        # Stage B: full runs (may use global neighbor pool from Stage A)
        rc = 0
        for ds in selected:
            env_over = {"APP_MVS_CONFIG": base_mvs_yaml}
            if need_global_pool and os.path.exists(global_pose_csv):
                env_over["GLOBAL_NEIGHBOR_POOL_CSV"] = global_pose_csv
            rc = _run_one(ds, env_over, "Run dataset")
            if rc != 0:
                break

        if rc != 0:
            print("Some sessions failed; skip merging outputs.")
            sys.exit(rc)

        # --- Merge outputs into group folder ---
        # 1) Merge point clouds
        session_plys = [
            os.path.join(group_output_dir, ds, "point_cloud", "output.ply") for ds in selected
        ]
        merged_ply = os.path.join(group_output_dir, "point_cloud", "output.ply")
        if _merge_point_clouds(session_plys, merged_ply):
            print(f"Merged point cloud saved: {merged_ply}")
        else:
            print("No point clouds found to merge.")

        # 2) Merge selected frames CSV
        session_selected_csvs = [
            os.path.join(group_output_dir, ds, "csv", "selected_frames.csv") for ds in selected
        ]
        merged_csv = os.path.join(group_output_dir, "csv", "selected_frames_merged.csv")
        if _merge_selected_frames_csv(session_selected_csvs, selected, merged_csv):
            print(f"Merged selected frames CSV saved: {merged_csv}")
        else:
            print("No selected_frames.csv found to merge.")

        # 3) Merged pose plot (per-session selections)
        plane = str(mvs_cfg.get("POSE_PLOT_PLANE", "xz") or "xz")
        arrow_stride = int(mvs_cfg.get("POSE_PLOT_ARROW_STRIDE", 10) or 10)
        arrow_scale = float(mvs_cfg.get("POSE_PLOT_ARROW_SCALE", 0.25) or 0.25)
        pose_csvs = [
            os.path.join(home_dir, "data", ds, "txt", "left_camera_poses.csv") for ds in selected
        ]
        merged_plot = os.path.join(group_output_dir, "plots", "selected_camera_poses_merged.png")
        if _save_merged_pose_plot(
            session_names=selected,
            session_pose_csvs=pose_csvs,
            session_selected_csvs=session_selected_csvs,
            out_path=merged_plot,
            plane=plane,
            arrow_stride=arrow_stride,
            arrow_scale=arrow_scale,
            title=f"Group={group_name} (sessions={len(selected)})",
        ):
            print(f"Merged pose plot saved: {merged_plot}")
        else:
            print("Merged pose plot skipped (missing data or dependencies).")

        # 4) Global re-index (integrated load) CSV + plot (post-run refresh)
        try:
            rows = _build_global_selected_pose_rows(
                project_root=project_root,
                home_dir=home_dir,
                group_output_dir=group_output_dir,
                session_names=selected,
            )
            if _write_global_pose_csv(rows, global_pose_csv):
                print(f"Global re-indexed pose CSV saved: {global_pose_csv} (count={len(rows)})")
            global_plot = os.path.join(group_output_dir, "plots", "global_selected_camera_poses.png")
            if _save_global_pose_plot_from_rows(
                rows=rows,
                out_path=global_plot,
                plane=plane,
                arrow_stride=arrow_stride,
                arrow_scale=arrow_scale,
                title=f"Group={group_name} global_idx=0..{max(0, len(rows)-1)}",
            ):
                print(f"Global re-indexed pose plot saved: {global_plot}")
        except Exception as e:
            print(f"Global re-index outputs skipped: {e}")

        sys.exit(0)

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
