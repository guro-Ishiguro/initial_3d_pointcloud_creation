import os
import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt

# Optional EMD
try:
    from pytorch3d.loss import emd_loss
    EMD_AVAILABLE = True
except ImportError:
    EMD_AVAILABLE = False
    print("Warning: PyTorch3D not found. EMD will be skipped.")

# ——— Configuration —————————————————————————————————————————————
RECON_PATH = "/home/geolab/Projects/initial_3d_pointcloud_creation/output/" \
             "3840_2160_16_74.73365_92_1.0/point_cloud/old_output.ply"
GT_MESH_PATH = "/home/geolab/Projects/initial_3d_pointcloud_creation/data/" \
               "3840_2160_16_74.73365_92_1.0/ground_truth.obj"
EMD_SAMPLES = 1024

# ——— Helper Functions ——————————————————————————————————————————

def compute_chamfer_distance(pcd_p, pcd_q):
    P = np.asarray(pcd_p.points)
    Q = np.asarray(pcd_q.points)
    tree_q = o3d.geometry.KDTreeFlann(pcd_q)
    tree_p = o3d.geometry.KDTreeFlann(pcd_p)
    d_pq = sum(np.sqrt(tree_q.search_knn_vector_3d(p, 1)[2][0]) for p in P) / len(P)
    d_qp = sum(np.sqrt(tree_p.search_knn_vector_3d(q, 1)[2][0]) for q in Q) / len(Q)
    return d_pq + d_qp


def compute_earth_movers_distance(pcd_p, pcd_q, num_samples=EMD_SAMPLES):
    if not EMD_AVAILABLE:
        raise RuntimeError("PyTorch3D not found. Cannot compute EMD.")
    def sample_or_pad(arr, n):
        if len(arr) >= n:
            return arr[np.random.choice(len(arr), n, replace=False)]
        pad = arr[np.random.choice(len(arr), n - len(arr), replace=True)]
        return np.vstack([arr, pad])
    P = np.asarray(pcd_p.points)
    Q = np.asarray(pcd_q.points)
    P_s = sample_or_pad(P, num_samples)
    Q_s = sample_or_pad(Q, num_samples)
    P_t = torch.from_numpy(P_s).float().unsqueeze(0)
    Q_t = torch.from_numpy(Q_s).float().unsqueeze(0)
    return emd_loss(P_t, Q_t, eps=1e-2, max_iter=100).item()


def compute_cloud_to_mesh(pcd, mesh):
    scene = o3d.t.geometry.RaycastingScene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(tmesh)
    pts = np.asarray(pcd.points, dtype=np.float32)
    query = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    signed = scene.compute_signed_distance(query).numpy().flatten()
    d = np.abs(signed)
    return d.mean(), np.sqrt((d**2).mean()), d


def compute_density(pcd, cell_size=0.1):
    pts = np.asarray(pcd.points)[:, :2]
    idx = np.floor(pts / cell_size).astype(int)
    _, counts = np.unique(idx, axis=0, return_counts=True)
    return counts.mean(), counts.std()


def compute_coverage(pcd, mesh, voxel_size=0.02, sample_pts=50000):
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=sample_pts)
    tree = o3d.geometry.KDTreeFlann(pcd)
    covered = sum(
        1 for p in np.asarray(mesh_pcd.points)
        if np.sqrt(tree.search_knn_vector_3d(p, 1)[2][0]) <= voxel_size
    )
    return covered / sample_pts


def compute_outlier_ratio(pcd, mesh, threshold=0.02):
    _, _, errors = compute_cloud_to_mesh(pcd, mesh)
    ratio = np.sum(errors > threshold) / len(errors)
    return ratio, errors


def compute_nn_stats(pcd):
    pts = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for p in pts:
        _, _, d2 = tree.search_knn_vector_3d(p, 2)
        dists.append(np.sqrt(d2[1]))
    arr = np.array(dists)
    return arr.mean(), np.median(arr), arr.std()


def compute_forward_backward_consistency(pcd_p, pcd_q):
    P_pts = np.asarray(pcd_p.points)
    Q_pts = np.asarray(pcd_q.points)
    tree_q = o3d.geometry.KDTreeFlann(pcd_q)
    tree_p = o3d.geometry.KDTreeFlann(pcd_p)
    dists = []
    for p in P_pts:
        _, idx_q, _ = tree_q.search_knn_vector_3d(p, 1)
        q = Q_pts[idx_q[0]]
        _, idx_p2, _ = tree_p.search_knn_vector_3d(q, 1)
        p2 = P_pts[idx_p2[0]]
        dists.append(np.linalg.norm(p - p2))
    dists = np.array(dists)
    return dists.mean(), dists.std(), dists


def plot_error_histogram(errors, out_path="error_histogram_roi.png", bins=50):
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=bins, edgecolor='black')
    plt.title("Point-to-Mesh Error Distribution (ROI)")
    plt.xlabel('Error (m)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Error histogram saved to {out_path}")

# ——— Main ——————————————————————————————————————————

def main():
    # 1) Load reconstructed point cloud
    pcd = o3d.io.read_point_cloud(RECON_PATH)
    pts = np.asarray(pcd.points)
    pts[:, 2] *= -1.0
    pcd.points = o3d.utility.Vector3dVector(pts)
    print(f"✔ Reconstructed point cloud: {len(pcd.points)} points")

    # 2) Load ground-truth mesh
    mesh = o3d.io.read_triangle_mesh(GT_MESH_PATH)
    mesh.compute_triangle_normals()
    print(f"✔ Ground-truth mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # 3) Define ROI by point cloud's AABB
    aabb = pcd.get_axis_aligned_bounding_box()

    # 4) Sample full mesh to point cloud
    num_pts = len(pcd.points)
    pcd_gt_full = mesh.sample_points_uniformly(number_of_points=num_pts)
    print(f"✔ Sampled full mesh into {len(pcd_gt_full.points)} points")

    # 5) Crop the sampled ground-truth point cloud by ROI
    pcd_gt_roi = pcd_gt_full.crop(aabb)
    print(f"✔ Cropped ground-truth point cloud: {len(pcd_gt_roi.points)} points")

    # 6) Chamfer Distance (ROI)
    cd_roi = compute_chamfer_distance(pcd, pcd_gt_roi)
    print(f"Chamfer Distance (ROI): {cd_roi:.6f} m")

    # 7) Earth Mover’s Distance (ROI)
    if EMD_AVAILABLE:
        emd_roi = compute_earth_movers_distance(pcd, pcd_gt_roi)
        print(f"EMD (ROI): {emd_roi:.6f} m")
    else:
        print("EMD (ROI): skipped")

    # 8) Cloud-to-Mesh (full mesh)
    mean_cm, rmse_cm, errors = compute_cloud_to_mesh(pcd, mesh)
    print(f"Cloud-to-Mesh Mean: {mean_cm:.6f} m, RMSE: {rmse_cm:.6f} m")

    # 9) Density
    avg_den, std_den = compute_density(pcd, cell_size=0.1)
    print(f"Average density: {avg_den:.1f} ±{std_den:.1f} pts/m²")

    # 10) Coverage (full mesh)
    cov = compute_coverage(pcd, mesh, voxel_size=0.02, sample_pts=50000)
    print(f"Coverage ratio: {cov*100:.1f}%")

    # 11) Outlier ratio (full mesh)
    outlier_ratio, _ = compute_outlier_ratio(pcd, mesh, threshold=0.1)
    print(f"Outlier ratio (>0.1m): {outlier_ratio*100:.1f}%")

    # 12) NN stats
    mean_nn, med_nn, std_nn = compute_nn_stats(pcd)
    print(f"NN distance mean: {mean_nn:.3f}, median: {med_nn:.3f}, std: {std_nn:.3f}")

    # 13) Forward-Backward Consistency
    fb_mean, fb_std, fb_dists = compute_forward_backward_consistency(pcd, pcd_gt_roi)
    print(f"Forward-Backward Consistency (ROI) mean: {fb_mean:.6f} m, std: {fb_std:.6f} m")

    # 14) Histogram
    plot_error_histogram(errors)

    # 15) Visualize ROI comparison
    pcd.paint_uniform_color([1.0, 0.706, 0])
    pcd_gt_roi.paint_uniform_color([0.0, 0.651, 0.929])
    o3d.visualization.draw_geometries(
        [pcd, pcd_gt_roi],
        window_name="Reconstruction vs Ground Truth (ROI)",
        width=1280, height=720,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()