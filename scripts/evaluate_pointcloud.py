import os
import numpy as np
import open3d as o3d
import torch

try:
    from pytorch3d.loss import emd_loss
    EMD_AVAILABLE = True
except ImportError:
    EMD_AVAILABLE = False
    print("Warning: PyTorch3D not found. EMD will be skipped.")

# ——— 設定 —————————————————————————————————————————————
# 再構成点群のパス
RECON_PATH = "/home/geolab/Projects/initial_3d_pointcloud_creation/output/" \
             "3840_2160_16_74.73365_92_1.0/point_cloud/old_output.ply"
# Unity から出力したグラウンドトゥルースメッシュのパス
GT_MESH_PATH = "/home/geolab/Projects/initial_3d_pointcloud_creation/data/" \
               "3840_2160_16_74.73365_92_1.0/ground_truth.obj"

# サンプリング点数（EMD用／Chamfer 用に同数パック）
EMD_SAMPLES = 1024

# ——— 関数定義 ——————————————————————————————————————————

def compute_chamfer_distance(pcd_p, pcd_q):
    P = np.asarray(pcd_p.points)
    Q = np.asarray(pcd_q.points)
    tree_q = o3d.geometry.KDTreeFlann(pcd_q)
    tree_p = o3d.geometry.KDTreeFlann(pcd_p)

    # P -> Q
    d_pq = 0.0
    for p in P:
        _, _, d2 = tree_q.search_knn_vector_3d(p, 1)
        d_pq += np.sqrt(d2[0])
    d_pq /= len(P)

    # Q -> P
    d_qp = 0.0
    for q in Q:
        _, _, d2 = tree_p.search_knn_vector_3d(q, 1)
        d_qp += np.sqrt(d2[0])
    d_qp /= len(Q)

    return d_pq + d_qp

def compute_earth_movers_distance(pcd_p, pcd_q, num_samples=EMD_SAMPLES):
    """
    Earth Mover’s Distance (EMD) using PyTorch3D の emd_loss。
    点群をランダムサンプリングまたはパディングして同一サイズに揃え、計算します。
    """
    if not EMD_AVAILABLE:
        raise RuntimeError("PyTorch3D が見つかりません。EMD を計算できません。")

    def sample_or_pad(arr, n):
        if len(arr) >= n:
            idx = np.random.choice(len(arr), n, replace=False)
            return arr[idx]
        else:
            pad = arr[np.random.choice(len(arr), n - len(arr), replace=True)]
            return np.vstack([arr, pad])

    P = np.asarray(pcd_p.points)
    Q = np.asarray(pcd_q.points)
    P_s = sample_or_pad(P, num_samples)
    Q_s = sample_or_pad(Q, num_samples)

    P_t = torch.from_numpy(P_s).float().unsqueeze(0)  # (1, N, 3)
    Q_t = torch.from_numpy(Q_s).float().unsqueeze(0)

    emd = emd_loss(P_t, Q_t, eps=1e-2, max_iter=100)
    return emd.item()

def compute_cloud_to_mesh(pcd, mesh):
    """
    Cloud-to-Mesh Distance を RaycastingScene で計算し、平均とRMSEを返す。
    """
    scene = o3d.t.geometry.RaycastingScene()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(tmesh)

    pts = np.asarray(pcd.points, dtype=np.float32)
    query = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    signed = scene.compute_signed_distance(query).numpy().flatten()
    d = np.abs(signed)
    return d.mean(), np.sqrt((d**2).mean())

# ——— メイン処理 ——————————————————————————————————————————

def main():
    # 1) 再構成点群の読み込み & Z 軸反転
    if not os.path.isfile(RECON_PATH):
        raise FileNotFoundError(f"Reconstructed point cloud not found:\n  {RECON_PATH}")
    pcd_recon = o3d.io.read_point_cloud(RECON_PATH)
    pts = np.asarray(pcd_recon.points)
    pts[:, 2] *= -1.0
    pcd_recon.points = o3d.utility.Vector3dVector(pts)
    if pcd_recon.is_empty():
        raise RuntimeError("Loaded reconstructed point cloud is empty.")
    print(f"✔ Reconstructed point cloud: {len(pcd_recon.points)} points")

    # 2) グラウンドトゥルースメッシュの読み込み & 法線計算
    if not os.path.isfile(GT_MESH_PATH):
        raise FileNotFoundError(f"Ground-truth mesh not found:\n  {GT_MESH_PATH}")
    mesh_gt = o3d.io.read_triangle_mesh(GT_MESH_PATH)
    if len(mesh_gt.vertices)==0 or len(mesh_gt.triangles)==0:
        raise RuntimeError("Loaded ground-truth mesh is empty.")
    mesh_gt.compute_triangle_normals()
    print(f"✔ Ground-truth mesh: {len(mesh_gt.vertices)} vertices, {len(mesh_gt.triangles)} triangles")

    # 3) Chamfer と EMD 用にメッシュをサンプリング
    num_pts = len(pcd_recon.points)
    pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=num_pts)
    print(f"✔ Sampled ground-truth mesh into point cloud: {len(pcd_gt.points)} points")

    # 4) Chamfer Distance
    chamfer = compute_chamfer_distance(pcd_recon, pcd_gt)
    print(f"Chamfer Distance:            {chamfer:.6f} m")

    # 5) Earth Mover's Distance (EMD)
    if EMD_AVAILABLE:
        emd = compute_earth_movers_distance(pcd_recon, pcd_gt, num_samples=EMD_SAMPLES)
        print(f"Earth Mover’s Distance (EMD): {emd:.6f} m")
    else:
        print("Earth Mover’s Distance (EMD): skipped (PyTorch3D not installed)")

    # 6) Cloud-to-Mesh Distance
    mean_cm, rmse_cm = compute_cloud_to_mesh(pcd_recon, mesh_gt)
    print(f"Cloud-to-Mesh Mean Distance: {mean_cm:.6f} m")
    print(f"Cloud-to-Mesh RMSE:          {rmse_cm:.6f} m")

    # 7) 可視化
    pcd_recon.paint_uniform_color([1.0, 0.706, 0])  # 黄色
    mesh_gt.paint_uniform_color([0.0, 0.651, 0.929])  # 青
    o3d.visualization.draw_geometries(
        [pcd_recon, mesh_gt],
        window_name="Reconstruction vs Ground Truth",
        width=1280, height=720,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()
