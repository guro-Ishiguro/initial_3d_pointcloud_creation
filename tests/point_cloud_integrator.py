import numpy as np
import open3d as o3d
import logging


class PointCloudIntegrator:
    def __init__(self, config):
        self.config = config

    def integrate_depth_maps_median(self, points_list, colors_list, voxel_size=0.1):
        """
        複数の深度マップから生成された点群を統合し、ボクセルグリッド内でメディアンを計算する。
        """
        if not points_list:
            logging.warning("No points to integrate.")
            return np.array([]), np.array([])

        all_pts = np.vstack(points_list)
        all_cols = np.vstack(colors_list)

        # NaNや無限大の点を除去
        valid = np.isfinite(all_pts).all(axis=1)
        pts = all_pts[valid]
        cols = all_cols[valid]

        if pts.shape[0] == 0:
            logging.warning("No valid points after filtering for integration.")
            return np.array([]), np.array([])

        # 各点を対応するボクセルIDに割り当てる
        vids = np.floor(pts / voxel_size).astype(int)

        voxel_dict = {}
        for i, vid in enumerate(map(tuple, vids)):
            voxel_dict.setdefault(vid, []).append(i)

        med_pts, med_cols = [], []
        for idxs in voxel_dict.values():
            voxel_pts = pts[idxs]
            voxel_cols = cols[idxs]
            # 各ボクセル内の点のメディアンを計算
            med_pts.append(np.median(voxel_pts, axis=0))
            med_cols.append(np.median(voxel_cols, axis=0))

        logging.info(f"Integrated {len(med_pts)} points from multiple depth maps.")
        return np.array(med_pts), np.array(med_cols)

    def process_and_save_final_point_cloud(self, points_list, colors_list, file_path):
        """最終的な点群を処理し、PLYファイルとして保存する"""
        if not points_list:
            logging.warning("No point clouds to process.")
            return None

        # リストを結合して単一の配列にする
        points = np.vstack(points_list)
        colors = np.vstack(colors_list)

        if points.shape[0] == 0:
            logging.warning("No points to process for final point cloud.")
            return None

        # Z軸の反転
        points[:, 2] = -points[:, 2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        logging.info(f"Initial merged point cloud size: {len(pcd.points)}")

        # 外れ値除去
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
        logging.info(f"Point cloud size after outlier removal: {len(pcd.points)}")

        # PLYファイルとして保存
        self.write_ply(file_path, np.asarray(pcd.points), np.asarray(pcd.colors))

        return pcd

    @staticmethod
    def write_ply(filename, vertices, colors):
        """点群データをPLYファイルとして書き込む"""
        assert (
            vertices.shape[0] == colors.shape[0]
        ), "Vertices and colors must have the same number of points."

        colors_uchar = (colors * 255).astype(np.uint8)

        header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        data = np.hstack((vertices, colors_uchar))

        with open(filename, "w") as f:
            f.write(header)
            np.savetxt(f, data, fmt="%f %f %f %d %d %d")
        logging.info(f"Final point cloud saved to {filename}")
