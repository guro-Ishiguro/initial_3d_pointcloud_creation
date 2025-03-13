import open3d as o3d
import numpy as np


class PointCloudCreator:
    def __init__(self):
        pass

    def grid_sampling(self, point_cloud, colors, confidences, grid_size):
        rounded_coords = np.floor(point_cloud / grid_size).astype(int)
        _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
        return (
            point_cloud[unique_indices],
            colors[unique_indices],
            confidences[unique_indices],
        )

    def process_point_cloud(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
        return pcd

    def voxel_downsample_points(self, points, colors, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.asarray(pcd_down.points), np.asarray(pcd_down.colors)

    def write_ply(self, filename, vertices, colors):
        assert vertices.shape[0] == colors.shape[0], "頂点数と色データの数が一致しません。"
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
        with open(filename, "w") as f:
            f.write(header)
            data = np.hstack((vertices, (colors * 255).astype(np.uint8)))
            np.savetxt(f, data, fmt="%f %f %f %d %d %d")
