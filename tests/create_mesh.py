import open3d as o3d
import logging
import config

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



def save_mesh(mesh, output_path):
    """生成されたメッシュをファイルに保存"""
    try:
        o3d.io.write_triangle_mesh(output_path, mesh)
        logging.info(f"Mesh saved to {output_path}.")
    except Exception as e:
        logging.error(f"Error saving mesh: {e}")


def create_mesh_from_point_cloud(pcd):
    """点群からメッシュを生成"""
    try:
        # 法線を計算
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=20)
        )
        logging.info("Normals estimated for the point cloud.")

        # ポアソン表面再構成でメッシュを作成
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10
        )
        logging.info(f"Mesh created with {len(mesh.triangles)} triangles.")

        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        logging.info("Bounding box cleaned the mesh.")

        return mesh
    except Exception as e:
        logging.error(f"Error creating mesh: {e}")
        return None
    
def load_point_cloud(file_path):
    """PLYファイルから三次元点群を読み込む"""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        logging.info(
            f"Loaded point cloud from {file_path} with {len(pcd.points)} points."
        )
        return pcd
    except Exception as e:
        logging.error(f"Error loading point cloud: {e}")
        return None


if __name__ == "__main__":
    # PLYファイルのパス
    input_file = config.POINT_CLOUD_FILE_PATH

    # 点群を読み込み
    pcd = load_point_cloud(input_file)

    o3d.visualization.draw_geometries([pcd])

    # 点群からメッシュを生成する
    mesh = create_mesh_from_point_cloud(pcd)

    # 生成したメッシュを保存する
    save_mesh(mesh, config.MESH_FILE_PATH)

    # メッシュの表示
    o3d.visualization.draw_geometries([mesh], window_name="Generated Mesh")