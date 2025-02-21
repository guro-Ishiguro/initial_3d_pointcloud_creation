import open3d as o3d
import logging
import config

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
