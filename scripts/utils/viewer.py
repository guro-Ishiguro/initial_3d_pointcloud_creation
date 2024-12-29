import cv2
import numpy as np
import open3d as o3d


class ViewerRecorder:
    def __init__(
        self,
        width=640,
        height=480,
        video_filename="output.avi",
        fps=10,
        record_video=False,
    ):
        """
        ビューワーと録画機能を初期化
        :param width: ビューワーと録画の横解像度
        :param height: ビューワーと録画の縦解像度
        :param video_filename: 録画ファイル名
        :param fps: 録画のフレームレート
        :param record_video: 録画を有効にするかどうか
        """
        self.width = width
        self.height = height
        self.record_video = record_video
        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(width=self.width, height=self.height)

        # 録画が有効な場合にビデオライターを初期化
        self.video_writer = None
        if self.record_video:
            self.video_writer = cv2.VideoWriter(
                video_filename,
                cv2.VideoWriter_fourcc(*"XVID"),
                fps,
                (self.width, self.height),
            )

    def update_viewer(self, points):
        """ビューワーに点群データを表示・更新し、必要に応じて録画"""
        # 点群データを表示
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        self.viewer.clear_geometries()
        self.viewer.add_geometry(pcd)
        self.viewer.poll_events()
        self.viewer.update_renderer()

        # 録画が有効な場合はフレームを記録
        if self.record_video and self.video_writer:
            self.record_frame()

    def record_frame(self):
        """ビューワーの現在のフレームを録画"""
        image = (
            np.asarray(self.viewer.capture_screen_float_buffer(do_render=True)) * 255
        )
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(image, (self.width, self.height))
        self.video_writer.write(resized_image)

    def close(self):
        """ビューワーとビデオライターを閉じる"""
        self.viewer.destroy_window()
        if self.video_writer:
            self.video_writer.release()
