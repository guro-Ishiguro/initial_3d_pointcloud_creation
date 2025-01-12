import cv2
import os

def video_to_frames(video_path, output_folder, prefix):
    """
    動画をフレームに分割して保存する関数

    Parameters:
        video_path (str): 動画ファイルのパス
        output_folder (str): フレームを保存するフォルダのパス
    """
    # フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"動画を開けませんでした: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        # フレームを画像として保存
        frame_file = os.path.join(output_folder, f"{prefix}_{frame_count:05d}.png")
        cv2.imwrite(frame_file, frame)
        frame_count += 1

    cap.release()
    print(f"動画のフレームを保存しました: {video_path} -> {output_folder}")

# 動画ファイルのパスを指定
video1_path = "/home/geolab/Projects/initial_3d_pointcloud_creation/data/3840_2160_16_74.73365_92_0.3/videos/Left.webm"  
video2_path = "/home/geolab/Projects/initial_3d_pointcloud_creation/data/3840_2160_16_74.73365_92_0.3/videos/Right.webm" 

# 出力フォルダを指定
output_folder = "/home/geolab/Projects/initial_3d_pointcloud_creation/data/3840_2160_16_74.73365_92_0.3/images" 

# フレームを保存
video_to_frames(video1_path, output_folder, "left")
video_to_frames(video1_path, output_folder, "right")
