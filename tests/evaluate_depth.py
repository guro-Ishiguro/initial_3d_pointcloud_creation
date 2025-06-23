import cv2
import numpy as np
import sys
import OpenEXR  
import Imath  
from utils import save_depth_map_as_image

def read_exr_depth(file_path):
    """
    OpenEXRライブラリを使用して、単一チャンネルのEXR深度ファイルを読み込む関数。
    """
    try:
        exr_file = OpenEXR.InputFile(file_path)
        header = exr_file.header()
        
        available_channels = list(header['channels'].keys())
        
        target_channel = ''
        if 'R' in available_channels:
            target_channel = 'R'
        elif 'Y' in available_channels:
            target_channel = 'Y'
        else:
            print(f"エラー: 深度情報を持つ 'R' または 'Y' チャンネルが見つかりませんでした。")
            print(f"利用可能なチャンネル: {available_channels}")
            return None
        
        print(f"情報: ファイル内に '{target_channel}' チャンネルを検出しました。これを深度データとして読み込みます。")

        dw = header['dataWindow']
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_bytes = exr_file.channel(target_channel, pt)
        
        depth_map = np.frombuffer(channel_bytes, dtype=np.float32)
        depth_map = depth_map.reshape(size)
        
        return depth_map
    except Exception as e:
        print(f"EXRファイルの読み込み中にエラーが発生しました: {e}")
        return None

def main():
    depth_map_path = '/home/geolab/Projects/initial_3d_pointcloud_creation/data/1280_720_45_74.73365_92_1.0/images/depth/depth_000002.exr' 

    depth_map = read_exr_depth(depth_map_path)
    
    if depth_map is None:
        print("処理を終了します。")
        sys.exit()

    save_depth_map_as_image(depth_map, 'output_depth_image.png')

if __name__ == '__main__':
    main()