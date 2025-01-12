import os
import cv2

def process_images(input_dir, output_dir, size=(960, 540)):
    # 入力ディレクトリの存在を確認
    if not os.path.exists(input_dir):
        print(f"入力ディレクトリ {input_dir} が存在しません。")
        return
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 入力ディレクトリ内の画像を処理
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        
        # ファイルが画像かどうかを確認
        if os.path.isfile(input_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')) and filename.startswith('left'):
            try:
                # 画像を読み込み
                img = cv2.imread(input_path)
                
                # 画像が読み込めなかった場合をチェック
                if img is None:
                    print(f"{filename} を読み込めませんでした。スキップします。")
                    continue
                
                # 白黒（グレースケール）に変換
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # サイズ変更
                resized_img = cv2.resize(gray_img, size)
                
                # 出力パス
                output_path = os.path.join(output_dir, filename)
                
                # 画像を保存
                cv2.imwrite(output_path, resized_img)
                print(f"{filename} を処理して保存しました。")
            except Exception as e:
                print(f"{filename} の処理中にエラーが発生しました: {e}")
        else:
            print(f"{filename} は画像ではないためスキップされました。")

# 左と右の画像をそれぞれ処理
process_images('/home/geolab/Projects/initial_3d_pointcloud_creation/data/3840_2160_16_74.73365_92_0.3/images', '/home/geolab/Projects/catkin_ws/src/process_images/data/left')