import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import config


def create_disparity_image(image_L, image_R, window_size, min_disp, num_disp):
    """左・右画像から視差画像を生成する"""
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=16 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disparity


def save_disparity_colormap(disparity, output_path):
    """視差画像をカラーマップで保存する"""
    disparity_normalized = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    disparity_normalized = disparity_normalized.astype(np.uint8)
    disparity_colormap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    plt.imsave(output_path, disparity_colormap)
    print(f"Disparity colormap saved as {output_path}")


# パラメータの読み込み
camera_height = config.camera_height
window_size, min_disp, num_disp = config.window_size, config.min_disp, config.num_disp

# 左右の画像を読み込み
img_id = 87
left_image = cv2.imread(
    os.path.join(config.IMAGE_DIR, f"left_{str(img_id).zfill(6)}.png"),
    cv2.IMREAD_GRAYSCALE,
)
right_image = cv2.imread(
    os.path.join(config.IMAGE_DIR, f"right_{str(img_id).zfill(6)}.png"),
    cv2.IMREAD_GRAYSCALE,
)

# 視差画像を生成
disparity = create_disparity_image(
    left_image,
    right_image,
    window_size=window_size,
    min_disp=min_disp,
    num_disp=num_disp,
)

# 視差画像をカラーマップとして保存
output_path = os.path.join(config.DISPARITY_IMAGE_DIR, f"disparity_{img_id}.png")
save_disparity_colormap(disparity, output_path)
