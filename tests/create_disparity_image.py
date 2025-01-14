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
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=200,
        speckleRange=2,
    )
    disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
    return disparity


def to_orthographic_projection(depth, camera_height):
    """中心投影から正射投影への変換を適用する"""
    rows, cols = depth.shape
    mid_idx = cols // 2
    mid_idy = rows // 2
    col_indices = np.vstack([np.arange(cols)] * rows)
    row_indices = np.vstack([np.arange(rows)] * cols).transpose()
    shift_x = (
        (camera_height - depth) * (mid_idx - col_indices) / camera_height
    ).astype(int)
    shift_y = (
        (camera_height - depth) * (mid_idy - row_indices) / camera_height
    ).astype(int)
    new_x = np.clip(col_indices + shift_x, 0, cols - 1)
    new_y = np.clip(row_indices + shift_y, 0, rows - 1)
    ortho_depth = np.full_like(depth, np.inf)
    np.minimum.at(ortho_depth, (new_y, new_x), depth)
    ortho_depth[ortho_depth == np.inf] = 0
    return ortho_depth


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
img_id = 102
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
