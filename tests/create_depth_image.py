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


def to_orthographic_projection(depth, color_image, camera_height):
    """中心投影から正射投影への変換を適用し、カラー画像を正射投影にマッピング"""
    rows, cols = depth.shape
    mid_idx = cols // 2
    mid_idy = rows // 2
    col_indices = np.vstack([np.arange(cols)] * rows)
    row_indices = np.vstack([np.arange(rows)] * cols).transpose()

    shift_x = np.where(
        (depth > 40) | (depth < 0),
        0,
        ((camera_height - depth) * (mid_idx - col_indices) / camera_height).astype(int),
    )
    shift_y = np.where(
        (depth > 40) | (depth < 0),
        0,
        ((camera_height - depth) * (mid_idy - row_indices) / camera_height).astype(int),
    )

    new_x = col_indices + shift_x
    new_y = row_indices + shift_y

    valid_mask = (new_x >= 0) & (new_x < cols) & (new_y >= 0) & (new_y < rows)
    new_x = np.where(valid_mask, new_x, -1)
    new_y = np.where(valid_mask, new_y, -1)

    ortho_color_image = np.zeros_like(color_image)
    valid_depth_mask = (depth > 0) & (depth < 40) & valid_mask

    ortho_color_image[new_y[valid_depth_mask], new_x[valid_depth_mask]] = color_image[
        row_indices[valid_depth_mask], col_indices[valid_depth_mask]
    ]

    ortho_depth = np.full_like(depth, np.inf)
    np.minimum.at(ortho_depth, (new_y[valid_mask], new_x[valid_mask]), depth[valid_mask])
    ortho_depth[ortho_depth == np.inf] = 0

    ortho_color_image[ortho_depth == 0] = [0, 0, 0]

    return ortho_depth, ortho_color_image


def save_depth_colormap(depth, output_path):
    """視差画像をカラーマップで保存する"""
    depth_normalized = cv2.normalize(
        depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # カラーマップの範囲を表示
    plt.imshow(depth_colormap)
    plt.colorbar(label='Depth (normalized)')  # カラーバーを表示
    plt.title("Disparity Colormap with Depth Values")
    plt.savefig(output_path)

# パラメータの読み込み
(
    B,
    focal_length,
    camera_height,
    K,
    R,
) = (config.B, config.focal_length, config.camera_height, config.K, config.R)
window_size, min_disp, num_disp = config.window_size, config.min_disp, config.num_disp

# 左右の画像を読み込み
img_id = 26
left_image = cv2.imread(
    os.path.join(config.IMAGE_DIR, f"left_{str(img_id).zfill(6)}.png")
)
left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_image = cv2.imread(
    os.path.join(config.IMAGE_DIR, f"right_{str(img_id).zfill(6)}.png"),
    cv2.IMREAD_GRAYSCALE,
)

# 視差画像を生成
disparity = create_disparity_image(
    left_image_gray,
    right_image,
    window_size=window_size,
    min_disp=min_disp,
    num_disp=num_disp,
)

# 深度画像を生成
depth = B * focal_length / (disparity + 1e-6)
depth[(depth < 10) | (depth > 40)] = 0
ortho_depth, ortho_color_image = to_orthographic_projection(depth, left_image, camera_height)

# 深度画像をカラーマップとして保存
output_path = os.path.join(config.DEPTH_IMAGE_DIR, f"depth_{img_id}.png")
save_depth_colormap(depth, output_path)
