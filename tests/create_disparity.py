import cv2
import numpy as np
import logging
from numba import njit


class DisparityCreator:
    def __init__(self, window_size, min_disp, num_disp):
        self.window_size = window_size
        self.min_disp = min_disp
        self.num_disp = num_disp

    def create_disparity(self, image_L, image_R, img_id):
        if image_L is None or image_R is None:
            logging.error(f"Error: One or both images for ID {img_id} are None.")
            return None
        stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.window_size,
            P1=8 * 3 * self.window_size ** 2,
            P2=16 * 3 * self.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )
        disparity = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
        return disparity

    @staticmethod
    @njit
    def census_transform_numba(img, window_size=7):
        half = window_size // 2
        rows, cols = img.shape
        census = np.zeros((rows, cols), dtype=np.uint64)
        for y in range(half, rows - half):
            for x in range(half, cols - half):
                center = img[y, x]
                descriptor = 0
                for i in range(-half, half + 1):
                    for j in range(-half, half + 1):
                        if i == 0 and j == 0:
                            continue
                        descriptor = descriptor << 1
                        if img[y + i, x + j] < center:
                            descriptor |= 1
                census[y, x] = descriptor
        return census

    @staticmethod
    @njit
    def compute_census_cost(
        left, right, disparity, census_left, census_right, window_size
    ):
        half = window_size // 2
        rows, cols = left.shape
        cost_image = np.zeros((rows, cols), dtype=np.float32)
        for y in range(half, rows - half):
            for x in range(half, cols - half):
                d_val = disparity[y, x]
                d = int(np.round(d_val))
                if d >= 0:
                    xr = x - d
                    if (xr - half >= 0) and (xr + half < cols):
                        cost = 0
                        xor_val = census_left[y, x] ^ census_right[y, xr]
                        while xor_val:
                            cost += xor_val & 1
                            xor_val = xor_val >> 1
                        cost_image[y, x] = cost
        return cost_image
