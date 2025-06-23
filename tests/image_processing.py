import cv2
import numpy as np
from numba import njit


class ImageProcessor:
    def __init__(self, config):
        self.config = config

    def create_disparity(self, image_L, image_R):
        """ステレオ画像から視差マップを生成する"""
        stereo = cv2.StereoSGBM_create(
            minDisparity=self.config.min_disp,
            numDisparities=self.config.num_disp,
            blockSize=self.config.window_size,
            P1=8 * 3 * self.config.window_size ** 2,
            P2=16 * 3 * self.config.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )
        disp = stereo.compute(image_L, image_R).astype(np.float32) / 16.0
        return disp

    @staticmethod
    @njit
    def census_transform_numba(img, window_size=7):
        """Census変換を行う (Numbaによる高速化)"""
        half = window_size // 2
        rows, cols = img.shape
        census = np.zeros((rows, cols), np.uint64)
        for y in range(half, rows - half):
            for x in range(half, cols - half):
                center = img[y, x]
                desc = 0
                for i in range(-half, half + 1):
                    for j in range(-half, half + 1):
                        if i == 0 and j == 0:
                            continue
                        desc <<= 1
                        if img[y + i, x + j] < center:
                            desc |= 1
                census[y, x] = desc
        return census

    @staticmethod
    @njit
    def compute_census_cost(left, right, disparity, c_left, c_right, window_size):
        """Census変換によるコストを計算する (Numbaによる高速化)"""
        half = window_size // 2
        rows, cols = left.shape
        cost = np.zeros((rows, cols), np.float32)
        for y in range(half, rows - half):
            for x in range(half, cols - half):
                d = int(round(disparity[y, x]))
                if d >= 0:
                    xr = x - d
                    if xr - half >= 0 and xr + half < cols:
                        xorv = c_left[y, x] ^ c_right[y, xr]
                        c = 0
                        while xorv:
                            c += xorv & 1
                            xorv >>= 1
                        cost[y, x] = c
        return cost
