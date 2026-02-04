"""
視差推定モジュール。
"""

import cv2
import numpy as np


class ImageProcessor:
    """
    視差マップ生成を行うクラス。
    """

    def __init__(self, config):
        self.config = config

    def create_disparity(self, image_L, image_R):
        """
        左右ステレオ画像から OpenCV SGBM で視差マップを生成する。
        """
        stereo = cv2.StereoSGBM_create(
            minDisparity=self.config.MIN_DISP,
            numDisparities=self.config.NUM_DISP,
            blockSize=self.config.WINDOW_SIZE,
            P1=8 * 3 * self.config.WINDOW_SIZE**2,
            P2=16 * 3 * self.config.WINDOW_SIZE**2,
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
        """
        画像の各ピクセルについて、窓内の周辺ピクセルが中心より暗いかをビット列で表現する Census 変換を行う。
        """
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
        """
        左画像の各ピクセルで、視差に従って右画像の対応点を参照し、Census 記述子の XOR の popcount をコストとする
        """
        half = window_size // 2
        rows, cols = left.shape
        cost = np.zeros((rows, cols), np.float32)
        for y in range(half, rows - half):
            for x in range(half, cols - half):
                d = int(round(disparity[y, x]))
                if d >= 0:
                    xr = x - d  # 右画像の対応点の x 座標
                    if xr - half >= 0 and xr + half < cols:
                        xorv = c_left[y, x] ^ c_right[y, xr]
                        c = 0
                        while xorv:
                            c += xorv & 1
                            xorv >>= 1
                        cost[y, x] = c  # XOR の popcount（ビットの 1 の個数）
        return cost
