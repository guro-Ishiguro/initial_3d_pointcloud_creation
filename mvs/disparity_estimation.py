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
