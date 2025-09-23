import numpy as np
import os
import logging
from typing import List, Optional

from utils import save_depth_map_as_image


class CameraPlaneMedianFuser:
    """
    参照カメラの画像平面に、各ビューの深度マップをポーズでワープして逐次中央値融合する。

    - 参照: (K_ref, R_ref, T_ref) と解像度 (H,W)
    - 追加: add_depth_from_source(depth_src, R_src, T_src)
      -> src 深度を3D化→ワールド→参照カメラに投影→深度イメージとしてラスタ化
      -> ラスタ結果をレイヤとして保持し、np.nanmedian で融合
    """

    def __init__(
        self,
        height: int,
        width: int,
        K: np.ndarray,
        R_ref: np.ndarray,
        T_ref: np.ndarray,
    ):
        self.h = int(height)
        self.w = int(width)
        self.K = K.astype(np.float32)
        self.R_ref = R_ref.astype(np.float32)
        self.T_ref = T_ref.astype(np.float32)
        self.Kinv = np.linalg.inv(self.K).astype(np.float32)
        self._layers: List[np.ndarray] = []
        self._fused: Optional[np.ndarray] = None

        # precompute pixel grid (homogeneous)
        u = np.arange(self.w, dtype=np.float32)
        v = np.arange(self.h, dtype=np.float32)
        self.U, self.V = np.meshgrid(u, v)

    def _warp_to_ref(
        self, depth_src: np.ndarray, R_src: np.ndarray, T_src: np.ndarray
    ) -> np.ndarray:
        img_ref = np.full((self.h, self.w), np.nan, dtype=np.float32)
        d = depth_src.astype(np.float32)
        mask = np.isfinite(d) & (d > 0)
        if not np.any(mask):
            return img_ref
        u = self.U[mask]
        v = self.V[mask]
        z = d[mask]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        # backproject to source camera
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_src = np.stack([x, y, z], axis=0)  # (3,N)
        R_src = R_src.astype(np.float32)
        T_src = T_src.astype(np.float32).reshape(3, 1)
        # camera->world (R is world->camera)
        pts_world = R_src.T @ (pts_src - T_src)
        # world->ref camera
        pts_ref = self.R_ref @ pts_world + self.T_ref.reshape(3, 1)
        Zr = pts_ref[2]
        valid = Zr > 1e-6
        if not np.any(valid):
            return img_ref
        Xr = pts_ref[0][valid]
        Yr = pts_ref[1][valid]
        Zr = Zr[valid]
        ur = self.K[0, 0] * Xr / Zr + cx
        vr = self.K[1, 1] * Yr / Zr + cy
        ui = np.rint(ur).astype(np.int32)
        vi = np.rint(vr).astype(np.int32)
        inb = (ui >= 0) & (ui < self.w) & (vi >= 0) & (vi < self.h)
        ui = ui[inb]
        vi = vi[inb]
        zr = Zr[inb].astype(np.float32)
        # z-buffer like: keep nearest (min z)
        for uu, vv, zz in zip(ui, vi, zr):
            if np.isnan(img_ref[vv, uu]) or zz < img_ref[vv, uu]:
                img_ref[vv, uu] = zz
        return img_ref

    def add_depth_from_source(
        self, depth_src: np.ndarray, R_src: np.ndarray, T_src: np.ndarray
    ) -> np.ndarray:
        warped = self._warp_to_ref(depth_src, R_src, T_src)
        self._layers.append(warped)
        stack = np.stack(self._layers, axis=0)
        self._fused = np.nanmedian(stack, axis=0).astype(np.float32)
        return self._fused

    def get_fused_depth(self) -> Optional[np.ndarray]:
        return self._fused

    def save_fused(
        self, save_dir: str, filename: str = "fused_camera_plane_running.png"
    ) -> Optional[str]:
        if self._fused is None:
            return None
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        save_depth_map_as_image(self._fused, path)
        logging.info(f"Saved camera-plane fused depth to {path}")
        return path


class OrthoDepthMedianFuser:
    """
    正射投影（オルソ）深度マップを逐次的に中央値融合する。
    すべて同一グリッド・同一解像度であることを前提とする。
    """

    def __init__(self):
        self._layers: List[np.ndarray] = []
        self._fused: Optional[np.ndarray] = None

    def add_depth_map(self, ortho_depth: np.ndarray) -> np.ndarray:
        if ortho_depth is None:
            return self._fused if self._fused is not None else np.array([])
        if self._fused is None:
            self._fused = ortho_depth.astype(np.float32).copy()
            self._layers.append(self._fused)
            return self._fused
        if ortho_depth.shape != self._fused.shape:
            raise ValueError(
                f"Ortho depth shape mismatch: got {ortho_depth.shape}, expected {self._fused.shape}"
            )
        self._layers.append(ortho_depth.astype(np.float32))
        stack = np.stack(self._layers, axis=0)
        self._fused = np.nanmedian(stack, axis=0).astype(np.float32)
        return self._fused

    def save_fused(
        self, save_dir: str, filename: str = "fused_ortho_running.png"
    ) -> Optional[str]:
        if self._fused is None:
            return None
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        save_depth_map_as_image(self._fused, path)
        logging.info(f"Saved fused ortho depth to {path}")
        return path


class WorldOrthoMedianFuser:
    """
    世界座標(XY)に固定したグリッドへ各ビューのワールド点群をラスタ化し、
    各セルの深さ(Z)を逐次的に中央値融合する。
    """

    def __init__(
        self,
        x_min: Optional[float],
        x_max: Optional[float],
        y_min: Optional[float],
        y_max: Optional[float],
        pixel_size: float,
        plane_axes=(0, 2),
        depth_axis=1,
        depth_invert=False,
    ):
        # 平面に使う世界座標軸（デフォルト: (X,Z)）、深度に使う軸（デフォルト: Y）
        self.axis_u = int(plane_axes[0])  # 横軸
        self.axis_v = int(plane_axes[1])  # 縦軸
        self.axis_d = int(depth_axis)  # 深度軸
        self.depth_invert = bool(depth_invert)
        self.u_min = float(x_min) if x_min is not None else None
        self.v_min = float(y_min) if y_min is not None else None
        self.pixel_size = float(pixel_size)
        if x_min is None or x_max is None or y_min is None or y_max is None:
            self.w = 0
            self.h = 0
            self.initialized = False
        else:
            self.w = int(np.ceil((x_max - x_min) / pixel_size))
            self.h = int(np.ceil((y_max - y_min) / pixel_size))
            self.initialized = True
        self._layers: List[np.ndarray] = []
        self._fused: Optional[np.ndarray] = None

    def _ensure_initialized(self, u: np.ndarray, v: np.ndarray):
        if self.initialized:
            return
        if u.size == 0:
            # fallback default box
            self.u_min = -50.0
            self.v_min = -50.0
            self.w = int(np.ceil(100.0 / self.pixel_size))
            self.h = int(np.ceil(100.0 / self.pixel_size))
            self.initialized = True
            return
        umin, umax = float(np.nanmin(u)), float(np.nanmax(u))
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        # margin 10%
        du = max(umax - umin, 1e-3)
        dv = max(vmax - vmin, 1e-3)
        umin -= 0.1 * du
        umax += 0.1 * du
        vmin -= 0.1 * dv
        vmax += 0.1 * dv
        self.u_min = umin
        self.v_min = vmin
        self.w = int(np.ceil((umax - umin) / self.pixel_size))
        self.h = int(np.ceil((vmax - vmin) / self.pixel_size))
        self.initialized = True

    def _rasterize_min(self, world_points: np.ndarray) -> np.ndarray:
        if world_points is None or world_points.size == 0:
            if self.initialized:
                return np.full((self.h, self.w), np.nan, dtype=np.float32)
            return np.full((1, 1), np.nan, dtype=np.float32)
        u = world_points[:, self.axis_u]
        v = world_points[:, self.axis_v]
        d = world_points[:, self.axis_d]
        if self.depth_invert:
            d = -d
        self._ensure_initialized(u, v)
        grid = np.full((self.h, self.w), np.nan, dtype=np.float32)
        gx = np.floor((u - self.u_min) / self.pixel_size).astype(np.int32)
        gy = np.floor((v - self.v_min) / self.pixel_size).astype(np.int32)
        valid = (gx >= 0) & (gx < self.w) & (gy >= 0) & (gy < self.h) & np.isfinite(d)
        gx = gx[valid]
        gy = gy[valid]
        dv = d[valid].astype(np.float32)
        for x, y, z in zip(gx, gy, dv):
            if np.isnan(grid[y, x]) or z < grid[y, x]:
                grid[y, x] = z
        return grid

    def add_world_points(self, world_points: np.ndarray) -> np.ndarray:
        layer = self._rasterize_min(world_points)
        self._layers.append(layer)
        stack = np.stack(self._layers, axis=0)
        self._fused = np.nanmedian(stack, axis=0).astype(np.float32)
        return self._fused

    def get_fused_depth(self) -> Optional[np.ndarray]:
        return self._fused

    def save_fused_depth(
        self,
        save_path: str,
        *,
        swap_axes: bool = False,
        flip_x: bool = True,
        flip_y: bool = False,
    ) -> Optional[str]:
        fused = self.get_fused_depth()
        if fused is None:
            logging.warning("No fused world-ortho depth to save yet.")
            return None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img = fused.copy()
        if swap_axes:
            img = img.T
        if flip_x:
            img = np.fliplr(img)
        if flip_y:
            img = np.flipud(img)
        save_depth_map_as_image(img, save_path)
        logging.info(f"Saved fused world-orthographic depth to {save_path}")
        return save_path
