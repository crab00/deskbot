"""USB 摄像头采集（OpenCV）。"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.logging_setup import get_logger

log = get_logger("vision.camera")


class CameraUnavailable(Exception):
    pass


class Camera:
    def __init__(self, index: int = 0):
        self.index = index
        self._cap = None

    def _open(self):
        if self._cap is None:
            try:
                import cv2
            except Exception as e:
                raise CameraUnavailable(f"未安装 opencv: {e}")
            self._cap = cv2.VideoCapture(self.index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self._cap

    def capture(self) -> Optional[np.ndarray]:
        """抓一帧 BGR 图像；失败返回 None。"""
        cap = self._open()
        if not cap.isOpened():
            raise CameraUnavailable("摄像头打不开")
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
