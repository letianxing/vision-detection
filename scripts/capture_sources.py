from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CaptureSample:
    color_bgr: np.ndarray
    depth_m: np.ndarray | None = None
    depth_source: str = "none"


class OpenCvCaptureSource:
    def __init__(self, index: int, width: int, height: int):
        self.capture = cv2.VideoCapture(int(index))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read(self) -> tuple[bool, CaptureSample | None]:
        ok, frame = self.capture.read()
        return ok, CaptureSample(frame) if ok and frame is not None else None

    def release(self) -> None:
        self.capture.release()


class RealSenseCaptureSource:
    def __init__(self, serial: str, width: int, height: int, fps: int = 30):
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise RuntimeError(f"pyrealsense2 is required for RealSense capture: {exc}") from exc
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))
        profile = self.pipeline.start(config)
        sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(sensor.get_depth_scale())
        self.align = rs.align(rs.stream.color)
        self.opened = True

    def is_opened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, CaptureSample | None]:
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=1000))
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return False, None
        color_bgr = np.asanyarray(color.get_data()).copy()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return True, CaptureSample(color_bgr=color_bgr, depth_m=depth_m, depth_source="realsense")

    def release(self) -> None:
        if self.opened:
            self.pipeline.stop()
            self.opened = False


def create_capture_source(source_type: str, source_id: str, width: int, height: int):
    if source_type == "realsense":
        return RealSenseCaptureSource(source_id, width, height)
    return OpenCvCaptureSource(int(source_id), width, height)


def list_realsense_sources() -> list[dict[str, Any]]:
    try:
        import pyrealsense2 as rs
    except Exception:
        return []
    devices = []
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        devices.append(
            {
                "index": serial,
                "name": name,
                "source_type": "realsense",
                "available": True,
                "depth": True,
            }
        )
    return devices
