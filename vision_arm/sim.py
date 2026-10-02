"""合成相机：把电机角度实时渲染成画面，用来在没有硬件时跑通整条链路。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .motor import MotorBackend, SimMotor
from .vision import Detection, Scene


@dataclass
class SyntheticCamera:
    scene: Scene
    motor: MotorBackend
    brightness: float = 1.0
    ambient_hue: int = 0
    save_dir: str | None = None
    frames: dict[float, np.ndarray] = field(default_factory=dict)
    cam_deg: float = 0.0          # 最近一帧曝光时刻的电机角度

    def sync(self, motor_deg: float) -> None:
        """仿真下相机与电机刚性同轴，无需做任何事（保留接口以便换真实相机）。"""
        self.cam_deg = float(motor_deg)

    def grab(self, dt: float = 1 / 30) -> np.ndarray:
        self.motor.pump(dt)
        angle = round(float(self.motor.read_pos()), 3)
        self.cam_deg = angle
        key = round(angle, 2)
        frame = self.frames.get(key)
        if frame is None:
            frame = self.scene.render(angle, brightness=self.brightness,
                                      ambient_hue=self.ambient_hue)
            self.frames[key] = frame
        return frame.copy()

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# 可视化
# --------------------------------------------------------------------------
def draw_detections(frame: np.ndarray, dets: list[Detection],
                    cam_deg: float | None = None) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 220, 0), 1)
    cv2.line(vis, (int(w * 0.47), 0), (int(w * 0.47), h), (0, 140, 0), 1)
    cv2.line(vis, (int(w * 0.53), 0), (int(w * 0.53), h), (0, 140, 0), 1)
    for d in dets:
        x, y, bw, bh = d.bbox
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
        cv2.circle(vis, (int(d.px_center[0]), int(d.px_center[1])), 4, (0, 0, 255), -1)
        cv2.putText(vis, f"{d.name} {d.score:.2f}", (x, max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    label = f"cam={cam_deg:.1f} deg" if cam_deg is not None else ""
    cv2.putText(vis, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    return vis


def hstack(images: list[np.ndarray], target_h: int = 360) -> np.ndarray:
    out = []
    for img in images:
        scale = target_h / img.shape[0]
        out.append(cv2.resize(img, (int(img.shape[1] * scale), target_h)))
    return np.hstack(out)


def pack_motion_trace(images: list[np.ndarray], labels: list[str],
                      target_h: int = 300) -> np.ndarray:
    """把伺服过程中的关键帧拼成一张长图，方便直接看结果。"""
    vis = []
    for img, text in zip(images, labels):
        scale = target_h / img.shape[0]
        small = cv2.resize(img, (int(img.shape[1] * scale), target_h))
        cv2.rectangle(small, (0, 0), (small.shape[1], 26), (255, 255, 255), -1)
        cv2.putText(small, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        vis.append(small)
    return hstack(vis, target_h)
