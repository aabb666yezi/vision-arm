"""视觉伺服：让电机转到"目标物品位于画面正中"的角度。

两个能力：
  * scan()  全行程扫描定位。相机视野有限，物品可能一开始根本不在画面里，
            所以先左右扫一遍，把每件已登记物品的绝对方位角逐个记下来。
  * move()  闭环对位。读到物品 -> 算画面偏差 -> 换算成角度增量 -> 下发 ->
            等稳定 -> 再看一眼，直到偏差进容差带并连续 N 帧稳定。

角度约定（一定要实测确认，装反了就改 move_sign）：
    move_sign = +1 ：电机角度增大 -> 视野向右移 -> 物品在画面里向左移
    move_sign = -1 ：反过来（相机或电机反向安装时用）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .motor import MotorBackend
from .vision import Detection, Detector


# --------------------------------------------------------------------------
# 相机抽象（真实相机与合成相机同一接口）
# --------------------------------------------------------------------------
class Camera(Protocol):  # pragma: no cover
    """相机接口。关键点：cam_deg 必须是"这一帧曝光时刻"的电机角度。

    真实相机实现里，在 grab() 里读一次电机位置并缓存下来给 cam_deg 用；
    如果直接拿指令角度代替实际角度，机械回差会变成 1~3° 的系统误差。
    """

    cam_deg: float

    def grab(self, dt: float = 1 / 30) -> np.ndarray: ...
    def sync(self, motor_deg: float) -> None: ...
    def close(self) -> None: ...


@dataclass
class ServoConfig:
    hfov_deg: float = 53.0       # 相机水平视场角（务必实测标定，见 README）
    move_sign: int = 1           # 电机转向与画面偏移的符号关系
    gain: float = 0.85           # 比例增益，震荡就调小
    max_step_deg: float = 6.0    # 单次最大角度增量，防止大幅甩动
    tol_px: float = 22.0         # 像素容差（画面宽 640 时约 ±3.4%）
    stable_frames: int = 3       # 连续多少帧进容差才判定到位
    settle_s: float = 0.45       # 每次下发指令后的最短等待时间
    max_settle_s: float = 4.0    # 最长等待（等电机到位），防止无限阻塞
    max_steps: int = 40          # 最大迭代步数，兜底防死循环
    ema_alpha: float = 0.55      # 像素偏差的指数平滑系数，抗抖动
    speed_dps: float = 30.0      # 电机速度
    lost_limit: int = 12         # 连续丢失目标多少帧就放弃
    search_step_deg: float = 8.0  # 丢失后每轮扩大多少度做搜索（12 帧 ≈ ±48°）

    # settle_s 必须大于 "max_step_deg / speed_dps + 机械响应时间"，
    # 否则下一帧是在电机还没停稳时拍的，误差读数虚高、收敛会明显变慢。


@dataclass
class MotionResult:
    ok: bool
    name: str
    angle_deg: float
    steps: int
    message: str
    trace: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# 伺服控制器
# --------------------------------------------------------------------------
class Servo:
    def __init__(self, detector: Detector, motor: MotorBackend,
                 cfg: ServoConfig | None = None, log=None) -> None:
        self.detector = detector
        self.motor = motor
        self.cfg = cfg or ServoConfig()
        self.log = log or (lambda *a, **k: None)

    # -- 单帧识别 --
    def look(self, frame: np.ndarray, cam_deg: float,
             names: list[str] | None = None) -> list[Detection]:
        return self.detector.detect(
            frame, cam_deg=cam_deg,
            hfov_deg=self.cfg.hfov_deg, move_sign=self.cfg.move_sign, names=names,
        )

    # ------------------------------------------------------------------
    # 闭环对位
    # ------------------------------------------------------------------
    def move(self, camera: Camera, name: str,
             hint_deg: float | None = None) -> MotionResult:
        cfg = self.cfg
        trace: list[dict] = []

        # 有扫描结果就直接先转到大概位置，省时间也提高识别率
        if hint_deg is not None:
            self._goto(camera, hint_deg)
        ema = None
        stable = 0
        lost = 0
        cam_deg = self.motor.read_pos()
        # 搜索基准点：丢失恢复时围绕"最后看到目标的位置"来回扫。
        # 必须用固定基准点算偏移，不能在当前位置上累加 ——
        # 否则每次 nudge 都会把基准点一起推走，变成单向发散而不是来回搜索。
        anchor_deg = cam_deg

        for step in range(1, cfg.max_steps + 1):
            frame = camera.grab()
            cam_deg = float(getattr(camera, "cam_deg", self.motor.read_pos()))
            dets = self.look(frame, cam_deg, names=[name])
            det = dets[0] if dets else None
            if det is None:
                lost += 1
                trace.append({"step": step, "cam": round(cam_deg, 2), "found": None})
                if lost >= cfg.lost_limit:
                    self.log(f"  第{step}步：连续 {lost} 帧没看到「{name}」，放弃")
                    return MotionResult(False, name, cam_deg, step,
                                        f"连续 {lost} 帧未识别到「{name}」", trace)
                # 丢失恢复：不要只原地微转（那样最多找回 ±12°），
                # 而是围绕最后已知位置做逐步扩大的来回搜索：
                # 偏移序列 +1,-1,+2,-2,... × search_step_deg，
                # 这样即使初值偏差 30° 以上也还能捞回来。
                nudge = cfg.search_step_deg * ((lost + 1) // 2)
                direction = cfg.move_sign * (1 if lost % 2 else -1)
                probe = anchor_deg + direction * nudge
                self.log(f"  第{step}步：没看到「{name}」，搜索到 {probe:+.1f}°"
                         f"（第 {lost} 次）")
                self._goto(camera, probe)
                continue
            lost = 0
            anchor_deg = cam_deg

            h, w = frame.shape[:2]
            cx = det.px_center[0]
            err_px = cx - w * 0.5
            ema = err_px if ema is None else cfg.ema_alpha * err_px + (1 - cfg.ema_alpha) * ema
            trace.append({"step": step, "cam": round(cam_deg, 2),
                          "found": det.name, "score": round(det.score, 3),
                          "px_err": round(err_px, 1), "ema": round(ema, 1)})
            self.log(f"  第{step}步：{det.name} score={det.score:.2f} "
                     f"像素偏差={err_px:+.0f} 平滑后={ema:+.0f} 当前角度={cam_deg:.2f}°")

            if abs(ema) <= cfg.tol_px:
                stable += 1
                if stable >= cfg.stable_frames:
                    self.log(f"  ✔ 到位：{name} 已居中，电机 {cam_deg:.2f}°（{step} 步）")
                    return MotionResult(True, name, cam_deg, step, "已居中到位", trace)
                time.sleep(0.05)
                continue
            stable = 0

            # 像素偏差 -> 角度增量
            deg_per_px = cfg.hfov_deg / w
            delta = cfg.gain * ema * deg_per_px * cfg.move_sign
            delta = float(np.clip(delta, -cfg.max_step_deg, cfg.max_step_deg))
            target = self.motor.clamp(cam_deg + delta)
            if abs(target - cam_deg) < 0.05:
                self.log(f"  ✔ 收敛：角度变化小于 0.05°，电机 {cam_deg:.2f}°")
                return MotionResult(True, name, cam_deg, step, "已收敛（最小步进）", trace)
            self._goto(camera, target)

        return MotionResult(False, name, cam_deg, cfg.max_steps,
                            f"超过最大迭代次数 {cfg.max_steps} 仍未居中", trace)

    def _goto(self, camera: Camera, deg: float) -> None:
        cfg = self.cfg
        self.motor.move_to(deg, cfg.speed_dps)
        self._advance(camera, cfg.settle_s)

    def _advance(self, camera: Camera, seconds: float | None = None) -> None:
        """等到电机真的走到目标位置，再让相机取下一帧。

        不是"固定等 settle_s"：固定等待时间没有物理意义（转 5° 和转 60° 差好几倍），
        等不够就会出现"请求 35° 只走到 23.4°"，表现为对准永远差一点，
        很容易被误判成增益调不对。seconds 只作为最短等待时间。
        """
        cfg = self.cfg
        min_wait = cfg.settle_s if seconds is None else seconds
        self.motor.wait_for(min_wait, tick=camera.grab, timeout_s=cfg.max_settle_s)

    # ------------------------------------------------------------------
    # 全行程扫描
    # ------------------------------------------------------------------
    def scan(self, camera: Camera, *, lo: float = -90.0, hi: float = 90.0,
             step_deg: float = 8.0, dwell_s: float = 0.22,
             names: list[str] | None = None) -> dict[str, float]:
        """左右扫一遍，返回 {物品名: 绝对方位角}。"""
        found: dict[str, list[float]] = {}
        angles = list(np.arange(lo, hi + 1e-6, step_deg))
        self.log(f"扫描 {lo:.0f}° ~ {hi:.0f}°，步长 {step_deg:.0f}°，共 {len(angles)} 个采样点")
        for a in angles:
            self._goto(camera, float(a))
            frame = camera.grab()
            cam_deg = float(getattr(camera, "cam_deg", self.motor.read_pos()))
            for det in self.look(frame, cam_deg, names=names):
                # 让该物品居中的绝对角度
                est = cam_deg + self.cfg.move_sign * det.px_err_norm * (self.cfg.hfov_deg * 0.5)
                # 只看画面中央区域的结果，边缘畸变大、误差大
                if abs(det.px_err_norm) < 0.75:
                    found.setdefault(det.name, []).append(float(est))
            if found:
                self.log(f"  相机 {cam_deg:6.2f}° -> 已看到 {sorted(found)}")
        result = {k: float(np.median(v)) for k, v in found.items()}
        self.log(f"扫描完成，定位到 {len(result)} 件："
                 + "、".join(f"{k}={v:.1f}°" for k, v in result.items()))
        return result

    def save_map(self, mapping: dict[str, float], path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"angle_map": mapping}, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    def load_map(self, path: str | Path) -> dict[str, float]:
        p = Path(path)
        if not p.exists():
            return {}
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {k: float(v) for k, v in raw.get("angle_map", {}).items()}
