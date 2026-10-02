"""任务编排：状态机 + 安全域 + 行人模拟。这是演示时"主角"跑的那一层。

设计原则（答辩要讲的）：
    1. 每个状态都必须有超时出口 —— 这是"不卡死"的唯一保证；
    2. 任何异常一律导向"停住并保持"，而不是"做完当前动作"（fail-safe）；
    3. 安全优先级最高：行人进入停止域时，机械臂停在原地并保持姿态稳定，
       等安全域恢复后再继续，而不是急停甩动（对开口容器来说，急停=洒水）。

结构：
    SafetyArbiter   分层安全域仲裁（正常/降速/停止/远离）
    Pedestrian      行人模拟（含位置与速度估计，用于预测）
    Mission         "语音点名 → 定位 → 拾取 → 搬运 → 放置"的完整状态机
    DemoEngine      给网页控制台用的实时引擎（后台线程推进模拟时间）
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .balance import (
    RiskProfile, SloshModel, Trajectory, classify_target, plan_transport,
    simulate_trajectory,
)
from .motor import SimMotor
from .registry import Registry
from .servo import Servo, ServoConfig
from .sim import SyntheticCamera, draw_detections
from .vision import Detector, Scene

# --------------------------------------------------------------------------
# 安全域阈值（米）—— 答辩时会被问到，背下来
# --------------------------------------------------------------------------
SAFE_NORMAL = 0.80     # 大于此距离：正常速度
SAFE_SLOW = 0.50       # 0.5~0.8：降速到 30%
SAFE_STOP = 0.30       # 0.3~0.5：停止前进，原地保持姿态
SAFE_RETREAT = 0.20    # 小于此距离：缓速远离


@dataclass
class SafetyDecision:
    level: str            # NORMAL / SLOW / STOP / RETREAT
    speed_scale: float
    message: str

    @property
    def blocked(self) -> bool:
        return self.level in ("STOP", "RETREAT")


class SafetyArbiter:
    """把"人到机械臂的距离"翻译成"当前允许的动作"。

    比"重新规划路径"更可靠：宿舍空间狭小，路径重规划来不及，也验证不了。
    分层限速是工业协作机器人（ISO/TS 15066 的速度与分离监控）的成熟做法。
    """

    def __init__(self, normal: float = SAFE_NORMAL, slow: float = SAFE_SLOW,
                 stop: float = SAFE_STOP, retreat: float = SAFE_RETREAT) -> None:
        self.normal = normal
        self.slow = slow
        self.stop = stop
        self.retreat = retreat
        self.history: list[str] = []

    def evaluate(self, distance_m: float) -> SafetyDecision:
        if distance_m >= self.normal:
            d = SafetyDecision("NORMAL", 1.0, "环境安全，正常搬运")
        elif distance_m >= self.slow:
            d = SafetyDecision("SLOW", 0.30, f"有人靠近（{distance_m:.2f}m），速度降至 30%")
        elif distance_m >= self.stop:
            d = SafetyDecision("STOP", 0.0,
                               f"行人进入停止域（{distance_m:.2f}m），原地保持姿态")
        else:
            d = SafetyDecision("RETREAT", 0.15,
                               f"距离过近（{distance_m:.2f}m），缓速远离")
        if not self.history or self.history[-1] != d.level:
            self.history.append(d.level)
        return d


# --------------------------------------------------------------------------
# 行人模拟
# --------------------------------------------------------------------------
@dataclass
class Pedestrian:
    """一个会走动的室友：走近 → 在停止域停一会 → 退回安全距离。

    为什么要"退回"（这是演示设计的关键）：
        如果行人一直站在停止域里，机械臂会永远停在暂停态，任务卡死；
        演示就只能看到"停住"，看不到"安全之后继续完成"。
        真实使用中室友也总会走开，所以完整的行为是"暂停 → 恢复 → 送达"。
    """

    start_m: float = 1.3        # 起点：约 1.6s 走进停止域，正好压在搬运过程中
    target_m: float = 0.25      # 走到停止域（0.3~0.5m）甚至远离域，触发避让
    speed_mps: float = 0.5      # 按真实时间走（见 Mission.tick），与引擎加速无关
    dwell_s: float = 5.0        # 在最近处停留多久（够演示"暂停"）
    pos_m: float = 1.3
    vel_mps: float = 0.0
    estimated_m: float = 1.3
    phase: str = "approach"
    _t_dwell: float = 0.0
    _noise: float = 0.0

    def reset(self) -> None:
        self.pos_m = self.start_m
        self.vel_mps = 0.0
        self.estimated_m = self.start_m
        self.phase = "approach"
        self._t_dwell = 0.0

    def tick(self, dt: float) -> None:
        if self.phase == "approach":
            if self.pos_m <= self.target_m:
                self.phase = "dwell"
                self.vel_mps = 0.0
            else:
                self.vel_mps = -self.speed_mps
                self.pos_m = max(self.target_m, self.pos_m + self.vel_mps * dt)
        elif self.phase == "dwell":
            self._t_dwell += dt
            self.vel_mps = 0.0
            if self._t_dwell >= self.dwell_s:
                self.phase = "retreat"
        else:                                    # retreat
            self.vel_mps = self.speed_mps
            self.pos_m = min(self.start_m, self.pos_m + self.vel_mps * dt)
            if self.pos_m >= self.start_m - 1e-6:
                self.phase = "away"
                self.vel_mps = 0.0
        # 模拟跟踪器的观测噪声（确定性，保证演示可复现）
        self._noise += dt * 7.0
        self.estimated_m = max(0.0, self.pos_m + 0.012 * math.sin(self._noise))

    def predicted_in(self, seconds: float) -> float:
        """按当前速度预测多少秒后的距离（用于提前减速，而不是撞上才停）。"""
        return max(0.0, self.estimated_m + self.vel_mps * seconds)


# --------------------------------------------------------------------------
# 语音/文本指令解析
# --------------------------------------------------------------------------
VOICE_ACTIONS = ("拿", "取", "帮我", "递", "要", "来", "带")


def parse_command(text: str, known: list[str]) -> tuple[str | None, str]:
    """把一句自然语言解析成 (物品名, 说明)。不需要大模型，槽位模板足够。"""
    t = (text or "").strip()
    if not t:
        return None, "没听清，请再说一遍"
    for name in sorted(known, key=len, reverse=True):     # 长名字优先，避免子串误判
        if name in t:
            return name, f"识别到指令：取「{name}」"
    if any(k in t for k in VOICE_ACTIONS):
        return None, f"听懂了动作，但没说清物品。已登记：{'、'.join(known)}"
    return None, f"没听懂「{t}」。试试说：帮我拿水杯"


# --------------------------------------------------------------------------
# 任务状态机
# --------------------------------------------------------------------------
PHASES = ("IDLE", "LISTEN", "SCAN", "APPROACH", "LIFT", "TRANSFER", "PLACE", "DONE", "FAILED")

PHASE_LABEL = {
    "IDLE": "待命", "LISTEN": "听指令", "SCAN": "定位物品", "APPROACH": "对准物品",
    "LIFT": "拾取（平衡约束）", "TRANSFER": "搬运上楼", "PLACE": "放置到床面",
    "DONE": "完成", "FAILED": "失败/中止",
}


@dataclass
class MissionState:
    phase: str = "IDLE"
    t_phase: float = 0.0
    item: str = ""
    detail: str = "等待语音指令"
    progress: float = 0.0
    motor_deg: float = 0.0
    motor_target: float = 0.0
    risk: str = "-"
    peak_tilt_deg: float = 0.0
    tilt_now_deg: float = 0.0
    planned_duration_s: float = 0.0
    peak_acc_dps2: float = 0.0
    safety: SafetyDecision = field(default_factory=lambda: SafetyDecision("NORMAL", 1.0, ""))
    person_m: float = 2.6
    person_vel: float = 0.0
    paused: bool = False
    paused_count: int = 0
    detections: list = field(default_factory=list)
    log: list[str] = field(default_factory=list)


@dataclass
class StepResult:
    """状态机一次 tick 的产物：一张画面 + 一份状态快照。"""

    frame: np.ndarray
    state: MissionState


class Mission:
    """完整取件任务的状态机。

    用法（同步，方便命令行测试）：
        m = Mission(scene, registry, detector)
        m.start("水杯")
        for r in m.run():          # 每次 tick 产出一帧
            ...
    用法（异步，给网页用）：m.start(...); m.tick(dt)
    """

    def __init__(self, scene: Scene, registry: Registry, detector: Detector,
                 motor: SimMotor | None = None, servo: Servo | None = None,
                 time_scale: float = 1.0, seed: int = 3) -> None:
        self.scene = scene
        self.registry = registry
        self.detector = detector
        self.motor = motor or SimMotor()
        self.camera = SyntheticCamera(scene=scene, motor=self.motor)
        self.servo = servo or Servo(
            detector, self.motor,
            ServoConfig(hfov_deg=scene.hfov_deg, settle_s=0.45, max_steps=60))
        self.safety = SafetyArbiter()
        self.person = Pedestrian()
        self.slosh = SloshModel()
        self.time_scale = time_scale
        self.state = MissionState()
        self._rng = np.random.default_rng(seed)
        self._traj: Trajectory | None = None
        self._traj_t = 0.0
        self._lift_from = 0.0
        self._lift_to = 0.0
        self._frames = 0

        # 演示用的初始摆放：3 件已登记物品 + 1 件故意不登记
        self.placements = {it.name: it.angle_deg for it in scene.items}
        self.motor._angle = self.motor._target = 0.0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = MissionState()
        self.person.reset()
        self.motor._angle = self.motor._target = 0.0
        self._traj = None
        self._traj_t = 0.0

    def start(self, item: str) -> None:
        self.reset()
        self.state.item = item
        self.state.phase = "LISTEN"
        self._say(f"收到指令：取「{item}」")

    def tick(self, dt: float) -> StepResult:
        """推进 dt 秒真实时间，返回一帧画面与状态快照。"""
        dt = max(0.0, min(0.2, float(dt))) * self.time_scale
        st = self.state
        st.t_phase += dt

        self.person.tick(dt / max(1e-6, self.time_scale))   # 行人按真实时间走
        st.person_m = self.person.estimated_m
        st.person_vel = self.person.vel_mps
        st.safety = self.safety.evaluate(self.person.estimated_m)

        handler = getattr(self, f"_phase_{st.phase.lower()}", None)
        if handler is not None:
            handler(dt)

        st.motor_deg = float(self.motor.read_pos())
        st.motor_target = float(self.motor._target)

        # 保持液面姿态：播放轨迹时按当前指令加速度估算倾角（给界面显示用）
        if self._traj is not None:
            idx = min(len(self._traj.acc) - 1, int(self._traj_t / 0.002))
            a = float(self._traj.acc[idx]) * math.pi / 180.0
            st.tilt_now_deg = math.degrees(-a * 0.5 / 9.81)

        frame = self.camera.grab(0.0)
        st.detections = self.detector.detect(
            frame, cam_deg=self.camera.cam_deg, hfov_deg=self.scene.hfov_deg,
            names=[st.item] if st.item else None)
        self._frames += 1
        return StepResult(frame=frame, state=st)

    def annotated_frame(self, res: StepResult | None = None) -> np.ndarray:
        """画好识别框的画面，网页直接显示这一张。"""
        if res is None:
            frame = self.camera.grab(0.0)
            dets = self.detector.detect(frame, cam_deg=self.camera.cam_deg,
                                        hfov_deg=self.scene.hfov_deg,
                                        names=[self.state.item] if self.state.item else None)
        else:
            frame, dets = res.frame, res.state.detections
        return draw_detections(frame, dets, self.camera.cam_deg)

    def run(self, dt: float = 1 / 30):
        """同步迭代器版本（命令行测试用），直到任务结束。"""
        guard = 0
        while self.state.phase not in ("DONE", "FAILED") and guard < 20000:
            guard += 1
            yield self.tick(dt)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        stamp = f"[{self.state.phase}]"
        self.state.log.append(f"{stamp} {msg}")
        if len(self.state.log) > 200:
            del self.state.log[:-200]

    def _goto_sync(self, deg: float, seconds: float) -> None:
        """等电机真的转到位（仿真里靠 pump 推进时间）。"""
        self.motor.move_to(deg, self.servo.cfg.speed_dps)
        self.motor.wait_for(seconds, timeout_s=self.servo.cfg.max_settle_s)

    def _enter(self, phase: str, detail: str, progress: float = 0.0) -> None:
        self.state.phase = phase
        self.state.t_phase = 0.0
        self.state.detail = detail
        self.state.progress = progress
        self._say(detail)

    # ------------------------------------------------------------------
    # 各状态
    # ------------------------------------------------------------------
    def _phase_listen(self, dt: float) -> None:
        """校验物品是否已登记，没登记就明确拒绝（演示健壮性的关键一幕）。"""
        if self.state.t_phase < 0.3:
            return
        item = self.state.item
        if item not in self.registry.names():
            self._enter("FAILED", f"「{item}」不在已登记清单里，拒绝执行（这是安全设计）")
            return
        risk = classify_target(item)
        self.state.risk = risk.code
        self._say(f"风险分级：{risk.code} 级（{risk.label}），倾角上限 {risk.max_tilt_deg}°")
        self._enter("SCAN", "开始左右扫描，定位物品方位", 0.1)

    def _phase_scan(self, dt: float) -> None:
        """扫描定位：从当前位置左右扫，找到目标物品的方位角。"""
        item = self.state.item
        if self.state.t_phase < 0.3:
            return
        found = None
        for a in (-60, -40, -20, 0, 20, 40, 60):
            self._goto_sync(a, 0.15)
            frame = self.camera.grab(0.0)
            dets = self.detector.detect(frame, cam_deg=self.camera.cam_deg,
                                        hfov_deg=self.scene.hfov_deg, names=[item])
            if dets:
                found = dets[0]
                self.state.progress = 0.45
                self._say(f"扫描到「{item}」：相机 {self.camera.cam_deg:+.1f}°，"
                          f"score={found.score:.2f}")
                break
            self._say(f"扫描 {self.camera.cam_deg:+5.1f}°，暂未发现「{item}」")
        if found is None:
            self._enter("FAILED", f"扫描一圈没找到「{item}」，请把它放到视野范围内")
            return
        self.state.peak_acc_dps2 = 0.0
        self._enter("APPROACH", f"闭环对准「{item}」", 0.55)

    def _phase_approach(self, dt: float) -> None:
        """视觉伺服闭环：转到物品居中。这里用同步快速版，模拟里一步到位。"""
        item = self.state.item
        for _ in range(30):
            frame = self.camera.grab(0.0)
            dets = self.detector.detect(frame, cam_deg=self.camera.cam_deg,
                                        hfov_deg=self.scene.hfov_deg, names=[item])
            if not dets:
                self._enter("FAILED", f"对准过程中丢失「{item}」")
                return
            d = dets[0]
            w = frame.shape[1]
            err = d.px_center[0] - w * 0.5
            if abs(err) <= 22.0:
                self.state.progress = 0.75
                self._say(f"已对准：相机 {self.camera.cam_deg:+.2f}°，像素偏差 {err:+.0f}")
                break
            delta = float(np.clip(0.85 * err * (self.scene.hfov_deg / w), -6.0, 6.0))
            self._goto_sync(self.motor.read_pos() + delta, 0.45)
        else:
            self._enter("FAILED", f"对准「{item}」超时未收敛")
            return

        # 计算搬运轨迹：查离线表 -> 生成 S 曲线 -> 输入整形
        risk = classify_target(item)
        self._lift_from = float(self.motor.read_pos())
        self._lift_to = float(self.placements.get(item, 0.0))
        delta = self._lift_to - self._lift_from
        traj = plan_transport(risk, delta, self.slosh, shaper=True)
        res = simulate_trajectory(traj, self.slosh)
        self._traj = traj
        self._traj_t = 0.0
        self.state.planned_duration_s = traj.duration_s
        self.state.peak_acc_dps2 = traj.peak_acc_dps2
        self.state.peak_tilt_deg = res.max_tilt_deg
        self._say(f"平衡规划完成：转角 {abs(delta):.1f}°（{self._lift_from:+.1f}° → "
                  f"{self._lift_to:+.1f}°），峰值加速度 "
                  f"{traj.peak_acc_dps2:.1f}°/s²，预计液面峰值 {res.max_tilt_deg:.2f}°，"
                  f"用时 {traj.duration_s:.1f}s")
        self._enter("LIFT", "按平衡轨迹拾取并抬升", 0.8)

    def _phase_lift(self, dt: float) -> None:
        """按规划好的轨迹播放，同时受安全域约束（这是演示的高潮段落）。"""
        traj = self._traj
        if traj is None:
            self._enter("FAILED", "轨迹丢失")
            return

        if self.state.safety.blocked:
            self.state.paused = True
            if self.state.t_phase % 0.5 < dt:
                self._say(f"{self.state.safety.message} —— 已暂停在原位，保持液面稳定")
            return
        if self.state.paused:
            self.state.paused = False
            self.state.paused_count += 1
            self._say("安全域恢复，继续搬运")

        scale = max(0.15, self.state.safety.speed_scale)
        self._traj_t += dt * scale
        idx = min(len(traj.t) - 1, int(self._traj_t / 0.002))
        self.motor._angle = self._lift_from + float(traj.angle[idx])
        self.motor._target = self._lift_from + float(traj.angle[-1])
        self.state.progress = 0.8 + 0.15 * min(1.0, idx / max(1, len(traj.t) - 1))

        if idx >= len(traj.t) - 2:
            self._goto_sync(self._lift_from + float(traj.angle[-1]), 0.15)
            self._enter("TRANSFER", "已抬升至目标高度，正在转移到床面上方", 0.95)

    def _phase_transfer(self, dt: float) -> None:
        if self.state.t_phase < 0.6:
            return
        self._enter("PLACE", "放置到床面", 0.98)

    def _phase_place(self, dt: float) -> None:
        if self.state.t_phase < 0.6:
            return
        self._enter("DONE",
                    f"完成：{self.state.item} 已送达，液面峰值 "
                    f"{self.state.peak_tilt_deg:.2f}°（阈值 "
                    f"{classify_target(self.state.item).max_tilt_deg}°）", 1.0)

    def _phase_done(self, dt: float) -> None:
        pass

    def _phase_failed(self, dt: float) -> None:
        pass


# --------------------------------------------------------------------------
# 网页控制台用的实时引擎
# --------------------------------------------------------------------------
class DemoEngine:
    """在后台线程里按真实时间推进 Mission，把最新画面和状态共享给 HTTP 服务。"""

    def __init__(self, data_dir, base_dir, hz: float = 12.0,
                 time_scale: float = 1.5) -> None:
        self.registry = Registry(data_dir / "items.json")
        self.scene = Scene()
        self.detector = Detector(self.registry, base_dir=base_dir)
        self.mission = Mission(self.scene, self.registry, self.detector,
                               time_scale=time_scale)
        self.hz = hz
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._state: dict = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._last = time.monotonic()

    # -- 生命周期 --
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        import cv2
        period = 1.0 / self.hz
        while self._running:
            now = time.monotonic()
            dt = now - self._last
            self._last = now
            if dt <= 0:
                dt = period
            with self._lock:
                res = self.mission.tick(min(dt, 0.25))
                frame = self.mission.annotated_frame(res)
                ok, buf = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                if ok:
                    self._jpeg = buf.tobytes()
                self._state = self._snapshot(res.state)
            time.sleep(max(0.0, period - (time.monotonic() - now)))

    # -- 对外读取（HTTP 线程调用）--
    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def frame_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg or b""

    # -- 控制 --
    def command(self, text: str) -> dict:
        name, msg = parse_command(text, self.registry.names())
        with self._lock:
            if name is None:
                self.mission.state.log.append(f"[语音] {msg}")
                return {"ok": False, "message": msg}
            self.mission.start(name)
        return {"ok": True, "message": msg, "item": name}

    def reset(self) -> None:
        with self._lock:
            self.mission.reset()

    def set_safety(self, normal: float, slow: float, stop: float) -> None:
        with self._lock:
            self.mission.safety.normal = normal
            self.mission.safety.slow = slow
            self.mission.safety.stop = stop
            self.mission.safety.history.clear()

    def _snapshot(self, st: MissionState) -> dict:
        return {
            "phase": st.phase,
            "phase_label": PHASE_LABEL.get(st.phase, st.phase),
            "detail": st.detail,
            "progress": round(st.progress, 3),
            "item": st.item,
            "risk": st.risk,
            "motor_deg": round(st.motor_deg, 2),
            "motor_target": round(st.motor_target, 2),
            "tilt_now_deg": round(st.tilt_now_deg, 3),
            "peak_tilt_deg": round(st.peak_tilt_deg, 3),
            "peak_acc_dps2": round(st.peak_acc_dps2, 1),
            "planned_duration_s": round(st.planned_duration_s, 2),
            "safety": {
                "level": st.safety.level,
                "speed_scale": st.safety.speed_scale,
                "message": st.safety.message,
            },
            "person_m": round(st.person_m, 3),
            "person_vel": round(st.person_vel, 3),
            "person_phase": self.mission.person.phase,
            "paused": st.paused,
            "paused_count": st.paused_count,
            "detections": [
                {"name": d.name, "score": round(d.score, 3),
                 "x": round(d.px_center[0], 1), "err": round(d.px_err_norm, 3)}
                for d in st.detections
            ],
            "log": st.log[-14:],
            "scene": {
                "items": [{"name": i.name, "angle": i.angle_deg,
                           "registered": i.registered} for i in self.scene.items],
                "hfov": self.scene.hfov_deg,
            },
            "limits": {
                "normal": self.mission.safety.normal,
                "slow": self.mission.safety.slow,
                "stop": self.mission.safety.stop,
            },
        }


def run_headless(items: str = "水杯", verbose: bool = True) -> int:
    """命令行自检：不启网页，直接跑一遍完整任务，打印每个状态。"""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    reg = Registry(root / "data" / "items.json")
    if not len(reg):
        print("台账为空，先运行：python run.py register --synthetic")
        return 2
    scene = Scene()
    det = Detector(reg, base_dir=root)
    m = Mission(scene, reg, det)
    m.start(items)
    last_phase = None
    for _ in m.run(1 / 30):
        if m.state.phase != last_phase:
            last_phase = m.state.phase
            if verbose:
                print(f"  {PHASE_LABEL.get(last_phase, last_phase):<16} "
                      f"{m.state.detail}")
    print(f"\n最终状态：{PHASE_LABEL.get(m.state.phase)}")
    print(f"液面峰值倾角：{m.state.peak_tilt_deg:.2f}°  用时 {m.state.planned_duration_s:.2f}s")
    print(f"安全暂停次数：{m.state.paused_count}")
    return 0 if m.state.phase == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(run_headless())
