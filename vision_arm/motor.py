"""电机抽象层：同一套 ASCII 协议驱动仿真电机与真实串口电机。

协议（每行一条，\n 结尾，全部 ASCII）：
    上位机 -> 下位机
        MOVE A <角度> F <速度>    转动到绝对角度（度），速度单位 度/秒
        HOME A                    回零（角度 0）
        STOP A                    立即停止（保留当前位置）
        STATUS A                  查询状态
        PING                      心跳
    下位机 -> 上位机
        OK MOVE A <角度>
        OK STOP A
        POS A <角度> MOVING <0|1>
        OK PONG
        ERR <原因>

这套协议刻意做得很笨：下位机只管"转到某个角度"，不需要通信往返，
上位机崩了也不会让电机飞车（详见 firmware/arduino_motor_servo.ino）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

DEFAULT_AXIS = "A"


# --------------------------------------------------------------------------
# 协议
# --------------------------------------------------------------------------
def build_move(axis: str, deg: float, speed_dps: float) -> str:
    """生成 MOVE 指令，例如 'MOVE A 12.500 F 30.0'"""
    return f"MOVE {axis} {deg:.3f} F {speed_dps:.2f}"


def build_stop(axis: str) -> str:
    return f"STOP {axis}"


def build_home(axis: str) -> str:
    return f"HOME {axis}"


def build_status(axis: str) -> str:
    return f"STATUS {axis}"


def parse_pos(line: str, axis: str = DEFAULT_AXIS) -> float | None:
    """从 'POS A 12.5 MOVING 1' 里解析出角度；解析失败返回 None。"""
    parts = line.replace("=", " ").split()
    for i, tok in enumerate(parts):
        if tok.upper() == axis.upper() and i + 1 < len(parts):
            try:
                val = float(parts[i + 1])
            except ValueError:
                return None
            # 通讯噪声可能解析出荒唐值，超出物理范围就当作无效
            if abs(val) > 10000.0:
                return None
            return val
    return None


def parse_ok(line: str) -> bool:
    return line.strip().upper().startswith("OK")


# --------------------------------------------------------------------------
# 类型协议
# --------------------------------------------------------------------------
class SerialLike(Protocol):  # pragma: no cover - 仅用于类型提示
    is_open: bool
    timeout: float

    def write(self, data: bytes) -> int: ...
    def readline(self) -> bytes: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------
# 抽象后端
# --------------------------------------------------------------------------
class MotorBackend:
    """所有电机后端的公共接口。角度单位统一为度。"""

    axis: str = DEFAULT_AXIS
    limit_min: float = -180.0
    limit_max: float = 180.0

    def __init__(self, axis: str = DEFAULT_AXIS,
                 limit_min: float = -180.0, limit_max: float = 180.0) -> None:
        self.axis = axis
        self.limit_min = limit_min
        self.limit_max = limit_max

    # --- 必须实现 ---
    def move_to(self, deg: float, speed_dps: float = 30.0) -> None:
        raise NotImplementedError

    def read_pos(self) -> float:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def settle(self, seconds: float) -> None:
        """等待电机稳定。真实后端为阻塞等待，仿真后端由 pump() 推进。"""
        time.sleep(max(0.0, seconds))

    def wait_for(self, seconds: float, tick=None, timeout_s: float = 4.0) -> None:
        """等到电机"真的走到目标"，而不是干等一个固定时间。

        为什么必须这样（这是仿真里踩出来的坑）：
            固定等待时间是没有物理意义的 —— 转 5° 和转 60° 需要的时间差好几倍。
            本工程早期版本固定等 0.45s，结果请求 35° 只走到了 23.4°，
            于是"对准"永远差一点，看起来像增益调不对，实际是时间没给够。
        做法：
            仿真后端用 pump(dt) 推进内部时间，每步检查是否到位；
            真实后端没有 pump（无操作），退化成 sleep(seconds) —— 相当于"给足时间"。
        """
        import time as _time
        deadline = _time.monotonic() + max(seconds, timeout_s)
        dt = 0.02
        while not self.at_target() and _time.monotonic() < deadline:
            self.pump(dt)
            if tick is not None:
                tick(dt)
            else:
                _time.sleep(0)

    def at_target(self, tol: float = 1e-6) -> bool:
        """是否已停在目标位置。真实后端默认返回 True（由下位机负责走到位）。"""
        return True

    # --- 可选 ---
    def home(self) -> None:
        self.move_to(0.0)

    def pump(self, dt: float) -> None:
        """推进内部时间（仿真后端用；真实后端无操作）。"""

    def close(self) -> None:
        pass

    # --- 公共工具 ---
    def clamp(self, deg: float) -> float:
        return max(self.limit_min, min(self.limit_max, float(deg)))

    def __enter__(self) -> "MotorBackend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# 仿真后端
# --------------------------------------------------------------------------
@dataclass
class SimMotor(MotorBackend):
    """仿真电机：模拟匀速转动 + 可配置延迟 + 角度读数噪声。

    注意：仿真里"物体在画面中的位置"必须由使用者随真实角度一起更新，
    因此这里提供 angle 只读属性，合成场景会读它来渲染画面。
    """

    speed_dps: float = 45.0          # 最大角速度
    latency_s: float = 0.12          # 指令到开始运动的延迟（模拟串口+舵机响应）
    noise_deg: float = 0.0           # 位置读数的随机噪声（度）
    _angle: float = 0.0
    _target: float = 0.0
    _pending_delay: float = 0.0
    _last_cmd: str = ""
    _stalled: bool = False
    log: list[str] = field(default_factory=list)

    # -- 只读状态 --
    @property
    def angle(self) -> float:
        return self._angle

    @property
    def target(self) -> float:
        return self._target

    @property
    def moving(self) -> bool:
        return abs(self._target - self._angle) > 1e-6 or self._pending_delay > 0.0

    # -- MotorBackend 接口 --
    def move_to(self, deg: float, speed_dps: float = 30.0) -> None:
        deg = self.clamp(deg)
        self._target = deg
        self.speed_dps = max(1e-3, float(speed_dps))
        self._pending_delay = self.latency_s
        self._last_cmd = build_move(self.axis, deg, speed_dps)
        self.log.append(self._last_cmd)

    def read_pos(self) -> float:
        if self.noise_deg <= 0:
            return self._angle
        # 用确定性伪噪声，保证自检可复现
        import math
        return self._angle + self.noise_deg * math.sin(self._angle * 12.9898)

    def stop(self) -> None:
        self._target = self._angle
        self._pending_delay = 0.0
        self.log.append(build_stop(self.axis))

    def pump(self, dt: float) -> None:
        """把时间往前推 dt 秒，按最大角速度逼近目标角。"""
        if dt <= 0:
            return
        if self._pending_delay > 0:
            used = min(self._pending_delay, dt)
            self._pending_delay -= used
            dt -= used
            if dt <= 0:
                return
        delta = self._target - self._angle
        max_step = self.speed_dps * dt
        if abs(delta) <= max_step:
            self._angle = self._target
        else:
            self._angle += max_step * (1.0 if delta > 0 else -1.0)

    def at_target(self, tol: float = 1e-6) -> bool:
        """到位判断：既没有剩余行程，也没有未消耗的指令延迟。"""
        return self._pending_delay <= 0.0 and abs(self._target - self._angle) <= tol


# --------------------------------------------------------------------------
# 串口后端
# --------------------------------------------------------------------------
class SerialMotor(MotorBackend):
    """真实硬件后端。需要 pyserial：pip install pyserial"""

    def __init__(self, port: str, baudrate: int = 115200, axis: str = DEFAULT_AXIS,
                 limit_min: float = -180.0, limit_max: float = 180.0,
                 timeout: float = 0.3, dry_run: bool = False) -> None:
        super().__init__(axis=axis, limit_min=limit_min, limit_max=limit_max)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.dry_run = dry_run
        self._ser: SerialLike | None = None
        self._last_pos = 0.0

        if self.dry_run:
            return
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 pyserial，请执行：pip install pyserial（或用 --dry-run 空跑）"
            ) from exc
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(2.0)  # Arduino 复位后 bootloader 会占用约 1.5~2 秒
        self._flush_input()

    # -- 内部 --
    def _flush_input(self) -> None:
        ser = self._ser
        if ser is None:
            return
        try:
            while True:
                line = ser.readline()
                if not line:
                    break
        except Exception:
            pass

    def _send(self, cmd: str) -> str:
        if self.dry_run or self._ser is None:
            return "OK DRYRUN"
        self._ser.write((cmd + "\n").encode("ascii"))
        line = self._ser.readline().decode("ascii", errors="replace").strip()
        if line and line.upper().startswith("ERR"):
            raise RuntimeError(f"下位机报错：{line}（指令 {cmd}）")
        return line

    # -- MotorBackend 接口 --
    def move_to(self, deg: float, speed_dps: float = 30.0) -> None:
        deg = self.clamp(deg)
        self._send(build_move(self.axis, deg, speed_dps))
        self._last_pos = deg

    def read_pos(self) -> float:
        line = self._send(build_status(self.axis))
        val = parse_pos(line, self.axis)
        if val is not None:
            self._last_pos = val
        return self._last_pos

    def stop(self) -> None:
        self._send(build_stop(self.axis))

    def home(self) -> None:
        self._send(build_home(self.axis))
        self._last_pos = 0.0

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None


def open_motor(spec: str, axis: str = DEFAULT_AXIS,
               limit_min: float = -180.0, limit_max: float = 180.0,
               dry_run: bool = False) -> MotorBackend:
    """按 'sim' 或 'COM5' / 'COM5@115200' 打开对应后端。"""
    if spec.strip().lower() in ("sim", "simulate", "simulation"):
        return SimMotor(axis=axis, limit_min=limit_min, limit_max=limit_max)
    port, _, baud = spec.partition("@")
    baudrate = int(baud) if baud else 115200
    return SerialMotor(port=port.strip(), baudrate=baudrate, axis=axis,
                       limit_min=limit_min, limit_max=limit_max, dry_run=dry_run)
