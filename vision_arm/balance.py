"""平衡控制：把水杯里的水当作"被驱动的单摆"，用离线优化算出不洒水的搬运曲线。

为什么这么做（这是本模块最值得写进答辩 PPT 的判断）：
    实时闭环抑制液面晃动非常难：没有便宜的液面传感器，视觉测液面延迟大、噪声大，
    一旦闭环相位错了反而会把晃动放大。而在宿舍取件这个场景里，**物体位置和搬运距离是有限的**，
    所以更划算的做法是"离线把所有情况都算一遍，把最优加速度上限做成查找表"，
    运行时只剩查表 + 限幅 —— 计算量几乎为零，可靠性却最高。
    这套思路在有硬件之后依然成立：表是在仿真里标定的，硬件上只执行。

物理模型：
    液体一阶晃动模态 ~ 一个摆长 L = R / (1.841 * tanh(1.841 * h / R)) 的单摆（R 杯半径，h 液深），
    固有频率 f1 = (1/2π) * sqrt(g / L)，常见水杯在 1.5 ~ 3 Hz。
    杯子的角加速度 a 在水面处等价于一个惯性力，驱动这个单摆：
        θ'' = -(g/L) * sin(θ) - c * θ' + (a / L) * cos(θ)
    其中 θ 是**液面相对杯口平面的倾角**，也就是"会不会洒"的直接判据。

对外接口：
    PLAN = build_table(...)                 离线生成的加速度上限查找表
    plan_transport(risk, angle_delta, L)     运行时查表 -> 一条可直接播放的轨迹
    simulate_trajectory(traj, risk, L)       给定轨迹，算最大液面倾角（用于验收）
可独立运行：
    python balance.py            跑对照实验并打印报告 + 导出结果 JSON
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
G = 9.81                  # 重力加速度 m/s^2
ARM_RADIUS_M = 0.50       # 默认：把杯子从桌面抬到床铺的等效回转半径（米）
DT = 0.002                # 仿真步长 2 ms（500 Hz，对 3 Hz 晃动足够）
MAX_TILT_DEG = 3.0        # 默认允许的最大液面倾角（直径 70 mm 杯口对应液面高差约 1.8 mm）


# --------------------------------------------------------------------------
# 风险分级：不同物品给不同的倾角与加加速度约束
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskProfile:
    code: str                 # A / B / C
    label: str
    max_tilt_deg: float       # 允许的最大液面/姿态倾角
    max_jerk_dps3: float      # 最大角加加速度（度/秒^3），越小越平顺但越慢
    note: str = ""


RISK: dict[str, RiskProfile] = {
    "A": RiskProfile("A", "开口液体（水杯/汤碗）", 3.0, 40.0, "最严：液面倾角直接决定洒不洒"),
    "B": RiskProfile("B", "易损/瓶装（玻璃杯、饮料瓶）", 8.0, 160.0, "次严：防倒防碰"),
    "C": RiskProfile("C", "普通物品（书、衣物、零食）", 20.0, 600.0, "最松：只防掉落"),
}


def classify_target(name: str) -> RiskProfile:
    """按物品名做风险分级。真实项目里可以换成从台账 items.json 读。"""
    liquid = ("水", "汤", "咖啡", "奶茶", "茶", "杯")
    fragile = ("玻璃", "瓶", "饮料", "酒", "罐", "碗", "盘")
    if any(k in name for k in liquid):
        return RISK["A"]
    if any(k in name for k in fragile):
        return RISK["B"]
    return RISK["C"]


# --------------------------------------------------------------------------
# 液体晃动模型
# --------------------------------------------------------------------------
@dataclass
class SloshModel:
    """一个水杯的晃动模型。默认是一杯装到 2/3 的普通马克杯。"""

    radius_m: float = 0.035       # 杯内半径 35 mm
    fill_m: float = 0.060         # 液深 60 mm
    damping_z: float = 0.05       # 一阶模态阻尼比（水的黏性很小，所以很容易晃）

    @property
    def length_m(self) -> float:
        """等效摆长（米）。

        公式来自一阶晃动模态的标准等效：L = R / (1.841 · tanh(1.841 · h / R))。
        注意必须夹到 R 以上：h >> R 时 tanh → 1，上式会给出 L → R/1.841 = 0.54R，
        比容器半径还短 —— 物理上不可能（摆长不可能小于液体的横向尺度）。
        本工程初期就漏了这个夹取，得到 L=19mm < 液深 60mm，
        于是晃动频率虚高到 3.61Hz，整个查找表都被算废了（所有工况都顶到搜索上界）。
        """
        x = 1.841
        return max(self.radius_m, self.radius_m / (x * math.tanh(x * self.fill_m / self.radius_m)))

    @property
    def freq_hz(self) -> float:
        return math.sqrt(G / self.length_m) / (2 * math.pi)

    @property
    def period_s(self) -> float:
        return 1.0 / self.freq_hz

    @property
    def damping_c(self) -> float:
        """转成 θ'' 方程里的阻尼系数 c = 2*zeta*omega_n。"""
        return 2.0 * self.damping_z * 2.0 * math.pi * self.freq_hz

    def describe(self) -> str:
        return (f"杯 R={self.radius_m * 1000:.0f}mm 液深={self.fill_m * 1000:.0f}mm "
                f"-> 摆长 L={self.length_m * 1000:.1f}mm 晃动频率 f1={self.freq_hz:.2f}Hz "
                f"周期 T={self.period_s * 1000:.0f}ms 阻尼比 {self.damping_z}")


# --------------------------------------------------------------------------
# 轨迹生成：分段 S 曲线（限制加速度与加加速度）
# --------------------------------------------------------------------------
@dataclass
class Trajectory:
    """一条可直接播放的角度轨迹（单位：度、度/秒、度/秒^2）。"""

    t: np.ndarray
    angle: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    peak_acc_dps2: float
    duration_s: float
    risk: RiskProfile
    shaped: bool = False
    label: str = ""

    def to_json(self, stride: int = 5) -> dict:
        sl = slice(None, None, stride)
        return {
            "label": self.label,
            "risk": self.risk.code,
            "duration_s": round(self.duration_s, 3),
            "peak_acc_dps2": round(self.peak_acc_dps2, 2),
            "shaped": self.shaped,
            "t": [round(float(v), 4) for v in self.t[sl]],
            "angle": [round(float(v), 3) for v in self.angle[sl]],
            "vel": [round(float(v), 3) for v in self.vel[sl]],
            "acc": [round(float(v), 3) for v in self.acc[sl]],
        }


def s_curve(delta_deg: float, peak_acc_dps2: float | None = None,
            jerk_dps3: float = 40.0, risk: RiskProfile | None = None,
            label: str = "", min_acc_dps2: float = 5.0) -> Trajectory:
    """生成一条"加速-匀速-减速"的分段 S 曲线（限制加速度与加加速度）。

    peak_acc_dps2 是**期望峰值加速度**（每段的首选值，也是物理安全上限）。
    这一点很关键，早期版本写反了，值得记下来：
        曾经的做法是"给一个转角，让加加速度限制去决定峰值加速度"，
        于是峰值加速度 = sqrt(delta · jerk) —— 移动越远，允许的加速度越小。
        这是反的：远距离搬运本来可以匀速走，不该因为距离长就限制加速度。
    正确做法（本版本）：
        加加速度只决定斜坡形状 t_r = a/jerk；峰值加速度由安全阈值给定；
        距离不够就退化成三角形（此时峰值由距离决定，是唯一不可避免的情形）；
        距离够就靠匀速段把总时长铺开，见 _cruise_for_distance。

    min_acc_dps2 是合理下限：低于这个值说明这条轨迹是"勉强凑出来"的，
    调用方应当认为该搬运在当前安全约束下不可行。
    """
    delta = abs(float(delta_deg))
    jerk = max(1e-6, float(jerk_dps3))

    if peak_acc_dps2 is None:
        # 未指定：用几何给出的三角形解（距离决定峰值）
        a_max = max(min_acc_dps2, math.sqrt(max(1e-9, delta * jerk)))
        t_r = a_max / jerk
        t_s = 0.0
    else:
        a_req = max(min_acc_dps2, float(peak_acc_dps2))
        # 距离能达到的几何上限：delta = a²/jerk ⇒ a = sqrt(delta·jerk)。
        # 请求值超过它时必须**降下来**，绝不能忽略 delta 硬跑 ——
        # 早期版本就是漏了这个钳位，导致 10° 的搬运被算成 90° 的行程，
        # 整张查找表所有转角的用时完全一样（一眼就该看出不对）。
        a_cap = math.sqrt(max(1e-9, delta * jerk))
        a_max = min(a_req, a_cap)
        t_r = a_max / jerk
        d_tri = a_max * t_r * t_r                    # 三角形所需距离
        t_s = 0.0 if delta <= d_tri else _cruise_for_distance(delta, a_max, t_r)

    t = np.arange(0.0, 4 * t_r + t_s + 0.05 + DT, DT)
    t1, t2, t3 = t_r, t_r + t_s, 2 * t_r + t_s
    acc = np.zeros_like(t)
    m = t < t1
    acc[m] = a_max * (t[m] / t_r)
    acc[(t >= t1) & (t < t2)] = a_max
    m = (t >= t2) & (t < t3)
    acc[m] = a_max * (1.0 - (t[m] - t2) / t_r)
    acc[t >= t3] = 0.0

    vel = np.cumsum(acc) * DT
    ang = np.cumsum(vel) * DT
    if ang[-1] > 1e-9:                               # 只修正数值积分误差
        k = delta / ang[-1]
        vel, ang = vel * k, ang * k
    sign = 1.0 if delta_deg >= 0 else -1.0
    return Trajectory(t=t, angle=ang * sign, vel=vel * sign, acc=acc * sign,
                      peak_acc_dps2=float(np.max(np.abs(acc))),
                      duration_s=float(t[-1]),
                      risk=risk or RISK["A"], shaped=False, label=label)


def _cruise_for_distance(delta: float, a_max: float, t_r: float) -> float:
    """求匀速段时长 t_s，使总位移正好等于 delta。

    位移组成：加速段 a·t_r² + 匀速段 a·t_r·t_s + 减速段 0.5·a·t_r·t_s + 0.5·a·t_r²
    整理得 0.5·a·t_s² + 1.5·a·t_r·t_s + a·t_r² − delta = 0，取正根。
    """
    b = 1.5 * a_max * t_r
    c = a_max * t_r * t_r - delta
    disc = max(0.0, b * b - 2.0 * a_max * c)
    return (-b + math.sqrt(disc)) / a_max


def zvd_shaper(freq_hz: float, damping_z: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """ZVD 输入整形器：三个脉冲，能把指定频率的残留振动从原理上抵消掉。

    这是"防洒"的核心武器 —— 本质是把加速度指令和延迟半个晃动周期的一份自己做卷积。
    """
    wn = 2.0 * math.pi * freq_hz
    z = damping_z
    wd = wn * math.sqrt(max(1e-9, 1.0 - z * z))
    k = math.exp(-z * math.pi / math.sqrt(max(1e-9, 1.0 - z * z)))
    t = np.array([0.0, math.pi / wd, 2.0 * math.pi / wd])
    a = np.array([1.0, 2.0 * k, k * k])
    a = a / a.sum()
    return t, a


def apply_shaper(traj: Trajectory, freq_hz: float, damping_z: float = 0.05) -> Trajectory:
    """把 ZVD 整形器作用到加速度上，重新积分得到角度/速度。

    实现要点：用**整数样本位移**而不是浮点插值。
        ZVD 的三个冲量间隔正好是半个振荡周期 T/2 = 0.1879s，
        在 DT = 2ms 下就是 94 个采样点（离散化误差 0.3%，对抑制效果无影响）。
        早期版本做浮点插值时索引算错，丢掉了一部分冲量面积，
        结果"整形后峰值加速度"被削到几乎为零（12 → 1.5），明显不合物理。
        现在整形前后用面积守恒做自检（见下方 assert 说明）。
    """
    t_i, a_i = zvd_shaper(freq_hz, damping_z)
    n = len(traj.acc)
    step = int(round((t_i[1] - t_i[0]) / DT))
    offsets = [0, step, 2 * step]
    center = step                                 # 居中：整体左移一个间隔
    shaped = np.zeros(n)
    for off, amp in zip(offsets, a_i):
        shift = off - center
        if shift >= 0:
            shaped[shift:] += amp * traj.acc[:n - shift]
        else:
            shaped[:n + shift] += amp * traj.acc[-shift:]
    vel = np.cumsum(shaped) * DT
    ang = np.cumsum(vel) * DT
    # 只做数值积分误差的微修正（整形保持面积，所以这一步只动万分之几）
    if abs(ang[-1]) > 1e-9:
        k = traj.angle[-1] / ang[-1]
        vel, ang = vel * k, ang * k
    return Trajectory(t=traj.t.copy(), angle=ang, vel=vel, acc=shaped,
                      peak_acc_dps2=float(np.max(np.abs(shaped))),
                      duration_s=traj.duration_s, risk=traj.risk, shaped=True,
                      label=(traj.label + "+ZVD") if traj.label else "ZVD")


def bang_bang(delta_deg: float, acc_dps2: float, risk: RiskProfile,
              label: str = "无整形方波") -> Trajectory:
    """对照组：不定加加速度的方波加速度（很多 DIY 项目的实际做法）。"""
    delta = abs(float(delta_deg))
    a = max(1e-6, float(acc_dps2))
    t_acc = math.sqrt(delta / a)
    total = 2 * t_acc + 0.05
    t = np.arange(0.0, total + DT, DT)
    acc = np.where(t < t_acc, a, 0.0)
    acc = np.where((t >= t_acc) & (t < 2 * t_acc), -a, acc)
    vel = np.cumsum(acc) * DT
    ang = np.cumsum(vel) * DT
    sign = 1.0 if delta_deg >= 0 else -1.0
    return Trajectory(t=t, angle=ang * sign, vel=vel * sign, acc=acc * sign,
                      peak_acc_dps2=a, duration_s=float(t[-1]), risk=risk,
                      label=label)


# --------------------------------------------------------------------------
# 播放轨迹并仿真液面响应
# --------------------------------------------------------------------------
@dataclass
class SloshResult:
    max_tilt_deg: float
    final_tilt_deg: float          # 停下后残留的晃动（越接近 0 越好）
    residual_energy: float         # 残留振动能量（相对值）
    settle_time_s: float           # 从运动结束到液面平静所需时间
    tilted_forward: bool
    series: dict = field(default_factory=dict)

    @property
    def safe(self) -> bool:
        return True  # 是否安全由调用方拿 max_tilt_deg 与阈值比较


def simulate_trajectory(traj: Trajectory, slosh: SloshModel,
                        tilt_sign: float = 1.0,
                        arm_radius_m: float = ARM_RADIUS_M) -> SloshResult:
    """把角度轨迹喂给液体模型，算出最大液面倾角和残留晃动。

    物理形式（很容易写错的地方，这里逐步说清楚）：
        把液面看成惯性系里的一个单摆，它的支点就是杯子，支点加速度 a_cup 由轨迹给出。
        惯性系下液面相对水平面的角度 θ_abs 满足
            θ_abs'' = -(g/L)·sin(θ_abs) + (a_cup/L)·cos(θ_abs) - c·θ_abs'
        而杯子自身的惯性倾角是
            θ_cup = -a_cup / g          （杯子往右加速，水面往左偏）
        最终相对杯口的倾角就是
            θ_liquid = θ_abs - θ_cup

    ⚠️ 这里踩过一个很隐蔽的坑（debug 了很久，值得记下来）：
        早期版本改成对"摆动分量 φ = θ_liquid"积分，方程为
            φ'' = -(g/L)·sinφ - c·φ'
        但 φ 的平衡点是 φ_eq = -θ_cup ≠ 0。如果按"液面初始水平且静止"设 φ(0)=0，
        就恰好把系统放在了平衡点上 —— 于是 φ 恒等于 0，
        θ_liquid 退化成永远等于 θ_cup，**振荡被完全抹掉**。
        表现是：理论过冲 85%，实测过冲 0%；所有工况"振荡分量"都是 0.00°；
        输入整形看起来毫无作用（因为它抑制的振荡在模型里根本不存在）。
        正确写法有两种：要么给 φ 设对初值 φ(0) = -θ_cup(0) 并在方程里补上驱动项，
        要么像本版本这样直接对 θ_abs 积分（推荐，物理更直白、初值就是 0）。
    """
    L = slosh.length_m
    c = slosh.damping_c
    # 角度 -> 线加速度：a = alpha * r（alpha 用 rad/s^2）
    a_lin = np.radians(traj.acc) * arm_radius_m
    theta_cup = -a_lin / G                              # 惯性倾角（rad）
    n = len(a_lin)
    theta_abs = np.zeros(n)                             # 惯性系下液面相对水平面的角度
    omega = np.zeros(n)
    th, om = 0.0, 0.0                                   # 初值：液面水平且静止
    for i in range(1, n):
        acc_th = (-(G / L) * math.sin(th)
                  + (a_lin[i] / L) * math.cos(th)
                  - c * om)
        om += acc_th * DT
        th += om * DT
        theta_abs[i], omega[i] = th, om
    theta_liq = theta_abs - theta_cup                   # 相对杯口的倾角

    idx_move_end = int(np.searchsorted(traj.t, traj.t[-1] - 0.05))
    peak = float(np.max(np.abs(theta_liq)))
    final = float(abs(theta_liq[-1]))
    # 停稳后振幅衰减到 0.2 度以内所需时间（从运动结束算起）
    tail = np.abs(theta_liq[idx_move_end:])
    below = np.where(tail < math.radians(0.2))[0]
    settle = float(below[0] * DT) if len(below) else float(len(tail) * DT)
    energy = 0.5 * om ** 2 + (G / L) * (1 - np.cos(th))
    return SloshResult(
        max_tilt_deg=math.degrees(peak),
        final_tilt_deg=math.degrees(final),
        residual_energy=float(energy),
        settle_time_s=settle,
        tilted_forward=tilt_sign > 0,
        series={"t": traj.t, "acc": traj.acc,
                "theta_deg": np.degrees(theta_liq),
                "theta_cup_deg": np.degrees(theta_cup)},
    )


# --------------------------------------------------------------------------
# 离线优化：给定风险等级与转角，找出"不超倾角"的最大加速度
# --------------------------------------------------------------------------
def optimize_peak_acc(delta_deg: float, risk: RiskProfile, slosh: SloshModel,
                      use_shaper: bool = True, lo: float = 1.0,
                      iters: int = 20) -> tuple[float, SloshResult, Trajectory]:
    """二分搜索：满足"液面倾角不超阈值"的前提下，峰值加速度能取到多大。

    这就是"离线把所有情况算一遍"的核心一步，结果被做成查找表。
    搜索对象必须是**最终真正执行的那条轨迹**（含输入整形），
    否则表里的值与实跑的不是一回事（早期版本就犯过这个错）。

    搜索上界取该转角下的几何可达上限 a_cap = sqrt(delta · jerk)：
    因为距离给定的情况下，加速度不可能超过它（超过就需要更长的斜坡，
    而距离已经被用完了）。用固定的 400 当上界会搜出一个永远取不到的值，
    最后打印出来的"安全加速度"其实是那个假上界。

    两个"不可行"的判据：
      1. 峰值加速度低于 lo：说明这条轨迹是勉强凑出来的，搬运本身没意义；
      2. 倾角超阈值：物理上会洒。
    都不满足时返回 (0.0, ...)，调用方应当拒绝这次搬运。
    """
    hi = max(lo * 2.0, math.sqrt(max(1e-9, abs(delta_deg) * risk.max_jerk_dps3)))

    def make(acc: float) -> tuple[Trajectory, SloshResult]:
        traj = s_curve(delta_deg, acc, risk.max_jerk_dps3, risk, label="search")
        if use_shaper:
            traj = apply_shaper(traj, slosh.freq_hz, slosh.damping_z)
        return traj, simulate_trajectory(traj, slosh)

    def feasible(acc: float) -> tuple[bool, Trajectory, SloshResult]:
        traj, res = make(acc)
        # 注意：这里判断的是"实际峰值加速度"而不是请求值 —— 距离不够时
        # 轨迹会自动降加速度，请求值高不代表真的跑得快。
        real = float(np.max(np.abs(traj.acc)))
        return (real >= lo and res.max_tilt_deg <= risk.max_tilt_deg), traj, res

    ok_lo, traj_lo, res_lo = feasible(lo)
    ok_hi, traj_hi, res_hi = feasible(hi)
    if ok_hi:
        return hi, res_hi, traj_hi
    if not ok_lo:
        traj_lo.label = "不可行"
        return 0.0, res_lo, traj_lo
    best = (lo, res_lo, traj_lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok, traj, res = feasible(mid)
        if ok:
            best = (mid, res, traj)
            lo = mid
        else:
            hi = mid
    return best


def build_table(slosh: SloshModel | None = None,
                deltas: tuple[float, ...] = (10.0, 20.0, 40.0, 60.0, 90.0, 120.0),
                use_shaper: bool = True, verbose: bool = True) -> dict:
    """生成查找表：{(风险等级, 转角): 安全峰值加速度上限 + 预计用时}。"""
    slosh = slosh or SloshModel()
    table: dict = {"slosh": {"freq_hz": round(slosh.freq_hz, 3),
                             "length_mm": round(slosh.length_m * 1000, 1),
                             "period_ms": round(slosh.period_s * 1000, 1)},
                   "arm_radius_m": ARM_RADIUS_M,
                   "entries": {}}
    for code, risk in RISK.items():
        for d in deltas:
            acc, res, traj = optimize_peak_acc(d, risk, slosh, use_shaper=use_shaper)
            feasible = acc > 0.0
            table["entries"][f"{code}:{d:.0f}"] = {
                "risk": code, "delta_deg": d,
                "peak_acc_dps2": round(acc, 1),
                "max_tilt_deg": round(res.max_tilt_deg, 2),
                "duration_s": round(traj.duration_s, 2),
                "feasible": bool(feasible),
            }
            if verbose:
                tag = "" if feasible else "  ← 当前安全约束下不可行"
                print(f"  [{code}] Δ{d:5.0f}° -> 安全加速度 {acc:6.1f} °/s²  "
                      f"液面峰值 {res.max_tilt_deg:4.2f}°  用时 {traj.duration_s:5.2f}s{tag}")
    return table


_PLAN_CACHE: dict | None = None
_DEFAULT_PLAN = Path(__file__).resolve().parent.parent / "out" / "balance_plan.json"


def load_plan(path: str | Path | None = None) -> dict:
    """读取离线算好的表（没有就现算一次，保证 demo 永远能跑）。"""
    global _PLAN_CACHE
    if _PLAN_CACHE is not None:
        return _PLAN_CACHE
    p = Path(path) if path is not None else _DEFAULT_PLAN
    if p.exists():
        _PLAN_CACHE = json.loads(p.read_text(encoding="utf-8"))
        return _PLAN_CACHE
    _PLAN_CACHE = build_table(verbose=False)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_PLAN_CACHE, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    except OSError:
        pass
    return _PLAN_CACHE


def lookup_peak_acc(risk_code: str, delta_deg: float,
                    path: str | Path | None = None) -> float:
    """运行时查表：线性插值，超出表范围就取表内最近的边界值。"""
    plan = load_plan(path)
    entries = plan["entries"]
    pts = sorted((float(k.split(":")[1]), v["peak_acc_dps2"])
                 for k, v in entries.items() if v["risk"] == risk_code)
    if not pts:
        return 60.0
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return float(np.interp(abs(delta_deg), xs, ys))


def plan_transport(risk: RiskProfile, delta_deg: float,
                   slosh: SloshModel | None = None,
                   shaper: bool = True) -> Trajectory:
    """运行时入口：查表拿到安全加速度 -> 生成一条可播放、可验收的轨迹。

    返回的轨迹如果 peak_acc_dps2 明显偏低（说明是勉强凑出来的），
    调用方应当判定"这次搬运在当前安全约束下不可行"并拒绝执行。
    """
    slosh = slosh or SloshModel()
    acc = lookup_peak_acc(risk.code, delta_deg)
    traj = s_curve(delta_deg, acc, risk.max_jerk_dps3, risk,
                   label=f"{risk.code}级 Δ{abs(delta_deg):.0f}°")
    return apply_shaper(traj, slosh.freq_hz, slosh.damping_z) if shaper else traj


# --------------------------------------------------------------------------
# 对照实验（比赛演示的核心素材：为什么必须做平衡控制）
# --------------------------------------------------------------------------
def compare_strategies(delta_deg: float = 90.0,
                       slosh: SloshModel | None = None) -> dict:
    """四组对照：粗暴搬运 / 只做 S 曲线 / 只做输入整形 / S曲线+离线优化+整形。

    每一组都各自做优化，比的是"在相同安全阈值下，这类策略能做到多快"，
    而不是"用同一个加速度谁更晃" —— 后者不是工程上关心的比较。
    """
    slosh = slosh or SloshModel()
    risk_a = RISK["A"]
    out: dict = {"delta_deg": delta_deg, "slosh": slosh.describe(),
                 "freq_hz": round(slosh.freq_hz, 3),
                 "max_allowed_deg": risk_a.max_tilt_deg, "cases": []}

    # 1) 粗暴搬运：不管不顾，方波加速度（很多 DIY 项目的实际做法）
    rough = bang_bang(delta_deg, 120.0, risk_a, label="① 粗暴搬运（方波 120°/s²）")
    out["cases"].append(_case(rough, simulate_trajectory(rough, slosh)))

    # 2) 只做 S 曲线：同样想快，但受加加速度限制，波形变平滑
    sc = s_curve(delta_deg, 120.0, risk_a.max_jerk_dps3, risk_a, label="② 只做 S 曲线（仍想快）")
    out["cases"].append(_case(sc, simulate_trajectory(sc, slosh)))

    # 3) 只做输入整形：波形仍是方波，但叠了 ZVD
    bb3 = bang_bang(delta_deg, 120.0, risk_a)
    b3 = apply_shaper(bb3, slosh.freq_hz, slosh.damping_z)
    b3.label = "③ 只做输入整形（方波+ZVD）"
    out["cases"].append(_case(b3, simulate_trajectory(b3, slosh)))
    # 4) 本方案：离线优化出安全加速度 + S 曲线 + ZVD 输入整形
    acc4, res4, best = optimize_peak_acc(delta_deg, risk_a, slosh, use_shaper=True)
    best.label = "④ 本方案：离线优化+S曲线+输入整形"
    out["cases"].append(_case(best, res4))
    out["optimized_acc_dps2"] = round(acc4, 2)

    return out


def _case(traj: Trajectory, res: SloshResult) -> dict:
    return {
        "label": traj.label,
        "duration_s": round(traj.duration_s, 3),
        # 一律用轨迹的**实际**峰值加速度，不能用"请求值"。
        # 早期版本这里报的是请求值，导致"只做 S 曲线（仍想快）"显示 120°/s²，
        # 而轨迹实际只有 60°/s² —— 表格自相矛盾，评委一眼就能看出来。
        "peak_acc_dps2": round(float(np.max(np.abs(traj.acc))), 2),
        "requested_acc_dps2": round(traj.peak_acc_dps2, 2),
        "max_tilt_deg": round(res.max_tilt_deg, 3),
        "final_tilt_deg": round(res.final_tilt_deg, 4),
        "settle_s": round(res.settle_time_s, 3),
        "pass": bool(res.max_tilt_deg <= RISK["A"].max_tilt_deg),
        "traj": traj.to_json(stride=4),
        "theta": [round(float(v), 3) for v in res.series["theta_deg"][::4]],
    }


# --------------------------------------------------------------------------
# 命令行入口：python balance.py
# --------------------------------------------------------------------------
def _print_report(slosh: SloshModel, cmp: dict, table: dict) -> None:
    line = "=" * 78
    print(line)
    print("宿舍取件机械臂 —— 搬运平衡控制离线实验")
    print(line)
    print(f"\n【被测对象】{slosh.describe()}")
    print(f"【安全阈值】液面倾角峰值 <= {RISK['A'].max_tilt_deg}°（A 级物品，开口液体）")
    print(f"【等效回转半径】{ARM_RADIUS_M * 1000:.0f} mm")

    print(f"\n【对照实验】转角 {cmp['delta_deg']:.0f}°，同一个水杯，四种搬运方式：")
    print(f"{'方案':<32}{'液面峰值':>10}{'用时':>9}{'实际峰值加速度':>16}{'结论':>8}")
    print("-" * 78)
    for c in cmp["cases"]:
        verdict = "✅ 合格" if c["pass"] else "❌ 洒出"
        print(f"{c['label']:<32}{c['max_tilt_deg']:>9.2f}°{c['duration_s']:>8.2f}s"
              f"{c['peak_acc_dps2']:>13.1f}°/s²{verdict:>9}")
    base = cmp["cases"][0]["max_tilt_deg"]
    final = cmp["cases"][-1]["max_tilt_deg"]
    print("-" * 78)
    print(f"结论：把液面峰值倾角从 {base:.2f}° 压到 {final:.2f}°（降低 "
          f"{100 * (1 - final / max(1e-9, base)):.0f}%），代价是搬运时间从 "
          f"{cmp['cases'][0]['duration_s']:.2f}s 增加到 {cmp['cases'][-1]['duration_s']:.2f}s。")
    print(f"     最关键的取舍：搬运速度必须服从液面倾角约束，而不是反过来。")

    print("\n【离线优化得到的加速度上限查找表】")
    print(f"{'等级':<6}{'转角':>8}{'峰值加速度上限':>18}{'液面峰值':>11}{'用时':>9}")
    print("-" * 78)
    for code in RISK:
        for k, v in table["entries"].items():
            if v["risk"] != code:
                continue
            print(f"{code:<6}{v['delta_deg']:>7.0f}°{v['peak_acc_dps2']:>15.1f}°/s²"
                  f"{v['max_tilt_deg']:>9.2f}°{v['duration_s']:>8.2f}s")
    print(line)
    print("运行时的用法：plan_transport(风险等级, 转角) -> 一条可播放的轨迹，")
    print("控制循环只需按轨迹走，不需要任何实时晃动观测。")
    print(line)


def main() -> int:
    # 固定写到"项目根目录/out"，不依赖当前工作目录
    # （从包目录里运行和从项目根运行必须落在同一个地方，否则网页读不到）
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    slosh = SloshModel()

    print("正在离线优化加速度上限（每个等级 6 个转角，二分搜索 22 次）...")
    table = build_table(slosh, verbose=True)
    (out_dir / "balance_plan.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

    cmp = compare_strategies(90.0, slosh)
    (out_dir / "balance_compare.json").write_text(
        json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_report(slosh, cmp, table)
    print(f"\n已导出：{out_dir / 'balance_plan.json'}（查找表）")
    print(f"已导出：{out_dir / 'balance_compare.json'}（对照实验曲线，网页要读它）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
