"""包初始化：对外暴露最常用的几个名字。"""

from .balance import (
    RISK, RiskProfile, SloshModel, Trajectory, classify_target, plan_transport,
    simulate_trajectory,
)
from .brain import DemoEngine, Mission, Pedestrian, SafetyArbiter, parse_command
from .motor import MotorBackend, SerialMotor, SimMotor, open_motor
from .registry import Item, Registry
from .servo import MotionResult, Servo, ServoConfig
from .vision import Detection, Detector, Scene, SceneItem

__all__ = [
    "MotorBackend", "SerialMotor", "SimMotor", "open_motor",
    "Item", "Registry",
    "Servo", "ServoConfig", "MotionResult",
    "Detector", "Detection", "Scene", "SceneItem",
    "RISK", "RiskProfile", "SloshModel", "Trajectory",
    "classify_target", "plan_transport", "simulate_trajectory",
    "Mission", "DemoEngine", "Pedestrian", "SafetyArbiter", "parse_command",
]

__version__ = "0.2.0"
