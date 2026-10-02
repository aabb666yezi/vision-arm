"""物品台账：登记 / 查询 / 落盘。

台账文件 items.json 结构：
{
  "version": 1,
  "items": {
    "水杯": {
      "name": "水杯",
      "ref_image": "data/ref/水杯.jpg",
      "target_deg": 0.0,          # 抬取时电机应停的角度
      "sig": {                    # 视觉签名（登记时算好，避免每次启动重算）
        "hist_h": [...],          # HSV 色相直方图（32 维）
        "hist_s": [...],          # 饱和度直方图（32 维）
        "kp": 213,                # 参考图关键点数，用于可信度分级
        "size": [86, 104]         # 参考图尺寸 [w, h]
      },
      "thresholds": {             # 每件物品独立阈值
        "min_matches": 12,
        "min_inliers": 6,
        "min_score": 0.55
      }
    }
  }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .imgio import imread_unicode

REGISTRY_VERSION = 1
DEFAULT_THRESHOLDS = {
    "min_matches": 12,
    "min_inliers": 6,
    "min_score": 0.55,
}


# --------------------------------------------------------------------------
# 小工具：图像 -> 直方图（与 vision.py 共用，放在这里避免循环依赖）
# --------------------------------------------------------------------------
def color_signature(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (色相直方图, 饱和度直方图)，均归一化且做了轻度平滑。

    注意：这里用一维卷积核自己平滑，而不是 cv2.GaussianBlur。
    后者要求二维 ksize，对 (1,N,1) 形状的直方图会直接报 ksize 参数错误。
    """
    if patch is None or patch.size == 0:
        return np.zeros(32, np.float32), np.zeros(32, np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], np.float32)
    kernel /= kernel.sum()
    for h in (hist_h, hist_s):
        h += 1e-6
        h /= h.sum()
        h[:] = np.convolve(h, kernel, mode="same")
        h /= h.sum()
    return hist_h.astype(np.float32), hist_s.astype(np.float32)


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """两个归一化直方图的相似度，1 表示完全一致。"""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    return float(cv2.compareHist(
        a.astype(np.float32).reshape(-1, 1),
        b.astype(np.float32).reshape(-1, 1),
        cv2.HISTCMP_CORREL,
    ))


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class Item:
    name: str
    ref_image: str
    target_deg: float = 0.0
    sig: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    @property
    def hist_h(self) -> np.ndarray:
        return np.asarray(self.sig.get("hist_h", []), dtype=np.float32)

    @property
    def hist_s(self) -> np.ndarray:
        return np.asarray(self.sig.get("hist_s", []), dtype=np.float32)

    @property
    def kp(self) -> int:
        return int(self.sig.get("kp", 0))

    @property
    def min_score(self) -> float:
        return float(self.thresholds.get("min_score", DEFAULT_THRESHOLDS["min_score"]))

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ref_image": self.ref_image,
            "target_deg": float(self.target_deg),
            "sig": self.sig,
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
        }


# --------------------------------------------------------------------------
# 台账
# --------------------------------------------------------------------------
class Registry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.items: dict[str, Item] = {}
        self.load()

    # -- 持久化 --
    def load(self) -> None:
        if not self.path.exists():
            self.items = {}
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        items: dict[str, Item] = {}
        for name, payload in raw.get("items", {}).items():
            items[name] = Item(
                name=payload.get("name", name),
                ref_image=payload.get("ref_image", ""),
                target_deg=float(payload.get("target_deg", 0.0)),
                sig=payload.get("sig", {}) or {},
                thresholds={**DEFAULT_THRESHOLDS, **(payload.get("thresholds") or {})},
            )
        self.items = items

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": REGISTRY_VERSION,
            "items": {name: item.to_json() for name, item in self.items.items()},
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -- 增删查 --
    def add(self, item: Item, overwrite: bool = True) -> None:
        if item.name in self.items and not overwrite:
            raise KeyError(f"物品已存在：{item.name}")
        self.items[item.name] = item

    def remove(self, name: str) -> bool:
        return self.items.pop(name, None) is not None

    def get(self, name: str) -> Item:
        if name not in self.items:
            hint = "、".join(self.names()) or "（空）"
            raise KeyError(f"台账里没有物品 '{name}'。已登记：{hint}")
        return self.items[name]

    def names(self) -> list[str]:
        return sorted(self.items.keys())

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterable[Item]:
        return iter(self.items.values())

    # -- 签名计算 --
    @staticmethod
    def build_signature(ref_image_path: str | Path, n_keypoints: int = 0) -> dict[str, Any]:
        """从参考图算出视觉签名。n_keypoints 由识别模块传入（SIFT 在那边）。"""
        img = imread_unicode(ref_image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"读不到参考图：{ref_image_path}")
        hist_h, hist_s = color_signature(img)
        h, w = img.shape[:2]
        return {
            "hist_h": [round(float(v), 6) for v in hist_h],
            "hist_s": [round(float(v), 6) for v in hist_s],
            "kp": int(n_keypoints),
            "size": [int(w), int(h)],
        }
