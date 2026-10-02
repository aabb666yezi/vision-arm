"""视觉识别：把画面里的东西对上"已登记物品"，并给出它在画面里的位置。

识别流程（四级过滤，逐级淘汰，速度快且误检少）：
  1. 参考图自动抠背景生成掩膜，**只在物品区域**提特征点。
     挡掉：背景相似导致的错误匹配（背景点根本不该参与"认这个物品"）。
  2. KNN(k=2) 找每个点最像的两个候选，做 Lowe ratio 检验：
     只有当"第一名距离 < 0.75 × 第二名距离"（即第一名至少比第二名好 25%）才采纳，
     否则**丢弃这一组匹配** —— 说明画面里存在多个相似候选、无法确定哪个才是对应点
     （典型来源：重复纹理，如条纹、格子）。再按参考图掩膜做空间过滤，
     丢掉"参考点落在物品区域之外"的匹配。
  3. RANSAC 求一个统一的单应矩阵，统计有多少组匹配能被**同一个变换**一致解释（内点）；
     再检查投影框的**面积与长宽比**是否合理。
     挡掉：(a) 位置关系**不自洽**的误配（各点方向互相矛盾）；
           (b) **退化解** —— 匹配点近似共线时单应矩阵退化，投影框会塌缩成线甚至一个点；
           (c) 同时容纳"物品角度与登记时不同"造成的透视变形（不误杀真物品）。
  4. 把投影框内的画面裁出来，和参考图的物品区域比 **HSV 直方图（色相 + 饱和度）**相关性。
     挡掉"**自洽但张冠李戴**"的匹配 —— 跨物体误配（画面里的 B 被配到参考图的 A）
     以及"一致错位"的匹配。这两种情况的共同点是**几何上完全自洽**，
     所以第 3 级拦不住，必须用颜色这个独立证据确认"这个位置上的东西到底是不是它"。

⚠️ 一条必须记住的边界（写文档/答辩时别搞错）：
    RANSAC 的"自洽"是一个**很弱**的保证 —— 只要匹配点能**一致地错**，它就会认可。
      · 第 3 级管"**自不自洽**"：挡位置互相矛盾的点、以及退化解；
      · 第 4 级管"**自洽的前提对不对**"：挡自洽但身份错误的匹配。
    把"结构性误配"归给第 3 级是不准确的 —— 结构性误配恰恰是自洽的。

已知局限（实测数据支撑，见 README「已知局限」）：
    · 重复纹理（条纹/圆点）会让内点比例剧烈波动：同一物品换登记尺度，
      1.0× 时内点比例 93%，1.4× 时只有 21%。这不是实现缺陷，
      而是"重复纹理天然有歧义"的固有代价。
    · 若场景里存在另一个**颜色和几何都相似**的物体，四级过滤都可能放它过去。
      对策在场景设计层面：登记有独特特征的面，或在物品上贴标记（AprilTag/二维码）。

输出 Detection 里同时给出两样东西：
  * px_center : 物品在画面中的像素中心（视觉伺服要用）
  * angle_deg : 要让电机把该物品转到画面正中，电机应当到达的绝对角度
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .imgio import imread_unicode, imwrite_unicode
from .registry import Item, Registry, hist_similarity

# --------------------------------------------------------------------------
# 配方参数（换相机/换场景时主要调这里）
# --------------------------------------------------------------------------
RATIO_TEST = 0.75          # Lowe ratio
MIN_MATCH_MARGIN = 6       # 少于这个匹配数直接不判
MASK_MARGIN = 0.06         # 参考图 mask 外扩比例
MAX_AREA_RATIO = 0.75      # 投影框面积 / 画面面积 上限（防"整个画面都匹配上"）
ASPECT_TOL = 2.6           # 投影框长宽比与参考图的偏差上限
SCORE_W = (0.50, 0.30, 0.20)  # 内点比例 / 内点数量 / 直方图 的权重
# 内点数量的归一化基准：取"参考图关键点数的 25%"，而不是写死一个常数。
# 理由：纹理丰富的物体本来就能匹配上更多点，纯色物体天然少；
# 写死常数会导致"小物体永远不达标、大物体随便达标"，实机上很难调。
INLIER_REF_FRACTION = 0.25


def seed_from_name(name: str) -> int:
    """按名字生成"跨进程稳定"的随机种子。

    绝对不要用 Python 内置的 hash()：字符串 hash 受 PYTHONHASHSEED 影响，
    每个进程都不一样。本工程初期就踩过这个坑 ——
    参考图和场景图的纹理颗粒在不同进程间不一致，
    导致识别得分在 0.49~0.93 之间跳，表现为"同一份数据两次跑结果不同"，
    排查了很久才定位到 hash 随机化。crc32 是确定的，可复现。
    """
    import zlib
    return int(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF)


def _make_sift(nfeatures: int = 2000):
    """SIFT 只在 opencv-contrib-python 里；给一个清楚的报错提示。

    contrastThreshold 取 0.01（默认 0.04）是为了在小尺寸目标上也能提出足够特征；
    代价是特征数变多、匹配稍慢，对本项目这种"少量物品 + 低帧率"的场景完全可接受。
    """
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "当前 OpenCV 没有 SIFT。请改装：pip install opencv-contrib-python\n"
            "（opencv-python 基础包不含 SIFT/FLANN 的专利期外实现）"
        )
    return cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.01, edgeThreshold=12)


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class Reference:
    """一件已登记物品的视觉模板。"""

    item: Item
    image: np.ndarray
    gray: np.ndarray
    mask: np.ndarray                  # uint8，1=物品区域
    keypoints: list
    descriptors: np.ndarray | None
    bbox: tuple[int, int, int, int]   # 参考图里物品的框 (x, y, w, h)
    aspect: float

    @property
    def name(self) -> str:
        return self.item.name


@dataclass
class Detection:
    name: str
    score: float
    px_center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    angle_deg: float
    px_err_norm: float
    inliers: int
    matches: int
    hist_corr: float
    sign: int = 1

    def __str__(self) -> str:
        x, y, w, h = self.bbox
        return (f"{self.name} score={self.score:.2f} 中心=({self.px_center[0]:.0f},"
                f"{self.px_center[1]:.0f}) 框=({x},{y},{w},{h}) "
                f"角度={self.angle_deg:.2f}° err={self.px_err_norm:+.3f}")


# --------------------------------------------------------------------------
# 识别器
# --------------------------------------------------------------------------
class Detector:
    def __init__(self, registry: Registry, base_dir: str | Path = ".",
                 ratio: float = RATIO_TEST, debug: bool = False) -> None:
        self.registry = registry
        self.base_dir = Path(base_dir)
        self.ratio = ratio
        self.debug = debug
        self.sift = _make_sift()
        index_params = dict(algorithm=1, trees=5)          # KDTree
        search_params = dict(checks=48)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.refs: dict[str, Reference] = {}
        self._reload()

    # -- 载入参考模板 --
    def _reload(self) -> None:
        self.refs = {}
        for item in self.registry:
            try:
                self.refs[item.name] = self._load_reference(item)
            except Exception as exc:  # 单件坏掉不影响其他件
                print(f"[warn] 载入参考图失败，跳过 {item.name}：{exc}")

    def _load_reference(self, item: Item) -> Reference:
        path = Path(item.ref_image)
        if not path.is_absolute():
            path = self.base_dir / path
        img = imread_unicode(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        size = item.sig.get("size") or [w, h]
        if [w, h] != list(size):
            img = cv2.resize(img, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        bbox = self._item_bbox(img)
        mask = np.zeros((h, w), np.uint8)
        mx, my = int(bbox[2] * MASK_MARGIN), int(bbox[3] * MASK_MARGIN)
        x0 = max(0, bbox[0] - mx)
        y0 = max(0, bbox[1] - my)
        x1 = min(w, bbox[0] + bbox[2] + mx)
        y1 = min(h, bbox[1] + bbox[3] + my)
        mask[y0:y1, x0:x1] = 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = self.sift.detectAndCompute(gray, mask)
        if desc is None or len(kps) < 8:
            raise ValueError(f"参考图特征太少（{0 if desc is None else len(kps)} 个关键点），"
                             f"请换一张更清晰、纹理更丰富的图")
        return Reference(
            item=item, image=img, gray=gray, mask=mask,
            keypoints=kps, descriptors=desc.astype(np.float32),
            bbox=bbox, aspect=max(1e-3, bbox[2] / max(1, bbox[3])),
        )

    @staticmethod
    def _item_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
        """自动估计参考图里物品的紧致框：先抠掉和四边同色的背景，再取非背景外接框。"""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 取四边像素的中位数作为背景亮度基准
        border = np.concatenate([blur[0, :], blur[-1, :], blur[:, 0], blur[:, -1]])
        bg = float(np.median(border))
        diff = np.abs(blur.astype(np.float32) - bg)
        spread = float(np.percentile(border, 90) - np.percentile(border, 10))
        thr = max(12.0, spread * 2.5)
        fg = (diff > thr).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts is None or len(cnts) == 0:
            return (0, 0, w, h)
        big = max(cnts, key=cv2.contourArea)
        if big is None or big.size == 0 or cv2.contourArea(big) < 0.02 * w * h:
            return (0, 0, w, h)
        x, y, bw, bh = cv2.boundingRect(big)
        x0 = max(0, x - 1)
        y0 = max(0, y - 1)
        bw = max(4, min(w - x0, bw + 2))
        bh = max(4, min(h - y0, bh + 2))
        return (x0, y0, bw, bh)

    # ------------------------------------------------------------------
    # 单帧识别
    # ------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        cam_deg: float = 0.0,
        *,
        hfov_deg: float = 53.0,
        move_sign: int = 1,
        names: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[Detection]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return []
        frame_kps, frame_desc = self.sift.detectAndCompute(gray, None)
        if frame_desc is None or len(frame_kps) < 4:
            return []
        frame_desc = frame_desc.astype(np.float32)

        results: list[Detection] = []
        for name, ref in self.refs.items():
            if names and name not in names:
                continue
            det = self._match_one(frame, gray, frame_kps, frame_desc, ref,
                                  cam_deg, hfov_deg, move_sign)
            if det is not None and det.score >= max(min_score, ref.item.min_score):
                results.append(det)

        # 同一件物品若被多件参考命中，保留分数最高的
        best: dict[str, Detection] = {}
        for det in results:
            if det.name not in best or det.score > best[det.name].score:
                best[det.name] = det
        return sorted(best.values(), key=lambda d: -d.score)

    def _match_one(self, frame, gray, frame_kps, frame_desc, ref: Reference,
                   cam_deg: float, hfov_deg: float, move_sign: int) -> Detection | None:
        k = 2
        if len(frame_desc) < k + 1:
            return None
        # FLANN 对描述子的内存布局很敏感，必须 float32 且连续，否则会抛底层异常
        ref_desc = np.ascontiguousarray(ref.descriptors, dtype=np.float32)
        frm_desc = np.ascontiguousarray(frame_desc, dtype=np.float32)
        try:
            raw = self.flann.knnMatch(ref_desc, frm_desc, k=k)
        except cv2.error as exc:
            self._dbg(ref.name, f"FLANN 匹配异常，本帧跳过：{exc}")
            return None
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        if len(good) < max(MIN_MATCH_MARGIN, ref.item.thresholds.get("min_matches", 12) // 2):
            return None

        # 用参考图 mask 过滤：只信落在物品区域的匹配点
        rh, rw = ref.gray.shape[:2]
        pts_ref, pts_frame = [], []
        for m in good:
            rx, ry = ref.keypoints[m.queryIdx].pt
            if not (0 <= int(ry) < rh and 0 <= int(rx) < rw):
                continue
            if ref.mask[int(ry), int(rx)] == 0:
                continue
            pts_ref.append((rx, ry))
            pts_frame.append(frame_kps[m.trainIdx].pt)
        try:
            pts_ref = np.asarray(pts_ref, np.float32).reshape(-1, 1, 2)
            pts_frame = np.asarray(pts_frame, np.float32).reshape(-1, 1, 2)
        except Exception:
            return None
        if len(pts_ref) < 4:
            return None

        H, inlier_mask = cv2.findHomography(pts_ref, pts_frame, cv2.RANSAC, 5.0)
        if H is None or inlier_mask is None:
            return None
        inliers = int(inlier_mask.sum())
        ratio_in = inliers / max(1, len(pts_ref))
        if inliers < ref.item.thresholds.get("min_inliers", 6):
            self._dbg(ref.name, f"内点不足 {inliers}")
            return None

        x, y, bw, bh = ref.bbox
        fh, fw = frame.shape[:2]
        ref_area = float(bw * bh)
        in_idx = np.where(inlier_mask.ravel() > 0)[0]
        pts_f = pts_frame.reshape(-1, 2)[in_idx]
        pts_r = pts_ref.reshape(-1, 2)[in_idx]

        # --- 位置估计："H 提供尺度与旋转 + 内点质心提供位置" ---
        # 为什么不用单纯的投影框：匹配点往往只落在物体上某一块局部纹理
        # （例如杯子上的条纹、书脊上的字），此时单应矩阵给出的框会偏大、位置也偏到纹理一侧。
        # 但 H 的**尺度**是可信的（由匹配点之间的间距决定），
        # 所以正确做法是：用 H 求出物体在当前尺度下有多大，再把这个框套到内点质心上。
        #
        # 尺度不能用 sqrt(|det(H[:2,:2])|) 求：透视变换下行列式不代表长度缩放。
        # 正确做法是把 H 作用到"单位水平/垂直基向量"上量长度：
        #     H = [[a, b, tx], [c, d, ty], [g, h, 1]]
        #     A = [[a, b], [c, d]]
        #     水平尺度 = |A @ [1,0]| / (g*1 + 1)，垂直尺度 = |A @ [0,1]| / (h*1 + 1)
        method = "homography"
        A = H[:2, :2].astype(np.float64)
        g, h_ = float(H[2, 0]), float(H[2, 1])
        col0, col1 = A @ np.array([1.0, 0.0]), A @ np.array([0.0, 1.0])
        s_x = float(np.hypot(*col0)) / max(1e-6, abs(1.0 + g))
        s_y = float(np.hypot(*col1)) / max(1e-6, abs(1.0 + h_))
        s = float(np.sqrt(max(1e-9, s_x * s_y)))
        px_w, px_h = bw * s_x, bh * s_y
        proj_area = px_w * px_h
        cand = None
        if inliers >= 4 and np.isfinite([s_x, s_y]).all() \
                and 0.05 * ref_area <= proj_area <= 12.0 * ref_area \
                and proj_area <= MAX_AREA_RATIO * fw * fh:
            # 修正公式（单位必须统一，早期版本在这里把参考图像素和场景像素混加过）：
            #   物体中心 = 内点质心 + 尺度 × (参考框中心 − 参考内点质心)
            r_cx, r_cy = float(pts_r[:, 0].mean()), float(pts_r[:, 1].mean())
            f_cx, f_cy = float(pts_f[:, 0].mean()), float(pts_f[:, 1].mean())
            cx = f_cx + s_x * ((x + bw * 0.5) - r_cx)
            cy = f_cy + s_y * ((y + bh * 0.5) - r_cy)
            if np.isfinite(cx) and np.isfinite(cy):
                cand = (cx, cy)

        if cand is None:
            # 兜底：直接取内点包围盒并外扩（几何上保守，但一定能给出一个框）
            if len(pts_f) < 3:
                self._dbg(ref.name, "定位失败：内点不足 3 个，无法给出包围盒")
                return None
            bx0, by0 = pts_f.min(axis=0)
            bx1, by1 = pts_f.max(axis=0)
            pad = max(6.0, 0.25 * max(bx1 - bx0, by1 - by0))
            cx, cy = float((bx0 + bx1) * 0.5), float((by0 + by1) * 0.5)
            px_w = max(px_w, float(bx1 - bx0) + 2 * pad)
            px_h = max(px_h, float(by1 - by0) + 2 * pad)
            proj_area = px_w * px_h
            method = "inlier-box"
            self._dbg(ref.name, f"回退内点包围盒 ({px_w:.0f}x{px_h:.0f})")

        px0, py0 = cx - px_w * 0.5, cy - px_h * 0.5
        px1, py1 = cx + px_w * 0.5, cy + px_h * 0.5

        # 尺度合理性（兜底路径可能放大，所以再查一次）
        if not (0.05 * ref_area <= proj_area <= 12.0 * ref_area) \
                or proj_area > MAX_AREA_RATIO * fw * fh:
            self._dbg(ref.name, f"框面积不合理 {proj_area:.0f} vs 参考 {ref_area:.0f}")
            return None
        aspect = px_w / max(1e-3, px_h)
        if not (ref.aspect / ASPECT_TOL < aspect < ref.aspect * ASPECT_TOL):
            self._dbg(ref.name, f"长宽比异常 {aspect:.2f} vs 参考 {ref.aspect:.2f}")
            return None
        self._dbg(ref.name, f"定位方式={method} 框 {px_w:.0f}x{px_h:.0f} "
                            f"尺度={s:.2f} 内点={inliers} 中心=({cx:.0f},{cy:.0f})")

        # 外观二级确认：把投影框内画面裁出来比色彩直方图
        x0 = int(max(0, np.floor(px0)))
        y0 = int(max(0, np.floor(py0)))
        x1 = int(min(fw, np.ceil(px1)))
        y1 = int(min(fh, np.ceil(py1)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        patch = frame[y0:y1, x0:x1]
        hh_f, hs_f = _patch_hists(patch)
        hh_r, hs_r = _patch_hists(ref.image[y:y + bh, x:x + bw])
        corr = 0.5 * (hist_similarity(hh_f, hh_r) + hist_similarity(hs_f, hs_r))
        corr = max(0.0, corr)

        n_ref = max(10, int(INLIER_REF_FRACTION * len(ref.keypoints)))
        n_norm = min(1.0, inliers / n_ref)
        score = SCORE_W[0] * ratio_in + SCORE_W[1] * n_norm + SCORE_W[2] * corr

        cx = float((px0 + px1) * 0.5)
        cy = float((py0 + py1) * 0.5)

        # 几何合理性：中心点必须落在画面（含少量余量）内。
        # 这一步很关键 —— 实测里出现过中心算到 2000+ 像素外的情况
        # （尺度估计被少数异常内点带偏），如果不拦，伺服会朝完全错误的方向转。
        if not (-0.25 * fw <= cx <= 1.25 * fw and -0.25 * fh <= cy <= 1.25 * fh):
            self._dbg(ref.name, f"中心点不合理 ({cx:.0f},{cy:.0f})，丢弃本帧")
            return None

        px_err = (cx - fw * 0.5) / (fw * 0.5)
        # 像素偏差 -> 角度增量（针孔反投影，与渲染器同一个模型）：
        #   Δ角度 = atan(Δ像素 / 焦距)，小角度下近似等于线性关系，大角度下必须用 tan。
        # 自洽性很重要：渲染器、识别器、伺服闭环三者必须用同一个投影模型，
        # 否则会出现"物体稍微偏离中心就认不出来"这类极难排查的问题。
        dx_px = cx - fw * 0.5
        f_h = (fw * 0.5) / math.tan(math.radians(max(1e-3, hfov_deg) * 0.5))
        d_deg = math.degrees(math.atan(dx_px / f_h)) if f_h > 1e-6 \
            else px_err * (hfov_deg * 0.5)
        angle = cam_deg + move_sign * d_deg

        self._dbg(ref.name, f"命中 inliers={inliers} ratio={ratio_in:.2f} "
                            f"hist={corr:.2f} score={score:.2f}")
        return Detection(
            name=ref.name, score=float(score), px_center=(cx, cy),
            bbox=(x0, y0, x1 - x0, y1 - y0), angle_deg=float(angle),
            px_err_norm=float(px_err), inliers=inliers, matches=len(pts_ref),
            hist_corr=float(corr), sign=move_sign,
        )

    def _dbg(self, name: str, msg: str) -> None:
        if self.debug:
            print(f"    [dbg] {name}: {msg}")


def _patch_hists(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from .registry import color_signature
    return color_signature(patch)


# --------------------------------------------------------------------------
# 合成场景（无硬件自检用）
# --------------------------------------------------------------------------
@dataclass
class SceneItem:
    name: str
    angle_deg: float      # 物品在真实世界中的方位角（度）
    width_mm: float
    height_mm: float
    hue: int              # OpenCV 色相 0~179
    sat: int
    val: int
    pattern: str          # stripes / checks / dots / solid / band / text
    registered: bool = True


def make_items() -> list[SceneItem]:
    """一套颜色与纹理都区分得开的默认场景物品（含一件故意不登记的）。

    选色原则：物体与浅色背景要有足够**亮度对比**。
    早期版本给饮料瓶配了浅黄绿（hue=55）压在米白背景上，对比度太低，
    场景里只能提出 48 个 SIFT 关键点（水杯有 133 个），导致识别不稳定。
    换成蓝绿色塑料 + 白色标签条，纹理对比立刻够用 —— 这也符合真实场景。
    """
    return [
        SceneItem("水杯",   angle_deg=-18.0, width_mm=85, height_mm=115,
                  hue=110, sat=200, val=230, pattern="stripes"),
        SceneItem("书",     angle_deg=12.0, width_mm=140, height_mm=95,
                  hue=15, sat=190, val=200, pattern="band"),
        SceneItem("饮料瓶", angle_deg=35.0, width_mm=70, height_mm=160,
                  hue=95, sat=220, val=150, pattern="bottle"),
        SceneItem("雨伞",   angle_deg=52.0, width_mm=90, height_mm=150,
                  hue=150, sat=120, val=170, pattern="checks", registered=False),
    ]


class Scene:
    """把"物品方位角 + 相机方位角"渲染成一张图，用来在没有硬件时验证整条链路。

    渲染规则就是针孔模型：物体在画面中的水平偏移 = (物品角 - 相机角) * 焦距。
    与真实转台一致：相机转到物品角度时，物品正好在画面正中。
    """

    def __init__(self, width: int = 640, height: int = 480, hfov_deg: float = 53.0,
                 distance_m: float = 0.60, seed: int = 7,
                 items: list[SceneItem] | None = None) -> None:
        self.w = width
        self.h = height
        self.hfov_deg = hfov_deg
        self.f = (width * 0.5) / np.tan(np.radians(hfov_deg) * 0.5)  # 焦距（像素）
        self.distance_m = distance_m
        self.items = items if items is not None else make_items()
        self.registered = {i.name for i in self.items if i.registered}
        self.noise_seed = seed
        self._noise_i = 0

        rng = np.random.default_rng(seed)
        self._tex = rng.integers(0, 26, size=(height, width), dtype=np.uint8)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = width * 0.5, height * 0.5
        r = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)
        self._vignette = np.clip(1.0 - 0.22 * r ** 2, 0.55, 1.0)[..., None]
        self._cache: dict[str, tuple[float, np.ndarray]] = {}

    # -- 投影换算（必须与真实针孔相机一致）--
    # 关键修复：早期版本用 "像素偏移 = 角度差 × 焦距"，但焦距本身是按
    # 水平视场角推出来的 f = (W/2)/tan(HFOV/2)，两者放在一起并不自洽 ——
    # 结果是这台"合成相机"的实际视场只有大约 1°：
    # 相机偏离物体 2° 时，物体就被画到 1284 像素外（正确值应是 22 像素）。
    # 表现就是"物体稍微偏离画面中心就完全认不出来"。
    # 正确关系是针孔投影：Δpx = f · tan(Δ角度)，反解 Δ角度 = atan(Δpx / f)。
    def px_offset(self, d_deg: float) -> float:
        return self.f * math.tan(math.radians(d_deg))

    def angle_of_offset(self, d_px: float) -> float:
        return math.degrees(math.atan(d_px / self.f))

    def px_per_m(self) -> float:
        return self.f / self.distance_m

    def object_px(self, it: SceneItem) -> tuple[int, int]:
        p = self.px_per_m()
        return max(8, int(it.width_mm / 1000.0 * p)), max(8, int(it.height_mm / 1000.0 * p))

    def px_center(self, it: SceneItem, cam_deg: float) -> tuple[float, float]:
        """物品在画面中的像素中心（针孔投影）。"""
        return (self.w * 0.5 + self.px_offset(it.angle_deg - cam_deg), self.h * 0.58)

    def in_frame(self, it: SceneItem, cam_deg: float, margin_px: float = 0.0) -> bool:
        bw, bh = self.object_px(it)
        cx, cy = self.px_center(it, cam_deg)
        return (cx - bw * 0.5 + margin_px > 0 and cx + bw * 0.5 - margin_px < self.w
                and cy - bh * 0.5 > -bh and cy + bh * 0.5 < self.h + bh)

    # -- 渲染 --
    def render(self, cam_deg: float, *, brightness: float = 1.0,
               ambient_hue: int = 0) -> np.ndarray:
        tex = self._tex.astype(np.int16)
        bg = np.full((self.h, self.w, 3), 236, np.uint8)
        bg[:, :, 0] = np.clip(232 + tex // 2, 0, 255).astype(np.uint8)
        bg[:, :, 1] = np.clip(236 - tex // 3, 0, 255).astype(np.uint8)
        bg[:, :, 2] = np.clip(240 - tex // 4, 0, 255).astype(np.uint8)
        cv2.line(bg, (0, int(self.h * 0.94)), (self.w, int(self.h * 0.94)), (200, 205, 210), 3)

        for it in sorted(self.items, key=lambda i: i.width_mm, reverse=True):
            if not self.in_frame(it, cam_deg, margin_px=-6):
                continue
            patch = self._patch(it)
            ph, pw = patch.shape[:2]
            cx, cy = self.px_center(it, cam_deg)
            x0, y0 = int(round(cx - pw * 0.5)), int(round(cy - ph * 0.5))
            x1, y1 = x0 + pw, y0 + ph
            sx0, sy0 = max(0, -x0), max(0, -y0)
            sx1, sy1 = pw - max(0, x1 - self.w), ph - max(0, y1 - self.h)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            roi = bg[max(0, y0):min(self.h, y1), max(0, x0):min(self.w, x1)]
            roi[:] = patch[sy0:sy1, sx0:sx1]

        out = bg.astype(np.float32) * self._vignette
        if ambient_hue:
            hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + ambient_hue) % 180
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
        out = np.clip(out * float(brightness), 0, 255).astype(np.uint8)
        # 稀疏传感器噪声：约 30% 像素加 ±3 的偏置，模拟真实相机的散粒噪声。
        # 用固定偏移量而不是真随机：演示和自检必须可复现，
        # 每次渲染重新抽随机数会让 SIFT 结果在阈值附近抖动。
        rng = np.random.default_rng(seed_from_name(f"noise{self.noise_seed}") + self._noise_i)
        self._noise_i += 1
        noise = np.zeros(out.shape[:2], np.uint8)
        mask = rng.random(out.shape[:2]) < 0.3
        noise[mask] = rng.integers(0, 7, size=int(mask.sum()), dtype=np.uint8)
        shifted = out.astype(np.int16) + noise[..., None].astype(np.int16) - 3
        return np.clip(shifted, 0, 255).astype(np.uint8)

    def _patch(self, it: SceneItem) -> np.ndarray:
        cached = self._cache.get(it.name)
        if cached is not None and abs(cached[0] - it.angle_deg) < 1e-9:
            return cached[1]
        bw, bh = self.object_px(it)
        patch = _draw_object(it, bw, bh)
        self._cache[it.name] = (it.angle_deg, patch)
        return patch


def _draw_object(it: SceneItem, bw: int, bh: int) -> np.ndarray:
    """画一个有纹理细节的"物品"，纹理要够丰富，SIFT 才有东西可提。"""
    patch = np.zeros((bh, bw, 3), np.uint8)
    base = np.uint8([[[it.hue, it.sat, it.val]]])
    bgr = tuple(int(v) for v in cv2.cvtColor(base, cv2.COLOR_HSV2BGR)[0, 0])
    patch[:] = bgr

    rng = np.random.default_rng(seed_from_name(it.name))
    if it.pattern == "stripes":
        step = max(5, bh // 9)
        for y in range(0, bh, step * 2):
            patch[y:y + step, :] = np.clip(
                np.array(bgr, np.int16) + 45, 0, 255).astype(np.uint8)
    elif it.pattern == "checks":
        step = max(6, bw // 6)
        for y in range(0, bh, step):
            for x in range(0, bw, step):
                if (x // step + y // step) % 2 == 0:
                    patch[y:y + step, x:x + step] = np.clip(
                        np.array(bgr, np.int16) - 55, 0, 255).astype(np.uint8)
    elif it.pattern == "bottle":
        # 饮料瓶：瓶盖 + 两条白色标签条 + 标签上的文字线
        # 用大块高对比结构，保证小尺寸下 SIFT 也能提出足够关键点
        cap_h = max(4, int(bh * 0.09))
        patch[0:cap_h, int(bw * 0.18):int(bw * 0.82)] = (70, 70, 75)
        lab0, lab1 = int(bh * 0.30), int(bh * 0.64)
        patch[lab0:lab1, :] = (238, 240, 242)
        line_h = max(2, bh // 40)
        for k in range(3):
            y = lab0 + int((lab1 - lab0) * (0.22 + 0.26 * k))
            patch[y:y + line_h, int(bw * 0.16):int(bw * 0.84)] = (45, 45, 50)
        patch[int(bh * 0.70):int(bh * 0.76), :] = (200, 60, 55)
        # 瓶身竖向暗纹，增加可匹配的梯度
        for x in range(0, bw, max(6, bw // 5)):
            patch[:, x:x + max(1, bw // 22)] = np.clip(
                np.array(bgr, np.int16) - 40, 0, 255).astype(np.uint8)
    elif it.pattern == "dots":
        for _ in range(max(6, (bw * bh) // 900)):
            cx, cy = int(rng.integers(4, max(5, bw - 4))), int(rng.integers(4, max(5, bh - 4)))
            cv2.circle(patch, (cx, cy), max(2, bw // 14),
                       tuple(int(v) for v in np.clip(np.array(bgr, np.int16) + 60, 0, 255)),
                       -1)
    elif it.pattern == "band":
        h0 = int(bh * 0.28)
        patch[h0:h0 + max(6, bh // 6), :] = (40, 40, 45)
        cv2.line(patch, (int(bw * 0.15), int(bh * 0.62)),
                 (int(bw * 0.85), int(bh * 0.62)), (30, 30, 30), max(2, bh // 22))
        patch[int(bh * 0.70):int(bh * 0.78), int(bw * 0.2):int(bw * 0.8)] = (90, 90, 95)
    else:  # solid 也要给点纹理
        patch[:] = np.clip(np.array(bgr, np.int16) + rng.integers(-8, 9), 0, 255)

    # 描边 + 内部纹理颗粒，避免纯色区域
    cv2.rectangle(patch, (0, 0), (bw - 1, bh - 1), (25, 25, 28), max(1, bw // 40))
    grain = rng.integers(-14, 15, size=(bh, bw, 1))
    return np.clip(patch.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def render_reference(scene: Scene, item: SceneItem, margin: float = 0.18) -> np.ndarray:
    """渲染单件参考图：物品居中，四周留白，用于登记。"""
    bw, bh = scene.object_px(item)
    pad = int(max(bw, bh) * margin) + 12
    canvas = np.full((bh + pad * 2, bw + pad * 2, 3), 234, np.uint8)
    tex = scene._tex[:canvas.shape[0], :canvas.shape[1], None].astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + tex, 0, 255).astype(np.uint8)
    patch = _draw_object(item, bw, bh)
    canvas[pad:pad + bh, pad:pad + bw] = patch
    return canvas


# --------------------------------------------------------------------------
# 合成场景一键生成 + 登记
# --------------------------------------------------------------------------
def build_synthetic_registry(data_dir: str | Path, scene: Scene | None = None,
                             rel_prefix: str = "data") -> list[SceneItem]:
    """生成参考图并写入 items.json，返回场景物品列表。

    rel_prefix: 台账里记录参考图所用的相对路径前缀（相对项目根目录）。

    target_deg 的取值说明（很重要，关系到演示效果）：
        单轴转台方案里，"取件位"就是物品自身的方位角（相机转过去对准它）；
        但"上层放置位"是床面上的一个固定角度。所以台账里的 target_deg
        指的是**放置位角度**，抬升阶段的转角 = target_deg − 物品方位角。
        早期版本直接把 target_deg 设成物品自身方位角，结果抬升转角恒为 0，
        平衡控制完全没被用上（液面峰值只有 0.27°，演示毫无说服力）。
    """
    from .registry import DEFAULT_THRESHOLDS, Item

    # 床面放置位的角度：刻意与物品方位角拉开，让搬运真正考验防洒能力
    delivery_deg = {"水杯": 24.0, "书": -28.0, "饮料瓶": 6.0}

    data_dir = Path(data_dir)
    ref_dir = data_dir / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    scene = scene or Scene()
    reg = Registry(data_dir / "items.json")

    for it in scene.items:
        if not it.registered:
            continue
        ref_img = render_reference(scene, it)
        ref_path = ref_dir / f"{it.name}.jpg"
        imwrite_unicode(ref_path, ref_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        sift = _make_sift()
        gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        sig = Registry.build_signature(ref_path, n_keypoints=0 if desc is None else len(kps))
        reg.add(Item(
            name=it.name,
            ref_image=str(Path(rel_prefix) / "ref" / f"{it.name}.jpg"),
            target_deg=float(delivery_deg.get(it.name, it.angle_deg)),
            sig=sig,
            thresholds=dict(DEFAULT_THRESHOLDS),
        ))
    reg.save()
    return scene.items
