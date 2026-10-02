"""图像读写工具：解决 Windows 上 OpenCV 打不开中文路径的问题。

问题现象：
    cv2.imread(r"G:\\...\\data\\ref\\书.jpg") 返回 None，并打印
    "can't open/read file: check file path/integrity"
    文件明明存在，用资源管理器能正常打开。

原因：
    OpenCV 在 Windows 上用系统的 ANSI 代码页（中文系统是 GBK）去打开文件，
    而 Python 传进去的是 UTF-8 字符串，路径里的中文就变成了乱码。

解决：
    用 Python 自己读字节（np.fromfile 支持 Unicode 路径），
    再交给 cv2.imdecode 解码；写文件同理走 cv2.imencode + tofile。
    这样中文名物品（水杯、书、饮料瓶）的参考图才能正常读写。

这是本项目在真实中文 Windows 环境里踩到的坑，实机迁移时同样会遇到。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """等价于 cv2.imread，但支持中文/任意 Unicode 路径。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray,
                    params: list[int] | None = None) -> bool:
    """等价于 cv2.imwrite，但支持中文/任意 Unicode 路径。"""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    ext = p.suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    try:
        buf.tofile(str(p))
        return True
    except OSError:
        return False
