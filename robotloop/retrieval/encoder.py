"""编码器：文本语义向量（CLIP）与轨迹向量。

- 文本：sentence-transformers 的 clip-ViT-B-32（512 维）—— 与 RobotLoop
  既有 Ray 流水线 / 检索 API 用的模型一致，保证向量空间同源。
- 无 GPU/无网络环境（CI、演示）：自动降级为确定性的 HashEncoder，
  接口完全一致，仅召回质量下降，链路可端到端跑通。
- 轨迹向量：动作序列的统计摘要（mean/std/min/max 拼接后 L2 归一化）。
  生产可替换为 TCN/Transformer 轨迹编码器，接口不变。
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import numpy as np

TEXT_DIM = 512


class HashEncoder:
    """离线降级编码器：token 哈希 trick 的 512 维词袋向量。

    确定性强，同词必同向量 —— 足够支撑"红色方块/ALOHA/抓取"这类
    关键词共现检索的链路验证，但不具备 CLIP 的泛化语义。
    """

    dim = TEXT_DIM

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokenize(text)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        for a, b in zip(tokens, tokens[1:]):
            h = int(hashlib.md5(f"{a}_{b}".encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        text = text.lower()
        en = re.findall(r"[a-z0-9]+", text)
        zh = re.findall(r"[一-鿿]", text)
        zh_bi = [a + b for a, b in zip(zh, zh[1:])]
        return en + zh + zh_bi


class CLIPEncoder:
    """sentence-transformers CLIP 文本编码（与 RobotLoop 平台同款模型）。"""

    dim = TEXT_DIM

    def __init__(self, model_path: str = "clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_path)

    def encode(self, text: str) -> np.ndarray:
        v = np.asarray(self._model.encode(text), dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


_default_encoder: Optional[object] = None


def get_text_encoder(prefer_clip: bool = True, model_path: str = "clip-ViT-B-32"):
    """优先 CLIP；无依赖/无权重时自动降级 HashEncoder（打日志不报错）。"""
    global _default_encoder
    if _default_encoder is not None:
        return _default_encoder
    if prefer_clip:
        try:
            _default_encoder = CLIPEncoder(model_path)
            return _default_encoder
        except Exception as e:  # noqa: BLE001
            import logging

            logging.getLogger("robotloop.retrieval").warning(
                "CLIP 不可用（%s），降级为 HashEncoder", e
            )
    _default_encoder = HashEncoder()
    return _default_encoder


def encode_trajectory(actions: Sequence[Sequence[float]]) -> np.ndarray:
    """动作序列 → 轨迹向量（统计摘要 + 归一化）。

    输入: [T, D] 动作序列；输出: 4D 维向量（mean|std|min|max）。
    """
    m = np.asarray(actions, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    if m.shape[0] == 0:
        return np.zeros(4, dtype=np.float32)
    feat = np.concatenate([m.mean(0), m.std(0), m.min(0), m.max(0)]).astype(np.float32)
    n = np.linalg.norm(feat)
    return feat / n if n > 0 else feat


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
