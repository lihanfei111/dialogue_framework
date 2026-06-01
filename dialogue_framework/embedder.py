# -*- coding: utf-8 -*-
"""
Embedding 层，带磁盘缓存。

CachedEmbedder 把已编码向量以 JSONL 格式持久化到磁盘（key=sha1(model_tag|text)）。
命中缓存时只做 dict 查找，不调用模型，天然线程安全，无需锁。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

import config


class CachedEmbedder:
    """对 SentenceTransformer 加一层内存 + 磁盘缓存。

    首次遇到未见过的文本时调用模型推理并写盘；
    已见过的文本直接从内存 dict 返回，不碰模型。
    """

    def __init__(self, cache_path: str = None):
        local = config.EMBEDDING_LOCAL_PATH
        self._model_id = local if (local and Path(local).is_dir()) else config.EMBEDDING_MODEL_NAME
        self._model = None  # 懒加载

        self._cache_path = Path(cache_path or config.FEWSHOT_EMBED_CACHE_PATH)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._tag = self._model_id  # 区分不同模型的缓存 key
        self._cache: dict[str, np.ndarray] = {}
        self.dim: int = 0
        self._load()

    # ------------------------------------------------------------------
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_id)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def _key(self, text: str) -> str:
        h = hashlib.sha1()
        h.update(self._tag.encode("utf-8"))
        h.update(b"|")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _load(self):
        """启动时从磁盘恢复缓存到内存。"""
        if not self._cache_path.exists():
            return
        with self._cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._cache[obj["k"]] = np.array(obj["v"], dtype=np.float32)
                except Exception:
                    continue
        if self._cache and self.dim == 0:
            self.dim = next(iter(self._cache.values())).shape[0]

    def _persist(self, new_items: dict[str, np.ndarray]):
        """把新条目追加写盘（append-only，不改写已有内容）。"""
        if not new_items:
            return
        with self._cache_path.open("a", encoding="utf-8") as f:
            for k, v in new_items.items():
                f.write(json.dumps({"k": k, "v": v.tolist()}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """批量编码。命中缓存则跳过推理，只对未见过的文本调用模型。"""
        if not texts:
            return np.zeros((0, max(self.dim, 1)), dtype=np.float32)

        keys = [self._key(t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_vecs = self._get_model().encode(
                missing_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            if self.dim == 0:
                self.dim = new_vecs.shape[1]
            new_items: dict[str, np.ndarray] = {}
            for j, i in enumerate(missing_idx):
                self._cache[keys[i]] = new_vecs[j]
                new_items[keys[i]] = new_vecs[j]
            self._persist(new_items)

        return np.stack([self._cache[k] for k in keys])

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


__all__ = ["CachedEmbedder"]
