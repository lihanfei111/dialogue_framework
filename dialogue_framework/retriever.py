# -*- coding: utf-8 -*-
"""动态 few-shot 召回器(支持单路 BGE / 多路 / 静态 三种模式)。

RETRIEVAL_MODE 控制使用哪种召回器:
  "bge"         — 单路 BGE embedding 召回(baseline)
  "multi_route" — BGE + BM25 + Rule 三路合并
  "static"      — 不召回,返回空列表,prompts 回退到静态 CLASSIFY_FEW_SHOTS
"""
import json
import os
from collections import defaultdict

import numpy as np

import config
from embedder import CachedEmbedder


# ============================================================================
# 基类
# ============================================================================

class BaseRetriever:
    def __init__(self, pool_path=None):
        pool_path = pool_path or config.FEWSHOT_POOL_PATH
        with open(pool_path, encoding="utf-8") as f:
            self.pool = json.load(f)
        self.texts = [e["text"] for e in self.pool]

    def _interleave_by_intent(self, idx_list, k):
        """候选按意图分桶 → round-robin 交织取 k 条。"""
        buckets = defaultdict(list)
        for i in idx_list:
            buckets[self.pool[i]["output"]["intent"]].append(i)
        intent_order = sorted(buckets.keys())
        result, pos = [], 0
        while len(result) < k:
            took = False
            for intent in intent_order:
                if pos < len(buckets[intent]):
                    result.append(buckets[intent][pos])
                    took = True
                    if len(result) >= k:
                        break
            if not took:
                break
            pos += 1
        return [self.pool[i] for i in result]


# ============================================================================
# Route 1:BGE 向量(使用 CachedEmbedder 保留磁盘缓存)
# ============================================================================

class BGERetriever(BaseRetriever):
    def __init__(self, pool_path=None, vectors_path=None, model_name=None):
        super().__init__(pool_path)
        vectors_path = vectors_path or config.FEWSHOT_VECTORS_PATH
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(
                f"向量库 {vectors_path} 不存在。请先跑: python _build_vectors.py"
            )
        self.matrix = np.load(vectors_path)
        if self.matrix.shape[0] != len(self.pool):
            raise RuntimeError(
                f"向量库 ({self.matrix.shape[0]} 条) 与召回库 ({len(self.pool)} 条) 不匹配。"
                f"请重新跑: python _build_vectors.py"
            )
        self._embedder: CachedEmbedder | None = None

    def _get_embedder(self) -> CachedEmbedder:
        if self._embedder is None:
            self._embedder = CachedEmbedder()
        return self._embedder

    def _encode(self, texts: list) -> np.ndarray:
        return self._get_embedder().encode(texts)

    def prewarm(self, texts: list) -> None:
        """批量预编码,确保后续 retrieve() 全为缓存命中。"""
        self._get_embedder().encode(texts)

    def topn_indices(self, query: str, n: int) -> list:
        if not query.strip():
            return list(range(min(n, len(self.pool))))
        qv = self._encode([query])[0]
        sims = self.matrix @ qv
        return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:n]

    def retrieve(self, query: str, k: int = None, candidate_pool: int = None) -> list:
        k = k or config.FEWSHOT_K
        candidate_pool = candidate_pool or (k * 3)
        return self._interleave_by_intent(self.topn_indices(query, candidate_pool), k)


# ============================================================================
# Route 2:BM25 词频召回
# ============================================================================

class BM25Retriever(BaseRetriever):
    def __init__(self, pool_path=None):
        super().__init__(pool_path)
        from rank_bm25 import BM25Okapi
        import jieba
        self._tokenize = lambda s: list(jieba.cut(s))
        corpus_tokens = [self._tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(corpus_tokens)

    def topn_indices(self, query: str, n: int) -> list:
        if not query.strip():
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]


# ============================================================================
# Route 3:规则关键词匹配
# ============================================================================

class RuleRetriever(BaseRetriever):
    """强信号词命中召回库示例。误差极小,专治 BGE 在边缘 case 召回偏移。"""
    RULES = [
        ({"ref", "撤单", "撤了", "撤掉", "撤个"}, "bond_inquiry"),
        ({"人工", "客服", "请求对象", "改请求", "换请求"}, "transfer_human"),
        ({"在挂", "挂着", "进行中", "还在挂"}, "bond_query"),
        ({"成交记录", "成了哪些", "成交情况", "已完成"}, "bond_query"),
        ({"选", "推荐", "筛选", "筛一下"}, "bond_selection"),
    ]

    def topn_indices(self, query: str, n: int) -> list:
        if not query.strip():
            return []
        hit_intents = []
        for keywords, intent in self.RULES:
            if any(k in query for k in keywords):
                hit_intents.append(intent)
        if not hit_intents:
            return []
        return [i for i, e in enumerate(self.pool)
                if e["output"]["intent"] in hit_intents][:n]


# ============================================================================
# 多路合并器
# ============================================================================

class MultiRouteRetriever:
    """合并 BGE + BM25 + Rule 三路召回,去重后意图均衡交织。"""
    def __init__(self):
        self.bge = BGERetriever()
        self.bm25 = BM25Retriever()
        self.rule = RuleRetriever()
        self.pool = self.bge.pool  # 借 BGE 的 pool 做最终交织

    def prewarm(self, texts: list) -> None:
        self.bge.prewarm(texts)

    def retrieve(self, query: str, k: int = None) -> list:
        k = k or config.FEWSHOT_K
        per_route = max(4, k)
        idx_bge = self.bge.topn_indices(query, per_route)
        idx_bm25 = self.bm25.topn_indices(query, per_route)
        idx_rule = self.rule.topn_indices(query, per_route)
        # 合并去重,BGE 优先,其余补充
        seen, merged = set(), []
        for lst in (idx_bge, idx_bm25, idx_rule):
            for i in lst:
                if i not in seen:
                    seen.add(i)
                    merged.append(i)
        return self.bge._interleave_by_intent(merged, k)


# ============================================================================
# 静态兜底
# ============================================================================

class StaticRetriever:
    """不召回,返回空列表 → build_classify_messages 回退到静态 CLASSIFY_FEW_SHOTS。"""
    def prewarm(self, texts: list) -> None:
        pass

    def retrieve(self, query: str, k: int = None) -> list:
        return []


# ============================================================================
# 单例 + 工厂
# ============================================================================

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        mode = config.RETRIEVAL_MODE
        if mode == "bge":
            _retriever = BGERetriever()
        elif mode == "multi_route":
            _retriever = MultiRouteRetriever()
        elif mode == "static":
            _retriever = StaticRetriever()
        else:
            raise ValueError(f"未知 RETRIEVAL_MODE: {mode!r},可选: bge / multi_route / static")
    return _retriever


# 旧 API 兼容
FewShotRetriever = BGERetriever
