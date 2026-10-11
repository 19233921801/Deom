# -*- coding: utf-8 -*-
"""
memory.py —— 分层记忆模块（面试核心之二）

原版 BabyAGI 把所有历史结果一股脑拼进 prompt，上下文迅速膨胀、记忆互相污染。
本模块拆成两层：
  短期工作记忆（short_term）  ：只保留最近 N 轮迭代上下文，随运行即时更新；
  长期向量记忆（Chroma）      ：对任务结果做"文本分块 -> 向量化存储"，
                                召回时做相似度阈值过滤 + 数量上限，只把高相关片段注入 prompt。

召回质量靠两层闸门保证（对应"向量检索优化"这个改造点）：
  1. n_results 限制数量；
  2. 相似度阈值（1 - 距离）过滤低相关记忆。
"""
import chromadb

from config import (
    CHUNK_SIZE,
    MEMORY_COLLECTION,
    MEMORY_DB_PATH,
    MEMORY_TOP_K,
    SHORT_TERM_SIZE,
    SIM_THRESHOLD,
)


class MemoryStore:
    def __init__(self, db_path=MEMORY_DB_PATH, collection_name=MEMORY_COLLECTION):
        # Chroma 本地持久化：数据落盘到 ./memory_db，跨运行保留
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        # 短期工作记忆：最近若干轮 (任务, 结果) 的列表
        self.short_term = []

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_text(text, size=CHUNK_SIZE):
        """按固定长度把长文本切成块，保证单条向量粒度可控，便于召回定位。"""
        if not text:
            return []
        if len(text) <= size:
            return [text]
        return [text[i:i + size] for i in range(0, len(text), size)]

    def save_memory(self, task_id, task_content, result):
        """写入长期向量记忆 + 更新短期工作记忆。"""
        doc = f"任务: {task_content} | 结果: {result}"
        # 分块后逐块 upsert，块级 id 避免重复插入
        for i, chunk in enumerate(self._chunk_text(doc)):
            cid = f"{task_id}_{i}_{abs(hash(chunk))}"
            self.collection.upsert(documents=[chunk], ids=[cid])

        # 短期记忆：追加并裁剪，防止无界膨胀
        self.short_term.append({"task": task_content, "result": result})
        if len(self.short_term) > SHORT_TERM_SIZE:
            self.short_term.pop(0)

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------
    def recall_memory(self, query, top_k=MEMORY_TOP_K, threshold=SIM_THRESHOLD):
        """
        召回高相关记忆。
        返回 (docs_text, best_score)：
          - docs_text：通过阈值过滤后的历史片段拼串，用于注入 prompt；
          - best_score：最高相似度（0~1），用于任务优先级计算中的 memory_score。
        """
        count = self.collection.count()
        if count == 0:
            return "", 0.0

        # Chroma 距离越小越相似；这里用 L2，相似度近似 = 1 - distance
        n = min(top_k, count)
        res = self.collection.query(query_texts=[query], n_results=n)

        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]

        filtered, best = [], 0.0
        for doc, dist in zip(docs, dists):
            sim = round(1 - float(dist), 3)
            if sim >= threshold:
                filtered.append(doc)
                best = max(best, sim)

        return "\n".join(filtered), best

    def short_term_context(self):
        """短期记忆转 prompt 文本（按最新在前）。"""
        if not self.short_term:
            return ""
        lines = []
        for item in reversed(self.short_term):
            lines.append(f"- 任务：{item['task']} → 结果：{item['result']}")
        return "\n".join(lines)

    def stats(self):
        return {"long_term_docs": self.collection.count(), "short_term_size": len(self.short_term)}
