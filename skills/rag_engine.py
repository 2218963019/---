"""
RAG检索增强引擎
纯Python实现，零重依赖，开箱即用。
支持：文档分块 → TF-IDF嵌入 → 向量存储 → 语义检索
预留：API嵌入接口，可无缝切换到LLM Embedding API
"""

import re
import math
import json
import os
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter
from skills.doc_parser import DocParserSkill


# ============================================================
# 1. 文档分块器
# ============================================================

@dataclass
class Chunk:
    """文本块"""
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: int = -1


class TextChunker:
    """
    文本分块器，支持三种策略：
    - fixed: 按固定字符数切分
    - paragraph: 按段落切分
    - section: 按章节关键词切分（推荐用于项目申报书）
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_fixed(self, text: str, source: str = "") -> list:
        """按固定字符数切分，带重叠"""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(
                    content=piece,
                    metadata={"source": source, "strategy": "fixed"},
                    chunk_id=idx,
                ))
                idx += 1
            start += self.chunk_size - self.overlap
        return chunks

    def split_paragraph(self, text: str, source: str = "") -> list:
        """按段落切分，短段落合并"""
        raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current = ""
        idx = 0
        for para in raw_paragraphs:
            if len(current) + len(para) + 1 <= self.chunk_size:
                current = (current + "\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(Chunk(
                        content=current,
                        metadata={"source": source, "strategy": "paragraph"},
                        chunk_id=idx,
                    ))
                    idx += 1
                current = para
        if current:
            chunks.append(Chunk(
                content=current,
                metadata={"source": source, "strategy": "paragraph"},
                chunk_id=idx,
            ))
        return chunks

    def split_section(self, text: str, section_keywords: list, source: str = "") -> list:
        """按章节关键词切分，每个章节作为一个chunk"""
        lines = text.split("\n")
        sections = {}
        current_key = "前言"
        sections[current_key] = []

        for line in lines:
            for kw in section_keywords:
                if kw in line:
                    current_key = kw
                    if current_key not in sections:
                        sections[current_key] = []
                    break
            sections.setdefault(current_key, []).append(line)

        chunks = []
        idx = 0
        for key, content_lines in sections.items():
            content = "\n".join(content_lines).strip()
            if not content:
                continue
            if len(content) > self.chunk_size:
                sub_chunks = self.split_fixed(content, source)
                for sc in sub_chunks:
                    sc.metadata["section"] = key
                    sc.metadata["strategy"] = "section"
                    sc.chunk_id = idx
                    chunks.append(sc)
                    idx += 1
            else:
                chunks.append(Chunk(
                    content=content,
                    metadata={"source": source, "section": key, "strategy": "section"},
                    chunk_id=idx,
                ))
                idx += 1
        return chunks


# ============================================================
# 2. 嵌入器（TF-IDF纯Python实现 + API接口预留）
# ============================================================

class TFIDFEmbedder:
    """
    纯Python TF-IDF嵌入器，零外部依赖。
    适合中文项目文档的语义表示。
    """

    def __init__(self):
        self._vocab: dict = {}
        self._idf: dict = {}
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> list:
        """
        中文分词：按字符bigram + 关键词提取。
        纯Python实现，无需jieba。
        """
        text = re.sub(r"[^\u4e00-\u9fff\w]", " ", text)
        tokens = []
        chars = re.findall(r"[\u4e00-\u9fff]+", text)
        for segment in chars:
            for i in range(len(segment)):
                tokens.append(segment[i])
                if i + 1 < len(segment):
                    tokens.append(segment[i:i + 2])
                if i + 2 < len(segment):
                    tokens.append(segment[i:i + 3])
        words = re.findall(r"[a-zA-Z0-9]+", text)
        tokens.extend(w.lower() for w in words)
        return tokens

    def fit(self, documents: list) -> "TFIDFEmbedder":
        """
        基于文档集合构建词汇表和IDF。

        Args:
            documents: 文本列表
        """
        n = len(documents)
        doc_freq = Counter()
        vocab_set = set()

        for doc in documents:
            tokens = set(self._tokenize(doc))
            vocab_set.update(tokens)
            doc_freq.update(tokens)

        self._vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}
        self._idf = {}
        for word, df in doc_freq.items():
            self._idf[word] = math.log((n + 1) / (df + 1)) + 1

        self._fitted = True
        return self

    def embed(self, text: str) -> list:
        """将文本转换为TF-IDF向量"""
        if not self._fitted:
            raise RuntimeError("请先调用fit()训练词汇表")

        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1

        vector = [0.0] * len(self._vocab)
        for word, count in tf.items():
            if word in self._vocab:
                idx = self._vocab[word]
                vector[idx] = (count / total) * self._idf.get(word, 1.0)
        return vector

    def embed_batch(self, texts: list) -> list:
        """批量嵌入"""
        return [self.embed(t) for t in texts]

    @property
    def dim(self) -> int:
        return len(self._vocab)


class APIEmbedder:
    """
    LLM API嵌入器（预留接口）。
    支持OpenAI兼容API（智谱/通义/DeepSeek等）。
    """

    def __init__(self, api_key: str, model: str = "embedding-3", base_url: str = ""):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://open.bigmodel.cn/api/paas/v4"
        self._dim: Optional[int] = None

    def embed(self, text: str) -> list:
        """调用API获取嵌入向量"""
        import requests
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"][0]["embedding"]
        if not self._dim:
            self._dim = len(data)
        return data

    def embed_batch(self, texts: list) -> list:
        """批量嵌入"""
        return [self.embed(t) for t in texts]

    @property
    def dim(self) -> int:
        return self._dim or 0


# ============================================================
# 3. 向量存储与检索
# ============================================================

def _cosine_similarity(a: list, b: list) -> float:
    """纯Python余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class RetrievalResult:
    """检索结果项"""
    chunk: Chunk
    score: float


class VectorStore:
    """
    内存向量存储，支持增删查。
    纯Python实现，无需外部数据库。
    """

    def __init__(self):
        self._chunks: list = []
        self._vectors: list = []

    def add(self, chunks: list, vectors: list) -> None:
        """添加文档块和对应向量"""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks({len(chunks)})与vectors({len(vectors)})数量不匹配")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query_vector: list, top_k: int = 5) -> list:
        """
        语义检索，返回top_k最相关的文档块。

        Args:
            query_vector: 查询向量
            top_k: 返回数量

        Returns:
            RetrievalResult列表，按相似度降序
        """
        if not self._vectors:
            return []

        scores = []
        for i, vec in enumerate(self._vectors):
            sim = _cosine_similarity(query_vector, vec)
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            results.append(RetrievalResult(chunk=self._chunks[idx], score=score))
        return results

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()

    @property
    def size(self) -> int:
        return len(self._chunks)

    def save(self, path: str) -> None:
        """持久化存储到JSON文件"""
        data = {
            "chunks": [
                {"content": c.content, "metadata": c.metadata, "chunk_id": c.chunk_id}
                for c in self._chunks
            ],
            "vectors": self._vectors,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从JSON文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = [
            Chunk(content=d["content"], metadata=d["metadata"], chunk_id=d["chunk_id"])
            for d in data["chunks"]
        ]
        self._vectors = data["vectors"]


# ============================================================
# 4. RAG引擎主类
# ============================================================

class RAGEngine:
    """
    RAG检索增强引擎，整合分块→嵌入→存储→检索全流程。

    使用方式:
        engine = RAGEngine()
        engine.index_document("申报书.docx")
        results = engine.retrieve("项目的创新点是什么？")
    """

    DEFAULT_SECTION_KEYWORDS = [
        "项目背景", "研究目标", "创新点", "技术路线", "算法",
        "实验", "系统架构", "应用场景", "局限性", "未来规划",
        "摘要", "引言", "方法", "结果", "讨论", "结论",
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedder: Optional[object] = None,
        persist_path: Optional[str] = None,
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedder = embedder or TFIDFEmbedder()
        self.store = VectorStore()
        self._persist_path = persist_path
        self._parser = DocParserSkill()
        self._indexed_sources: list = []

    def index_text(
        self,
        text: str,
        source: str = "direct_input",
        strategy: str = "section",
        section_keywords: Optional[list] = None,
    ) -> int:
        """
        索引纯文本。

        Args:
            text: 文本内容
            source: 来源标识
            strategy: 分块策略 (fixed/paragraph/section)
            section_keywords: 章节关键词（strategy=section时使用）

        Returns:
            索引的chunk数量
        """
        if strategy == "fixed":
            chunks = self.chunker.split_fixed(text, source)
        elif strategy == "paragraph":
            chunks = self.chunker.split_paragraph(text, source)
        elif strategy == "section":
            keywords = section_keywords or self.DEFAULT_SECTION_KEYWORDS
            chunks = self.chunker.split_section(text, keywords, source)
        else:
            raise ValueError(f"未知分块策略: {strategy}")

        return self._index_chunks(chunks)

    def index_document(
        self,
        file_path: str,
        strategy: str = "section",
        section_keywords: Optional[list] = None,
    ) -> int:
        """
        索引文档文件（docx/pdf），自动调用Skill1解析。

        Args:
            file_path: 文件路径
            strategy: 分块策略
            section_keywords: 章节关键词

        Returns:
            索引的chunk数量
        """
        text = self._parser.parse(file_path)
        self._indexed_sources.append(file_path)
        return self.index_text(text, source=file_path, strategy=strategy, section_keywords=section_keywords)

    def index_documents(self, file_paths: list, **kwargs) -> dict:
        """批量索引多个文档"""
        results = {}
        for fp in file_paths:
            try:
                n = self.index_document(fp, **kwargs)
                results[fp] = n
            except Exception as e:
                results[fp] = f"失败: {e}"
        return results

    def _index_chunks(self, chunks: list) -> int:
        """内部：对chunks进行嵌入并存储"""
        if not chunks:
            return 0

        texts = [c.content for c in chunks]

        if isinstance(self.embedder, TFIDFEmbedder):
            self.embedder.fit(texts)

        vectors = self.embedder.embed_batch(texts)
        self.store.add(chunks, vectors)

        if self._persist_path:
            self.save(self._persist_path)

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list:
        """
        语义检索，返回与query最相关的文档片段。

        Args:
            query: 查询文本
            top_k: 返回数量
            min_score: 最低相似度阈值

        Returns:
            RetrievalResult列表
        """
        query_vec = self.embedder.embed(query)
        results = self.store.search(query_vec, top_k=top_k)
        return [r for r in results if r.score >= min_score]

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> str:
        """
        检索并拼接为LLM上下文文本。

        Args:
            query: 查询文本
            top_k: 检索数量
            min_score: 最低相似度

        Returns:
            格式化的上下文字符串，可直接拼入LLM prompt
        """
        results = self.retrieve(query, top_k, min_score)
        if not results:
            return "未检索到相关内容。"

        lines = ["检索到以下相关内容：\n"]
        for i, r in enumerate(results, 1):
            section = r.chunk.metadata.get("section", "")
            source = r.chunk.metadata.get("source", "")
            header = f"[片段{i}]"
            if section:
                header += f" 章节:{section}"
            if source:
                header += f" 来源:{os.path.basename(source)}"
            header += f" 相关度:{r.score:.3f}"
            lines.append(header)
            lines.append(r.chunk.content)
            lines.append("")

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """持久化引擎状态"""
        self.store.save(path)

    def load(self, path: str) -> None:
        """加载引擎状态"""
        self.store.load(path)

    @property
    def stats(self) -> dict:
        """引擎统计信息"""
        return {
            "indexed_chunks": self.store.size,
            "indexed_sources": self._indexed_sources,
            "embedder_type": type(self.embedder).__name__,
            "vocab_size": self.embedder.dim if hasattr(self.embedder, "dim") else 0,
        }