import os
import sys
import json
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from functools import wraps
from abc import ABC, abstractmethod

import requests
import chromadb
from chromadb.config import Settings
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# 加载环境变量
load_dotenv()

# ==================== 日志配置 ====================
def setup_logger(name: str = "rag_system", level: int = logging.INFO) -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 文件处理器
    file_handler = logging.FileHandler("rag_system.log")
    file_handler.setLevel(logging.DEBUG)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

# ==================== 配置管理 ====================
class SystemConfig(BaseModel):
    """系统配置验证模型"""
    api_key: str = Field(..., min_length=1, description="SiliconFlow API密钥")
    llm_api_url: str = Field("https://api.siliconflow.cn/v1/chat/completions")
    embedding_api_url: str = Field("https://api.siliconflow.cn/v1/embeddings")
    reranker_api_url: str = Field("https://api.siliconflow.cn/v1/rerank")

    llm_model: str = Field("Qwen/Qwen3-8B")
    embedding_model: str = Field("Qwen/Qwen3-Embedding-8B")
    reranker_model: str = Field("Qwen/Qwen3-Reranker-8B")

    chroma_dir: str = Field("./chroma_db")
    collection_name: str = Field("my_documents")

    retrieve_k: int = Field(20, ge=1, le=100, description="初始检索文档数量")
    top_k: int = Field(5, ge=1, le=20, description="最终返回文档数量")

    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=8000)

    rerank_instruction: Optional[str] = Field(None)
    rerank_method: str = Field("true_reranker")

    timeout: int = Field(60, ge=10, le=300, description="API超时时间(秒)")
    max_retries: int = Field(3, ge=1, le=10, description="最大重试次数")

    @field_validator('rerank_method')
    @classmethod
    def validate_rerank_method(cls, v):
        """验证重排序方法"""
        allowed_methods = ['none', 'embedding', 'true_reranker']
        if v not in allowed_methods:
            raise ValueError(f'rerank_method必须是以下之一: {", ".join(allowed_methods)}')
        return v

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v, info):
        """验证top_k参数"""
        values = info.data
        if 'retrieve_k' in values and v > values['retrieve_k']:
            raise ValueError('top_k不能大于retrieve_k')
        return v

    @field_validator('llm_api_url', 'embedding_api_url', 'reranker_api_url')
    @classmethod
    def validate_urls(cls, v):
        """验证URL格式"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL必须以http://或https://开头')
        return v

    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """从环境变量创建配置"""
        # 设置日志级别
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logger.setLevel(getattr(logging, log_level))

        return cls(
            api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen3-8B"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
            reranker_model=os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B"),
            rerank_method=os.getenv("RERANK_METHOD", "true_reranker"),
            rerank_instruction=os.getenv("RERANK_INSTRUCTION"),
            chroma_dir=os.getenv("CHROMA_DIR", "./chroma_db"),
            collection_name=os.getenv("COLLECTION_NAME", "my_documents"),
            retrieve_k=int(os.getenv("RETRIEVE_K", "20")),
            top_k=int(os.getenv("TOP_K", "5")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            timeout=int(os.getenv("TIMEOUT", "60")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            llm_api_url=os.getenv("CHAT_API_URL", "https://api.siliconflow.cn/v1/chat/completions"),
            embedding_api_url=os.getenv("EMBEDDING_API_URL", "https://api.siliconflow.cn/v1/embeddings"),
            reranker_api_url=os.getenv("RERANKER_API_URL", "https://api.siliconflow.cn/v1/rerank")
        )

# ==================== 枚举和类型定义 ====================
class RerankMethod(Enum):
    """重排序方法枚举"""
    NONE = "none"
    EMBEDDING = "embedding"
    TRUE_RERANKER = "true_reranker"

class ConversationTurn(BaseModel):
    """对话轮次"""
    user: str
    assistant: str

class Document(BaseModel):
    """文档类型"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    original_distance: float = 0.0
    rerank_method: str = "none"

class QueryResult(BaseModel):
    """查询结果"""
    query: str
    standalone_query: str
    answer: str
    source_documents: List[Document]
    num_sources: int
    rerank_method: str

# ==================== 工具函数 ====================
def retry(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"函数 {func.__name__} 调用失败，{sleep_time:.1f}秒后重试... "
                            f"(尝试 {attempt + 1}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"函数 {func.__name__} 重试 {max_retries} 次后失败: {e}")
            raise last_exception or Exception("重试失败")
        return wrapper
    return decorator

# ==================== API客户端基类 ====================
class BaseAPIClient:
    """API客户端基类"""

    def __init__(self, api_url: str, api_key: str, timeout: int = 60):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _log_request(self, endpoint: str, payload: Dict):
        """记录请求日志（不记录敏感信息）"""
        safe_payload = {
            k: v if k not in ['input', 'query', 'documents'] else f"[{len(v)} items]"
            for k, v in payload.items()
        }
        logger.debug(f"请求 {endpoint}: {safe_payload}")

    def _log_response(self, endpoint: str, response: requests.Response):
        """记录响应日志"""
        logger.debug(f"响应 {endpoint}: {response.status_code}")

# ==================== LLM客户端 ====================
class LLMClient(BaseAPIClient):
    """大语言模型客户端"""

    def __init__(self, api_url: str, api_key: str, model_name: str = "Qwen/Qwen3-8B", **kwargs):
        super().__init__(api_url, api_key, **kwargs)
        self.model_name = model_name

    @retry(max_retries=3)
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """生成回答"""
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }

            self._log_request("LLM生成", payload)
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            self._log_response("LLM生成", response)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except requests.RequestException as e:
            logger.error(f"LLM请求失败: {e}")
            raise
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"解析LLM响应失败: {e}")
            raise

# ==================== 嵌入模型客户端（带缓存） ====================
class EmbeddingClient(BaseAPIClient):
    """嵌入模型客户端（带缓存）"""

    def __init__(self, api_url: str, api_key: str = None, cache_size: int = 1000, **kwargs):
        super().__init__(api_url, api_key, **kwargs)
        self._cache: Dict[str, List[List[float]]] = {}
        self.cache_size = cache_size

    def get_embeddings(self, texts: List[str], model: str = "Qwen/Qwen3-Embedding-8B") -> List[List[float]]:
        """获取文本嵌入向量（带缓存）"""
        cache_key = self._get_cache_key(texts, model)

        if cache_key in self._cache:
            logger.debug(f"缓存命中: {cache_key[:32]}...")
            return self._cache[cache_key]

        embeddings = self._call_embedding_api(texts, model)

        # 更新缓存（限制大小）
        if len(self._cache) >= self.cache_size:
            # 简单策略：移除第一个键
            self._cache.pop(next(iter(self._cache)))

        self._cache[cache_key] = embeddings
        return embeddings

    @retry(max_retries=3)
    def _call_embedding_api(self, texts: List[str], model: str) -> List[List[float]]:
        """调用嵌入API"""
        try:
            payload = {"model": model, "input": texts}

            self._log_request("嵌入生成", payload)
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            self._log_response("嵌入生成", response)
            response.raise_for_status()

            result = response.json()
            return [item['embedding'] for item in result['data']]

        except requests.RequestException as e:
            logger.error(f"嵌入API请求失败: {e}")
            raise
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"解析嵌入响应失败: {e}")
            raise

    def _get_cache_key(self, texts: List[str], model: str) -> str:
        """生成缓存键"""
        text_hash = hashlib.md5("|".join(texts).encode()).hexdigest()
        return f"{model}_{text_hash}"

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("嵌入缓存已清空")

# ==================== 重排序器基类 ====================
class BaseReranker(ABC):
    """重排序器基类"""

    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = 5, **kwargs) -> List[Tuple[int, float]]:
        """重排序方法"""
        pass

    def _fallback_ranking(self, documents: List[str], top_k: int) -> List[Tuple[int, float]]:
        """失败时的降级策略"""
        return list(zip(
            range(min(top_k, len(documents))),
            [1.0 - i * 0.1 for i in range(min(top_k, len(documents)))]
        ))

# ==================== 嵌入模型重排序器 ====================
class EmbeddingReranker(BaseReranker):
    """嵌入模型重排序器（原方式）"""

    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client

    def rerank(self, query: str, documents: List[str], top_k: int = 5, **kwargs) -> List[Tuple[int, float]]:
        """使用嵌入模型重新计算相似度并排序"""
        try:
            if not documents:
                return []

            # 获取查询和文档的嵌入向量
            query_embedding = self.embedding_client.get_embeddings([query])[0]
            doc_embeddings = self.embedding_client.get_embeddings(documents)

            # 计算余弦相似度
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(doc_embeddings)

            # 归一化
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)

            # 计算相似度
            similarities = np.dot(doc_norms, query_norm)

            # 排序并返回top_k
            ranked_indices = np.argsort(similarities)[::-1][:top_k]
            ranked_scores = similarities[ranked_indices]

            return list(zip(ranked_indices.tolist(), ranked_scores.tolist()))

        except Exception as e:
            logger.error(f"嵌入重排序失败: {e}")
            return self._fallback_ranking(documents, top_k)

# ==================== 官方重排序器 ====================
class TrueReranker(BaseReranker):
    """真正的重排序器（按照SiliconFlow官方API实现）"""

    def __init__(self, api_url: str, api_key: str, model_name: str = "Qwen/Qwen3-Reranker-8B", **kwargs):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = kwargs.get('timeout', 60)

    @retry(max_retries=3)
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        instruction: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """使用SiliconFlow官方重排序API"""
        try:
            if not documents:
                return []

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # 构建请求体
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": top_k,
                "return_documents": False
            }

            # 仅Qwen3-Reranker系列支持instruction参数
            if instruction and self.model_name.startswith("Qwen/Qwen3-Reranker"):
                payload["instruction"] = instruction

            self._log_request("重排序", payload)
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            self._log_response("重排序", response)
            response.raise_for_status()

            result = response.json()

            # 解析API响应
            ranked_results = []
            if "results" in result:
                for item in result["results"]:
                    index = item.get("index", 0)
                    score = item.get("relevance_score", 0.0)
                    ranked_results.append((index, score))

            # 按分数从高到低排序
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            return ranked_results[:top_k]

        except requests.RequestException as e:
            logger.error(f"官方重排序API请求失败: {e}")
            return self._fallback_ranking(documents, top_k)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"解析重排序响应失败: {e}")
            return self._fallback_ranking(documents, top_k)

    def _log_request(self, endpoint: str, payload: Dict):
        """记录请求日志"""
        safe_payload = payload.copy()
        if 'documents' in safe_payload:
            safe_payload['documents'] = f"[{len(safe_payload['documents'])} items]"
        if 'query' in safe_payload:
            safe_payload['query'] = f"[{len(safe_payload['query'])} chars]"
        logger.debug(f"请求 {endpoint}: {safe_payload}")

    def _log_response(self, endpoint: str, response: requests.Response):
        """记录响应日志"""
        logger.debug(f"响应 {endpoint}: {response.status_code}")

# ==================== 向量数据库检索器 ====================
class ChromaRetriever:
    """Chroma向量数据库检索器"""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents"):
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Chroma集合 '{collection_name}' 加载成功")
        except Exception as e:
            logger.error(f"加载Chroma集合失败: {e}")
            raise

    def retrieve(self, query_embedding: List[float], top_k: int = 10) -> Dict:
        """从向量库检索相关文档"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            return results
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            return {
                "collection_name": self.collection.name,
                "document_count": self.collection.count()
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"collection_name": "unknown", "document_count": 0}

# ==================== 提示词构建器 ====================
class PromptBuilder:
    """提示词构建器"""

    @staticmethod
    def build_rag_prompt(query: str, context_docs: List[Document], conversation_history: List[ConversationTurn] = None) -> str:
        """构建RAG提示词"""
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.metadata.get('source', 'unknown')
            content = doc.content[:500]  # 限制长度
            context_parts.append(f"[文档{i}] 来源: {source}\n{content}")

        context = "\n\n".join(context_parts)

        # 构建对话历史
        history_text = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-3:]:  # 只保留最近3轮
                history_parts.append(f"用户: {turn.user}\n助手: {turn.assistant}")
            if history_parts:
                history_text = "对话历史:\n" + "\n\n".join(history_parts) + "\n\n"

        # 组合提示词
        prompt = f"""{history_text}参考文档:
{context}

请基于以上参考文档回答用户问题。要求:
1. 答案必须基于参考文档内容，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 可以引用具体的文档来源

用户问题: {query}

回答:"""

        return prompt

    @staticmethod
    def build_standalone_query(query: str, conversation_history: List[ConversationTurn]) -> str:
        """构建独立查询（用于指代消解）"""
        if not conversation_history:
            return query

        history_parts = []
        for turn in conversation_history[-2:]:
            history_parts.append(f"用户: {turn.user}\n助手: {turn.assistant}")

        history_text = "\n".join(history_parts)

        prompt = f"""对话历史:
{history_text}

当前用户问题: {query}

请将当前问题改写为一个独立的、完整的问题，使其不依赖对话历史也能被理解。
只输出改写后的问题，不要有其他内容。

改写后的问题:"""

        return prompt

# ==================== RAG问答系统 ====================
class RAGQASystem:
    """完整的RAG问答系统"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logger

        # 验证配置
        if not config.api_key:
            raise ValueError("API密钥未设置，请设置SILICONFLOW_API_KEY环境变量")

        # 初始化各组件
        self.llm_client = LLMClient(
            config.llm_api_url,
            config.api_key,
            config.llm_model,
            timeout=config.timeout
        )

        self.embedding_client = EmbeddingClient(
            config.embedding_api_url,
            config.api_key,
            timeout=config.timeout
        )

        self.retriever = ChromaRetriever(config.chroma_dir, config.collection_name)

        # 设置重排序方法
        self.rerank_method = RerankMethod(config.rerank_method)
        self.reranker = self._init_reranker()

        # 对话历史
        self.conversation_history: List[ConversationTurn] = []

        self.logger.info(f"RAG系统初始化完成，重排序方式: {self.rerank_method.value}")

    def _init_reranker(self) -> Optional[BaseReranker]:
        """初始化重排序器"""
        if self.rerank_method == RerankMethod.EMBEDDING:
            return EmbeddingReranker(self.embedding_client)
        elif self.rerank_method == RerankMethod.TRUE_RERANKER:
            return TrueReranker(
                self.config.reranker_api_url,
                self.config.api_key,
                self.config.reranker_model,
                timeout=self.config.timeout
            )
        else:
            return None

    def query(self, user_query: str, use_history: bool = True) -> QueryResult:
        """处理用户查询"""
        self.logger.info(f"处理查询: {user_query[:100]}...")
        self.logger.info(f"重排序方式: {self.rerank_method.value}")

        # Step 1: 查询改写（指代消解）
        standalone_query = user_query
        if use_history and self.conversation_history:
            self.logger.debug("执行查询改写...")
            rewrite_prompt = PromptBuilder.build_standalone_query(
                user_query,
                self.conversation_history
            )
            try:
                standalone_query = self.llm_client.generate(
                    rewrite_prompt,
                    temperature=0.3,
                    max_tokens=200
                ).strip()
                self.logger.info(f"改写后查询: {standalone_query}")
            except Exception as e:
                self.logger.warning(f"查询改写失败: {e}，使用原始查询")
                standalone_query = user_query

        # Step 2: 生成查询向量
        self.logger.debug("生成查询向量...")
        query_embedding = self.embedding_client.get_embeddings(
            [standalone_query],
            self.config.embedding_model
        )[0]

        # Step 3: 向量检索
        self.logger.debug(f"向量检索 (Top-{self.config.retrieve_k})...")
        results = self.retriever.retrieve(query_embedding, top_k=self.config.retrieve_k)

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        self.logger.info(f"检索到 {len(documents)} 个候选文档")

        # Step 4: 重排序
        final_docs = []
        if self.reranker and len(documents) > self.config.top_k:
            self.logger.debug(f"{self.rerank_method.value} 重排序 (Top-{self.config.top_k})...")

            # 调用相应的重排序器
            if self.rerank_method == RerankMethod.TRUE_RERANKER:
                reranked = self.reranker.rerank(
                    standalone_query,
                    documents,
                    self.config.top_k,
                    instruction=self.config.rerank_instruction
                )
            else:
                reranked = self.reranker.rerank(standalone_query, documents, self.config.top_k)

            # 重新组织文档
            for orig_idx, score in reranked:
                final_docs.append(Document(
                    content=documents[orig_idx],
                    metadata=metadatas[orig_idx] if orig_idx < len(metadatas) else {},
                    score=float(score),
                    original_distance=float(distances[orig_idx]) if orig_idx < len(distances) else 0.0,
                    rerank_method=self.rerank_method.value
                ))
        else:
            self.logger.debug("不使用重排序")
            for doc, meta, dist in zip(
                documents[:self.config.top_k],
                metadatas[:self.config.top_k],
                distances[:self.config.top_k]
            ):
                final_docs.append(Document(
                    content=doc,
                    metadata=meta,
                    score=1 - float(dist),
                    original_distance=float(dist),
                    rerank_method='none'
                ))

        # 记录选中的文档
        self.logger.info(f"最终选择 {len(final_docs)} 个文档:")
        for i, doc in enumerate(final_docs, 1):
            self.logger.info(
                f"  [{i}] {doc.rerank_method} 相似度: {doc.score:.4f} | "
                f"来源: {doc.metadata.get('source', 'unknown')}"
            )

        # Step 5: 生成答案
        self.logger.debug("生成答案...")
        history = self.conversation_history if use_history else None
        prompt = PromptBuilder.build_rag_prompt(user_query, final_docs, history)

        answer = self.llm_client.generate(
            prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # 保存对话历史
        self.conversation_history.append(ConversationTurn(
            user=user_query,
            assistant=answer
        ))

        # 返回结果
        result = QueryResult(
            query=user_query,
            standalone_query=standalone_query,
            answer=answer,
            source_documents=final_docs,
            num_sources=len(final_docs),
            rerank_method=self.rerank_method.value
        )

        self.logger.info("查询处理完成")
        return result

    def set_rerank_method(self, method: RerankMethod, **kwargs):
        """动态设置重排序方法"""
        old_method = self.rerank_method
        self.rerank_method = method

        if method == RerankMethod.EMBEDDING:
            self.reranker = EmbeddingReranker(self.embedding_client)
            self.logger.info("已切换到嵌入模型重排序")
        elif method == RerankMethod.TRUE_RERANKER:
            api_url = kwargs.get('reranker_api_url', self.config.reranker_api_url)
            api_key = kwargs.get('reranker_api_key', self.config.api_key)
            model_name = kwargs.get('reranker_model_name', self.config.reranker_model)

            if api_url and api_key:
                self.reranker = TrueReranker(
                    api_url,
                    api_key,
                    model_name,
                    timeout=self.config.timeout
                )
                self.logger.info(f"已切换到官方重排序API (模型: {model_name})")
            else:
                self.logger.warning("缺少重排序API配置，无法切换")
                self.rerank_method = old_method
        else:
            self.reranker = None
            self.logger.info("已禁用重排序")

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        self.logger.info("对话历史已清空")

    def get_history(self) -> List[ConversationTurn]:
        """获取对话历史"""
        return self.conversation_history

    def print_stats(self):
        """打印系统统计信息"""
        stats = self.retriever.get_stats()
        self.logger.info("系统状态:")
        self.logger.info(f"  向量库: {stats['collection_name']}")
        self.logger.info(f"  文档数: {stats['document_count']}")
        self.logger.info(f"  历史轮次: {len(self.conversation_history)}")
        self.logger.info(f"  重排序方式: {self.rerank_method.value}")
        self.logger.info(f"  嵌入模型: {self.config.embedding_model}")

    def clear_embedding_cache(self):
        """清空嵌入缓存"""
        self.embedding_client.clear_cache()

# ==================== 交互式对话 ====================
def interactive_chat(qa_system: RAGQASystem):
    """交互式对话界面"""
    print("\n" + "=" * 60)
    print("RAG问答系统已启动")
    print("=" * 60)
    print("命令:")
    print("  - 直接输入问题开始对话")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'stats' 查看系统状态")
    print("  - 输入 'cache:clear' 清空嵌入缓存")
    print("  - 输入 'mode:none' 关闭重排序")
    print("  - 输入 'mode:embedding' 使用嵌入模型重排序")
    print("  - 输入 'mode:true' 使用官方重排序API")
    print("  - 输入 'exit' 或 'quit' 或 'q' 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\n👤 用户: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n再见！")
                break

            if user_input.lower() == 'clear':
                qa_system.clear_history()
                print("对话历史已清空")
                continue

            if user_input.lower() == 'stats':
                qa_system.print_stats()
                continue

            if user_input.lower() == 'cache:clear':
                qa_system.clear_embedding_cache()
                print("嵌入缓存已清空")
                continue

            # 切换重排序模式
            if user_input.lower().startswith('mode:'):
                mode_str = user_input.lower()[5:]
                if mode_str == 'none':
                    qa_system.set_rerank_method(RerankMethod.NONE)
                elif mode_str == 'embedding':
                    qa_system.set_rerank_method(RerankMethod.EMBEDDING)
                elif mode_str == 'true':
                    qa_system.set_rerank_method(RerankMethod.TRUE_RERANKER)
                else:
                    print(f"未知模式: {mode_str}，可用模式: none, embedding, true")
                continue

            # 处理查询
            print("\n" + "=" * 40)
            print("处理中...")
            print("=" * 40)

            result = qa_system.query(user_input)

            print(f"\n🤖 助手: {result.answer}")

            if result.source_documents:
                print(f"\n📚 参考文档 ({len(result.source_documents)}个):")
                for i, doc in enumerate(result.source_documents, 1):
                    source = doc.metadata.get('source', '未知')
                    print(f"  {i}. [{doc.rerank_method}] 相似度: {doc.score:.4f} | 来源: {source}")
                    if i == 1:  # 只显示第一个文档的预览
                        preview = doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
                        print(f"     预览: {preview}")

            print("=" * 40)

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            logger.exception("交互式对话出错")

# ==================== 主函数 ====================
def main():
    """主函数"""
    try:
        # 加载配置
        config = SystemConfig.from_env()

        # 显示配置信息（隐藏敏感信息）
        print("=" * 60)
        print("RAG系统配置信息:")
        print(f"  LLM模型: {config.llm_model}")
        print(f"  嵌入模型: {config.embedding_model}")
        print(f"  重排序方式: {config.rerank_method}")
        if config.rerank_method == "true_reranker":
            print(f"  重排序模型: {config.reranker_model}")
        print(f"  Chroma目录: {config.chroma_dir}")
        print(f"  集合名称: {config.collection_name}")
        print(f"  检索参数: retrieve_k={config.retrieve_k}, top_k={config.top_k}")
        print("=" * 60)

        # 检查API密钥
        if not config.api_key:
            print("错误：未设置SILICONFLOW_API_KEY环境变量")
            print("请创建.env文件或在环境变量中设置：")
            print("SILICONFLOW_API_KEY=your_api_key_here")
            sys.exit(1)

        # 初始化问答系统
        qa_system = RAGQASystem(config)

        # 检查Chroma集合
        try:
            stats = qa_system.retriever.get_stats()
            print(f"✓ Chroma集合 '{stats['collection_name']}' 加载成功")
            print(f"✓ 文档数量: {stats['document_count']}")
        except Exception as e:
            print(f"✗ 加载Chroma集合失败: {e}")
            print("请确保已创建向量数据库并导入文档")
            sys.exit(1)

        # 启动交互式对话
        interactive_chat(qa_system)

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        logger.exception("程序运行出错")
        print(f"程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()