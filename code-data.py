import os
from pathlib import Path
from typing import List, Dict, Any
import requests
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class DocumentProcessor:
    """文档处理器：负责加载和分块"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "50"))

    def load_document(self, file_path: str) -> str:
        """加载文档内容"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif file_path.suffix.lower() == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif file_path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                print("请安装 PyPDF2: pip install PyPDF2")
                return ""

        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """智能分块文本"""
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            # 如果单个段落就超过chunk_size，需要进一步切分
            if para_length > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'length': current_length
                    })
                    current_chunk = ""
                    current_length = 0

                # 按句子切分长段落
                sentences = self._split_long_paragraph(para)
                for sent in sentences:
                    if current_length + len(sent) > self.chunk_size and current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'length': current_length
                        })
                        # 保留重叠部分
                        overlap_text = self._get_overlap(current_chunk)
                        current_chunk = overlap_text + " " + sent
                        current_length = len(current_chunk)
                    else:
                        current_chunk += " " + sent if current_chunk else sent
                        current_length = len(current_chunk)

            # 正常大小的段落
            elif current_length + para_length > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'length': current_length
                    })
                    # 保留重叠部分
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = overlap_text + " " + para
                    current_length = len(current_chunk)
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_length += para_length

        # 添加最后一个chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'length': current_length
            })

        return chunks

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """切分长段落为句子"""
        import re
        # 按中英文句子结束符切分
        sentences = re.split(r'([。！？.!?]+)', paragraph)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            result.append(sentences[i] + sentences[i + 1])
        if len(sentences) % 2 == 1:
            result.append(sentences[-1])
        return [s.strip() for s in result if s.strip()]

    def _get_overlap(self, text: str) -> str:
        """获取重叠部分文本"""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]


class EmbeddingClient:
    """嵌入模型客户端：调用 vLLM 部署的 Qwen3-Embedding-4B"""

    def __init__(self, api_url: str = None, api_key: str = None, model: str = None):
        self.api_url = api_url or os.getenv("EMBEDDING_API_URL")
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.model = model or os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

        if not self.api_url:
            raise ValueError("EMBEDDING_API_URL 未配置")
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY 未配置")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入向量"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "input": texts
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                timeout=int(os.getenv("TIMEOUT", "60"))
            )
            response.raise_for_status()

            result = response.json()
            embeddings = [item['embedding'] for item in result['data']]
            return embeddings

        except Exception as e:
            print(f"获取嵌入向量失败: {e}")
            raise


class ChromaVectorStore:
    """Chroma 向量存储"""

    def __init__(self, persist_directory: str = None, collection_name: str = None):
        self.persist_directory = persist_directory or os.getenv("CHROMA_DIR", "./chroma_db")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "documents")

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )

    def add_documents(self, texts: List[str], embeddings: List[List[float]],
                      metadatas: List[Dict] = None, ids: List[str] = None):
        """添加文档到向量库"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        print(f"成功添加 {len(texts)} 个文档块到向量库")

    def query(self, query_embedding: List[float], n_results: int = None):
        """查询相似文档"""
        n_results = n_results or int(os.getenv("QUERY_TOP_K", "5"))
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def get_stats(self):
        """获取向量库统计信息"""
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": self.collection.count()
        }

class RAGPipeline:
    """完整的 RAG 处理流程"""

    def __init__(self, embedding_api_url: str = None, sificonflow_api_key: str = None,
                 chroma_dir: str = None, collection_name: str = None,
                 chunk_size: int = None, chunk_overlap: int = None):

        self.doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_client = EmbeddingClient(api_url=embedding_api_url, api_key=sificonflow_api_key)
        self.vector_store = ChromaVectorStore(persist_directory=chroma_dir, collection_name=collection_name)

    def process_file(self, file_path: str, metadata: Dict = None) -> int:
        """处理单个文件"""
        print(f"\n开始处理文件: {file_path}")

        # 1. 加载文档
        print("1. 加载文档...")
        text = self.doc_processor.load_document(file_path)
        print(f"   文档长度: {len(text)} 字符")

        # 2. 分块
        print("2. 文本分块...")
        chunks = self.doc_processor.split_text(text)
        print(f"   分块数量: {len(chunks)} 个")

        # 3. 获取嵌入向量
        print("3. 生成嵌入向量...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_client.get_embeddings(texts)
        print(f"   嵌入维度: {len(embeddings[0])}")

        # 4. 存储到向量库
        print("4. 存储到 Chroma...")
        file_name = Path(file_path).name
        metadatas = [
            {
                "source": file_name,
                "chunk_id": i,
                "length": chunk['length'],
                **(metadata or {})
            }
            for i, chunk in enumerate(chunks)
        ]

        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        self.vector_store.add_documents(texts, embeddings, metadatas, ids)

        print(f"✓ 文件处理完成!")
        return len(chunks)

    def process_directory(self, directory: str, file_extensions: List[str] = None) -> int:
        """批量处理目录下的文件"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.pdf']

        directory = Path(directory)
        files = []
        for ext in file_extensions:
            files.extend(directory.glob(f"**/*{ext}"))

        print(f"找到 {len(files)} 个文件待处理")

        total_chunks = 0
        for file_path in files:
            try:
                chunks = self.process_file(str(file_path))
                total_chunks += chunks
            except Exception as e:
                print(f"处理文件失败 {file_path}: {e}")

        print(f"\n总计处理: {len(files)} 个文件, {total_chunks} 个文档块")
        return total_chunks

    def search(self, query: str, top_k: int = None) -> Dict:
        """搜索相关文档"""
        print(f"\n搜索查询: {query}")

        # 获取查询的嵌入向量
        query_embedding = self.embedding_client.get_embeddings([query])[0]

        # 向量检索
        results = self.vector_store.query(query_embedding, n_results=top_k)

        print(f"\n找到 {len(results['documents'][0])} 个相关文档:\n")
        for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
        )):
            print(f"[{i + 1}] 相似度: {1 - distance:.4f}")
            print(f"    来源: {metadata.get('source', 'unknown')}")
            print(f"    内容预览: {doc[:100]}...")
            print()

        return results

    def get_stats(self) -> Dict:
        """获取整个系统的统计信息"""
        vector_stats = self.vector_store.get_stats()
        return {
            "vector_store": vector_stats,
            "chunk_size": self.doc_processor.chunk_size,
            "chunk_overlap": self.doc_processor.chunk_overlap,
            "embedding_model": self.embedding_client.model
        }


# 使用示例
if __name__ == "__main__":
    # 初始化 RAG Pipeline（会自动从.env读取配置）
    rag = RAGPipeline()

    # 示例1: 处理单个文件
    # rag.process_file("./测试数据/基于大模型的具身智能系统综述.pdf", metadata={"category": "tutorial"})

    # 示例2: 批量处理目录
    data_dir = os.getenv("DATA_DIRECTORY", "./data")
    if os.path.exists(data_dir):
        rag.process_directory(data_dir)
    else:
        print(f"数据目录不存在: {data_dir}")

    # 示例3: 搜索
    # rag.search("如何使用RAG系统？", top_k=3)

    # 查看系统统计
    print("\n系统统计信息:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")