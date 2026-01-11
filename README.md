# RAG 问答系统演示项目

一个基于检索增强生成（Retrieval-Augmented Generation, RAG）的智能问答系统，支持文档导入、向量检索、重排序和智能问答等功能。

## 📋 项目简介

本项目实现了一个完整的 RAG 系统，包含两个核心模块：

1. **文档处理与向量化** (`code-data.py`)：将文档分块、生成嵌入向量并存储到 ChromaDB
2. **智能问答系统** (`code-test.py`)：基于向量检索和 LLM 的问答系统，支持多种重排序策略

## ✨ 功能特性

### 文档处理模块 (`code-data.py`)
- ✅ 支持多种文档格式（PDF、TXT、MD）
- ✅ 智能文本分块（支持重叠策略）
- ✅ 批量文档处理
- ✅ 使用 Qwen 嵌入模型生成向量
- ✅ ChromaDB 持久化存储

### 问答系统模块 (`code-test.py`)
- ✅ 基于向量相似度的文档检索
- ✅ 多种重排序策略（无重排序、嵌入模型重排序、官方重排序 API）
- ✅ 对话历史管理（支持上下文理解）
- ✅ 查询改写（指代消解）
- ✅ 交互式命令行界面
- ✅ 完整的日志记录
- ✅ 嵌入向量缓存机制
- ✅ 自动重试机制

## 🏗️ 项目结构

```
RAG-demo/
├── code-data.py              # 文档处理和向量化脚本
├── code-test.py              # RAG 问答系统主程序
├── chroma_db/                # ChromaDB 向量数据库存储目录
├── 测试数据/                  # 测试文档目录
│   └── 基于大模型的具身智能系统综述.pdf
├── rag_system.log            # 系统运行日志
└── README.md                 # 项目说明文档
```

## 🔧 环境要求

- Python 3.8+
- 依赖包：
  - `chromadb` - 向量数据库
  - `requests` - HTTP 请求
  - `python-dotenv` - 环境变量管理
  - `PyPDF2` - PDF 文档解析
  - `pydantic` - 数据验证
  - `numpy` - 数值计算

## 📦 安装步骤

### 1. 克隆或下载项目

```bash
git clone <repository-url>
cd RAG-demo
```

### 2. 安装依赖

```bash
pip install chromadb requests python-dotenv PyPDF2 pydantic numpy
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# SiliconFlow API 配置（必需）
SILICONFLOW_API_KEY=your_api_key_here

# API 端点配置（可选，有默认值）
EMBEDDING_API_URL=https://api.siliconflow.cn/v1/embeddings
CHAT_API_URL=https://api.siliconflow.cn/v1/chat/completions
RERANKER_API_URL=https://api.siliconflow.cn/v1/rerank

# 模型配置（可选，有默认值）
LLM_MODEL=Qwen/Qwen3-8B
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
RERANKER_MODEL=Qwen/Qwen3-Reranker-8B

# ChromaDB 配置（可选）
CHROMA_DIR=./chroma_db
COLLECTION_NAME=my_documents

# 检索参数（可选）
RETRIEVE_K=20          # 初始检索文档数量
TOP_K=5                # 最终返回文档数量
RERANK_METHOD=true_reranker  # 重排序方式：none, embedding, true_reranker

# LLM 参数（可选）
TEMPERATURE=0.7
MAX_TOKENS=2000

# 其他配置（可选）
TIMEOUT=60             # API 超时时间（秒）
MAX_RETRIES=3          # 最大重试次数
LOG_LEVEL=INFO         # 日志级别：DEBUG, INFO, WARNING, ERROR

# 文档处理参数（code-data.py 使用）
CHUNK_SIZE=500         # 文本分块大小
CHUNK_OVERLAP=50       # 分块重叠大小
DATA_DIRECTORY=./data  # 数据目录
```

## 🚀 使用方法

### 第一步：构建向量数据库

使用 `code-data.py` 处理文档并构建向量数据库：

```bash
python code-data.py
```

**示例用法：**

```python
from code_data import RAGPipeline

# 初始化 RAG Pipeline
rag = RAGPipeline()

# 处理单个文件
rag.process_file("./测试数据/基于大模型的具身智能系统综述.pdf")

# 或批量处理目录
rag.process_directory("./测试数据", file_extensions=['.pdf', '.txt', '.md'])

# 查看系统统计
stats = rag.get_stats()
print(stats)
```

### 第二步：启动问答系统

使用 `code-test.py` 启动交互式问答系统：

```bash
python code-test.py
```

**交互式命令：**

- 直接输入问题开始对话
- `clear` - 清空对话历史
- `stats` - 查看系统状态
- `cache:clear` - 清空嵌入缓存
- `mode:none` - 关闭重排序
- `mode:embedding` - 使用嵌入模型重排序
- `mode:true` - 使用官方重排序 API
- `exit` / `quit` / `q` - 退出程序

**代码调用示例：**

```python
from code_test import RAGQASystem, SystemConfig

# 从环境变量加载配置
config = SystemConfig.from_env()

# 初始化问答系统
qa_system = RAGQASystem(config)

# 进行查询
result = qa_system.query("什么是具身智能？")

print(f"答案: {result.answer}")
print(f"来源文档数: {result.num_sources}")
for i, doc in enumerate(result.source_documents, 1):
    print(f"文档 {i}: {doc.metadata.get('source')} (相似度: {doc.score:.4f})")
```

## 📚 核心组件说明

### DocumentProcessor
文档处理器，负责：
- 加载不同格式的文档（PDF、TXT、MD）
- 智能分块处理（保持语义完整性）
- 支持分块重叠策略

### EmbeddingClient
嵌入模型客户端：
- 调用 SiliconFlow API 生成文本嵌入向量
- 支持批量处理
- 内置缓存机制（避免重复计算）

### ChromaVectorStore
向量数据库管理器：
- 使用 ChromaDB 持久化存储
- 支持余弦相似度检索
- 文档元数据管理

### RAGQASystem
完整的 RAG 问答系统：
- **查询改写**：将上下文相关的查询改写为独立查询
- **向量检索**：从向量库中检索相关文档
- **重排序**：使用不同策略对检索结果进行重排序
- **答案生成**：基于检索到的文档生成答案
- **对话历史**：维护多轮对话上下文

### 重排序策略

1. **none**：不进行重排序，直接使用向量检索结果
2. **embedding**：使用嵌入模型重新计算相似度并排序
3. **true_reranker**：使用官方重排序 API（推荐，效果最好）

## ⚙️ 配置说明

### 重排序方法对比

| 方法 | 速度 | 准确性 | 成本 |
|------|------|--------|------|
| none | 最快 | 中等 | 最低 |
| embedding | 中等 | 较高 | 中等 |
| true_reranker | 较慢 | 最高 | 较高 |

### 检索参数建议

- `RETRIEVE_K`：建议 15-30，根据文档库大小调整
- `TOP_K`：建议 3-7，用于最终生成答案的文档数量
- `CHUNK_SIZE`：建议 300-800，根据文档类型调整

## 📝 日志系统

系统会自动记录运行日志到 `rag_system.log` 文件，包括：
- API 请求和响应
- 文档处理状态
- 检索和重排序结果
- 错误和警告信息

可通过环境变量 `LOG_LEVEL` 控制日志级别。

## 🔍 故障排查

### 问题：API 调用失败
- 检查 `SILICONFLOW_API_KEY` 是否正确设置
- 确认网络连接正常
- 查看日志文件了解详细错误信息

### 问题：向量库加载失败
- 确保已运行 `code-data.py` 构建向量库
- 检查 `CHROMA_DIR` 和 `COLLECTION_NAME` 配置
- 确认 ChromaDB 文件完整

### 问题：文档解析失败
- PDF 文件需要安装 `PyPDF2`
- 确保文件编码正确（TXT、MD 文件使用 UTF-8）
- 检查文件路径是否正确

## 📄 许可证

本项目仅供学习和演示使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

如有问题或建议，请通过 Issue 联系。

---

**注意**：使用本系统需要有效的 SiliconFlow API 密钥。请访问 [SiliconFlow](https://siliconflow.cn) 获取 API 密钥。