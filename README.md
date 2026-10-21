# RAG QA System Demo

English | [中文](README.zh.md)

An intelligent QA system built on Retrieval-Augmented Generation (RAG), supporting document ingestion, vector retrieval, reranking and question answering.

## 📋 Overview

This project implements a complete RAG system with two core modules:

1. **Document processing & vectorization** (`code-data.py`): chunks documents, generates embedding vectors and stores them in ChromaDB
2. **QA system** (`code-test.py`): a retrieval + LLM QA system with multiple reranking strategies

## ✨ Features

### Document processing module (`code-data.py`)
- ✅ Multiple document formats (PDF, TXT, MD)
- ✅ Smart text chunking (with overlap strategy)
- ✅ Batch document processing
- ✅ Embeddings generated with Qwen embedding models
- ✅ Persistent storage in ChromaDB

### QA system module (`code-test.py`)
- ✅ Vector-similarity document retrieval
- ✅ Multiple reranking strategies (none / embedding-model / official reranker API)
- ✅ Conversation history management (context-aware)
- ✅ Query rewriting (coreference resolution)
- ✅ Interactive command-line interface
- ✅ Full logging
- ✅ Embedding cache
- ✅ Automatic retries

## 🏗️ Project structure

```
RAG-demo/
├── code-data.py              # document processing & vectorization script
├── code-test.py              # RAG QA system main program
├── chroma_db/                # ChromaDB vector store directory
├── 测试数据/                  # test documents
│   └── 基于大模型的具身智能系统综述.pdf
├── rag_system.log            # runtime log
└── README.md                 # this file
```

## 🔧 Requirements

- Python 3.8+
- Dependencies:
  - `chromadb` - vector database
  - `requests` - HTTP requests
  - `python-dotenv` - environment variables
  - `PyPDF2` - PDF parsing
  - `pydantic` - data validation
  - `numpy` - numeric computing

## 📦 Installation

### 1. Clone the project

```bash
git clone <repository-url>
cd RAG-demo
```

### 2. Install dependencies

```bash
pip install chromadb requests python-dotenv PyPDF2 pydantic numpy
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# SiliconFlow API config (required)
SILICONFLOW_API_KEY=your_api_key_here

# API endpoints (optional, defaults provided)
EMBEDDING_API_URL=https://api.siliconflow.cn/v1/embeddings
CHAT_API_URL=https://api.siliconflow.cn/v1/chat/completions
RERANKER_API_URL=https://api.siliconflow.cn/v1/rerank

# Model config (optional, defaults provided)
LLM_MODEL=Qwen/Qwen3-8B
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
RERANKER_MODEL=Qwen/Qwen3-Reranker-8B

# ChromaDB config (optional)
CHROMA_DIR=./chroma_db
COLLECTION_NAME=my_documents

# Retrieval params (optional)
RETRIEVE_K=20          # initial number of retrieved docs
TOP_K=5                # final number of returned docs
RERANK_METHOD=true_reranker  # none, embedding, true_reranker

# LLM params (optional)
TEMPERATURE=0.7
MAX_TOKENS=2000

# Other (optional)
TIMEOUT=60             # API timeout (seconds)
MAX_RETRIES=3          # max retries
LOG_LEVEL=INFO         # DEBUG, INFO, WARNING, ERROR

# Document processing params (used by code-data.py)
CHUNK_SIZE=500         # chunk size
CHUNK_OVERLAP=50       # chunk overlap
DATA_DIRECTORY=./data  # data directory
```

## 🚀 Usage

### Step 1: build the vector database

Use `code-data.py` to process documents and build the vector store:

```bash
python code-data.py
```

**Example:**

```python
from code_data import RAGPipeline

# Initialize the RAG pipeline
rag = RAGPipeline()

# Process a single file
rag.process_file("./测试数据/基于大模型的具身智能系统综述.pdf")

# Or batch-process a directory
rag.process_directory("./测试数据", file_extensions=['.pdf', '.txt', '.md'])

# View system stats
stats = rag.get_stats()
print(stats)
```

### Step 2: start the QA system

Use `code-test.py` to launch the interactive QA system:

```bash
python code-test.py
```

**Interactive commands:**

- Type a question directly to start a conversation
- `clear` - clear conversation history
- `stats` - view system status
- `cache:clear` - clear the embedding cache
- `mode:none` - disable reranking
- `mode:embedding` - rerank with the embedding model
- `mode:true` - rerank with the official reranker API
- `exit` / `quit` / `q` - quit

**Programmatic usage:**

```python
from code_test import RAGQASystem, SystemConfig

# Load config from environment variables
config = SystemConfig.from_env()

# Initialize the QA system
qa_system = RAGQASystem(config)

# Run a query
result = qa_system.query("What is embodied intelligence?")

print(f"Answer: {result.answer}")
print(f"Sources: {result.num_sources}")
for i, doc in enumerate(result.source_documents, 1):
    print(f"Doc {i}: {doc.metadata.get('source')} (score: {doc.score:.4f})")
```

## 📚 Core components

### DocumentProcessor
- Loads documents in various formats (PDF, TXT, MD)
- Smart chunking (semantically coherent)
- Overlap strategy support

### EmbeddingClient
- Calls the SiliconFlow API to generate embeddings
- Batch processing
- Built-in cache (avoids redundant computation)

### ChromaVectorStore
- Persistent storage via ChromaDB
- Cosine-similarity retrieval
- Document metadata management

### RAGQASystem
- **Query rewriting**: rewrites context-dependent queries into standalone ones
- **Vector retrieval**: retrieves relevant documents from the store
- **Reranking**: reranks results with different strategies
- **Answer generation**: generates answers from retrieved documents
- **Conversation history**: maintains multi-turn context

### Reranking strategies

1. **none**: no reranking, use raw vector-search results
2. **embedding**: recompute similarity with the embedding model and sort
3. **true_reranker**: official reranker API (recommended, best quality)

## ⚙️ Configuration notes

### Reranking comparison

| Method | Speed | Accuracy | Cost |
|--------|-------|----------|------|
| none | fastest | medium | lowest |
| embedding | medium | higher | medium |
| true_reranker | slower | highest | higher |

### Retrieval tips

- `RETRIEVE_K`: 15-30 recommended, adjust to corpus size
- `TOP_K`: 3-7 recommended, documents used for final answer generation
- `CHUNK_SIZE`: 300-800 recommended, depends on document type

## 📝 Logging

The system writes runtime logs to `rag_system.log`, including:
- API requests and responses
- Document processing status
- Retrieval and reranking results
- Errors and warnings

Log level is controlled by the `LOG_LEVEL` environment variable.

## 🔍 Troubleshooting

### API calls fail
- Check that `SILICONFLOW_API_KEY` is set correctly
- Verify your network connection
- See the log file for detailed errors

### Vector store fails to load
- Make sure `code-data.py` has been run to build the store
- Check `CHROMA_DIR` and `COLLECTION_NAME`
- Verify ChromaDB files are intact

### Document parsing fails
- Install `PyPDF2` for PDF files
- Use UTF-8 encoding for TXT/MD files
- Check file paths

## 📄 License

For learning and demonstration purposes only.

## 🤝 Contributing

Issues and pull requests are welcome!

## 📞 Contact

Open an issue for questions or suggestions.

---

**Note**: a valid SiliconFlow API key is required. Get one at [SiliconFlow](https://siliconflow.cn).
