# Nemotron-3-nano RAG with MiniLM-L6-v2 RAG Chat Interface

Enterprise document question-answering system combining MiniLM-L6-v2 embeddings with Nemotron-3-nano via Ollama. Features a clean Streamlit interface, parallel PDF processing, and incremental indexing.

## Features

- **Streamlit Web Interface**: Clean, interactive chat interface with source citation
- **Parallel PDF Processing**: Multi-core PDF extraction and chunking with PyMuPDF
- **Incremental Indexing**: Smart file tracking to avoid re-indexing unchanged documents
- **Local Inference**: Uses Ollama for fast local LLM inference with Nemotron-3-nano
- **Semantic Search**: ChromaDB vector store with MiniLM-L6-v2 embeddings
- **Source Attribution**: Automatic citation of source documents in responses

## System Architecture

```
┌─────────────┐
│   PDFs      │
│  (data/)    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  pdf_chunker    │  ← Parallel extraction & chunking
│  (PyMuPDF)      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  embeddings     │  ← MiniLM-L6-v2 via sentence-transformers
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  vector_store   │  ← ChromaDB + incremental indexing
│  (Chroma)       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   retriever     │  ← Ollama (Nemotron-3-nano) + RAG pipeline
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Streamlit UI    │  ← Web interface with chat + sources
└─────────────────┘
```

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running
- **CUDA-capable GPU** (recommended for 2x RTX 3090 setup)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/kylanj7/MiniLM-L6-v2-with-nemotron-3-nano-RAG
cd MiniLM-L6-v2-with-nemotron-3-nano-RAG
```

### 2. Set up Python environment
```bash
conda create -n nemotron-rag python=3.10
conda activate nemotron-rag
pip install -r requirements.txt
```

### 3. Install and configure Ollama
```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Nemotron-3-nano model
ollama pull nemotron-3-nano:latest

# Verify Ollama is running
curl http://127.0.0.1:11434/api/tags
```

### 4. Verify installation
```bash
python src/embeddings.py  # Test embeddings
python src/retriever.py   # Test RAG pipeline
```

## Quick Start

### 1. Add Your Documents
Place PDF files in the `data/pdfs/` directory:
```bash
mkdir -p data/pdfs
cp your-documents/*.pdf data/pdfs/
```

### 2. Index Documents
Index your documents into the vector database:
```bash
# Index new or modified PDFs only (recommended)
python main.py index

# Force re-index all PDFs (slower, use if needed)
python main.py index --force
```

### 3. Launch Web Interface
Start the Streamlit application:
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the chat interface.

### 4. Alternative: Command Line Interface
For command-line usage:
```bash
python main.py query
```

## Configuration

Edit `config/settings.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `nemotron-3-nano:latest` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server endpoint |
| `CHUNK_SIZE` | 1000 | Characters per document chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K_RESULTS` | 1 | Number of chunks to retrieve |
| `MAX_TOKENS` | 2048 | Maximum generation length |
| `TEMPERATURE` | 0.7 | LLM sampling temperature |
| `NUMBER_OF_GPUs` | 2 | GPU configuration for Ollama |

## Module Overview

### `app.py`
- Streamlit web interface with chat functionality
- Real-time model loading with caching
- Source citation display with expandable sections
- Clean chat history management

### `src/pdf_chunker.py`
- Parallel PDF text extraction using PyMuPDF
- Multi-core processing with `ProcessPoolExecutor`
- Recursive character-based text splitting
- Metadata tracking for source attribution

### `src/embeddings.py`
- HuggingFace `sentence-transformers` integration
- MiniLM-L6-v2 embedding generation
- Standalone testing capability

### `src/vector_store.py`
- ChromaDB persistence management
- File hash-based change detection
- Incremental indexing with `indexed_files.json` log
- Batch document insertion with progress tracking

### `src/retriever.py`
- Ollama integration for Nemotron-3-nano
- Context-aware prompt construction
- Semantic similarity search with ChromaDB
- Source citation tracking and response formatting

## Performance Considerations

### Ollama Configuration
Configure Ollama for your hardware setup:
```bash
# Set GPU memory allocation (optional)
export OLLAMA_GPU_LAYERS=32
export OLLAMA_NUM_PARALLEL=2

# For dual RTX 3090 setup
export OLLAMA_GPU_MEMORY="20GB,20GB"
```

### Memory Usage
- **MiniLM-L6-v2**: ~22MB RAM for embeddings
- **Nemotron-3-nano**: ~2GB VRAM via Ollama
- **ChromaDB**: Scales with document corpus size

### CPU Optimization
PDF processing uses all available CPU cores. Limit if needed:
```python
# In pdf_chunker.py
max_workers = 4  # Instead of multiprocessing.cpu_count()
```

## Incremental Updates

The system automatically tracks indexed files. To add new documents:

1. Add PDFs to `data/pdfs/`
2. Re-run indexing - only new/modified files will be processed
3. New chunks are appended to existing vector store

To force re-indexing everything:
```bash
rm -rf vectorstore/chroma_db
python main.py index
```

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# Restart Ollama service
ollama serve

# Verify model is available
ollama list
```

### Streamlit Issues
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with specific port
streamlit run app.py --server.port 8502
```

### ChromaDB Persistence Issues
```bash
# Clear and rebuild vector store
rm -rf vectorstore/chroma_db
python src/pdf_chunker.py
```

### GPU Memory Issues
Reduce batch size or use CPU-only mode:
```python
# In config/settings.py
NUMBER_OF_GPUs = 0  # Force CPU mode
MEMORY_UTILIZATION = 0.6  # Reduce memory usage
```

## Project Structure

```
.
├── app.py                   # Streamlit web interface
├── main.py                  # Command-line interface
├── config/
│   └── settings.py          # Configuration parameters
├── data/
│   └── pdfs/                # Input PDF directory
├── models/                  # Local model storage (Nemotron-3-nano)
├── src/
│   ├── pdf_chunker.py       # PDF processing
│   ├── embeddings.py        # MiniLM-L6-v2 embeddings
│   ├── vector_store.py      # ChromaDB management
│   └── retriever.py         # RAG pipeline with Ollama
├── vectorstore/
│   └── chroma_db/           # Persisted vector database
├── requirements.txt
└── README.md
```

## Dependencies

Core libraries:
- `streamlit` - Web interface framework
- `langchain` - RAG framework and document processing
- `langchain-ollama` - Ollama LLM integration
- `langchain-chroma` - ChromaDB vector store
- `langchain-huggingface` - HuggingFace embeddings
- `sentence-transformers` - MiniLM-L6-v2 embedding model
- `PyMuPDF` - PDF text extraction
- `chromadb` - Vector database

## Model Details

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - 384-dimensional embeddings
  - Optimized for semantic similarity
  - Fast inference on CPU/GPU

- **Language Model**: `nemotron-3-nano:latest`
  - NVIDIA's efficient small language model
  - Runs locally via Ollama
  - Optimized for dual RTX 3090 setup

## License

MIT License

## Contributing

Contributions welcome! Please submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- Built with [Ollama](https://ollama.ai/) for local LLM inference
- Powered by [ChromaDB](https://www.trychroma.com/) vector database  
- Uses HuggingFace [sentence-transformers](https://www.sbert.net/) for embeddings
- NVIDIA [Nemotron-3-nano](https://ollama.ai/library/nemotron-mini) for generation
