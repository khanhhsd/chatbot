# RAG (Retrieval-Augmented Generation) Implementation Guide

## Overview

RAG is a technique that combines information retrieval with language models. Instead of relying solely on the model's training data, RAG retrieves relevant documents from your database and uses them to augment the prompt before generation.

### How RAG Works

```
User Query
    ↓
[Embedding Model] → Query Embedding
    ↓
[Vector Database] → Similarity Search
    ↓
Retrieved Documents (Top-K Results)
    ↓
Augmented Prompt = Query + Retrieved Context
    ↓
[Language Model] → Generate Response
```

## Architecture

Your setup uses:
- **Data Source**: Food database (5 CSV files)
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Storage**: In-memory NumPy arrays
- **LLM**: Your local language model
- **UI**: Tkinter with RAG context display

## Quick Start

### 1. Install Dependencies

```bash
pip install sentence-transformers
pip install scikit-learn
```

For GPU acceleration:
```bash
pip install torch scikit-learn sentence-transformers
```

### 2. Enable RAG in Your Application

Edit `config.py`:
```python
USE_RAG = True
RAG_TOP_K = 5  # Number of documents to retrieve
```

### 3. Run with RAG

Update `main.py` to use the RAG-enabled GUI:

```python
from gui_rag import ChatbotGUI  # Instead of from gui import ChatbotGUI
```

## How to Use

### Option A: Simple Integration (Current Implementation)

The `rag_manager.py` provides a complete RAG solution:

```python
from rag_manager import RAGManager

# Initialize
rag = RAGManager()

# Retrieve documents
docs, scores, metadata = rag.retrieve("What foods are high in protein?")

# Get augmented prompt
augmented_prompt = rag.augment_prompt("What foods are high in protein?")

# Get both
augmented_prompt, metadata = rag.get_augmented_response("What foods are high in protein?")
```

### Option B: Advanced Vector Database (FAISS - Production Recommended)

For better performance with large databases, use FAISS:

```python
pip install faiss-cpu  # or faiss-gpu for GPU
```

Create `rag_manager_faiss.py`:

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGManagerFAISS:
    def __init__(self, db_path, embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.metadata = []
        
        # Load data
        self._load_database(db_path)
        
        # Create FAISS index
        self.embeddings = self.embedding_model.encode(
            self.documents, 
            batch_size=32
        )
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype(np.float32))
    
    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype(np.float32), 
            top_k
        )
        
        results = [
            (self.documents[i], distances[0][j], self.metadata[i])
            for j, i in enumerate(indices[0])
        ]
        return results
```

### Option C: Milvus (Scalable for Enterprise)

For large-scale deployments:

```bash
# Install Milvus Python client
pip install pymilvus
```

Setup with Docker:
```bash
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

## Configuration Options

### In `config.py`:

```python
# Enable/Disable RAG
USE_RAG = True

# Database path
RAG_DATABASE_PATH = r"D:\hocj\AI\TTCS\DataBase\archive\FINAL FOOD DATASET"

# Embedding model (lightweight options)
RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 22MB
# Other options:
# - "all-mpnet-base-v2" (Better quality, slower)
# - "paraphrase-MiniLM-L6-v2" (Good balance)
# - "sentence-transformers/all-distilroberta-v1" (Lightweight)

# Number of documents to retrieve
RAG_TOP_K = 5

# Minimum similarity threshold
RAG_SIMILARITY_THRESHOLD = 0.3

# Vector database type
RAG_VECTOR_DB_TYPE = "simple"  # Options: "faiss", "milvus", "simple"
```

## Customization Examples

### 1. Custom Document Processing

Modify `_create_document_text()` in `rag_manager.py`:

```python
def _create_document_text(self, row: pd.Series) -> str:
    # Example: Only use specific columns
    doc_parts = []
    
    important_fields = ['Food_Name', 'Protein', 'Calories', 'Nutrients']
    for field in important_fields:
        if field in row.index and pd.notna(row[field]):
            doc_parts.append(f"{field}: {row[field]}")
    
    return " | ".join(doc_parts)
```

### 2. Custom Similarity Threshold

```python
def retrieve_filtered(self, query: str, threshold=0.5) -> List[str]:
    docs, scores, metadata = self.retrieve(query)
    
    # Filter by threshold
    filtered = [
        (doc, score, meta)
        for doc, score, meta in zip(docs, scores, metadata)
        if score > threshold
    ]
    
    return filtered
```

### 3. Reranking Results

Add semantic reranking:

```python
from sentence_transformers.util import semantic_search

def retrieve_with_rerank(self, query: str, top_k=5):
    # Initial retrieval
    docs, scores, metadata = self.retrieve(query, top_k=top_k*2)
    
    # Rerank using cross-encoder for better quality
    # (requires CrossEncoder import)
    
    return docs[:top_k], scores[:top_k], metadata[:top_k]
```

## Performance Tips

### 1. Batch Processing

```python
# Embed documents in batches
self.embeddings = self.embedding_model.encode(
    self.documents,
    batch_size=32,  # Adjust based on GPU memory
    show_progress_bar=True,
    convert_to_tensor=False
)
```

### 2. Caching

```python
import pickle

def save_embeddings(self, path):
    with open(path, 'wb') as f:
        pickle.dump({
            'embeddings': self.embeddings,
            'documents': self.documents,
            'metadata': self.metadata
        }, f)

def load_embeddings(self, path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.documents = data['documents']
        self.metadata = data['metadata']
```

### 3. Asynchronous Retrieval

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def retrieve_async(self, query: str):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, 
            self.retrieve,
            query
        )
    return result
```

## Prompt Augmentation Strategies

### Strategy 1: Simple Context (Current)

```
Based on the following food database information:
- [Doc 1]
- [Doc 2]
- [Doc 3]

Please answer: {query}
```

### Strategy 2: Detailed Context

```
Food Database Context:
Source files: {sources}
Relevant information:
{formatted_documents}

Additional context about the query {analysis}

User Question: {query}

Answer based on the retrieved context:
```

### Strategy 3: Few-Shot with Examples

```
Here are relevant examples from our database:
{examples}

Based on these patterns, answer the question:
{query}
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce embedding batch size or use FAISS GPU
```python
self.embeddings = self.embedding_model.encode(
    self.documents,
    batch_size=8,  # Reduce from 32
    show_progress_bar=True
)
```

### Issue: Slow Retrieval

**Solution**: Use FAISS or cache embeddings
```python
# Use FAISS implementation
from rag_manager_faiss import RAGManagerFAISS
rag = RAGManagerFAISS(db_path)
```

### Issue: Irrelevant Results

**Solution**: Increase similarity threshold or adjust system prompt
```python
# In config.py
RAG_SIMILARITY_THRESHOLD = 0.5  # More strict

# Or customize prompt augmentation
def augment_prompt(self, query, context_docs):
    return f"""You are a food expert. Use ONLY the following information:
{context_docs}

If the information doesn't answer the question, say so.
Question: {query}"""
```

## Integration Checklist

- [ ] Install dependencies: `pip install sentence-transformers scikit-learn`
- [ ] Copy `rag_manager.py` to UI folder
- [ ] Update `config.py` with RAG settings
- [ ] Use `gui_rag.py` instead of `gui.py` in `main.py`
- [ ] Test with sample queries
- [ ] Adjust `RAG_TOP_K` and similarity thresholds
- [ ] Monitor performance and optimize

## Advanced Topics

### Custom Embedding Models

```python
# Fast models (< 100MB)
"all-MiniLM-L6-v2"           # 22MB
"all-distilroberta-v1"        # 27MB
"paraphrase-MiniLM-L6-v2"     # 46MB

# Quality models (> 400MB)
"all-mpnet-base-v2"           # 424MB
"all-roberta-large-v1"        # 498MB
```

### Multi-Modal RAG

For food images + descriptions:
```python
from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer('clip-ViT-B-32')
text_embedding = model.encode("high protein food")
image_embedding = model.encode(Image.open("food.jpg"))
```

### Hybrid Search

Combine keyword search with semantic search:
```python
def hybrid_retrieve(self, query, alpha=0.5):
    # Semantic search
    semantic_scores = cosine_similarity([query_emb], self.embeddings)
    
    # BM25 keyword search
    keyword_scores = bm25.get_scores(query.split())
    
    # Combine
    combined = alpha * semantic_scores + (1-alpha) * keyword_scores
    return combined.argsort()[::-1][:self.top_k]
```

## References

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [RAG Survey Paper](https://arxiv.org/abs/2312.10997)
- [LlamaIndex](https://www.llamaindex.ai/) - Python RAG framework
