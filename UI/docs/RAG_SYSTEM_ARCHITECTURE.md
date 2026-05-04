# RAG System Architecture - Visual Guide

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     YOUR FOOD DATABASE                              │
│  FOOD-DATA-GROUP[1-5].csv (1000s of food items)                     │
│  Columns: Name, Protein, Carbs, Fats, Vitamins, etc.               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  RAG MANAGER (rag_manager*.py)                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. DATA LOADING                                             │   │
│  │    ├─ Read all CSV files                                   │   │
│  │    └─ Parse into document strings                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 2. EMBEDDING (Sentence Transformers)                       │   │
│  │    ├─ Convert documents → 384-dim vectors                  │   │
│  │    └─ Store in index (NumPy or FAISS)                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐        ┌──────────────────────┐
        │  rag_manager.py     │        │ rag_manager_faiss.py │
        │  (In-memory RAG)    │        │  (Production RAG)    │
        │                     │        │                      │
        │ • NumPy arrays      │        │ • FAISS index        │
        │ • 50-100ms latency  │        │ • 5-10ms latency     │
        │ • <5K docs ideal    │        │ • Any size           │
        └─────────────────────┘        │ • GPU accelerated    │
                                        └──────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │   RETRIEVAL INTERFACE           │
                    │  retrieve(query) → top-k docs   │
                    └─────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   ┌─────────┐           ┌──────────────────┐         ┌─────────────┐
   │Documents│      Similarity │Metadata    │         │ Scores      │
   │  List   │      Scores [0.8]│  ({src,   │         │ [0.8, 0.7,  │
   │         │             [0.7]│   row})   │         │  0.65, ...] │
   │ "Food:  │             [0.65]│           │         │             │
   │  X"     │                 │           │         │             │
   │ "Food:  │                 │           │         │             │
   │  Y"     │                 │           │         │             │
   └─────────┘           └──────────────────┘         └─────────────┘
        │                           │                           │
        └───────────────┬───────────┼───────────────────────────┘
                        ▼           ▼
                    ┌────────────────────────────┐
                    │  PROMPT AUGMENTATION       │
                    │                            │
                    │  Original: "What's high    │
                    │  in protein?"              │
                    │           │                │
                    │           ▼                │
                    │  Augmented: "Based on      │
                    │  database:                 │
                    │  - Chicken: 31g protein   │
                    │  - Fish: 25g protein      │
                    │  - Eggs: 13g protein      │
                    │  What's high in protein?" │
                    └────────────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────┐
                    │  LANGUAGE MODEL            │
                    │  (Your local model)        │
                    │  Generates response with   │
                    │  database context          │
                    └────────────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────┐
                    │  RESPONSE + CONTEXT UI     │
                    │  (gui_rag.py)              │
                    │  ┌──────────────────────┐  │
                    │  │ Chat              │  │  │
                    │  │────────────────────│  │  │
                    │  │ AI: Chicken, fish..│  │  │
                    │  │                    │  │  │
                    │  │ Retrieved Context: │  │  │
                    │  │ [1] Score: 0.89    │  │  │
                    │  │     Chicken data   │  │  │
                    │  │ [2] Score: 0.87    │  │  │
                    │  │     Fish data      │  │  │
                    │  └──────────────────────┘  │
                    └────────────────────────────┘
```

## Implementation Timeline

```
IMMEDIATE (5 min):
  ✓ Install: pip install sentence-transformers scikit-learn
  ✓ Edit config.py: USE_RAG = True
  ✓ Run main.py with gui_rag.py

SHORT TERM (30 min):
  ✓ Test RAG: python test_rag_quick_start.py
  ✓ Tune RAG_TOP_K and similarity threshold
  ✓ Try different queries

MEDIUM TERM (1-2 hours):
  ✓ Switch to FAISS for production
  ✓ Enable caching
  ✓ Test performance

LONG TERM (optional):
  ✓ Implement cross-encoder reranking
  ✓ Add semantic filtering
  ✓ Multi-hop retrieval
  ✓ GPU acceleration
```

## Code Integration Points

### In main_with_rag.py:
```python
from config import USE_RAG

if USE_RAG:
    from gui_rag import ChatbotGUI  # With RAG context
else:
    from gui import ChatbotGUI      # Standard UI
```

### In gui_rag.py:
```python
self.rag_manager = RAGManager()  # Initialize

# In _generate_response():
augmented_prompt, metadata = self.rag_manager.get_augmented_response(query)
# Use augmented_prompt instead of regular prompt
```

### In config.py:
```python
USE_RAG = True                      # Toggle
RAG_TOP_K = 5                       # How many docs
RAG_EMBEDDING_MODEL = "all..."      # Which model
RAG_SIMILARITY_THRESHOLD = 0.3      # Min score
```

## Comparison: Simple vs FAISS

| Feature | Simple RAG | FAISS RAG |
|---------|-----------|-----------|
| Setup Time | Seconds | 1-2 min |
| Query Latency | 50-100ms | 5-10ms |
| Memory | ~400MB-2GB | ~200MB-1GB |
| Max Docs | ~5000 | Unlimited |
| GPU Support | No | Yes |
| Caching | Manual | Built-in |
| Scalability | :x: | :check_mark: |
| Production Ready | :x: | :check_mark: |

**Recommendation**: Start with simple, upgrade to FAISS when speed becomes critical.

## Customization Points

1. **Document Processing** - Modify `_create_document_text()`
2. **Embedding Model** - Change `RAG_EMBEDDING_MODEL` in config
3. **Similarity Threshold** - Tune `RAG_SIMILARITY_THRESHOLD`
4. **Result Reranking** - Add in `retrieve()` method
5. **Prompt Template** - Customize `augment_prompt()`
6. **UI Display** - Enhance `_update_rag_context()` in gui_rag.py

## Performance Optimization Checklist

- [ ] Use FAISS instead of simple RAG
- [ ] Cache embeddings on disk
- [ ] Adjust batch size to GPU memory
- [ ] Profile with test_rag_quick_start.py
- [ ] Try faster embedding models
- [ ] Implement result deduplication
- [ ] Add query caching
- [ ] Monitor memory usage

## Next Steps

1. **Quick Start** (now):
   ```bash
   python setup_rag.py
   python test_rag_quick_start.py
   ```

2. **Integration** (next):
   ```bash
   # Update main.py to use gui_rag.py
   python main.py
   ```

3. **Optimization** (after testing):
   - Review `RAG_IMPLEMENTATION_GUIDE.md`
   - Switch to FAISS
   - Adjust parameters

## Files Reference

| File | Purpose |
|------|---------|
| rag_manager.py | Core RAG logic (simple) |
| rag_manager_faiss.py | Production RAG (fast) |
| gui_rag.py | Enhanced UI with RAG |
| config.py | Configuration |
| RAG_README.md | Quick start guide |
| RAG_IMPLEMENTATION_GUIDE.md | Technical docs |
| test_rag_quick_start.py | Testing tool |
| setup_rag.py | Dependency installer |

All files are in: `d:\hocj\AI\TTCS\UI\`
