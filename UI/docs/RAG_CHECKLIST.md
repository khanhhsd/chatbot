# RAG Implementation Checklist

## Setup Checklist

### Prerequisites
- [ ] Python 3.7+ installed
- [ ] VS Code open with workspace
- [ ] Internet connection (for downloading embedding models)

### Installation
- [ ] Run setup: `python setup_rag.py` OR manually install
  - [ ] `pip install sentence-transformers`
  - [ ] `pip install scikit-learn`
  - [ ] `pip install pandas` (if not already installed)
  - [ ] `pip install faiss-cpu` (optional but recommended)

### Files Verification
In `d:\hocj\AI\TTCS\UI\` verify you have:
- [ ] rag_manager.py
- [ ] rag_manager_faiss.py
- [ ] gui_rag.py
- [ ] config.py (with RAG settings)
- [ ] main_with_rag.py
- [ ] test_rag_quick_start.py
- [ ] setup_rag.py
- [ ] RAG_README.md
- [ ] RAG_IMPLEMENTATION_GUIDE.md
- [ ] RAG_SYSTEM_ARCHITECTURE.md
- [ ] GET_STARTED_WITH_RAG.md

### Configuration
- [ ] Open `config.py`
- [ ] Verify: `USE_RAG = True`
- [ ] Check: `RAG_TOP_K = 5` (or adjust as needed)
- [ ] Check: `RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"`

### Testing
- [ ] Run: `python test_rag_quick_start.py`
- [ ] Verify: "Simple RAG test completed successfully!" appears
- [ ] Verify: Stats show documents loaded
- [ ] (Optional) Test FAISS if installed

### Integration  
- [ ] Update `main.py`:
  ```python
  from gui_rag import ChatbotGUI  # Instead of gui.ChatbotGUI
  ```
- [ ] Or use: `python main_with_rag.py` directly

### Final Test
- [ ] Run: `python main.py` (or main_with_rag.py)
- [ ] Wait for model to load
- [ ] Wait for RAG to initialize
- [ ] Try a query about food
- [ ] Verify retrieved documents appear in right panel
- [ ] Verify chatbot uses context in response

## Configuration Tuning Checklist

### For Better Quality
- [ ] Lower `RAG_SIMILARITY_THRESHOLD` (0.2-0.4)
- [ ] Increase `RAG_TOP_K` (7-10)
- [ ] Use better model: `RAG_EMBEDDING_MODEL = "all-mpnet-base-v2"`

### For Better Speed  
- [ ] Use FAISS: `pip install faiss-cpu`
- [ ] Reduce `RAG_TOP_K` (2-3)
- [ ] Use lightweight model: `RAG_EMBEDDING_MODEL = "all-distilroberta-v1"`

### For Lower Memory
- [ ] Use lightweight model
- [ ] Reduce `RAG_TOP_K`
- [ ] Switch to FAISS (uses less memory)

## Troubleshooting Checklist

### ImportError or ModuleNotFoundError
- [ ] Run: `pip install sentence-transformers scikit-learn`
- [ ] Verify Python version >= 3.7: `python --version`
- [ ] Check pip: `pip --version`

### RAG Not Working/No Context Shown
- [ ] Check config.py: `USE_RAG = True` ?
- [ ] Check database exists: `d:\hocj\AI\TTCS\DataBase\archive\FINAL FOOD DATASET`
- [ ] Run test: `python test_rag_quick_start.py`
- [ ] Check console for errors

### Slow Performance
- [ ] Install FAISS: `pip install faiss-cpu`
- [ ] Set in config: `RAG_VECTOR_DB_TYPE = "faiss"`
- [ ] Run test for performance comparison

### Out of Memory
- [ ] Reduce `RAG_TOP_K` to 3
- [ ] Use lighter model: `"all-distilroberta-v1"`
- [ ] Switch to FAISS (more memory efficient)
- [ ] Restart Python (clears memory)

### No Documents Retrieved
- [ ] Check food CSV files exist and have data
- [ ] Try test: `python test_rag_quick_start.py`
- [ ] Lower `RAG_SIMILARITY_THRESHOLD` in config
- [ ] Check queries are food-related

### GUI Issues
- [ ] Verify gui_rag.py exists in UI folder
- [ ] Check main imports gui_rag correctly
- [ ] Run: `python gui_rag.py` to test GUI directly

## Performance Checklist

### Before Optimization
- [ ] Run test: `python test_rag_quick_start.py`
- [ ] Note retrieval times
- [ ] Check memory usage

### Optimization Steps
- [ ] Try FAISS if available
- [ ] Adjust batch sizes
- [ ] Use caching
- [ ] Profile with test script

### After Optimization
- [ ] Verify faster responses
- [ ] Confirm memory usage acceptable
- [ ] Test quality still good

## Feature Checklist

### Basic RAG (rag_manager.py)
- [ ] Documents load from CSV
- [ ] Embeddings created
- [ ] Retrieval works
- [ ] Prompts augmented
- [ ] Test script passes

### Advanced RAG (rag_manager_faiss.py)
- [ ] FAISS installed
- [ ] Index created
- [ ] Faster retrieval (~5-10ms)
- [ ] Caching works
- [ ] Stats displayed

### UI with RAG (gui_rag.py)
- [ ] Chat interface works
- [ ] RAG context panel visible
- [ ] Documents displayed
- [ ] Similarity scores shown
- [ ] Toggle RAG on/off works
- [ ] Fall back to standard GUI if RAG disabled

## Documentation Checklist

- [ ] Read: GET_STARTED_WITH_RAG.md (START HERE)
- [ ] Read: RAG_README.md (quick reference)
- [ ] Reference: RAG_IMPLEMENTATION_GUIDE.md (when customizing)
- [ ] Check: RAG_SYSTEM_ARCHITECTURE.md (understand flow)

## Customization Checklist

### Modify Document Processing
- [ ] Edit: `rag_manager.py` → `_create_document_text()`
- [ ] Test: `python test_rag_quick_start.py`
- [ ] Verify: Document format changed

### Change Embedding Model
- [ ] Edit: `config.py` → `RAG_EMBEDDING_MODEL`
- [ ] Install model if needed (auto-downloads on first use)
- [ ] Test: `python test_rag_quick_start.py`
- [ ] Compare: Quality vs speed

### Customize Prompt Template
- [ ] Edit: `rag_manager.py` → `augment_prompt()`
- [ ] Test with manual retrieval
- [ ] Check: Responses use custom format

### Add Custom Retrieval Logic
- [ ] Create method in RAGManager
- [ ] Call from gui_rag.py
- [ ] Test functionality

## Deployment Checklist

### Development → Production
- [ ] Ensure FAISS installed for speed
- [ ] Cache embeddings on disk
- [ ] Test with production data
- [ ] Monitor memory usage
- [ ] Check retrieval quality
- [ ] Optimize RAG_TOP_K for use case

### Performance Validation
- [ ] Retrieval latency < 100ms
- [ ] Memory usage stable
- [ ] Quality metrics acceptable
- [ ] Error handling in place

### Testing Complete
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] UI test passes
- [ ] Performance acceptable
- [ ] Documentation complete

## Quick Status Check

Run this to verify everything:

```python
# test_status.py
import sys
sys.path.insert(0, r'd:\hocj\AI\TTCS\UI')

# Check imports
try:
    import sentence_transformers
    print("✓ sentence-transformers")
except: print("✗ sentence-transformers")

try:
    import sklearn
    print("✓ scikit-learn")
except: print("✗ scikit-learn")

try:
    import faiss
    print("✓ FAISS (optional)")
except: print("○ FAISS (optional)")

# Check files
from pathlib import Path
files = [
    "rag_manager.py",
    "rag_manager_faiss.py", 
    "gui_rag.py",
    "config.py"
]

for f in files:
    exists = Path(f"d:\\hocj\\AI\\TTCS\\UI\\{f}").exists()
    print(f"{'✓' if exists else '✗'} {f}")

# Check config
from config import USE_RAG, RAG_TOP_K
print(f"\nConfig:")
print(f"  USE_RAG = {USE_RAG}")
print(f"  RAG_TOP_K = {RAG_TOP_K}")
```

## Sign-Off

When all items checked:
- [ ] RAG is fully installed and configured
- [ ] Tests pass successfully
- [ ] UI displays retrieved documents
- [ ] System is ready for use
- [ ] Documentation reviewed

**You're all set! Start using RAG in your chatbot.**

---

**Questions?** Check the documentation files or run the test script.
