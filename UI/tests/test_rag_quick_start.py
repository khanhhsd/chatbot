"""
Quick Start Guide for RAG Implementation
Run this script to test RAG functionality before integrating with the full UI.
"""

import sys
from pathlib import Path

# Ensure the script can find the modules
sys.path.insert(0, str(Path(__file__).parent))


def test_simple_rag():
    """Test the simple in-memory RAG implementation."""
    print("\n" + "="*60)
    print("Testing Simple RAG Manager (in-memory)")
    print("="*60)
    
    try:
        from rag.rag_manager import RAGManager
        
        # Initialize
        print("\n1. Initializing RAG Manager...")
        rag = RAGManager(top_k=3)
        
        # Test retrieval
        test_queries = [
            "What foods are high in protein?",
            "Which foods contain vitamin C?",
            "Tell me about carbohydrates in food",
        ]
        
        print("\n2. Testing retrievals...")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            docs, scores, metadata = rag.retrieve(query)
            
            print(f"Found {len(docs)} documents:")
            for i, (doc, score, meta) in enumerate(zip(docs, scores, metadata), 1):
                print(f"\n  [{i}] Score: {score:.4f}")
                print(f"      Source: {meta['source']}")
                print(f"      Preview: {doc[:150]}...")
        
        print("\n3. Testing augmented prompts...")
        augmented = rag.augment_prompt("What are the benefits of eating protein?")
        print(f"Augmented prompt length: {len(augmented)} chars")
        print(f"Preview: {augmented[:200]}...")
        
        print("\n✓ Simple RAG test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Install with: pip install sentence-transformers scikit-learn")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_faiss_rag():
    """Test the FAISS-based RAG implementation."""
    print("\n" + "="*60)
    print("Testing FAISS RAG Manager (production-grade)")
    print("="*60)
    
    try:
        from rag.rag_manager_faiss import RAGManagerFAISS
        
        # Initialize
        print("\n1. Initializing FAISS RAG Manager...")
        rag = RAGManagerFAISS(top_k=3)
        
        # Show stats
        stats = rag.get_stats()
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Embedding Dimension: {stats['embedding_dimension']}")
        print(f"   FAISS Vectors: {stats['faiss_vectors']}")
        print(f"   Using GPU: {stats['using_gpu']}")
        
        # Test retrieval
        test_queries = [
            "What foods are high in protein?",
            "Which foods contain vitamin C?",
        ]
        
        print("\n2. Testing fast retrievals...")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            docs, scores, metadata = rag.retrieve(query)
            
            for i, (doc, score, meta) in enumerate(zip(docs, scores, metadata), 1):
                print(f"  [{i}] Similarity: {score:.4f} | {meta['source']}")
        
        # Test batch retrieval
        print("\n3. Testing batch retrieval (more efficient)...")
        batch_results = rag.batch_retrieve(test_queries, top_k=2)
        print(f"Batch processed {len(batch_results)} queries")
        
        print("\n✓ FAISS RAG test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ FAISS not installed: {e}")
        print("  Install with: pip install faiss-cpu")
        print("  (or pip install faiss-gpu for GPU acceleration)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_implementations():
    """Compare simple vs FAISS RAG performance."""
    print("\n" + "="*60)
    print("Comparing RAG Implementations")
    print("="*60)
    
    import time
    
    test_query = "What are the benefits of eating fiber rich foods?"
    num_runs = 5
    
    # Test simple RAG
    print("\n1. Simple RAG (in-memory numpy)...")
    try:
        from rag.rag_manager import RAGManager
        rag_simple = RAGManager()
        
        start = time.time()
        for _ in range(num_runs):
            rag_simple.retrieve(test_query)
        simple_time = (time.time() - start) / num_runs
        print(f"   Average retrieval time: {simple_time*1000:.2f} ms")
    except Exception as e:
        print(f"   Error: {e}")
        simple_time = None
    
    # Test FAISS RAG
    print("\n2. FAISS RAG (optimized)...")
    try:
        from rag.rag_manager_faiss import RAGManagerFAISS
        rag_faiss = RAGManagerFAISS()
        
        start = time.time()
        for _ in range(num_runs):
            rag_faiss.retrieve(test_query)
        faiss_time = (time.time() - start) / num_runs
        print(f"   Average retrieval time: {faiss_time*1000:.2f} ms")
    except Exception as e:
        print(f"   Error: {e}")
        faiss_time = None
    
    # Compare
    if simple_time and faiss_time:
        speedup = simple_time / faiss_time
        print(f"\n   FAISS is {speedup:.1f}x faster" if speedup > 1 else f"   Simple is {1/speedup:.1f}x faster")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# RAG Implementation Quick Start")
    print("#"*60)
    
    # Test simple RAG
    simple_ok = test_simple_rag()
    
    # Test FAISS RAG
    faiss_ok = test_faiss_rag()
    
    # Compare if both available
    if simple_ok and faiss_ok:
        compare_implementations()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Simple RAG: {'✓ Available' if simple_ok else '✗ Not available'}")
    print(f"FAISS RAG:  {'✓ Available' if faiss_ok else '✗ Not available'}")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Review RAG_IMPLEMENTATION_GUIDE.md for detailed documentation")
    print("2. Enable RAG in config.py by setting USE_RAG = True")
    print("3. Run the full UI with: python main.py")
    print("4. For production: Use FAISS instead of simple RAG")
    print("5. Customize embedding model and processing in rag_manager*.py")
    

if __name__ == "__main__":
    main()
