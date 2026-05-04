"""
Production-grade RAG using FAISS 
Faster and more scalable than in-memory numpy arrays.
"""
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch

try:
    import faiss
except ImportError:
    print("FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")
    faiss = None

from sentence_transformers import SentenceTransformer
from core.config import DEVICE


class RAGManagerFAISS:
    """
    Production-grade RAG Manager using FAISS for fast similarity search.
    Recommended for large-scale deployments.
    """
    
    def __init__(
        self, 
        db_path: str = None, 
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        use_gpu: bool = False,
        cache_path: str = None
    ):
        """
        Initialize FAISS-based RAG Manager.
        
        Args:
            db_path: Path to database folder
            embedding_model: Sentence transformer model name
            top_k: Number of documents to retrieve
            use_gpu: Use GPU-accelerated FAISS (requires GPU)
            cache_path: Path to cache embeddings file
        """
        if faiss is None:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        self.db_path = db_path or r"D:\hocj\AI\TTCS\DataBase\archive\FINAL FOOD DATASET"
        self.top_k = top_k
        self.use_gpu = use_gpu
        self.cache_path = cache_path or os.path.join(
            self.db_path, ".cache_faiss.pkl"
        )
        
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        
        # Load embedding model
        print("Loading embedding model (FAISS)...")
        self.embedding_model = SentenceTransformer(embedding_model)
        if DEVICE == "cuda":
            self.embedding_model.to(DEVICE)
        
        # Load or create index
        if os.path.exists(self.cache_path):
            print("Loading cached embeddings...")
            self._load_from_cache()
        else:
            print("Building new index...")
            self._load_database()
            self._create_faiss_index()
            self._save_to_cache()
    
    def _load_database(self):
        """Load CSV files from database."""
        print(f"Loading food database from {self.db_path}...")
        
        csv_files = [
            "FOOD-DATA-GROUP1.csv",
            "FOOD-DATA-GROUP2.csv",
            "FOOD-DATA-GROUP3.csv",
            "FOOD-DATA-GROUP4.csv",
            "FOOD-DATA-GROUP5.csv"
        ]
        
        for csv_file in csv_files:
            filepath = os.path.join(self.db_path, csv_file)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    print(f"  Loaded {csv_file}: {len(df)} rows")
                    
                    for idx, row in df.iterrows():
                        doc_text = self._create_document_text(row)
                        self.documents.append(doc_text)
                        self.metadata.append({
                            'source': csv_file,
                            'row_index': idx,
                            'data': row.to_dict()
                        })
                except Exception as e:
                    print(f"  Error loading {csv_file}: {e}")
        
        print(f"Total documents loaded: {len(self.documents)}")
    
    def _create_document_text(self, row: pd.Series) -> str:
        """Convert CSV row to document text."""
        doc_parts = []
        for col, value in row.items():
            if pd.notna(value):
                doc_parts.append(f"{col}: {str(value)[:100]}")
        return " | ".join(doc_parts)
    
    def _create_faiss_index(self):
        """Create FAISS index for fast similarity search."""
        print("Creating embeddings...")
        
        # Encode all documents
        self.embeddings = self.embedding_model.encode(
            self.documents,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        
        if self.use_gpu and torch.cuda.is_available():
            print("Creating GPU FAISS index...")
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            print("Creating CPU FAISS index...")
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype(np.float32))
        print(f"FAISS index created with {self.index.ntotal} vectors")
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        threshold: float = 0.0
    ) -> Tuple[List[str], List[float], List[dict]]:
        """
        Fast similarity search using FAISS.
        
        Args:
            query: User query
            top_k: Number of results
            threshold: Minimum similarity score to include
            
        Returns:
            Tuple of (documents, scores, metadata)
        """
        if top_k is None:
            top_k = self.top_k
        
        # Search for more candidates to allow filtering
        search_k = min(top_k * 2, self.index.ntotal)
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )[0]
        
        # Search FAISS
        distances, indices = self.index.search(
            np.array([query_embedding]).astype(np.float32),
            search_k
        )
        
        # Convert L2 distances to similarity scores
        # Distance -> Similarity: sim = 1 / (1 + distance)
        similarities = 1.0 / (1.0 + distances[0])
        
        # Filter by threshold
        valid_mask = similarities >= threshold
        valid_indices = indices[0][valid_mask]
        valid_similarities = similarities[valid_mask]
        
        # Sort by similarity descending and take top_k
        if len(valid_similarities) > 0:
            sort_order = np.argsort(valid_similarities)[::-1][:top_k]
            top_indices = valid_indices[sort_order]
            top_similarities = valid_similarities[sort_order]
        else:
            top_indices = np.array([])
            top_similarities = np.array([])
        
        results_docs = [self.documents[i] for i in top_indices]
        results_scores = [top_similarities[j] for j in range(len(top_indices))]
        results_meta = [self.metadata[i] for i in top_indices]
        
        return results_docs, results_scores, results_meta
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: Optional[int] = None
    ) -> List[Tuple[List[str], List[float], List[dict]]]:
        """
        Batch retrieve for multiple queries (efficient).
        
        Args:
            queries: List of queries
            top_k: Number of results per query
            
        Returns:
            List of (documents, scores, metadata) tuples
        """
        if top_k is None:
            top_k = self.top_k
        
        # Encode all queries at once
        query_embeddings = self.embedding_model.encode(
            queries,
            convert_to_numpy=True,
            batch_size=64
        )
        
        # Search FAISS
        distances, indices = self.index.search(
            query_embeddings.astype(np.float32),
            top_k
        )
        
        results = []
        for i, (query_distances, query_indices) in enumerate(zip(distances, indices)):
            similarities = 1.0 / (1.0 + query_distances)
            
            result_docs = [self.documents[j] for j in query_indices]
            result_scores = [similarities[k] for k in range(len(query_indices))]
            result_meta = [self.metadata[j] for j in query_indices]
            
            results.append((result_docs, result_scores, result_meta))
        
        return results
    
    def augment_prompt(self, query: str, context_docs: List[str] = None) -> str:
        """Augment prompt with context."""
        if context_docs is None:
            context_docs, _, _ = self.retrieve(query)
        
        context_text = "\n\n".join([f"- {doc}" for doc in context_docs])
        
        return f"""Based on the following food database information:

{context_text}

Please answer this question: {query}"""
    
    def get_augmented_response(self, query: str) -> Tuple[str, List[dict]]:
        """Get augmented prompt and metadata."""
        docs, scores, metadata = self.retrieve(query)
        augmented_prompt = self.augment_prompt(query, docs)
        return augmented_prompt, metadata
    
    def _save_to_cache(self):
        """Save embeddings and index to cache."""
        print(f"Saving cache to {self.cache_path}...")
        cache_data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'embeddings': self.embeddings,
            'top_k': self.top_k
        }
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Save FAISS index separately (binary format)
        index_path = self.cache_path.replace('.pkl', '.faiss')
        faiss.write_index(self.index, index_path)
        print(f"Cache saved successfully")
    
    def _load_from_cache(self):
        """Load embeddings and index from cache."""
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.documents = cache_data['documents']
            self.metadata = cache_data['metadata']
            self.embeddings = cache_data['embeddings']
            self.top_k = cache_data.get('top_k', self.top_k)
            
            # Load FAISS index
            index_path = self.cache_path.replace('.pkl', '.faiss')
            self.index = faiss.read_index(index_path)
            
            print(f"Loaded {len(self.documents)} documents from cache")
            print(f"FAISS index loaded with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Rebuilding index...")
            self._load_database()
            self._create_faiss_index()
            self._save_to_cache()
    
    def rebuild_cache(self):
        """Force rebuild of cache."""
        print("Rebuilding RAG index...")
        self.documents = []
        self.metadata = []
        self._load_database()
        self._create_faiss_index()
        self._save_to_cache()
        print("Index rebuilt")
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'faiss_vectors': self.index.ntotal if self.index is not None else 0,
            'cache_path': self.cache_path,
            'using_gpu': self.use_gpu
        }
