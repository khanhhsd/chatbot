"""
RAG (Retrieval-Augmented Generation) Manager for the chatbot.
Handles loading database, creating embeddings, and retrieving relevant documents.
"""
import os
import pickle
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from core.config import DEVICE, RAG_TOP_K


class RAGManager:
    """
    Manages retrieval-augmented generation using food database.
    """
    
    def __init__(self, db_path: str = None, embedding_model: str = "all-MiniLM-L6-v2", top_k: int = None):
        """
        Initialize RAG Manager.
        
        Args:
            db_path: Path to the database folder
            embedding_model: Model name for sentence embeddings
            top_k: Number of documents to retrieve
        """
        self.db_path = db_path or r"D:\hocj\AI\TTCS\DataBase\archive\FINAL FOOD DATASET"
        self.top_k = top_k or RAG_TOP_K
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.cache_dir = Path(__file__).parent / ".rag_cache"
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_model.to(DEVICE)
        
        # Load and process database
        self._load_database()
        
        # Try to load embeddings from cache first
        if not self._load_embeddings_from_cache():
            self._create_embeddings()
            self._save_embeddings_to_cache()
        
    def _load_database(self):
        """Load CSV files from the database and prepare documents."""
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
                    
                    # Convert each row to a document
                    for idx, row in df.iterrows():
                        # Create document text from relevant columns
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
        """Convert a CSV row to document text."""
        doc_parts = []
        
        # Focus on meaningful columns: food name and key nutritional info
        important_cols = ['food', 'Caloric Value', 'Fat', 'Protein', 'Carbohydrates', 'Sugars']
        
        for col in important_cols:
            if col in row.index and pd.notna(row[col]):
                doc_parts.append(f"{col}: {row[col]}")
        
        # Add a few more nutritional values if available
        additional_cols = ['Dietary Fiber', 'Vitamin C', 'Calcium', 'Iron']
        for col in additional_cols:
            if col in row.index and pd.notna(row[col]) and row[col] != 0:
                doc_parts.append(f"{col}: {row[col]}")
        
        return " | ".join(doc_parts)
    
    def _create_embeddings(self):
        """Create embeddings for all documents."""
        if not self.documents:
            print("No documents to embed!")
            return
        
        print("Creating embeddings for documents...")
        self.embeddings = self.embedding_model.encode(
            self.documents,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=32
        )
        print(f"Embeddings created: shape {self.embeddings.shape}")
    
    def _load_embeddings_from_cache(self) -> bool:
        """
        Load embeddings from cache file.
        
        Returns:
            True if cache was loaded successfully, False otherwise
        """
        cache_path = str(self.cache_file)
        print(f"Checking for cache at: {cache_path}")
        
        if not self.cache_file.exists():
            print(f"Cache file not found at {cache_path}")
            return False
        
        try:
            print(f"Loading embeddings from cache ({self.cache_file})...")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache has the same number of documents
            cached_doc_count = len(cache_data.get('documents', []))
            current_doc_count = len(self.documents)
            
            print(f"  Cached documents: {cached_doc_count}, Current documents: {current_doc_count}")
            
            if cached_doc_count != current_doc_count:
                print(f"  [MISMATCH] Cache size mismatch. Regenerating...")
                return False
            
            # Verify embeddings exist and have correct shape
            cached_embeddings = cache_data.get('embeddings')
            if cached_embeddings is None or len(cached_embeddings) == 0:
                print(f"  [ERROR] Cache embeddings invalid. Regenerating...")
                return False
            
            self.embeddings = cached_embeddings
            self.metadata = cache_data.get('metadata', self.metadata)
            print(f"  [OK] Embeddings loaded from cache: shape {self.embeddings.shape}")
            return True
        except Exception as e:
            print(f"  [ERROR] Error loading cache: {e}")
            return False
    
    def _save_embeddings_to_cache(self):
        """Save embeddings to cache file."""
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[OK] Embeddings cached to {self.cache_file} ({self.embeddings.nbytes / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def clear_cache(self):
        """Clear the embedding cache. Embeddings will be regenerated on next launch."""
        if self.cache_file.exists():
            try:
                self.cache_file.unlink()
                print(f"Cache cleared: {self.cache_file}")
            except Exception as e:
                print(f"Error clearing cache: {e}")
        else:
            print("No cache file found.")
    
    def regenerate_embeddings(self):
        """Force regeneration of embeddings and update cache."""
        print("Regenerating embeddings...")
        self._create_embeddings()
        self._save_embeddings_to_cache()
        print("Embeddings regenerated and cached.")
    
    def _parse_nutrition_needs(self, query: str) -> Dict[str, float]:
        """
        Parse nutrition-related numeric requirements from a query.
        """
        patterns = {
            'protein': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?protein',
            'carbs': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?(?:carbs?|carbohydrates?)',
            'fat': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?fat',
            'calories': r'(\d+(?:\.\d+)?)\s*calories?',
            'fiber': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?(?:fiber|fibre)',
            'sugar': r'(\d+(?:\.\d+)?)\s*g(?:rams?)?\s*(?:of\s*)?sugar',
        }

        needs = {}
        text_lower = query.lower()
        for nutrient, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                needs[nutrient] = float(match.group(1))
        return needs

    def _is_retrievable(self, meta: dict) -> bool:
        """
        Determine whether a metadata entry should be returned by RAG.
        """
        row = meta.get('data', {})
        caloric_value = row.get('Caloric Value')
        try:
            if caloric_value is not None and float(caloric_value) > 400:
                return False
        except (TypeError, ValueError):
            pass
        return True

    def _category_boost(self, row: dict, needs: Dict[str, float]) -> float:
        """
        Apply a score boost for category-matching foods.
        """
        text = str(row.get('food', '')).lower()
        boost = 1.0

        if needs.get('protein', 0) > 0:
            meat_keywords = ['meat', 'chicken', 'beef', 'pork', 'turkey', 'fish', 'salmon', 'tuna', 'lamb', 'shrimp', 'duck', 'bacon', 'ham']
            if any(keyword in text for keyword in meat_keywords):
                boost *= 1.4

        if needs.get('carbs', 0) > 0:
            carb_keywords = ['bread', 'rice', 'pasta', 'noodle', 'cereal', 'oat', 'bagel', 'tortilla', 'cracker', 'bun', 'toast', 'potato']
            if any(keyword in text for keyword in carb_keywords):
                boost *= 1.4

        return boost

    def _retrieve_by_nutrition(self, needs: Dict[str, float], top_k: int) -> Tuple[List[str], List[float], List[dict]]:
        """
        Retrieve top foods by nutrient density for the specified needs.
        """
        if not needs:
            return [], [], []

        candidate_scores = {}
        for nutrient, target in needs.items():
            col = {
                'protein': 'Protein',
                'carbs': 'Carbohydrates',
                'fat': 'Fat',
                'calories': 'Caloric Value',
                'fiber': 'Dietary Fiber',
                'sugar': 'Sugars'
            }.get(nutrient)
            if not col:
                continue

            for i, meta in enumerate(self.metadata):
                if not self._is_retrievable(meta):
                    continue

                row = meta.get('data', {})
                if col not in row or not pd.notna(row[col]):
                    continue

                value = float(row[col])
                if value <= 0:
                    continue

                if target > 0:
                    score = value / target
                else:
                    score = value / 100.0

                score *= self._category_boost(row, needs)
                candidate_scores[i] = candidate_scores.get(i, 0.0) + score

        candidate_scores = [(idx, score) for idx, score in candidate_scores.items() if score > 0]

        if not candidate_scores:
            return [], [], []

        candidate_scores.sort(key=lambda item: item[1], reverse=True)
        top_candidates = candidate_scores[:top_k]
        top_indices = [item[0] for item in top_candidates]
        scores = [item[1] for item in top_candidates]

        results_docs = [self.documents[i] for i in top_indices]
        results_meta = [self.metadata[i] for i in top_indices]
        return results_docs, scores, results_meta

    def retrieve(self, query: str, top_k: int = None, threshold: float = 0.0) -> Tuple[List[str], List[float], List[dict]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of results to return
            threshold: Minimum similarity score to include (0.0 to 1.0)
            
        Returns:
            Tuple of (documents, scores, metadata)
        """
        if top_k is None:
            top_k = self.top_k

        nutrition_needs = self._parse_nutrition_needs(query)
        if nutrition_needs:
            docs, scores, meta = self._retrieve_by_nutrition(nutrition_needs, top_k)
            if docs:
                return docs, scores, meta
            # Fall back to semantic retrieval if strict nutrient retrieval returns nothing

        # Embed the query
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)[0]
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Filter by threshold and exclude high-calorie foods
        valid_indices = [i for i in range(len(similarities))
                         if similarities[i] >= threshold and self._is_retrievable(self.metadata[i])]
        if not valid_indices:
            return [], [], []

        # Sort valid indices by similarity (descending)
        valid_indices.sort(key=lambda i: similarities[i], reverse=True)

        # Take top_k from valid
        top_indices = valid_indices[:top_k]
        
        results_docs = [self.documents[i] for i in top_indices]
        results_scores = [similarities[i] for i in top_indices]
        results_meta = [self.metadata[i] for i in top_indices]
        
        return results_docs, results_scores, results_meta
    
    def augment_prompt(self, query: str, context_docs: List[str] = None) -> str:
        """
        Augment the user query with retrieved context.
        
        Args:
            query: Original user query
            context_docs: Documents to use as context (retrieved if None)
            
        Returns:
            Augmented prompt with context
        """
        if context_docs is None:
            context_docs, _, _ = self.retrieve(query)
        
        # Build augmented prompt
        context_text = "\n\n".join([f"- {doc}" for doc in context_docs])
        
        augmented_prompt = f"""Based on the following food database information:

{context_text}

Please answer this question: {query}"""
        
        return augmented_prompt
    
    def get_augmented_response(self, query: str, threshold: float = 0.0) -> Tuple[str, List[dict]]:
        """
        Get retrieval results and augmented prompt.
        
        Args:
            query: User query
            threshold: Minimum similarity score to include
            
        Returns:
            Tuple of (augmented_prompt, metadata)
        """
        docs, scores, metadata = self.retrieve(query, threshold=threshold)
        augmented_prompt = self.augment_prompt(query, docs)
        
        return augmented_prompt, metadata
