import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from threading import Lock
import hashlib

from config.settings import EMBEDDING_MODEL, MODEL_DIR
from src.utils.text_utils import clean_text, normalize_whitespace
from src.models.lazy_loader import LazyModelLoader 


class EmbeddingModel:
    """Optimized embedding model with lazy loading, caching and batch processing."""
    
    def __init__(self, model_name: str = None, cache_embeddings: bool = True):
        self.model_name = model_name or EMBEDDING_MODEL
        self.cache_embeddings = cache_embeddings
        self.logger = logging.getLogger(__name__)
        
        # NEW: Lazy model loader instead of direct model loading
        self.lazy_loader = LazyModelLoader(cache_size_limit=1)
        self.model = None  # Will be loaded on demand
        
        # Embedding caching (separate from model caching)
        self.embedding_cache = {}
        self.cache_file = MODEL_DIR / "embedding_cache.pkl"
        self.cache_lock = Lock()
        
        # Model info
        self.embedding_dim = None
        self.max_seq_length = 512
        self.device = self._get_optimal_device()
        
        # Performance tracking
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_embeddings": 0,
            "total_time": 0.0
        }
        
        # Initialize (but don't load model yet)
        self._initialize_model_info()
        self._load_cache()
    
    def _get_optimal_device(self) -> str:
        """Determine the best device for embeddings."""
        if torch.cuda.is_available():
            # Check if we have enough GPU memory
            try:
                torch.cuda.empty_cache()
                return "cuda"
            except Exception:
                self.logger.warning("CUDA available but unusable, falling back to CPU")
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        
        return "cpu"
    
    def _initialize_model_info(self) -> None:
        """Initialize model info without loading the actual model."""
        try:
            self.logger.info(f"Preparing lazy loading for model: {self.model_name}")
            
            # Create model directory if it doesn't exist
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            # Set default values (will be updated when model is actually loaded)
            self.embedding_dim = 768 # Updated from 384 to 768
            self.max_seq_length = 512
            
            self.logger.info(f"Model info initialized - Device: {self.device}, Estimated Dim: {self.embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model info: {e}")
            self.embedding_dim = 768  # Updated from 384 to 768
    
    def _get_model(self) -> Optional[SentenceTransformer]:
        """Get model using lazy loader (loads on first access)."""
        if self.model is None:
            self.logger.info(f"Loading model on demand: {self.model_name}")
            self.model = self.lazy_loader.load_on_demand(self.model_name, self.device)
            
            # Update actual model info once loaded
            if self.model is not None:
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.max_seq_length = self.model.max_seq_length
                self.logger.info(f"Model loaded - Actual Dim: {self.embedding_dim}, Max Length: {self.max_seq_length}")
        
        return self.model
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if not self.cache_embeddings:
            return
        
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            self.logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if not self.cache_embeddings or not self.embedding_cache:
            return
        
        try:
            with self.cache_lock:
                # Create directory if it doesn't exist
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.embedding_cache, f)
                
                self.logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            self.logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Normalize text for consistent caching
        normalized_text = normalize_whitespace(clean_text(text))
        return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
    
    def encode(self, texts: Union[str, List[str]], 
               batch_size: int = 32,
               show_progress: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode text(s) to embeddings with lazy loading, caching and optimization."""
        
        # NEW: Get model using lazy loader
        model = self._get_model()
        if model is None:
            self.logger.warning("Model not available, returning zero embeddings")
            if isinstance(texts, str):
                return np.zeros(self.embedding_dim)
            else:
                return [np.zeros(self.embedding_dim) for _ in texts]
        
        # Handle single string input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        start_time = time.time()
        embeddings = []
        texts_to_embed = []
        cache_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                embeddings.append(np.zeros(self.embedding_dim))
                continue
            
            cache_key = self._get_cache_key(text)
            
            if self.cache_embeddings and cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                self.stats["cache_hits"] += 1
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                cache_indices.append((i, cache_key))
                self.stats["cache_misses"] += 1
        
        # Embed uncached texts in batches
        if texts_to_embed:
            try:
                with torch.no_grad():  # Disable gradient computation for inference
                    new_embeddings = model.encode(
                        texts_to_embed,
                        batch_size=batch_size,
                        show_progress_bar=show_progress,
                        convert_to_tensor=False,
                        normalize_embeddings=True  # L2 normalization for better similarity
                    )
                
                # Store new embeddings in cache and result array
                for j, (original_idx, cache_key) in enumerate(cache_indices):
                    embedding = new_embeddings[j]
                    embeddings[original_idx] = embedding
                    
                    # Cache the embedding
                    if self.cache_embeddings:
                        with self.cache_lock:
                            self.embedding_cache[cache_key] = embedding
            
            except Exception as e:
                self.logger.error(f"Failed to compute embeddings: {e}")
                # Fill with zero embeddings as fallback
                for original_idx, _ in cache_indices:
                    embeddings[original_idx] = np.zeros(self.embedding_dim)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats["total_embeddings"] += len(texts)
        self.stats["total_time"] += processing_time
        
        # Save cache periodically
        if len(self.embedding_cache) % 100 == 0:
            self._save_cache()
        
        # Return single embedding or list
        if single_input:
            return embeddings[0]
        else:
            return embeddings
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        try:
            embedding1 = self.encode(text1)
            embedding2 = self.encode(text2)
            
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Failed to compute similarity: {e}")
            return 0.0
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute pairwise similarity matrix for a list of texts."""
        try:
            embeddings = self.encode(texts)
            embeddings_array = np.array(embeddings)
            
            similarity_matrix = cosine_similarity(embeddings_array)
            return similarity_matrix
            
        except Exception as e:
            self.logger.warning(f"Failed to compute similarity matrix: {e}")
            return np.zeros((len(texts), len(texts)))
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar texts to query."""
        try:
            query_embedding = self.encode(query_text)
            candidate_embeddings = self.encode(candidate_texts)
            
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    candidate_embedding.reshape(1, -1)
                )[0][0]
                similarities.append((candidate_texts[i], float(similarity)))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.warning(f"Failed to find similar texts: {e}")
            return []
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 5) -> Dict[int, List[str]]:
        """Cluster texts based on semantic similarity."""
        try:
            from sklearn.cluster import KMeans
            
            embeddings = self.encode(texts)
            embeddings_array = np.array(embeddings)
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Group texts by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(texts[i])
            
            return clusters
            
        except Exception as e:
            self.logger.warning(f"Failed to cluster texts: {e}")
            return {0: texts}  # Return all texts in one cluster as fallback
    
    def get_text_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features from text using the embedding."""
        try:
            embedding = self.encode(text)
            
            features = {
                "embedding_norm": float(np.linalg.norm(embedding)),
                "embedding_mean": float(np.mean(embedding)),
                "embedding_std": float(np.std(embedding)),
                "embedding_dim": len(embedding),
                "text_length": len(text),
                "word_count": len(text.split()),
            }
            
            # Compute similarity with common heading patterns
            heading_patterns = [
                "introduction", "conclusion", "methodology", "results",
                "chapter", "section", "appendix", "references"
            ]
            
            pattern_similarities = {}
            for pattern in heading_patterns:
                similarity = self.compute_similarity(text, pattern)
                pattern_similarities[f"sim_to_{pattern}"] = similarity
            
            features.update(pattern_similarities)
            return features
            
        except Exception as e:
            self.logger.warning(f"Failed to extract text features: {e}")
            return {}
    
    def optimize_for_inference(self) -> None:
        """Optimize model for faster inference."""
        model = self._get_model()
        if model is None:
            return
        
        try:
            # Optimize using lazy loader
            self.lazy_loader.optimize_for_inference(self.model_name)
            self.logger.info("Model optimized for inference")
            
        except Exception as e:
            self.logger.warning(f"Failed to optimize model: {e}")
    
    def preload_model(self) -> bool:
        """Preload the model for faster first access."""
        try:
            self.logger.info(f"Preloading model: {self.model_name}")
            return self.lazy_loader.preload_model(self.model_name, self.device)
        except Exception as e:
            self.logger.warning(f"Failed to preload model: {e}")
            return False
    
    def warmup_model(self, sample_texts: List[str] = None) -> None:
        """Warm up the model with sample texts."""
        try:
            self.lazy_loader.warmup(self.model_name, sample_texts)
        except Exception as e:
            self.logger.warning(f"Failed to warm up model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        base_info = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.max_seq_length,
            "device": self.device,
            "embedding_cache_size": len(self.embedding_cache),
            "model_loaded": self.lazy_loader.is_model_loaded(self.model_name),
            "stats": self.stats.copy()
        }
        
        # Add lazy loader stats
        loader_stats = self.lazy_loader.get_cache_stats()
        base_info.update(loader_stats)
        
        return base_info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (self.stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        avg_time_per_embedding = (self.stats["total_time"] / self.stats["total_embeddings"]) if self.stats["total_embeddings"] > 0 else 0
        
        embedding_stats = {
            "embedding_stats": {
                "total_embeddings_computed": self.stats["total_embeddings"],
                "cache_hit_rate": f"{cache_hit_rate:.1f}%",
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "total_processing_time": f"{self.stats['total_time']:.2f}s",
                "avg_time_per_embedding": f"{avg_time_per_embedding * 1000:.2f}ms",
                "embedding_cache_size": len(self.embedding_cache)
            }
        }
        
        # Add lazy loader performance stats
        loader_stats = self.lazy_loader.get_cache_stats()
        embedding_stats.update(loader_stats)
        
        # Add memory usage
        memory_stats = self.lazy_loader.get_memory_usage()
        embedding_stats["memory_stats"] = memory_stats
        
        return embedding_stats
    
    def clear_cache(self) -> None:
        """Clear both embedding and model caches."""
        with self.cache_lock:
            self.embedding_cache.clear()
            self.stats["cache_hits"] = 0
            self.stats["cache_misses"] = 0
        
        # Clear model cache
        self.lazy_loader.clear_all_cache()
        self.model = None
        
        self.logger.info("All caches cleared")
    
    def clear_model_cache_only(self) -> None:
        """Clear only the model cache, keep embedding cache."""
        self.lazy_loader.clear_all_cache()
        self.model = None
        self.logger.info("Model cache cleared")
    
    def precompute_embeddings(self, texts: List[str], batch_size: int = 32) -> None:
        """Precompute embeddings for a list of texts."""
        self.logger.info(f"Precomputing embeddings for {len(texts)} texts")
        
        # Filter out already cached texts
        texts_to_compute = []
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key not in self.embedding_cache:
                texts_to_compute.append(text)
        
        if texts_to_compute:
            self.logger.info(f"Computing {len(texts_to_compute)} new embeddings")
            self.encode(texts_to_compute, batch_size=batch_size, show_progress=True)
            self._save_cache()
        else:
            self.logger.info("All embeddings already cached")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'embedding_cache') and self.embedding_cache:
                self._save_cache()
            if hasattr(self, 'lazy_loader'):
                self.lazy_loader.clear_all_cache()
        except Exception:
            pass  # Ignore errors during cleanup