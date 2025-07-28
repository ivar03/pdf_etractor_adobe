import logging
import gc
import time
from typing import Optional, Dict, Any
from pathlib import Path
import threading
from sentence_transformers import SentenceTransformer

from config.settings import MODEL_DIR


class LazyModelLoader:
    """
    Edge-Cache Model Loader for optimized memory management and faster startup.
    
    Features:
    - Lazy loading: Models loaded only when needed
    - Memory management: Automatic cleanup and garbage collection
    - Thread-safe: Multiple threads can safely access the loader
    - Cache management: Intelligent model caching with size limits
    - Performance tracking: Load time monitoring and statistics
    """
    
    def __init__(self, cache_size_limit: int = 1):
        """
        Initialize the lazy model loader.
        
        Args:
            cache_size_limit: Maximum number of models to keep in memory
        """
        self.logger = logging.getLogger(__name__)
        self.cache_size_limit = cache_size_limit
        
        # Thread-safe model cache
        self._models: Dict[str, SentenceTransformer] = {}
        self._model_load_times: Dict[str, float] = {}
        self._access_count: Dict[str, int] = {}
        self._lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            "models_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_load_time": 0.0,
            "memory_clears": 0
        }
        
        # Model directory setup
        self.model_dir = Path(MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"LazyModelLoader initialized with cache limit: {cache_size_limit}")
    
    def load_on_demand(self, model_name: str, device: str = "cpu") -> Optional[SentenceTransformer]:
        """
        Load model on demand with intelligent caching.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to load model on ("cpu", "cuda", "mps")
            
        Returns:
            Loaded SentenceTransformer model or None if loading fails
        """
        with self._lock:
            # Check if model is already cached
            if model_name in self._models:
                self.stats["cache_hits"] += 1
                self._access_count[model_name] += 1
                self.logger.debug(f"Model '{model_name}' loaded from cache")
                return self._models[model_name]
            
            # Cache miss - need to load model
            self.stats["cache_misses"] += 1
            self.logger.info(f"Loading model '{model_name}' on device '{device}'...")
            
            try:
                start_time = time.time()
                
                # Check if we need to clear cache first
                self._manage_cache_size()
                
                # Load the model
                model = SentenceTransformer(model_name, device=device)
                model.eval()  # Set to evaluation mode for inference
                
                # Track loading performance
                load_time = time.time() - start_time
                self._model_load_times[model_name] = load_time
                self._access_count[model_name] = 1
                self.stats["models_loaded"] += 1
                self.stats["total_load_time"] += load_time
                
                # Cache the model
                self._models[model_name] = model
                
                self.logger.info(f"Model '{model_name}' loaded successfully in {load_time:.2f}s")
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model '{model_name}': {e}")
                return None
    
    def _manage_cache_size(self) -> None:
        """Manage cache size by removing least recently used models."""
        if len(self._models) >= self.cache_size_limit:
            # Find least recently used model
            lru_model = min(self._access_count.items(), key=lambda x: x[1])[0]
            self.clear_model(lru_model)
            self.logger.info(f"Removed LRU model '{lru_model}' from cache")
    
    def clear_model(self, model_name: str) -> None:
        """
        Clear a specific model from cache.
        
        Args:
            model_name: Name of the model to clear
        """
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                del self._access_count[model_name]
                if model_name in self._model_load_times:
                    del self._model_load_times[model_name]
                
                # Force garbage collection
                gc.collect()
                self.stats["memory_clears"] += 1
                
                self.logger.info(f"Cleared model '{model_name}' from cache")
    
    def clear_all_cache(self) -> None:
        """Clear all models from cache and force garbage collection."""
        with self._lock:
            model_count = len(self._models)
            
            self._models.clear()
            self._access_count.clear()
            self._model_load_times.clear()
            
            # Aggressive garbage collection
            gc.collect()
            self.stats["memory_clears"] += model_count
            
            self.logger.info(f"Cleared all {model_count} models from cache")
    
    def preload_model(self, model_name: str, device: str = "cpu") -> bool:
        """
        Preload a model for faster access later.
        
        Args:
            model_name: Name of the model to preload
            device: Device to load model on
            
        Returns:
            True if preloading successful, False otherwise
        """
        self.logger.info(f"Preloading model '{model_name}'...")
        model = self.load_on_demand(model_name, device)
        return model is not None
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded in cache.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is loaded, False otherwise
        """
        with self._lock:
            return model_name in self._models
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded model names."""
        with self._lock:
            return list(self._models.keys())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        with self._lock:
            total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
            hit_rate = (self.stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
            
            avg_load_time = (self.stats["total_load_time"] / self.stats["models_loaded"]) if self.stats["models_loaded"] > 0 else 0
            
            return {
                "cache_stats": {
                    "models_currently_loaded": len(self._models),
                    "cache_size_limit": self.cache_size_limit,
                    "loaded_models": list(self._models.keys()),
                    "cache_hit_rate": f"{hit_rate:.1f}%",
                    "cache_hits": self.stats["cache_hits"],
                    "cache_misses": self.stats["cache_misses"],
                    "total_models_loaded": self.stats["models_loaded"],
                    "memory_clears": self.stats["memory_clears"]
                },
                "performance_stats": {
                    "total_load_time": f"{self.stats['total_load_time']:.2f}s",
                    "avg_load_time": f"{avg_load_time:.2f}s",
                    "model_load_times": dict(self._model_load_times),
                    "access_counts": dict(self._access_count)
                }
            }
    
    def optimize_for_inference(self, model_name: str) -> bool:
        """
        Optimize a loaded model for faster inference.
        
        Args:
            model_name: Name of the model to optimize
            
        Returns:
            True if optimization successful, False otherwise
        """
        with self._lock:
            if model_name not in self._models:
                self.logger.warning(f"Model '{model_name}' not loaded, cannot optimize")
                return False
            
            try:
                model = self._models[model_name]
                
                # Set to evaluation mode
                model.eval()
                
                # Additional optimizations for specific devices
                if hasattr(model, '_modules'):
                    for module in model._modules.values():
                        if hasattr(module, 'eval'):
                            module.eval()
                
                self.logger.info(f"Model '{model_name}' optimized for inference")
                return True
                
            except Exception as e:
                self.logger.warning(f"Failed to optimize model '{model_name}': {e}")
                return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get estimated memory usage of loaded models.
        
        Returns:
            Dictionary with memory usage information
        """
        memory_info = {
            "loaded_models": len(self._models),
            "estimated_memory_mb": len(self._models) * 80,  # Rough estimate for MiniLM
            "cache_utilization": f"{len(self._models)}/{self.cache_size_limit}"
        }
        
        return memory_info
    
    def warmup(self, model_name: str, sample_texts: list = None) -> None:
        """
        Warm up the model with sample encoding to optimize performance.
        
        Args:
            model_name: Name of the model to warm up
            sample_texts: Optional list of sample texts for warmup
        """
        model = self.load_on_demand(model_name)
        if model is None:
            return
        
        if sample_texts is None:
            sample_texts = [
                "Introduction",
                "Chapter 1: Overview", 
                "Methodology and Approach",
                "Results and Analysis",
                "Conclusion"
            ]
        
        try:
            self.logger.info(f"Warming up model '{model_name}' with {len(sample_texts)} samples...")
            start_time = time.time()
            
            # Encode sample texts to warm up the model
            model.encode(sample_texts, show_progress_bar=False)
            
            warmup_time = time.time() - start_time
            self.logger.info(f"Model warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    def __del__(self):
        """Cleanup when loader is destroyed."""
        try:
            self.clear_all_cache()
        except Exception:
            pass  # Ignore errors during cleanup