# Import configuration modules
try:
    from .settings import (
        # Directories
        BASE_DIR,
        DATA_DIR,
        OUTPUT_DIR,
        MODEL_DIR,
        
        # Processing settings
        MAX_PROCESSING_TIME,
        BATCH_SIZE,
        MAX_FILE_SIZE_MB,
        
        # Detection thresholds
        FONT_SIZE_THRESHOLD_RATIO,
        BOLD_WEIGHT_THRESHOLD,
        MIN_HEADING_LENGTH,
        MAX_HEADING_LENGTH,
        
        # Semantic settings
        EMBEDDING_MODEL,
        SEMANTIC_SIMILARITY_THRESHOLD,
        CONTEXT_WINDOW,
        
        # Hierarchy settings
        MAX_HIERARCHY_LEVELS,
        TITLE_POSITION_THRESHOLD,
        CENTER_ALIGNMENT_TOLERANCE,
        
        # Output settings
        OUTPUT_FORMAT,
        INCLUDE_CONFIDENCE_SCORES,
        INCLUDE_DEBUG_INFO
    )
    
    from .cultural_patterns import (
        CULTURAL_PATTERNS,
        TEST_SAMPLES
    )
    
    __all__ = [
        # Directory settings
        "BASE_DIR",
        "DATA_DIR", 
        "OUTPUT_DIR",
        "MODEL_DIR",
        
        # Processing configuration
        "MAX_PROCESSING_TIME",
        "BATCH_SIZE",
        "MAX_FILE_SIZE_MB",
        
        # Detection parameters
        "FONT_SIZE_THRESHOLD_RATIO",
        "BOLD_WEIGHT_THRESHOLD", 
        "MIN_HEADING_LENGTH",
        "MAX_HEADING_LENGTH",
        
        # Semantic parameters
        "EMBEDDING_MODEL",
        "SEMANTIC_SIMILARITY_THRESHOLD",
        "CONTEXT_WINDOW",
        
        # Hierarchy parameters
        "MAX_HIERARCHY_LEVELS",
        "TITLE_POSITION_THRESHOLD",
        "CENTER_ALIGNMENT_TOLERANCE",
        
        # Output parameters
        "OUTPUT_FORMAT",
        "INCLUDE_CONFIDENCE_SCORES", 
        "INCLUDE_DEBUG_INFO",
        
        # Cultural patterns
        "CULTURAL_PATTERNS",
        "TEST_SAMPLES"
    ]
    
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Configuration modules not available: {e}")
    __all__ = []

# Version info
__version__ = "1.0.0"

# Configuration defaults (fallback values)
DEFAULT_CONFIG = {
    "max_processing_time": 10,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "font_size_threshold_ratio": 1.2,
    "semantic_similarity_threshold": 0.3,
    "max_hierarchy_levels": 6,
    "include_confidence_scores": True
}

# Module documentation
__doc__ = """
Configuration and cultural patterns for PDF heading extraction.

Components:
    settings: Core configuration parameters and thresholds
    cultural_patterns: Multilingual heading recognition patterns
    
Key Settings:
    - Processing timeouts and file size limits
    - Font analysis thresholds and detection parameters  
    - Semantic filtering and embedding model configuration
    - Hierarchy assignment rules and level limits
    - Output formatting and export options
    
Cultural Support:
    - English, Japanese, Chinese, Arabic, Hindi patterns
    - Language-specific numbering and heading styles
    - Cultural layout preferences and text directions
    
Usage:
    from src.config import MAX_PROCESSING_TIME, CULTURAL_PATTERNS
    
    # Use processing timeout
    processor.process(pdf_path, timeout=MAX_PROCESSING_TIME)
    
    # Access cultural patterns
    japanese_patterns = CULTURAL_PATTERNS['japanese']
"""