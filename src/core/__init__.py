try:
    from src.core.pdf_processor import PDFProcessor
    from src.core.candidate_generator import CandidateGenerator, HeadingCandidate
    from src.core.semantic_filter import SemanticFilter
    from src.core.hierarchy_assigner import HierarchyAssigner, HierarchyNode
    from src.core.output_formatter import OutputFormatter
    from src.models.embedding_model import EmbeddingModel
    from src.models.font_analyzer import FontAnalyzer, FontInfo, FontStatistics
    
    __all__ = [
        "PDFProcessor",
        "CandidateGenerator", 
        "HeadingCandidate",
        "SemanticFilter",
        "HierarchyAssigner",
        "HierarchyNode", 
        "OutputFormatter",
        "EmbeddingModel",
        "FontAnalyzer",
        "FontInfo",
        "FontStatistics"
    ]
    
except ImportError as e:
    # Log import issues but don't fail completely
    import logging
    logging.getLogger(__name__).warning(f"Some core modules not available: {e}")
    __all__ = []

# Version info
__version__ = "1.0.0"

# Module information
__doc__ = """
Core processing pipeline for PDF heading extraction.

Main Components:
    PDFProcessor: Central orchestrator with Adobe-style intelligence
    CandidateGenerator: Fast font and layout-based detection  
    SemanticFilter: Smart semantic verification using embeddings
    HierarchyAssigner: Multi-strategy hierarchy level assignment
    OutputFormatter: Clean output in multiple formats
    
Usage:
    from src.core import PDFProcessor
    
    processor = PDFProcessor(language='auto', debug=False)
    result = processor.process('document.pdf')
"""