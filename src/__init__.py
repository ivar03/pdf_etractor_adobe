
__version__ = "1.0.0"
__author__ = "PDF Extractor Team"
__description__ = "Advanced PDF heading extraction using hybrid heuristic and semantic approach"

# Package metadata
__all__ = [
    "core",
    "utils", 
    "config",
    "__version__",
    "__author__",
    "__description__"
]

# Optional: Import main classes for convenience
try:
    from src.core.pdf_processor import PDFProcessor
    from src.utils.validation import validate_pdf
    __all__.extend(["PDFProcessor", "validate_pdf"])
except ImportError:
    # Graceful handling if dependencies aren't available yet
    pass