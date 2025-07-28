try:
    from src.utils.validation import (
        validate_pdf, 
        validate_extraction_result,
        get_pdf_info,
        get_result_validation,
        PDFValidator,
        ResultValidator,
        ValidationError
    )
    
    from src.utils.text_utils import (
        clean_text,
        normalize_whitespace,
        extract_sentences,
        extract_words,
        detect_language,
        is_likely_heading,
        extract_key_phrases,
        calculate_text_similarity,
        get_text_statistics,
        contains_url_or_email,
        is_mostly_numeric
    )
    
    from src.utils.layout_utils import (
        LayoutUtils,
        LayoutRegion,
        ColumnInfo, 
        LayoutStructure,
        Position
    )
    
    __all__ = [
        # Validation utilities
        "validate_pdf",
        "validate_extraction_result", 
        "get_pdf_info",
        "get_result_validation",
        "PDFValidator",
        "ResultValidator",
        "ValidationError",
        
        # Text processing utilities
        "clean_text",
        "normalize_whitespace",
        "extract_sentences",
        "extract_words", 
        "detect_language",
        "is_likely_heading",
        "extract_key_phrases",
        "calculate_text_similarity",
        "get_text_statistics",
        "contains_url_or_email",
        "is_mostly_numeric",
        
        # Layout analysis utilities
        "LayoutUtils",
        "LayoutRegion",
        "ColumnInfo",
        "LayoutStructure", 
        "Position"
    ]
    
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Some utility modules not available: {e}")
    __all__ = []

# Version info
__version__ = "1.0.0"

# Module documentation
__doc__ = """
Utility functions and classes for PDF processing support.

Key Modules:
    validation: Comprehensive PDF and result validation
    text_utils: Multilingual text processing and analysis
    layout_utils: Advanced spatial layout analysis
    
Usage:
    from src.utils import validate_pdf, clean_text, LayoutUtils
    
    # Validate PDF file
    if validate_pdf('document.pdf'):
        print("Valid PDF")
    
    # Clean text
    cleaned = clean_text(raw_text)
    
    # Analyze layout
    layout_analyzer = LayoutUtils()
    structure = layout_analyzer.analyze_page_layout(page)
"""