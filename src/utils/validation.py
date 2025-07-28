import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import mimetypes

try:
    import magic
except ImportError:
    magic = None  # fallback

import fitz  # PyMuPDF
import json
import re
from datetime import datetime

from config.settings import MAX_FILE_SIZE_MB, MIN_HEADING_LENGTH, MAX_HEADING_LENGTH
from src.utils.text_utils import clean_text, detect_language, is_likely_heading



class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class PDFValidator:
    """Comprehensive PDF file validation."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # File validation settings
        self.max_file_size = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.min_file_size = 1024  # 1KB minimum
        self.supported_extensions = {'.pdf'}
        self.supported_mime_types = {'application/pdf'}
        
        # PDF validation settings
        self.min_pages = 1
        self.max_pages = 1000  # Reasonable limit
        self.min_text_content = 100  # Minimum characters for meaningful processing
        
        # Initialize libmagic if available
        self.magic_available = self._check_magic_availability()
    
    def _check_magic_availability(self) -> bool:
        """Check if python-magic is available for MIME type detection."""
        if magic is None:
            self.logger.warning("python-magic not installed, using mimetypes fallback")
            return False
        try:
            magic.from_file(__file__, mime=True)
            return True
        except Exception:
            self.logger.warning("python-magic failed to initialize, using mimetypes fallback")
            return False

    
    def validate_pdf_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Comprehensive PDF file validation."""
        file_path = Path(file_path)
        validation_result = {
            "is_valid": False,
            "file_path": str(file_path),
            "errors": [],
            "warnings": [],
            "file_info": {},
            "pdf_info": {},
            "content_info": {}
        }
        
        try:
            # Basic file existence and accessibility
            file_info = self._validate_file_basics(file_path)
            validation_result["file_info"] = file_info
            
            if file_info.get("errors"):
                validation_result["errors"].extend(file_info["errors"])
                return validation_result
            
            # PDF-specific validation
            pdf_info = self._validate_pdf_structure(file_path)
            validation_result["pdf_info"] = pdf_info
            
            if pdf_info.get("errors"):
                validation_result["errors"].extend(pdf_info["errors"])
                return validation_result
            
            # Content validation
            content_info = self._validate_pdf_content(file_path)
            validation_result["content_info"] = content_info
            
            if content_info.get("errors"):
                validation_result["errors"].extend(content_info["errors"])
            
            if content_info.get("warnings"):
                validation_result["warnings"].extend(content_info["warnings"])
            
            # Overall validation result
            validation_result["is_valid"] = len(validation_result["errors"]) == 0
            
        except Exception as e:
            validation_result["errors"].append(f"Validation failed: {str(e)}")
            self.logger.error(f"PDF validation error for {file_path}: {e}")
        
        return validation_result
    
    def _validate_file_basics(self, file_path: Path) -> Dict[str, Any]:
        """Validate basic file properties."""
        info = {
            "exists": False,
            "readable": False,
            "size_bytes": 0,
            "extension": "",
            "mime_type": "",
            "errors": [],
            "warnings": []
        }
        
        # Check existence
        if not file_path.exists():
            info["errors"].append(f"File does not exist: {file_path}")
            return info
        
        info["exists"] = True
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            info["errors"].append(f"Path is not a file: {file_path}")
            return info
        
        # Check readability
        if not os.access(file_path, os.R_OK):
            info["errors"].append(f"File is not readable: {file_path}")
            return info
        
        info["readable"] = True
        
        # Get file size
        try:
            info["size_bytes"] = file_path.stat().st_size
        except Exception as e:
            info["errors"].append(f"Cannot get file size: {e}")
            return info
        
        # Validate file size
        if info["size_bytes"] < self.min_file_size:
            info["errors"].append(f"File too small: {info['size_bytes']} bytes (minimum: {self.min_file_size})")
        elif info["size_bytes"] > self.max_file_size:
            info["warnings"].append(f"Large file: {info['size_bytes'] / (1024*1024):.1f}MB (may be slow to process)")
        
        # Check file extension
        info["extension"] = file_path.suffix.lower()
        if info["extension"] not in self.supported_extensions:
            info["errors"].append(f"Unsupported file extension: {info['extension']}")
        
        # Check MIME type
        info["mime_type"] = self._get_mime_type(file_path)
        if info["mime_type"] not in self.supported_mime_types:
            info["warnings"].append(f"Unexpected MIME type: {info['mime_type']}")
        
        return info
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type of the file."""
        try:
            if self.magic_available:
                return magic.from_file(str(file_path), mime=True)
            else:
                # Fallback to mimetypes module
                mime_type, _ = mimetypes.guess_type(str(file_path))
                return mime_type or "application/octet-stream"
        except Exception as e:
            self.logger.warning(f"MIME type detection failed: {e}")
            return "unknown"
    
    def _validate_pdf_structure(self, file_path: Path) -> Dict[str, Any]:
        """Validate PDF document structure."""
        info = {
            "can_open": False,
            "page_count": 0,
            "is_encrypted": False,
            "has_text": False,
            "pdf_version": "",
            "metadata": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            doc = fitz.open(str(file_path))
            
            info["can_open"] = True
            info["page_count"] = len(doc)
            info["is_encrypted"] = doc.needs_pass
            
            # Check PDF version
            if hasattr(doc, 'pdf_version'):
                info["pdf_version"] = f"{doc.pdf_version()[0]}.{doc.pdf_version()[1]}"
            
            # Get metadata
            try:
                metadata = doc.metadata
                info["metadata"] = {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", "")
                }
            except Exception as e:
                info["warnings"].append(f"Cannot read metadata: {e}")
            
            # Validate page count
            if info["page_count"] < self.min_pages:
                info["errors"].append(f"Too few pages: {info['page_count']} (minimum: {self.min_pages})")
            elif info["page_count"] > self.max_pages:
                info["warnings"].append(f"Many pages: {info['page_count']} (may be slow to process)")
            
            # Check if PDF is encrypted
            if info["is_encrypted"]:
                info["errors"].append("PDF is password-protected")
            
            # Quick check for text content
            if info["page_count"] > 0 and not info["is_encrypted"]:
                try:
                    # Check first few pages for text
                    text_found = False
                    for page_num in range(min(3, info["page_count"])):
                        page = doc.load_page(page_num)
                        page_text = page.get_text().strip()
                        if len(page_text) > 10:  # At least some meaningful text
                            text_found = True
                            break
                    
                    info["has_text"] = text_found
                    if not text_found:
                        info["warnings"].append("No readable text found in first few pages (may be image-only PDF)")
                
                except Exception as e:
                    info["warnings"].append(f"Cannot extract text for validation: {e}")
            
            doc.close()
            
        except Exception as e:
            info["errors"].append(f"Cannot open PDF: {str(e)}")
        
        return info
    
    def _validate_pdf_content(self, file_path: Path) -> Dict[str, Any]:
        """Validate PDF content for heading extraction."""
        info = {
            "total_text_length": 0,
            "total_blocks": 0,
            "pages_with_text": 0,
            "estimated_headings": 0,
            "languages_detected": [],
            "font_diversity": 0,
            "layout_complexity": "unknown",
            "errors": [],
            "warnings": []
        }
        
        try:
            doc = fitz.open(str(file_path))
            
            if doc.needs_pass:
                info["errors"].append("Cannot analyze encrypted PDF content")
                doc.close()
                return info
            
            total_text = ""
            total_blocks = 0
            pages_with_text = 0
            font_sizes = set()
            font_families = set()
            potential_headings = 0
            
            # Analyze content from first 5 pages (for performance)
            pages_to_analyze = min(5, len(doc))
            
            for page_num in range(pages_to_analyze):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    pages_with_text += 1
                    total_text += page_text
                
                # Analyze text blocks
                try:
                    blocks = page.get_text("dict")["blocks"]
                    total_blocks += len([b for b in blocks if "lines" in b])
                    
                    # Analyze fonts and potential headings
                    for block in blocks:
                        if "lines" not in block:
                            continue
                        
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["text"].strip():
                                    font_sizes.add(span["size"])
                                    font_families.add(span["font"])
                                    
                                    # Quick heading detection
                                    text = span["text"].strip()
                                    if self._is_potential_heading(text, span):
                                        potential_headings += 1
                
                except Exception as e:
                    info["warnings"].append(f"Cannot analyze page {page_num + 1} structure: {e}")
            
            doc.close()
            
            # Populate analysis results
            info["total_text_length"] = len(total_text)
            info["total_blocks"] = total_blocks
            info["pages_with_text"] = pages_with_text
            info["estimated_headings"] = potential_headings
            info["font_diversity"] = len(font_families)
            
            # Content validation
            if info["total_text_length"] < self.min_text_content:
                info["warnings"].append(f"Limited text content: {info['total_text_length']} characters")
            
            if info["pages_with_text"] == 0:
                info["errors"].append("No pages contain readable text")
            
            if info["estimated_headings"] == 0:
                info["warnings"].append("No obvious headings detected")
            
            # Language detection
            if total_text.strip():
                try:
                    detected_lang = detect_language(total_text[:2000])  # Sample for detection
                    info["languages_detected"] = [detected_lang]
                except Exception as e:
                    info["warnings"].append(f"Language detection failed: {e}")
            
            # Layout complexity assessment
            info["layout_complexity"] = self._assess_layout_complexity(
                len(font_families), len(font_sizes), total_blocks, pages_to_analyze
            )
            
        except Exception as e:
            info["errors"].append(f"Content analysis failed: {str(e)}")
        
        return info
    
    def _is_potential_heading(self, text: str, span: Dict[str, Any]) -> bool:
        """Quick check if a text span might be a heading."""
        if not text or len(text) < MIN_HEADING_LENGTH or len(text) > MAX_HEADING_LENGTH:
            return False
        
        # Check font size (assuming larger fonts are headings)
        font_size = span.get("size", 12)
        if font_size < 10:  # Too small to be a heading
            return False
        
        # Check if bold
        is_bold = bool(span.get("flags", 0) & 16)  # Bold flag
        
        # Quick linguistic check
        heading_analysis = is_likely_heading(text)
        
        # Combine factors
        score = 0
        if font_size > 14:
            score += 2
        if is_bold:
            score += 2
        if heading_analysis.get("is_heading", False):
            score += 3
        
        return score >= 3
    
    def _assess_layout_complexity(self, font_families: int, font_sizes: int, 
                                 total_blocks: int, pages_analyzed: int) -> str:
        """Assess layout complexity based on document characteristics."""
        
        avg_blocks_per_page = total_blocks / pages_analyzed if pages_analyzed > 0 else 0
        
        complexity_score = 0
        
        # Font diversity
        if font_families > 5:
            complexity_score += 2
        elif font_families > 2:
            complexity_score += 1
        
        if font_sizes > 8:
            complexity_score += 2
        elif font_sizes > 4:
            complexity_score += 1
        
        # Block density
        if avg_blocks_per_page > 20:
            complexity_score += 2
        elif avg_blocks_per_page > 10:
            complexity_score += 1
        
        # Classify complexity
        if complexity_score >= 5:
            return "complex"
        elif complexity_score >= 3:
            return "moderate"
        else:
            return "simple"


class ResultValidator:
    """Validate heading extraction results."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
    
    def validate_extraction_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the structure and content of extraction results."""
        validation = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Validate top-level structure
            required_keys = ["document_info", "headings", "metadata"]
            for key in required_keys:
                if key not in result:
                    validation["errors"].append(f"Missing required key: {key}")
            
            if validation["errors"]:
                return validation
            
            # Validate document_info
            doc_info_validation = self._validate_document_info(result["document_info"])
            validation["errors"].extend(doc_info_validation["errors"])
            validation["warnings"].extend(doc_info_validation["warnings"])
            
            # Validate headings
            headings_validation = self._validate_headings(result["headings"])
            validation["errors"].extend(headings_validation["errors"])
            validation["warnings"].extend(headings_validation["warnings"])
            validation["statistics"].update(headings_validation["statistics"])
            
            # Validate metadata
            metadata_validation = self._validate_metadata(result["metadata"])
            validation["errors"].extend(metadata_validation["errors"])
            validation["warnings"].extend(metadata_validation["warnings"])
            
            # Validate hierarchy tree if present
            if "hierarchy_tree" in result:
                tree_validation = self._validate_hierarchy_tree(result["hierarchy_tree"])
                validation["warnings"].extend(tree_validation["warnings"])
            
            validation["is_valid"] = len(validation["errors"]) == 0
            
        except Exception as e:
            validation["errors"].append(f"Result validation failed: {str(e)}")
        
        return validation
    
    def _validate_document_info(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document_info section."""
        validation = {"errors": [], "warnings": []}
        
        required_fields = ["filename", "total_pages", "processing_time"]
        for field in required_fields:
            if field not in doc_info:
                validation["errors"].append(f"Missing document_info field: {field}")
        
        # Validate data types and ranges
        if "total_pages" in doc_info:
            if not isinstance(doc_info["total_pages"], int) or doc_info["total_pages"] < 1:
                validation["errors"].append("total_pages must be a positive integer")
        
        if "processing_time" in doc_info:
            if not isinstance(doc_info["processing_time"], (int, float)) or doc_info["processing_time"] < 0:
                validation["errors"].append("processing_time must be a non-negative number")
            elif doc_info["processing_time"] > 60:
                validation["warnings"].append(f"Long processing time: {doc_info['processing_time']:.1f}s")
        
        return validation
    
    def _validate_headings(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate headings list."""
        validation = {
            "errors": [],
            "warnings": [],
            "statistics": {
                "total_headings": len(headings),
                "level_distribution": {},
                "avg_confidence": 0.0,
                "empty_headings": 0,
                "duplicate_headings": 0
            }
        }
        
        if not isinstance(headings, list):
            validation["errors"].append("headings must be a list")
            return validation
        
        if len(headings) == 0:
            validation["warnings"].append("No headings found")
            return validation
        
        # Track statistics
        levels = []
        confidences = []
        texts = []
        empty_count = 0
        
        # Validate each heading
        for i, heading in enumerate(headings):
            heading_validation = self._validate_single_heading(heading, i)
            validation["errors"].extend(heading_validation["errors"])
            validation["warnings"].extend(heading_validation["warnings"])
            
            # Collect statistics
            if isinstance(heading, dict):
                level = heading.get("level", 1)
                confidence = heading.get("confidence", 0.0)
                text = heading.get("text", "")
                
                levels.append(level)
                if isinstance(confidence, (int, float)):
                    confidences.append(confidence)
                
                if not text or not text.strip():
                    empty_count += 1
                else:
                    texts.append(text.strip().lower())
        
        # Calculate statistics
        if levels:
            level_counts = {}
            for level in levels:
                level_counts[level] = level_counts.get(level, 0) + 1
            validation["statistics"]["level_distribution"] = level_counts
        
        if confidences:
            validation["statistics"]["avg_confidence"] = sum(confidences) / len(confidences)
        
        validation["statistics"]["empty_headings"] = empty_count
        validation["statistics"]["duplicate_headings"] = len(texts) - len(set(texts))
        
        # Validation warnings
        if empty_count > 0:
            validation["warnings"].append(f"{empty_count} headings have empty text")
        
        if validation["statistics"]["duplicate_headings"] > 0:
            validation["warnings"].append(f"{validation['statistics']['duplicate_headings']} duplicate headings found")
        
        return validation
    
    def _validate_single_heading(self, heading: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Validate a single heading object."""
        validation = {"errors": [], "warnings": []}
        
        if not isinstance(heading, dict):
            validation["errors"].append(f"Heading {index} is not a dictionary")
            return validation
        
        # Required fields
        required_fields = ["text", "level", "page", "bbox"]
        for field in required_fields:
            if field not in heading:
                validation["errors"].append(f"Heading {index} missing field: {field}")
        
        # Validate text
        if "text" in heading:
            text = heading["text"]
            if not isinstance(text, str):
                validation["errors"].append(f"Heading {index} text must be a string")
            elif not text.strip():
                validation["warnings"].append(f"Heading {index} has empty text")
            elif len(text) > MAX_HEADING_LENGTH:
                validation["warnings"].append(f"Heading {index} text is very long: {len(text)} characters")
        
        # Validate level
        if "level" in heading:
            level = heading["level"]
            if not isinstance(level, int) or level < 0 or level > 6:
                validation["errors"].append(f"Heading {index} level must be integer 0-6, got: {level}")
        
        # Validate page
        if "page" in heading:
            page = heading["page"]
            if not isinstance(page, int) or page < 1:
                validation["errors"].append(f"Heading {index} page must be positive integer, got: {page}")
        
        # Validate bbox
        if "bbox" in heading:
            bbox = heading["bbox"]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                validation["errors"].append(f"Heading {index} bbox must be list/tuple of 4 numbers")
            else:
                try:
                    bbox_nums = [float(x) for x in bbox]
                    if bbox_nums[2] <= bbox_nums[0] or bbox_nums[3] <= bbox_nums[1]:
                        validation["warnings"].append(f"Heading {index} has invalid bbox dimensions")
                except (ValueError, TypeError):
                    validation["errors"].append(f"Heading {index} bbox contains non-numeric values")
        
        # Validate confidence if present
        if "confidence" in heading:
            confidence = heading["confidence"]
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                validation["errors"].append(f"Heading {index} confidence must be number 0-1, got: {confidence}")
        
        return validation
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata section."""
        validation = {"errors": [], "warnings": []}
        
        if not isinstance(metadata, dict):
            validation["errors"].append("metadata must be a dictionary")
            return validation
        
        # Check for required metadata fields
        recommended_fields = ["extraction_version", "format_version", "generator"]
        for field in recommended_fields:
            if field not in metadata:
                validation["warnings"].append(f"Missing recommended metadata field: {field}")
        
        return validation
    
    def _validate_hierarchy_tree(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hierarchy tree structure."""
        validation = {"errors": [], "warnings": []}
        
        if not isinstance(tree, dict):
            validation["errors"].append("hierarchy_tree must be a dictionary")
            return validation
        
        # Check for reasonable tree structure
        if len(tree) == 0:
            validation["warnings"].append("Empty hierarchy tree")
        elif len(tree) > 100:
            validation["warnings"].append("Very large hierarchy tree - may indicate over-detection")
        
        return validation


# Convenience functions
def validate_pdf(file_path: Union[str, Path]) -> bool:
    """Quick PDF validation - returns True if valid, False otherwise."""
    try:
        validator = PDFValidator()
        result = validator.validate_pdf_file(file_path)
        return result["is_valid"]
    except Exception:
        return False


def validate_extraction_result(result: Dict[str, Any]) -> bool:
    """Quick result validation - returns True if valid, False otherwise."""
    try:
        validator = ResultValidator()
        validation = validator.validate_extraction_result(result)
        return validation["is_valid"]
    except Exception:
        return False


def get_pdf_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get comprehensive PDF information and validation results."""
    validator = PDFValidator(debug=True)
    return validator.validate_pdf_file(file_path)


def get_result_validation(result: Dict[str, Any]) -> Dict[str, Any]:
    """Get comprehensive result validation information."""
    validator = ResultValidator(debug=True)
    return validator.validate_extraction_result(result)