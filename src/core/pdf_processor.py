import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import fitz  
import pdfplumber
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from src.core.candidate_generator import CandidateGenerator
from src.core.semantic_filter import SemanticFilter
from src.core.hierarchy_assigner import HierarchyAssigner
from src.core.output_formatter import OutputFormatter
from src.utils.validation import validate_pdf, detect_language
from src.utils.text_utils import clean_text, normalize_whitespace
from config.settings import (
    MAX_PROCESSING_TIME, MAX_FILE_SIZE_MB, 
    OUTPUT_DIR, INCLUDE_DEBUG_INFO
)


class PDFProcessor:
    """Main orchestrator for PDF heading extraction using hybrid approach with accessibility support."""
    
    def __init__(self, language: str = 'auto', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.candidate_generator = CandidateGenerator(language=language, debug=debug)
        self.semantic_filter = SemanticFilter(language=language, debug=debug) if not self._is_fast_mode() else None
        self.hierarchy_assigner = HierarchyAssigner(language=language, debug=debug)
        self.output_formatter = OutputFormatter(debug=debug)
        
        # Processing statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "processing_stages": [],
            "warnings": [],
            "document_info": {}
        }
    
    def process(self, pdf_path: str, timeout: Optional[int] = None, 
                include_metadata: Optional[bool] = None) -> Dict[str, Any]:
        """Main processing pipeline with timeout protection and optional metadata inclusion."""
        self.stats["start_time"] = time.time()
        timeout = timeout or MAX_PROCESSING_TIME
        
        # Check metadata inclusion preference
        if include_metadata is None:
            include_metadata = self._should_include_metadata()
        
        if include_metadata:
            self.logger.info(f"Starting PDF processing with full metadata and accessibility support: {pdf_path}")
        else:
            self.logger.info(f"Starting PDF processing in simple mode: {pdf_path}")
        
        try:
            # Use ThreadPoolExecutor for timeout control
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._process_internal, pdf_path, include_metadata)
                result = future.result(timeout=timeout)
                
        except FutureTimeoutError:
            self.logger.error(f"Processing timeout after {timeout}s")
            raise TimeoutError(f"PDF processing exceeded {timeout} second limit")
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise
        
        self.stats["end_time"] = time.time()
        self.stats["processing_time"] = self.stats["end_time"] - self.stats["start_time"]
        
        self.logger.info(f"Processing completed in {self.stats['processing_time']:.2f}s")
        return result
    
    def process_for_round1a(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF for Round 1A format (clean format without accessibility)."""
        self.logger.info("Processing for Round 1A format (no accessibility metadata)")
        
        # Process with simple format (no metadata)
        result = self.process(pdf_path, include_metadata=False)
        
        # Return only the clean outline format for Round 1A
        return {
            "title": result.get("title", "Document"),
            "outline": result.get("outline", [])
        }
    
    def _process_internal(self, pdf_path: str, include_metadata: bool = False) -> Dict[str, Any]:
        """Internal processing pipeline with optional metadata and accessibility tagging."""
        
        # Stage 1: Validate and analyze PDF
        self._add_stage("pdf_validation")
        document_info = self._analyze_pdf(pdf_path)
        
        # Stage 2: Check for structured PDF tags (Adobe approach)
        self._add_stage("structure_detection")
        structured_headings = self._extract_structured_headings(pdf_path)
        
        if structured_headings:
            self.logger.info("Found structured PDF tags, using native extraction")
            headings = structured_headings
            hierarchy_tree = self._build_simple_tree(headings) if include_metadata else None
        else:
            # Stage 3: Generate candidates using heuristics
            self._add_stage("candidate_generation")
            candidates = self.candidate_generator.generate_candidates(pdf_path)
            
            if not candidates:
                self.logger.warning("No heading candidates found")
                return self._create_empty_result(document_info, include_metadata)
            
            title = self._extract_title_from_candidates(candidates, document_info)
            document_info["title"] = title  # Set the definitive title
            
            other_candidates = [c for c in candidates if c.text != title]
            
            # Stage 4: Apply semantic filtering (if enabled and not in fast mode)
            if self.semantic_filter and not self._is_fast_mode():
                self._add_stage("semantic_filtering")
                    
                filtered_candidates = self.semantic_filter.filter_candidates(
                    other_candidates, pdf_path
                )
            else:
                filtered_candidates = other_candidates
            
            # Stage 5: Assign hierarchy levels
            self._add_stage("hierarchy_assignment")
            headings = self.hierarchy_assigner.assign_hierarchy(filtered_candidates)
            
            # Stage 6: Generate hierarchy tree (only if metadata is requested)
            if include_metadata:
                hierarchy_tree = self.hierarchy_assigner.generate_hierarchy_tree(
                    [self._dict_to_node(h) for h in headings]
                )
            else:
                hierarchy_tree = None
        
        # Stage 7: Format output with or without metadata
        self._add_stage("output_formatting")
        
        # Compile processing statistics (only if metadata is requested)
        processing_stats = None
        if include_metadata:
            processing_stats = {
                **self.stats,
                "hierarchy_stats": self.hierarchy_assigner.get_hierarchy_statistics(
                    [self._dict_to_node(h) for h in headings]
                ) if headings else {}
            }
        
        # Format final result with optional metadata
        result = self.output_formatter.format_results(
            headings=headings,
            document_info={**document_info, **self.stats},
            hierarchy_tree=hierarchy_tree,
            processing_stats=processing_stats if (include_metadata and INCLUDE_DEBUG_INFO) else None,
            include_metadata=include_metadata
        )
        
        # Log accessibility summary (only if metadata is included)
        if include_metadata and "accessibility" in result:
            acc_summary = result["accessibility"]["compliance_summary"]
            self.logger.info(f"Accessibility Score: {acc_summary['accessibility_score']:.1f}/100")
            self.logger.info(f"Compliance - WCAG: {'✓' if acc_summary['wcag_2_1_aa'] else '✗'}, "
                           f"PDF/UA: {'✓' if acc_summary['pdf_ua'] else '✗'}, "
                           f"Section 508: {'✓' if acc_summary['section_508'] else '✗'}")
        elif not include_metadata:
            self.logger.info(f"Simple format generated with {len(headings)} headings")
        
        return result
    
    def _analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF document and extract metadata."""
        
        # Basic validation
        if not validate_pdf(pdf_path):
            raise ValueError(f"Invalid PDF file: {pdf_path}")
        
        file_size = os.path.getsize(pdf_path)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            self.logger.warning(f"Large file size: {file_size / (1024*1024):.1f}MB")
        
        document_info = {
            "filename": Path(pdf_path).name,
            "file_path": str(pdf_path),
            "file_size": file_size,
            "processing_method": "hybrid"
        }
        
        # Extract PDF metadata using PyMuPDF
        try:
            with fitz.open(pdf_path) as doc:
                metadata = doc.metadata
                document_info.update({
                    "total_pages": len(doc),
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", ""),
                })
                
                # Detect language if set to auto
                if self.language == 'auto':
                    detected_lang = self._detect_document_language(doc)
                    document_info["language"] = detected_lang
                    self.language = detected_lang
                else:
                    document_info["language"] = self.language
                
                # Analyze document structure (only if metadata will be used)
                if self._should_include_metadata():
                    structure_info = self._analyze_document_structure(doc)
                    document_info.update(structure_info)
                
        except Exception as e:
            self.logger.warning(f"Failed to extract PDF metadata: {e}")
            document_info.update({
                "total_pages": 0,
                "language": self.language,
            })
        
        return document_info
    
    def _extract_structured_headings(self, pdf_path: str) -> Optional[List[Dict[str, Any]]]:
        """Try to extract headings from PDF structure/tags (Adobe approach)."""
        
        try:
            with fitz.open(pdf_path) as doc:
                # Check if document has structural information
                if not doc.is_pdf or not hasattr(doc, 'get_toc'):
                    return None
                
                # Extract table of contents
                toc = doc.get_toc()
                if not toc:
                    return None
                
                self.logger.info(f"Found structured TOC with {len(toc)} entries")
                
                structured_headings = []
                for i, (level, title, page_num) in enumerate(toc):
                    # Find the actual text position on the page
                    try:
                        page = doc.load_page(page_num - 1)  # 0-indexed
                        
                        # Search for the title text on the page
                        text_instances = page.search_for(title.strip())
                        if text_instances:
                            bbox = text_instances[0]  # First occurrence
                        else:
                            # Fallback bbox if text not found
                            bbox = [0, 0, 100, 20]
                        
                        heading = {
                            "text": clean_text(title),
                            "level": max(1, min(level, 6)),  # Clamp to 1-6
                            "page": page_num,
                            "bbox": list(bbox),
                            "font_info": {
                                "size": 14,  # Default size for structured headings
                                "weight": "bold",
                                "family": "unknown"
                            },
                            "confidence": 1.0,  # High confidence for structured data
                            "features": {
                                "source": "pdf_structure",
                                "toc_index": i
                            }
                        }
                        
                        structured_headings.append(heading)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process TOC entry '{title}': {e}")
                        continue
                
                return structured_headings if structured_headings else None
                
        except Exception as e:
            self.logger.debug(f"Structured extraction failed: {e}")
            return None
    
    def _detect_document_language(self, doc: fitz.Document) -> str:
        """Detect document language from content."""
        
        # Sample text from first few pages
        sample_text = ""
        for page_num in range(min(3, len(doc))):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            sample_text += page_text[:1000]  # First 1000 chars per page
        
        if len(sample_text.strip()) < 100:
            return 'en'  # Default to English if insufficient text
        
        # Use language detection utility
        detected_language = detect_language(sample_text)
        self.logger.debug(f"Detected language: {detected_language}")
        
        return detected_language
    
    def _analyze_document_structure(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze document structure and layout characteristics."""
        
        structure_info = {
            "has_images": False,
            "has_tables": False,
            "is_multi_column": False,
            "avg_line_height": 0,
            "font_analysis": {},
            "layout_complexity": "simple"
        }
        
        try:
            font_sizes = []
            font_families = set()
            line_heights = []
            has_images = False
            
            # Analyze first 3 pages for structure
            for page_num in range(min(3, len(doc))):
                page = doc.load_page(page_num)
                
                # Check for images
                if page.get_images():
                    has_images = True
                
                # Analyze text blocks
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        if len(line["spans"]) > 0:
                            # Collect font information
                            for span in line["spans"]:
                                font_sizes.append(span["size"])
                                font_families.add(span["font"])
                            
                            # Calculate line height
                            bbox = line["bbox"]
                            line_heights.append(bbox[3] - bbox[1])
                
                # Check for multi-column layout
                if self._detect_multi_column_layout(page):
                    structure_info["is_multi_column"] = True
            
            # Compile font analysis
            if font_sizes:
                structure_info["font_analysis"] = {
                    "unique_sizes": len(set(font_sizes)),
                    "size_range": [min(font_sizes), max(font_sizes)],
                    "avg_size": sum(font_sizes) / len(font_sizes),
                    "font_families": list(font_families)
                }
            
            if line_heights:
                structure_info["avg_line_height"] = sum(line_heights) / len(line_heights)
            
            structure_info["has_images"] = has_images
            
            # Determine layout complexity
            complexity_score = 0
            if structure_info["is_multi_column"]: complexity_score += 2
            if has_images: complexity_score += 1
            if len(font_families) > 3: complexity_score += 1
            if structure_info["font_analysis"].get("unique_sizes", 0) > 5: complexity_score += 1
            
            if complexity_score >= 4:
                structure_info["layout_complexity"] = "complex"
            elif complexity_score >= 2:
                structure_info["layout_complexity"] = "moderate"
            
        except Exception as e:
            self.logger.warning(f"Structure analysis failed: {e}")
        
        return structure_info
    
    def _detect_multi_column_layout(self, page: fitz.Page) -> bool:
        """Detect if page has multi-column layout."""
        
        try:
            blocks = page.get_text("dict")["blocks"]
            text_blocks = [b for b in blocks if "lines" in b]
            
            if len(text_blocks) < 4:  # Need sufficient blocks
                return False
            
            # Group blocks by horizontal position
            left_blocks = []
            right_blocks = []
            page_width = page.rect.width
            middle = page_width / 2
            
            for block in text_blocks:
                bbox = block["bbox"]
                block_center = (bbox[0] + bbox[2]) / 2
                
                if block_center < middle * 0.8:  # Left side
                    left_blocks.append(block)
                elif block_center > middle * 1.2:  # Right side
                    right_blocks.append(block)
            
            # Multi-column if we have blocks on both sides
            return len(left_blocks) >= 2 and len(right_blocks) >= 2
            
        except Exception:
            return False
    
    def _build_simple_tree(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build simple hierarchy tree for structured headings."""
        
        tree = {}
        stack = []
        
        for heading in headings:
            level = heading["level"]
            text = heading["text"]
            
            # Adjust stack to current level
            while len(stack) >= level:
                stack.pop()
            
            # Create node
            node = {
                "level": level,
                "page": heading["page"],
                "children": {}
            }
            
            if not stack:
                # Root level
                tree[text] = node
                stack.append((text, node))
            else:
                # Child node
                parent_name, parent_node = stack[-1]
                parent_node["children"][text] = node
                stack.append((text, node))
        
        return tree
    
    def _create_empty_result(self, document_info: Dict[str, Any], include_metadata: bool = False) -> Dict[str, Any]:
        """Create empty result when no headings found."""
        
        self.stats["warnings"].append("No headings detected in document")
        
        return self.output_formatter.format_results(
            headings=[],
            document_info={**document_info, **self.stats},
            hierarchy_tree={} if include_metadata else None,
            processing_stats=self.stats if (include_metadata and INCLUDE_DEBUG_INFO) else None,
            include_metadata=include_metadata
        )
    
    def _dict_to_node(self, heading_dict: Dict[str, Any]):
        """Convert heading dictionary to HierarchyNode for tree building."""
        from src.core.hierarchy_assigner import HierarchyNode
        
        return HierarchyNode(
            text=heading_dict["text"],
            level=heading_dict["level"],
            page=heading_dict["page"],
            bbox=tuple(heading_dict["bbox"]),
            font_size=heading_dict["font_info"]["size"],
            confidence=heading_dict.get("confidence", 0.0)
        )
    
    def _add_stage(self, stage_name: str) -> None:
        """Add processing stage with timestamp."""
        
        stage_info = {
            "name": stage_name,
            "timestamp": time.time(),
            "duration": None
        }
        
        if self.stats["processing_stages"]:
            # Calculate duration of previous stage
            prev_stage = self.stats["processing_stages"][-1]
            prev_stage["duration"] = stage_info["timestamp"] - prev_stage["timestamp"]
        
        self.stats["processing_stages"].append(stage_info)
        self.logger.debug(f"Starting stage: {stage_name}")
    
    def _is_fast_mode(self) -> bool:
        """Check if running in fast mode (skip semantic filtering)."""
        return os.getenv("FAST_MODE", "false").lower() == "true"
    
    def _should_include_metadata(self) -> bool:
        """Check if metadata should be included based on environment variable."""
        return os.getenv("INCLUDE_METADATA", "false").lower() == "true"
    
    def save_output(self, result: Dict[str, Any], output_path: Optional[str] = None, 
           formats: Optional[List[str]] = None, 
           auto_filename: bool = True) -> Dict[str, str]:
        """Save processing results to file(s) with automatic path handling and accessibility support."""
        
        from config.settings import JSON_OUTPUT_DIR, OUTPUT_DIR
        
        if formats is None:
            formats = ["json"]
        
        # Handle automatic filename generation
        if output_path is None or auto_filename:
            # Generate filename from document title
            if "title" in result:
                filename = result["title"]
            elif "document_info" in result:
                filename = result["document_info"].get("filename", "outline")
            else:
                filename = "outline"
            
            # Remove extension and sanitize
            safe_name = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            
            # Add timestamp to avoid conflicts
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_name = f"{safe_name}_{timestamp}"
        else:
            final_name = Path(output_path).stem
        
        output_files = {}
        
        try:
            for format_type in formats:
                if format_type == "json":
                    output_dir = JSON_OUTPUT_DIR
                    extension = ".json"
                elif format_type == "pdf_ua_xml":
                    output_dir = OUTPUT_DIR
                    extension = "_accessibility.xml"
                else:
                    output_dir = OUTPUT_DIR
                    extension = f".{format_type}"
                
                # Ensure directory exists
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create full output path
                output_file = output_dir / f"{final_name}{extension}"
                
                # Save in the appropriate format
                if format_type == "json":
                    self.output_formatter.save_json(result, output_file)
                elif format_type == "pdf_ua_xml":
                    # Extract headings for accessibility XML (handle both formats)
                    headings = result.get("outline", result.get("headings", []))
                    self.output_formatter.save_pdf_ua_xml(headings, output_file)
                
                output_files[format_type] = str(output_file)
                self.logger.info(f"Saved {format_type.upper()} output to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save output: {e}")
            raise
        
        return output_files

    def save_output_to_custom_path(self, result: Dict[str, Any], custom_path: str, 
                                formats: Optional[List[str]] = None) -> Dict[str, str]:
        """Save output to a specific custom path with accessibility support."""
        
        if formats is None:
            formats = ["json"]
        
        custom_path = Path(custom_path)
        output_files = {}
        
        try:
            # Ensure the custom directory exists
            custom_path.parent.mkdir(parents=True, exist_ok=True)
            
            if len(formats) == 1:
                # Single format - use the exact path provided
                format_type = formats[0]
                
                if format_type == "json":
                    self.output_formatter.save_json(result, custom_path)
                elif format_type == "csv":
                    self.output_formatter.save_csv(result, custom_path)
                elif format_type == "xml":
                    self.output_formatter.save_xml(result, custom_path)
                elif format_type == "markdown":
                    self.output_formatter.save_markdown(result, custom_path)
                elif format_type == "html":
                    self.output_formatter.save_html_outline(result, custom_path)
                elif format_type == "pdf_ua_xml":
                    headings = result.get("outline", result.get("headings", []))
                    self.output_formatter.save_pdf_ua_xml(headings, custom_path)
                
                output_files[format_type] = str(custom_path)
            else:
                # Multiple formats - use base path and add extensions
                base_path = custom_path.with_suffix('')
                
                for format_type in formats:
                    if format_type == "pdf_ua_xml":
                        format_path = base_path.with_suffix('_accessibility.xml')
                    else:
                        format_path = base_path.with_suffix(f'.{format_type}')
                    
                    if format_type == "json":
                        self.output_formatter.save_json(result, format_path)
                    elif format_type == "csv":
                        self.output_formatter.save_csv(result, format_path)
                    elif format_type == "xml":
                        self.output_formatter.save_xml(result, format_path)
                    elif format_type == "markdown":
                        self.output_formatter.save_markdown(result, format_path)
                    elif format_type == "html":
                        self.output_formatter.save_html_outline(result, format_path)
                    elif format_type == "pdf_ua_xml":
                        headings = result.get("outline", result.get("headings", []))
                        self.output_formatter.save_pdf_ua_xml(headings, format_path)
                    
                    output_files[format_type] = str(format_path)
            
            self.logger.info(f"Saved output to custom path: {custom_path.parent}")
            
        except Exception as e:
            self.logger.error(f"Failed to save to custom path: {e}")
            raise
        
        return output_files

    def process_batch(self, pdf_paths: List[str], 
                     output_dir: Optional[str] = None,
                     max_workers: int = 2,
                     include_accessibility: bool = False,
                     include_metadata: bool = False) -> Dict[str, Any]:
        """Process multiple PDFs in batch mode with optional metadata and accessibility support."""
        
        output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        failed = {}
        
        self.logger.info(f"Starting batch processing of {len(pdf_paths)} files")
        if include_metadata:
            self.logger.info("Full metadata will be included")
        if include_accessibility:
            self.logger.info("Accessibility XML files will be generated")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(self.process, pdf_path, None, include_metadata): pdf_path 
                for pdf_path in pdf_paths
            }
            
            # Collect results
            for future in future_to_path:
                pdf_path = future_to_path[future]
                try:
                    result = future.result(timeout=MAX_PROCESSING_TIME)
                    results[pdf_path] = result
                    
                    # Save individual result
                    formats = ["json"]
                    if include_accessibility:
                        formats.append("pdf_ua_xml")
                    
                    output_file = output_dir / f"{Path(pdf_path).stem}_headings"
                    self.save_output_to_custom_path(result, output_file, formats)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {pdf_path}: {e}")
                    failed[pdf_path] = str(e)
        
        # Create batch summary with accessibility stats
        accessibility_summary = {}
        if include_accessibility and include_metadata and results:
            total_score = 0
            compliant_count = 0
            
            for pdf_path, result in results.items():
                if "accessibility" in result:
                    acc_data = result["accessibility"]["compliance_summary"]
                    total_score += acc_data["accessibility_score"]
                    if acc_data["wcag_2_1_aa"]:
                        compliant_count += 1
            
            accessibility_summary = {
                "average_accessibility_score": total_score / len(results),
                "wcag_compliant_documents": compliant_count,
                "compliance_rate": (compliant_count / len(results)) * 100
            }
        
        batch_summary = {
            "total_files": len(pdf_paths),
            "successful": len(results),
            "failed": len(failed),
            "success_rate": len(results) / len(pdf_paths) * 100,
            "failed_files": failed,
            "output_directory": str(output_dir),
            "metadata_included": include_metadata,
            "accessibility_included": include_accessibility,
            "accessibility_summary": accessibility_summary
        }
        
        # Save batch summary
        summary_file = output_dir / "batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Batch processing complete: {len(results)}/{len(pdf_paths)} successful")
        if accessibility_summary:
            self.logger.info(f"Average accessibility score: {accessibility_summary['average_accessibility_score']:.1f}/100")
            self.logger.info(f"WCAG compliant: {accessibility_summary['compliance_rate']:.1f}%")
        
        return {
            "results": results,
            "summary": batch_summary
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        
        if not self.stats["processing_stages"]:
            return {}
        
        # Calculate stage durations
        total_time = self.stats.get("processing_time", 0)
        stage_breakdown = {}
        
        for stage in self.stats["processing_stages"]:
            if stage["duration"] is not None:
                stage_breakdown[stage["name"]] = {
                    "duration": round(stage["duration"], 3),
                    "percentage": round((stage["duration"] / total_time) * 100, 1) if total_time > 0 else 0
                }
        
        return {
            "total_processing_time": total_time,
            "stage_breakdown": stage_breakdown,
            "warnings": self.stats["warnings"],
            "document_analysis": self.stats.get("document_info", {})
        }
    
    def clear_caches(self) -> None:
        """Clear all component caches to free memory."""
        
        try:
            if self.semantic_filter:
                self.semantic_filter.clear_cache()
            
            if hasattr(self.candidate_generator, 'clear_cache'):
                self.candidate_generator.clear_cache()
            
            if hasattr(self.hierarchy_assigner, 'clear_cache'):
                self.hierarchy_assigner.clear_cache()
            
            self.logger.info("All component caches cleared")
            
        except Exception as e:
            self.logger.warning(f"Failed to clear some caches: {e}")
            
    def get_accessibility_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract accessibility summary from processing result."""
        
        if "accessibility" not in result:
            return {"accessibility_available": False}
        
        acc_data = result["accessibility"]
        summary = {
            "accessibility_available": True,
            "compliance_summary": acc_data["compliance_summary"],
            "total_issues": len(acc_data["metadata"]["issues"]),
            "recommendations_count": len(acc_data["metadata"]["recommendations"]),
            "structure_xml_available": acc_data["structure_xml_available"]
        }
        
        return summary
    
    def _extract_title_from_candidates(self, candidates: List, document_info: Dict[str, Any]) -> str:
        """Extracts the document title from metadata or the highest-confidence candidate."""
        # Priority 1: Use existing metadata title if it's high quality.
        metadata_title = document_info.get("title", "")
        if metadata_title and len(metadata_title) > 5 and metadata_title != "Untitled":
            return metadata_title

        # Priority 2: Find the best candidate on the first page.
        # A title is typically at the top of the first page and has a high score.
        if candidates:
            first_page_candidates = sorted(
                [c for c in candidates if c.page == 1],
                key=lambda x: (x.position_ratio, -x.confidence_score) # Sort by position, then score
            )
            if first_page_candidates:
                # The best title candidate is usually the first one with a high score
                best_candidate = first_page_candidates[0]
                if best_candidate.confidence_score > 0.6 and best_candidate.position_ratio < 0.25:
                    return best_candidate.text

        # Priority 3: Fallback to the document filename.
        return document_info.get("filename", "Untitled Document").replace('_', ' ').replace('.pdf', '')