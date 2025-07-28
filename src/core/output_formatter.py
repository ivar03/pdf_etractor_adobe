import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import asdict
import csv
import os
from config.settings import OUTPUT_FORMAT, INCLUDE_CONFIDENCE_SCORES, INCLUDE_DEBUG_INFO
from src.core.accessibility_tagger import AccessibilityTagger


class OutputFormatter:
    """Formats and exports PDF heading extraction results in multiple formats with accessibility support."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.accessibility_tagger = AccessibilityTagger(debug=debug)
        
    def format_results(self, headings: List[Dict[str, Any]], 
                      document_info: Dict[str, Any],
                      hierarchy_tree: Optional[Dict[str, Any]] = None,
                      processing_stats: Optional[Dict[str, Any]] = None,
                      include_metadata: bool = False) -> Dict[str, Any]:
        """
        Format extraction results with optional metadata inclusion.
        
        Args:
            headings: List of extracted headings
            document_info: Document metadata
            hierarchy_tree: Optional hierarchical structure
            processing_stats: Optional processing statistics
            include_metadata: Whether to include full metadata and accessibility data
            
        Returns:
            Dictionary with formatted results (simple or full based on include_metadata)
        """
        
        # Check environment variable as fallback
        if include_metadata is None:
            include_metadata = os.getenv('INCLUDE_METADATA', 'false').lower() == 'true'
        
        if include_metadata:
            # Full format with all metadata and accessibility data
            self.logger.info("Generating full format with metadata and accessibility data")
            
            # Generate accessibility metadata
            accessibility_data = self.generate_accessibility_tags(headings)
            
            # Use the standard full format
            result = self.format_results_full(headings, document_info, hierarchy_tree, processing_stats)
            
            # Add accessibility metadata to result
            result["accessibility"] = accessibility_data
            
            return result
        else:
            # Simple format with just title and outline
            self.logger.info("Generating simple format with title and outline only")
            return self.format_results_simple(headings, document_info)
    
    def format_results_simple(self, headings: List[Dict[str, Any]], 
                             document_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format extraction results into simple clean format with only title and outline.
        
        Returns:
            Dictionary with just "title" and "outline" fields
        """
        
        # Extract document title
        title = self._extract_simple_title(headings, document_info)
        
        # Convert headings to simple outline format - ONLY level, text, and page
        outline = []
        for heading in headings:
            text = heading.get("text", "").strip()
            page = heading.get("page", 1)
            
            # Smart level mapping based on content
            level_str = self._determine_heading_level(text, heading.get("level", 1))
            
            # Create clean outline item with ONLY the 3 required fields
            outline_item = {
                "level": level_str,
                "text": text,
                "page": page
            }
            
            outline.append(outline_item)
        
        # Return ONLY title and outline - no metadata, no extra fields
        return {
            "title": title,
            "outline": outline
        }
    
    def format_results_full(self, headings: List[Dict[str, Any]], 
                           document_info: Dict[str, Any],
                           hierarchy_tree: Optional[Dict[str, Any]] = None,
                           processing_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format extraction results into full format with all metadata."""
        
        result = {
            "document_info": self._format_document_info(document_info),
            "headings": self._format_headings(headings),
            "metadata": self._generate_metadata(),
        }
        
        # Add optional sections
        if hierarchy_tree:
            result["hierarchy"] = hierarchy_tree
        
        if processing_stats:
            result["processing_stats"] = processing_stats
        
        if INCLUDE_DEBUG_INFO:
            result["debug_info"] = self._generate_debug_info(headings, document_info)
        
        # Validate output structure
        self._validate_output(result)
        
        return result
    
    def _extract_simple_title(self, headings: List[Dict[str, Any]], 
                             document_info: Dict[str, Any]) -> str:
        """Extract simple document title for clean output."""
        
        # First try document metadata
        if document_info.get("title"):
            doc_title = document_info["title"].strip()
            if doc_title and doc_title != "Untitled":
                return doc_title
        
        # Try filename without extension
        if document_info.get("filename"):
            filename = document_info["filename"]
            # Remove extension and clean up
            title = Path(filename).stem
            # Remove common prefixes like "Microsoft Word - "
            if title.startswith("Microsoft Word - "):
                title = title[17:]
            return title
        
        # Try first heading as title
        if headings:
            first_heading = headings[0]
            text = first_heading.get("text", "").strip()
            if text and len(text) < 100:  # Reasonable title length
                return text
        
        # Fallback
        return "Untitled Document"
    
    def _determine_heading_level(self, text: str, original_level: int) -> str:
        """Determine appropriate heading level based on content and context."""
        
        import re
        
        # Check if it's a subsection (like 2.1, 2.2, 3.1, 4.1, etc.) - these should be H2
        if re.match(r'^\d+\.\d+', text):  # Pattern for X.Y format (2.1, 3.1, etc.)
            return "H2"
        
        # Check if it's a main numbered section (like 1., 2., 3., 4.) - these should be H1  
        elif re.match(r'^\d+\.', text):  # Pattern for X. format (1., 2., etc.)
            return "H1"
        
        # Standard document sections are H1
        elif any(section in text.lower() for section in [
            "revision history", "table of contents", "acknowledgements", 
            "references", "introduction", "conclusion", "abstract", 
            "summary", "appendix", "bibliography"
        ]):
            return "H1"
        
        # Sub-subsections (like 2.1.1, 3.2.1) should be H3
        elif re.match(r'^\d+\.\d+\.\d+', text):
            return "H3"
        
        # Use original level as fallback, mapped to string
        else:
            level_map = {1: "H1", 2: "H2", 3: "H3", 4: "H4", 5: "H5", 6: "H6"}
            return level_map.get(original_level, "H1")
    
    def generate_accessibility_tags(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive accessibility metadata and tags."""
        
        self.logger.info("Generating accessibility tags and metadata")
        
        # Generate PDF/UA structure
        pdf_ua_structure = self.accessibility_tagger.generate_pdf_ua_structure(headings)
        
        # Generate accessibility metadata
        accessibility_metadata = self.accessibility_tagger.generate_accessibility_metadata(headings)
        
        # Generate ARIA labels
        aria_labels = self.accessibility_tagger.create_aria_labels(headings)
        
        return {
            "pdf_ua_structure": pdf_ua_structure,
            "metadata": accessibility_metadata,
            "aria_labels": aria_labels,
            "structure_xml_available": True,
            "compliance_summary": {
                "wcag_2_1_aa": accessibility_metadata["compliance"]["wcag_2.1_aa"],
                "pdf_ua": accessibility_metadata["compliance"]["pdf_ua"],
                "section_508": accessibility_metadata["compliance"]["section_508"],
                "accessibility_score": accessibility_metadata["accessibility_score"]
            }
        }
    
    def save_pdf_ua_xml(self, headings: List[Dict[str, Any]], output_path: str) -> None:
        """Save PDF/UA accessibility structure as XML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate structure XML
        structure_xml = self.accessibility_tagger.create_structure_xml(headings)
        
        # Write XML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(structure_xml)
        
        self.logger.info(f"PDF/UA accessibility XML saved to: {output_path}")
    
    def format_results_custom(self, headings: List[Dict[str, Any]], 
                             document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format extraction results into the custom outline format with only essential fields."""
        
        # This method is kept for backward compatibility
        # It now calls the simple format method
        return self.format_results_simple(headings, document_info)
    
    def _format_document_info(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format document information section."""
        formatted_info = {
            "filename": document_info.get("filename", "unknown.pdf"),
            "total_pages": document_info.get("total_pages", 0),
            "processing_time": round(document_info.get("processing_time", 0), 3),
            "file_size_mb": round(document_info.get("file_size", 0) / (1024 * 1024), 2),
            "language_detected": document_info.get("language", "auto"),
            "extraction_method": document_info.get("method", "hybrid"),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add optional fields if available
        optional_fields = ["title", "author", "subject", "creator", "creation_date"]
        for field in optional_fields:
            if field in document_info:
                formatted_info[field] = document_info[field]
        
        return formatted_info
    
    def _format_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format headings list with consistent structure."""
        formatted_headings = []
        
        for i, heading in enumerate(headings):
            formatted_heading = {
                "id": i + 1,
                "text": heading.get("text", "").strip(),
                "level": max(0, min(heading.get("level", 1), 6)),  # Clamp 0-6
                "page": heading.get("page", 1),
                "bbox": self._format_bbox(heading.get("bbox", [0, 0, 0, 0])),
                "font_info": self._format_font_info(heading.get("font_info", {})),
            }
            
            # Add confidence if enabled
            if INCLUDE_CONFIDENCE_SCORES:
                formatted_heading["confidence"] = round(
                    heading.get("confidence", 0.0), 3
                )
            
            # Add optional features
            features = heading.get("features", {})
            if features:
                formatted_heading["features"] = self._format_features(features)
            
            formatted_headings.append(formatted_heading)
        
        return formatted_headings
    
    def _format_bbox(self, bbox: List[float]) -> Dict[str, float]:
        """Format bounding box coordinates."""
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        
        return {
            "x0": round(bbox[0], 2),
            "y0": round(bbox[1], 2),
            "x1": round(bbox[2], 2),
            "y1": round(bbox[3], 2),
            "width": round(bbox[2] - bbox[0], 2),
            "height": round(bbox[3] - bbox[1], 2)
        }
    
    def _format_font_info(self, font_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format font information."""
        return {
            "size": font_info.get("size", 12),
            "weight": font_info.get("weight", "normal"),
            "family": font_info.get("family", "unknown"),
            "style": font_info.get("style", "normal")
        }
    
    def _format_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Format heading features and metadata."""
        formatted_features = {}
        
        # Numbering information
        if features.get("numbering_pattern"):
            formatted_features["numbering"] = {
                "pattern": features["numbering_pattern"],
                "has_numbering": True
            }
        
        # Semantic information
        if features.get("semantic_group"):
            formatted_features["semantic"] = {
                "group": features["semantic_group"],
                "keywords_matched": features.get("keywords_matched", [])
            }
        
        # Layout information
        layout_features = {}
        if "alignment" in features:
            layout_features["alignment"] = features["alignment"]
        if "indentation" in features:
            layout_features["indentation"] = round(features["indentation"], 2)
        if "spacing_before" in features:
            layout_features["spacing_before"] = round(features["spacing_before"], 2)
        if "spacing_after" in features:
            layout_features["spacing_after"] = round(features["spacing_after"], 2)
        
        if layout_features:
            formatted_features["layout"] = layout_features
        
        return formatted_features
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate extraction metadata."""
        return {
            "extraction_version": "1.0.0",
            "format_version": "1.0",
            "generator": "PDF-Heading-Extractor-Hybrid",
            "extraction_date": datetime.now().isoformat(),
            "settings": {
                "include_confidence": INCLUDE_CONFIDENCE_SCORES,
                "include_debug": INCLUDE_DEBUG_INFO,
                "output_format": OUTPUT_FORMAT
            }
        }
    
    def _generate_debug_info(self, headings: List[Dict[str, Any]], 
                           document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate debug information for troubleshooting."""
        return {
            "total_candidates_found": len(headings),
            "level_distribution": self._calculate_level_distribution(headings),
            "confidence_distribution": self._calculate_confidence_distribution(headings),
            "page_distribution": self._calculate_page_distribution(headings),
            "font_analysis": document_info.get("font_analysis", {}),
            "processing_stages": document_info.get("processing_stages", []),
            "warnings": document_info.get("warnings", [])
        }
    
    def _calculate_level_distribution(self, headings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of heading levels."""
        distribution = {}
        for heading in headings:
            level = heading.get("level", 1)
            distribution[f"level_{level}"] = distribution.get(f"level_{level}", 0) + 1
        return distribution
    
    def _calculate_confidence_distribution(self, headings: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence score statistics."""
        if not headings:
            return {}
        
        confidences = [h.get("confidence", 0.0) for h in headings]
        return {
            "min": round(min(confidences), 3),
            "max": round(max(confidences), 3),
            "mean": round(sum(confidences) / len(confidences), 3),
            "median": round(sorted(confidences)[len(confidences)//2], 3)
        }
    
    def _calculate_page_distribution(self, headings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of headings across pages."""
        distribution = {}
        for heading in headings:
            page = heading.get("page", 1)
            distribution[f"page_{page}"] = distribution.get(f"page_{page}", 0) + 1
        return distribution
    
    def _validate_output(self, result: Dict[str, Any]) -> None:
        """Validate output structure and content."""
        required_fields = ["document_info", "headings", "metadata"]
        
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate headings structure
        for i, heading in enumerate(result["headings"]):
            required_heading_fields = ["id", "text", "level", "page", "bbox"]
            for field in required_heading_fields:
                if field not in heading:
                    raise ValueError(f"Heading {i+1} missing required field: {field}")
        
        self.logger.debug("Output validation passed")
    
    def save_json(self, result: Dict[str, Any], output_path: str, 
                  pretty: bool = True) -> None:
        """Save results as JSON file (format depends on metadata inclusion)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(result, f, indent=4, ensure_ascii=False)
            else:
                json.dump(result, f, ensure_ascii=False)
        
        self.logger.info(f"Results saved to JSON: {output_path}")
    
    def save_json_custom(self, result: Dict[str, Any], output_path: str, 
                        pretty: bool = True) -> None:
        """Save results in custom outline format as JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(result, f, indent=4, ensure_ascii=False)
            else:
                json.dump(result, f, ensure_ascii=False)
        
        self.logger.info(f"Custom format results saved to JSON: {output_path}")
    
    def save_csv(self, result: Dict[str, Any], output_path: str) -> None:
        """Save headings as CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle both simple and full format
        if "outline" in result:
            # Simple format
            headings = result["outline"]
            fieldnames = ["level", "text", "page"]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for heading in headings:
                    writer.writerow(heading)
        else:
            # Full format
            headings = result["headings"]
            fieldnames = [
                "id", "text", "level", "page", "confidence",
                "font_size", "font_weight", "bbox_x0", "bbox_y0", 
                "bbox_x1", "bbox_y1", "width", "height"
            ]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for heading in headings:
                    row = {
                        "id": heading["id"],
                        "text": heading["text"],
                        "level": heading["level"],
                        "page": heading["page"],
                        "confidence": heading.get("confidence", 0.0),
                        "font_size": heading["font_info"]["size"],
                        "font_weight": heading["font_info"]["weight"],
                        "bbox_x0": heading["bbox"]["x0"],
                        "bbox_y0": heading["bbox"]["y0"],
                        "bbox_x1": heading["bbox"]["x1"],
                        "bbox_y1": heading["bbox"]["y1"],
                        "width": heading["bbox"]["width"],
                        "height": heading["bbox"]["height"]
                    }
                    writer.writerow(row)
        
        self.logger.info(f"Results saved to CSV: {output_path}")
    
    def save_xml(self, result: Dict[str, Any], output_path: str) -> None:
        """Save results as XML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        root = ET.Element("pdf_headings")
        
        # Handle both simple and full format
        if "outline" in result:
            # Simple format
            title_elem = ET.SubElement(root, "title")
            title_elem.text = result["title"]
            
            outline_elem = ET.SubElement(root, "outline")
            for heading in result["outline"]:
                heading_elem = ET.SubElement(outline_elem, "heading")
                heading_elem.set("level", heading["level"])
                heading_elem.set("page", str(heading["page"]))
                heading_elem.text = heading["text"]
        else:
            # Full format
            doc_info = ET.SubElement(root, "document_info")
            for key, value in result["document_info"].items():
                elem = ET.SubElement(doc_info, key)
                elem.text = str(value)
            
            headings_elem = ET.SubElement(root, "headings")
            for heading in result["headings"]:
                heading_elem = ET.SubElement(headings_elem, "heading")
                heading_elem.set("id", str(heading["id"]))
                heading_elem.set("level", str(heading["level"]))
                heading_elem.set("page", str(heading["page"]))
                
                text_elem = ET.SubElement(heading_elem, "text")
                text_elem.text = heading["text"]
                
                bbox_elem = ET.SubElement(heading_elem, "bbox")
                for coord, value in heading["bbox"].items():
                    coord_elem = ET.SubElement(bbox_elem, coord)
                    coord_elem.text = str(value)
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        self.logger.info(f"Results saved to XML: {output_path}")
    
    def save_markdown(self, result: Dict[str, Any], output_path: str) -> None:
        """Save headings as Markdown outline."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        
        # Handle both simple and full format
        if "outline" in result:
            # Simple format
            lines.append(f"# Document Outline: {result['title']}")
            lines.append("")
            lines.append(f"- **Total Headings:** {len(result['outline'])}")
            lines.append("")
            lines.append("---")
            lines.append("")
            
            for heading in result["outline"]:
                level = heading["level"]
                text = heading["text"]
                page = heading["page"]
                
                if level == "H1":
                    lines.append(f"# {text} *(Page {page})*")
                elif level == "H2":
                    lines.append(f"## {text} *(Page {page})*")
                elif level == "H3":
                    lines.append(f"### {text} *(Page {page})*")
                else:
                    indent = "  " * (int(level[1:]) - 1) if level.startswith("H") else "  "
                    lines.append(f"{indent}- **{text}** *(Page {page})*")
        else:
            # Full format
            lines.append(f"# Document Outline: {result['document_info']['filename']}")
            lines.append("")
            lines.append(f"- **Total Pages:** {result['document_info']['total_pages']}")
            lines.append(f"- **Processing Time:** {result['document_info']['processing_time']}s")
            lines.append(f"- **Total Headings:** {len(result['headings'])}")
            lines.append("")
            lines.append("---")
            lines.append("")
            
            for heading in result["headings"]:
                level = heading["level"]
                text = heading["text"]
                page = heading["page"]
                
                if level == 0:
                    lines.append(f"# {text} *(Page {page})*")
                else:
                    indent = "  " * (level - 1)
                    lines.append(f"{indent}- **{text}** *(Page {page})*")
        
        # Write markdown
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        self.logger.info(f"Results saved to Markdown: {output_path}")
    
    def save_html_outline(self, result: Dict[str, Any], output_path: str) -> None:
        """Save headings as HTML outline with collapsible tree."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle both simple and full format
        if "outline" in result:
            title = result["title"]
            headings = result["outline"]
            total_headings = len(headings)
            
            headings_html = []
            for heading in headings:
                level = heading["level"]
                text = heading["text"]
                page = heading["page"]
                
                level_class = f"level-{level.lower()}" if isinstance(level, str) else f"level-{level}"
                
                heading_html = f'''
                <div class="heading {level_class}">
                    {text} 
                    <span class="page-num">Page {page}</span>
                </div>'''
                headings_html.append(heading_html)
            
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Outline: {title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .outline {{ max-width: 800px; }}
        .heading {{ margin: 5px 0; padding: 5px; border-left: 3px solid #007acc; }}
        .level-h1 {{ font-size: 1.5em; font-weight: bold; color: #333; }}
        .level-h2 {{ font-size: 1.3em; font-weight: bold; margin-left: 20px; }}
        .level-h3 {{ font-size: 1.1em; font-weight: bold; margin-left: 40px; }}
        .level-h4 {{ font-size: 1.0em; margin-left: 60px; }}
        .level-h5 {{ font-size: 0.9em; margin-left: 80px; }}
        .level-h6 {{ font-size: 0.8em; margin-left: 100px; }}
        .page-num {{ color: #666; font-size: 0.8em; }}
        .stats {{ background: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>PDF Document Outline</h1>
    
    <div class="stats">
        <h3>Document Information</h3>
        <p><strong>Title:</strong> {title}</p>
        <p><strong>Total Headings:</strong> {total_headings}</p>
    </div>
    
    <div class="outline">
        <h3>Heading Structure</h3>
        {"".join(headings_html)}
    </div>
</body>
</html>"""
        else:
            # Full format (existing implementation)
            html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Outline: {filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .outline {{ max-width: 800px; }}
        .heading {{ margin: 5px 0; padding: 5px; border-left: 3px solid #007acc; }}
        .level-0 {{ font-size: 1.5em; font-weight: bold; color: #333; }}
        .level-1 {{ font-size: 1.3em; font-weight: bold; margin-left: 0px; }}
        .level-2 {{ font-size: 1.1em; font-weight: bold; margin-left: 20px; }}
        .level-3 {{ font-size: 1.0em; margin-left: 40px; }}
        .level-4 {{ font-size: 0.9em; margin-left: 60px; }}
        .level-5 {{ font-size: 0.9em; margin-left: 80px; }}
        .level-6 {{ font-size: 0.8em; margin-left: 100px; }}
        .page-num {{ color: #666; font-size: 0.8em; }}
        .confidence {{ color: #999; font-size: 0.7em; }}
        .stats {{ background: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>PDF Document Outline</h1>
    
    <div class="stats">
        <h3>Document Information</h3>
        <p><strong>File:</strong> {filename}</p>
        <p><strong>Pages:</strong> {total_pages}</p>
        <p><strong>Processing Time:</strong> {processing_time}s</p>
        <p><strong>Total Headings:</strong> {total_headings}</p>
        <p><strong>Language:</strong> {language}</p>
    </div>
    
    <div class="outline">
        <h3>Heading Structure</h3>
        {headings_html}
    </div>
</body>
</html>"""
            
            # Generate headings HTML
            headings_html = []
            for heading in result["headings"]:
                level = heading["level"]
                text = heading["text"]
                page = heading["page"]
                confidence = heading.get("confidence", 0.0)
                
                confidence_str = f'<span class="confidence">(conf: {confidence:.2f})</span>' if INCLUDE_CONFIDENCE_SCORES else ''
                
                heading_html = f'''
                <div class="heading level-{level}">
                    {text} 
                    <span class="page-num">Page {page}</span>
                    {confidence_str}
                </div>'''
                
                headings_html.append(heading_html)
            
            # Fill template
            html_content = html_template.format(
                filename=result["document_info"]["filename"],
                total_pages=result["document_info"]["total_pages"],
                processing_time=result["document_info"]["processing_time"],
                total_headings=len(result["headings"]),
                language=result["document_info"]["language_detected"],
                headings_html="".join(headings_html)
            )
        
        # Write HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Results saved to HTML: {output_path}")
    
    def export_multiple_formats(self, result: Dict[str, Any], 
                               base_path: str, formats: List[str]) -> Dict[str, str]:
        """Export results in multiple formats."""
        base_path = Path(base_path)
        output_files = {}
        
        for format_type in formats:
            if format_type == "json":
                output_path = base_path.with_suffix('.json')
                self.save_json(result, output_path)
                output_files["json"] = str(output_path)
                
            elif format_type == "json_custom":
                output_path = base_path.with_suffix('_custom.json')
                # Use simple format if not already in that format
                if "outline" in result:
                    custom_result = result
                else:
                    custom_result = self.format_results_simple(
                        result["headings"], result["document_info"]
                    )
                self.save_json_custom(custom_result, output_path)
                output_files["json_custom"] = str(output_path)
                
            elif format_type == "csv":
                output_path = base_path.with_suffix('.csv')
                self.save_csv(result, output_path)
                output_files["csv"] = str(output_path)
                
            elif format_type == "xml":
                output_path = base_path.with_suffix('.xml')
                self.save_xml(result, output_path)
                output_files["xml"] = str(output_path)
                
            elif format_type == "markdown":
                output_path = base_path.with_suffix('.md')
                self.save_markdown(result, output_path)
                output_files["markdown"] = str(output_path)
                
            elif format_type == "html":
                output_path = base_path.with_suffix('.html')
                self.save_html_outline(result, output_path)
                output_files["html"] = str(output_path)
                
            elif format_type == "pdf_ua_xml":
                output_path = base_path.with_suffix('_accessibility.xml')
                # Extract headings from either format
                headings = result.get("outline", result.get("headings", []))
                self.save_pdf_ua_xml(headings, output_path)
                output_files["pdf_ua_xml"] = str(output_path)
        
        return output_files