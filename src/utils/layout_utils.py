import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, namedtuple
from dataclasses import dataclass
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path

from config.settings import (
    TITLE_POSITION_THRESHOLD, CENTER_ALIGNMENT_TOLERANCE,
    MIN_HEADING_LENGTH, MAX_HEADING_LENGTH
)


@dataclass
class LayoutRegion:
    """Represents a region in the PDF layout."""
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    region_type: str  # 'header', 'footer', 'body', 'sidebar', 'column'
    page: int
    confidence: float
    elements: List[Dict[str, Any]]


@dataclass
class ColumnInfo:
    """Information about document columns."""
    column_count: int
    column_boundaries: List[Tuple[float, float]]  # List of (left, right) boundaries
    gutter_width: float
    is_consistent: bool
    confidence: float


@dataclass
class LayoutStructure:
    """Overall layout structure of the document."""
    page_margins: Tuple[float, float, float, float]  # top, right, bottom, left
    columns: ColumnInfo
    has_headers: bool
    has_footers: bool
    header_region: Optional[LayoutRegion]
    footer_region: Optional[LayoutRegion]
    body_region: LayoutRegion
    layout_type: str  # 'single', 'two_column', 'multi_column', 'complex'
    consistency_score: float


# Named tuple for position analysis
Position = namedtuple('Position', ['x', 'y', 'width', 'height'])


class LayoutUtils:
    """Advanced PDF layout analysis utilities."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Layout detection thresholds
        self.header_threshold = 0.15  # Top 15% of page
        self.footer_threshold = 0.85  # Bottom 15% of page
        self.margin_detection_threshold = 0.05  # 5% margin detection
        self.column_gap_threshold = 20  # Minimum gap between columns
        self.alignment_tolerance = 5  # Pixels tolerance for alignment
        
        # Text block classification
        self.min_text_block_height = 10
        self.min_text_block_width = 20
        
    def analyze_page_layout(self, page: fitz.Page) -> LayoutStructure:
        """Analyze the layout structure of a single page."""
        
        page_rect = page.rect
        text_blocks = self._extract_text_blocks(page)
        
        if not text_blocks:
            return self._create_default_layout(page_rect)
        
        # Detect margins
        margins = self._detect_margins(text_blocks, page_rect)
        
        # Detect columns
        columns = self._detect_columns(text_blocks, page_rect, margins)
        
        # Detect header/footer regions
        header_region = self._detect_header_region(text_blocks, page_rect)
        footer_region = self._detect_footer_region(text_blocks, page_rect)
        
        # Define body region
        body_region = self._define_body_region(page_rect, margins, header_region, footer_region)
        
        # Determine layout type
        layout_type = self._classify_layout_type(columns, text_blocks)
        
        # Calculate consistency score
        consistency_score = self._calculate_layout_consistency(text_blocks, columns)
        
        return LayoutStructure(
            page_margins=margins,
            columns=columns,
            has_headers=header_region is not None,
            has_footers=footer_region is not None,
            header_region=header_region,
            footer_region=footer_region,
            body_region=body_region,
            layout_type=layout_type,
            consistency_score=consistency_score
        )
    
    def _extract_text_blocks(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Extract text blocks with position and content information."""
        
        blocks = []
        dict_blocks = page.get_text("dict")["blocks"]
        
        for block_idx, block in enumerate(dict_blocks):
            if "lines" not in block:
                continue
            
            block_bbox = block["bbox"]
            block_text = ""
            block_spans = []
            
            # Collect all text and spans from the block
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        block_text += span["text"] + " "
                        block_spans.append(span)
            
            if block_text.strip() and len(block_spans) > 0:
                # Calculate block properties
                avg_font_size = np.mean([span["size"] for span in block_spans])
                is_bold = any(span["flags"] & 16 for span in block_spans)  # Bold flag
                
                blocks.append({
                    "id": block_idx,
                    "bbox": block_bbox,
                    "text": block_text.strip(),
                    "spans": block_spans,
                    "avg_font_size": avg_font_size,
                    "is_bold": is_bold,
                    "line_count": len(block["lines"]),
                    "position": Position(
                        x=block_bbox[0],
                        y=block_bbox[1],
                        width=block_bbox[2] - block_bbox[0],
                        height=block_bbox[3] - block_bbox[1]
                    )
                })
        
        return blocks
    
    def _detect_margins(self, text_blocks: List[Dict[str, Any]], 
                       page_rect: fitz.Rect) -> Tuple[float, float, float, float]:
        """Detect page margins based on text block positions."""
        
        if not text_blocks:
            # Default margins (1 inch = 72 points)
            return (72, 72, 72, 72)
        
        # Extract all block boundaries
        left_edges = [block["bbox"][0] for block in text_blocks]
        right_edges = [block["bbox"][2] for block in text_blocks]
        top_edges = [block["bbox"][1] for block in text_blocks]
        bottom_edges = [block["bbox"][3] for block in text_blocks]
        
        # Calculate margins with some tolerance
        left_margin = min(left_edges) if left_edges else 72
        right_margin = page_rect.width - max(right_edges) if right_edges else 72
        top_margin = min(top_edges) if top_edges else 72
        bottom_margin = page_rect.height - max(bottom_edges) if bottom_edges else 72
        
        # Ensure reasonable margins (minimum 18 points, maximum 144 points)
        margins = (
            max(18, min(144, top_margin)),
            max(18, min(144, right_margin)),
            max(18, min(144, bottom_margin)),
            max(18, min(144, left_margin))
        )
        
        return margins
    
    def _detect_columns(self, text_blocks: List[Dict[str, Any]], 
                       page_rect: fitz.Rect, 
                       margins: Tuple[float, float, float, float]) -> ColumnInfo:
        """Detect column structure in the page."""
        
        if not text_blocks:
            return ColumnInfo(1, [(margins[3], page_rect.width - margins[1])], 0, True, 1.0)
        
        # Get body text blocks (exclude headers/footers)
        body_blocks = self._filter_body_blocks(text_blocks, page_rect)
        
        if not body_blocks:
            return ColumnInfo(1, [(margins[3], page_rect.width - margins[1])], 0, True, 1.0)
        
        # Analyze horizontal distribution of text blocks
        x_positions = []
        for block in body_blocks:
            x_positions.extend([block["bbox"][0], block["bbox"][2]])
        
        if not x_positions:
            return ColumnInfo(1, [(margins[3], page_rect.width - margins[1])], 0, True, 1.0)
        
        # Sort positions and find clusters
        x_positions.sort()
        
        # Find potential column boundaries using clustering
        column_boundaries = self._find_column_boundaries(x_positions, margins, page_rect.width)
        
        # Validate column detection
        column_count = len(column_boundaries)
        
        # Calculate gutter width
        gutter_width = 0
        if column_count > 1:
            gutters = []
            for i in range(len(column_boundaries) - 1):
                gutter = column_boundaries[i+1][0] - column_boundaries[i][1]
                if gutter > 0:
                    gutters.append(gutter)
            gutter_width = np.mean(gutters) if gutters else 0
        
        # Check consistency
        consistency = self._check_column_consistency(body_blocks, column_boundaries)
        
        return ColumnInfo(
            column_count=column_count,
            column_boundaries=column_boundaries,
            gutter_width=gutter_width,
            is_consistent=consistency > 0.8,
            confidence=consistency
        )
    
    def _find_column_boundaries(self, x_positions: List[float], 
                               margins: Tuple[float, float, float, float],
                               page_width: float) -> List[Tuple[float, float]]:
        """Find column boundaries from x-positions."""
        
        # Use histogram to find natural breaks
        hist, bins = np.histogram(x_positions, bins=20)
        
        # Find significant gaps (potential column separators)
        gaps = []
        for i in range(len(bins) - 1):
            if hist[i] == 0 and i > 0 and i < len(hist) - 1:
                # Check if it's a significant gap
                left_density = np.sum(hist[max(0, i-2):i])
                right_density = np.sum(hist[i+1:min(len(hist), i+3)])
                
                if left_density > 0 and right_density > 0:
                    gaps.append((bins[i] + bins[i+1]) / 2)
        
        # If no clear gaps found, assume single column
        if not gaps:
            return [(margins[3], page_width - margins[1])]
        
        # Create column boundaries
        boundaries = []
        left_boundary = margins[3]
        
        for gap in sorted(gaps):
            if gap > left_boundary + 50:  # Minimum column width
                boundaries.append((left_boundary, gap - 10))  # Leave some margin
                left_boundary = gap + 10
        
        # Add final column
        boundaries.append((left_boundary, page_width - margins[1]))
        
        # Filter out very narrow columns
        boundaries = [(left, right) for left, right in boundaries if right - left > 50]
        
        return boundaries if boundaries else [(margins[3], page_width - margins[1])]
    
    def _check_column_consistency(self, text_blocks: List[Dict[str, Any]], 
                                 column_boundaries: List[Tuple[float, float]]) -> float:
        """Check how consistently text blocks align with detected columns."""
        
        if not text_blocks or len(column_boundaries) <= 1:
            return 1.0
        
        aligned_blocks = 0
        
        for block in text_blocks:
            block_left = block["bbox"][0]
            block_right = block["bbox"][2]
            
            # Check if block aligns with any column
            for col_left, col_right in column_boundaries:
                # Allow some tolerance
                if (abs(block_left - col_left) <= self.alignment_tolerance or
                    abs(block_right - col_right) <= self.alignment_tolerance or
                    (block_left >= col_left - self.alignment_tolerance and 
                     block_right <= col_right + self.alignment_tolerance)):
                    aligned_blocks += 1
                    break
        
        return aligned_blocks / len(text_blocks) if text_blocks else 0.0
    
    def _filter_body_blocks(self, text_blocks: List[Dict[str, Any]], 
                           page_rect: fitz.Rect) -> List[Dict[str, Any]]:
        """Filter out header/footer blocks to get body text."""
        
        page_height = page_rect.height
        header_boundary = page_height * self.header_threshold
        footer_boundary = page_height * self.footer_threshold
        
        body_blocks = []
        for block in text_blocks:
            block_top = block["bbox"][1]
            block_bottom = block["bbox"][3]
            
            # Skip header region
            if block_bottom < header_boundary:
                continue
            
            # Skip footer region
            if block_top > footer_boundary:
                continue
            
            body_blocks.append(block)
        
        return body_blocks
    
    def _detect_header_region(self, text_blocks: List[Dict[str, Any]], 
                             page_rect: fitz.Rect) -> Optional[LayoutRegion]:
        """Detect header region on the page."""
        
        page_height = page_rect.height
        header_boundary = page_height * self.header_threshold
        
        header_blocks = []
        for block in text_blocks:
            if block["bbox"][3] <= header_boundary:  # Block bottom within header region
                header_blocks.append(block)
        
        if not header_blocks:
            return None
        
        # Calculate header region bbox
        min_x = min(block["bbox"][0] for block in header_blocks)
        max_x = max(block["bbox"][2] for block in header_blocks)
        min_y = min(block["bbox"][1] for block in header_blocks)
        max_y = max(block["bbox"][3] for block in header_blocks)
        
        return LayoutRegion(
            bbox=(min_x, min_y, max_x, max_y),
            region_type="header",
            page=1,  # Will be set by caller
            confidence=0.8,
            elements=header_blocks
        )
    
    def _detect_footer_region(self, text_blocks: List[Dict[str, Any]], 
                             page_rect: fitz.Rect) -> Optional[LayoutRegion]:
        """Detect footer region on the page."""
        
        page_height = page_rect.height
        footer_boundary = page_height * self.footer_threshold
        
        footer_blocks = []
        for block in text_blocks:
            if block["bbox"][1] >= footer_boundary:  # Block top within footer region
                footer_blocks.append(block)
        
        if not footer_blocks:
            return None
        
        # Calculate footer region bbox
        min_x = min(block["bbox"][0] for block in footer_blocks)
        max_x = max(block["bbox"][2] for block in footer_blocks)
        min_y = min(block["bbox"][1] for block in footer_blocks)
        max_y = max(block["bbox"][3] for block in footer_blocks)
        
        return LayoutRegion(
            bbox=(min_x, min_y, max_x, max_y),
            region_type="footer",
            page=1,  # Will be set by caller
            confidence=0.8,
            elements=footer_blocks
        )
    
    def _define_body_region(self, page_rect: fitz.Rect, 
                           margins: Tuple[float, float, float, float],
                           header_region: Optional[LayoutRegion],
                           footer_region: Optional[LayoutRegion]) -> LayoutRegion:
        """Define the main body region of the page."""
        
        # Start with page margins
        top = margins[0]
        right = page_rect.width - margins[1]
        bottom = page_rect.height - margins[2]
        left = margins[3]
        
        # Adjust for header
        if header_region:
            top = max(top, header_region.bbox[3] + 10)  # 10pt gap
        
        # Adjust for footer
        if footer_region:
            bottom = min(bottom, footer_region.bbox[1] - 10)  # 10pt gap
        
        return LayoutRegion(
            bbox=(left, top, right, bottom),
            region_type="body",
            page=1,  # Will be set by caller
            confidence=1.0,
            elements=[]
        )
    
    def _classify_layout_type(self, columns: ColumnInfo, 
                             text_blocks: List[Dict[str, Any]]) -> str:
        """Classify the overall layout type."""
        
        if columns.column_count == 1:
            return "single_column"
        elif columns.column_count == 2:
            return "two_column"
        elif columns.column_count > 2:
            return "multi_column"
        else:
            # Analyze complexity based on text block distribution
            if len(text_blocks) > 20:
                return "complex"
            else:
                return "simple"
    
    def _calculate_layout_consistency(self, text_blocks: List[Dict[str, Any]], 
                                    columns: ColumnInfo) -> float:
        """Calculate overall layout consistency score."""
        
        if not text_blocks:
            return 1.0
        
        scores = []
        
        # Column consistency
        scores.append(columns.confidence)
        
        # Alignment consistency
        alignment_score = self._calculate_alignment_consistency(text_blocks)
        scores.append(alignment_score)
        
        # Spacing consistency
        spacing_score = self._calculate_spacing_consistency(text_blocks)
        scores.append(spacing_score)
        
        return np.mean(scores)
    
    def _calculate_alignment_consistency(self, text_blocks: List[Dict[str, Any]]) -> float:
        """Calculate how consistently text blocks are aligned."""
        
        if len(text_blocks) < 2:
            return 1.0
        
        # Group blocks by similar left edges
        left_edges = [block["bbox"][0] for block in text_blocks]
        unique_edges = []
        
        for edge in left_edges:
            # Check if this edge is similar to any existing unique edge
            similar_found = False
            for unique_edge in unique_edges:
                if abs(edge - unique_edge) <= self.alignment_tolerance:
                    similar_found = True
                    break
            
            if not similar_found:
                unique_edges.append(edge)
        
        # Fewer unique edges means better alignment
        alignment_score = 1.0 - (len(unique_edges) - 1) / len(text_blocks)
        return max(0.0, alignment_score)
    
    def _calculate_spacing_consistency(self, text_blocks: List[Dict[str, Any]]) -> float:
        """Calculate consistency of spacing between text blocks."""
        
        if len(text_blocks) < 2:
            return 1.0
        
        # Sort blocks by vertical position
        sorted_blocks = sorted(text_blocks, key=lambda x: x["bbox"][1])
        
        # Calculate vertical gaps between consecutive blocks
        gaps = []
        for i in range(len(sorted_blocks) - 1):
            current_bottom = sorted_blocks[i]["bbox"][3]
            next_top = sorted_blocks[i + 1]["bbox"][1]
            gap = next_top - current_bottom
            
            if gap > 0:  # Only positive gaps
                gaps.append(gap)
        
        if not gaps:
            return 1.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        if mean_gap == 0:
            return 1.0
        
        cv = std_gap / mean_gap
        consistency_score = max(0.0, 1.0 - cv)
        
        return consistency_score
    
    def _create_default_layout(self, page_rect: fitz.Rect) -> LayoutStructure:
        """Create default layout structure when no text blocks are found."""
        
        default_margins = (72, 72, 72, 72)  # 1 inch margins
        
        default_columns = ColumnInfo(
            column_count=1,
            column_boundaries=[(72, page_rect.width - 72)],
            gutter_width=0,
            is_consistent=True,
            confidence=1.0
        )
        
        body_region = LayoutRegion(
            bbox=(72, 72, page_rect.width - 72, page_rect.height - 72),
            region_type="body",
            page=1,
            confidence=1.0,
            elements=[]
        )
        
        return LayoutStructure(
            page_margins=default_margins,
            columns=default_columns,
            has_headers=False,
            has_footers=False,
            header_region=None,
            footer_region=None,
            body_region=body_region,
            layout_type="single_column",
            consistency_score=1.0
        )
    
    def analyze_text_alignment(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze text alignment patterns in the document."""
        
        if not text_blocks:
            return {"dominant_alignment": "left", "alignment_consistency": 1.0}
        
        # Categorize blocks by alignment
        left_aligned = 0
        center_aligned = 0
        right_aligned = 0
        justified = 0
        
        # Get page width (approximate from text blocks)
        max_right = max(block["bbox"][2] for block in text_blocks)
        min_left = min(block["bbox"][0] for block in text_blocks)
        page_width = max_right - min_left
        
        for block in text_blocks:
            block_left = block["bbox"][0] - min_left  # Normalize to 0
            block_right = block["bbox"][2] - min_left
            block_width = block_right - block_left
            block_center = (block_left + block_right) / 2
            page_center = page_width / 2
            
            # Determine alignment
            if abs(block_center - page_center) / page_width < CENTER_ALIGNMENT_TOLERANCE:
                center_aligned += 1
            elif block_right / page_width > 0.8:  # Near right edge
                right_aligned += 1
            elif block_width / page_width > 0.8:  # Nearly full width
                justified += 1
            else:
                left_aligned += 1
        
        total_blocks = len(text_blocks)
        alignments = {
            "left": left_aligned / total_blocks,
            "center": center_aligned / total_blocks,
            "right": right_aligned / total_blocks,
            "justified": justified / total_blocks
        }
        
        dominant_alignment = max(alignments, key=alignments.get)
        alignment_consistency = max(alignments.values())
        
        return {
            "dominant_alignment": dominant_alignment,
            "alignment_distribution": alignments,
            "alignment_consistency": alignment_consistency
        }
    
    def detect_reading_order(self, text_blocks: List[Dict[str, Any]], 
                           columns: ColumnInfo) -> List[Dict[str, Any]]:
        """Determine the reading order of text blocks."""
        
        if not text_blocks:
            return []
        
        # Sort blocks for reading order
        if columns.column_count == 1:
            # Single column: sort by vertical position
            ordered_blocks = sorted(text_blocks, key=lambda x: x["bbox"][1])
        else:
            # Multi-column: sort by column, then by vertical position within column
            ordered_blocks = []
            
            for col_left, col_right in columns.column_boundaries:
                # Find blocks in this column
                column_blocks = []
                for block in text_blocks:
                    block_center = (block["bbox"][0] + block["bbox"][2]) / 2
                    if col_left <= block_center <= col_right:
                        column_blocks.append(block)
                
                # Sort by vertical position within column
                column_blocks.sort(key=lambda x: x["bbox"][1])
                ordered_blocks.extend(column_blocks)
        
        # Add reading order index
        for i, block in enumerate(ordered_blocks):
            block["reading_order"] = i
        
        return ordered_blocks
    
    def calculate_whitespace_distribution(self, text_blocks: List[Dict[str, Any]], 
                                        page_rect: fitz.Rect) -> Dict[str, float]:
        """Calculate distribution of whitespace on the page."""
        
        if not text_blocks:
            return {"whitespace_ratio": 1.0, "text_density": 0.0}
        
        # Calculate total text area
        total_text_area = 0
        for block in text_blocks:
            bbox = block["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_text_area += area
        
        # Calculate page area
        page_area = page_rect.width * page_rect.height
        
        # Calculate ratios
        text_ratio = total_text_area / page_area
        whitespace_ratio = 1.0 - text_ratio
        
        # Calculate text density (blocks per unit area)
        text_density = len(text_blocks) / page_area * 10000  # Per 100x100 unit area
        
        return {
            "whitespace_ratio": whitespace_ratio,
            "text_ratio": text_ratio,
            "text_density": text_density,
            "total_text_area": total_text_area,
            "page_area": page_area
        }
    
    def get_layout_summary(self, layout: LayoutStructure) -> Dict[str, Any]:
        """Generate a human-readable summary of the layout analysis."""
        
        summary = {
            "layout_type": layout.layout_type,
            "columns": layout.columns.column_count,
            "has_headers": layout.has_headers,
            "has_footers": layout.has_footers,
            "consistency_score": round(layout.consistency_score, 3),
            "margins": {
                "top": layout.page_margins[0],
                "right": layout.page_margins[1],
                "bottom": layout.page_margins[2],
                "left": layout.page_margins[3]
            }
        }
        
        if layout.columns.column_count > 1:
            summary["column_info"] = {
                "boundaries": layout.columns.column_boundaries,
                "gutter_width": round(layout.columns.gutter_width, 1),
                "is_consistent": layout.columns.is_consistent
            }
        
        return summary