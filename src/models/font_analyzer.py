import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path

from config.settings import (
    FONT_SIZE_THRESHOLD_RATIO, BOLD_WEIGHT_THRESHOLD,
    MIN_HEADING_LENGTH, MAX_HEADING_LENGTH
)


@dataclass
class FontInfo:
    """Detailed font information."""
    family: str
    size: float
    weight: str
    style: str
    flags: int
    is_bold: bool
    is_italic: bool
    is_serif: bool
    is_monospace: bool
    is_sans_serif: bool


@dataclass
class FontStatistics:
    """Statistical analysis of document fonts."""
    total_fonts: int
    unique_families: Set[str]
    size_distribution: Dict[float, int]
    weight_distribution: Dict[str, int]
    most_common_size: float
    most_common_family: str
    size_range: Tuple[float, float]
    avg_size: float
    median_size: float
    heading_threshold_size: float
    body_text_size: float


class FontAnalyzer:
    """Advanced font analysis for PDF heading detection."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Font classification patterns
        self.serif_patterns = [
            "times", "serif", "georgia", "garamond", "palatino", 
            "bookman", "century", "minion"
        ]
        
        self.sans_serif_patterns = [
            "arial", "helvetica", "calibri", "verdana", "tahoma",
            "trebuchet", "segoe", "avenir", "futura", "optima"
        ]
        
        self.monospace_patterns = [
            "courier", "monaco", "consolas", "menlo", "inconsolata",
            "source code", "fira code", "jetbrains mono"
        ]
        
        # Font weight mappings (based on CSS standards)
        self.weight_mappings = {
            100: "thin",
            200: "extra-light",
            300: "light",
            400: "normal",
            500: "medium",
            600: "semi-bold",
            700: "bold",
            800: "extra-bold",
            900: "black"
        }
        
        # PyMuPDF font flags
        self.font_flags = {
            'superscript': 2**0,
            'italic': 2**1,
            'serifed': 2**2,
            'monospaced': 2**3,
            'bold': 2**4
        }
    
    def analyze_document_fonts(self, pdf_path: str) -> FontStatistics:
        """Analyze all fonts in the document and return statistics."""
        self.logger.info(f"Analyzing fonts in: {pdf_path}")
        
        try:
            with fitz.open(pdf_path) as doc:
                font_data = self._extract_all_font_data(doc)
                stats = self._calculate_font_statistics(font_data)
                
                if self.debug:
                    self._log_font_analysis(stats)
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Font analysis failed: {e}")
            return self._create_default_statistics()
    
    def _extract_all_font_data(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract font data from all pages."""
        font_data = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():  # Only non-empty text
                            font_info = self._extract_font_info(span)
                            font_data.append({
                                "page": page_num + 1,
                                "text": span["text"],
                                "bbox": span["bbox"],
                                "font_info": font_info,
                                "char_count": len(span["text"])
                            })
        
        return font_data
    
    def _extract_font_info(self, span: Dict[str, Any]) -> FontInfo:
        """Extract detailed font information from a span."""
        
        # Basic font properties
        font_family = span.get("font", "unknown").lower()
        font_size = span.get("size", 12.0)
        flags = span.get("flags", 0)
        
        # Determine font characteristics
        is_bold = bool(flags & self.font_flags['bold'])
        is_italic = bool(flags & self.font_flags['italic'])
        is_serif = bool(flags & self.font_flags['serifed']) or self._is_serif_font(font_family)
        is_monospace = bool(flags & self.font_flags['monospaced']) or self._is_monospace_font(font_family)
        is_sans_serif = not is_serif and not is_monospace
        
        # Determine font weight
        weight = self._determine_font_weight(font_family, flags, is_bold)
        
        # Determine font style
        style = "italic" if is_italic else "normal"
        
        return FontInfo(
            family=font_family,
            size=font_size,
            weight=weight,
            style=style,
            flags=flags,
            is_bold=is_bold,
            is_italic=is_italic,
            is_serif=is_serif,
            is_monospace=is_monospace,
            is_sans_serif=is_sans_serif
        )
    
    def _is_serif_font(self, font_family: str) -> bool:
        """Determine if font is serif based on name."""
        return any(pattern in font_family for pattern in self.serif_patterns)
    
    def _is_monospace_font(self, font_family: str) -> bool:
        """Determine if font is monospace based on name."""
        return any(pattern in font_family for pattern in self.monospace_patterns)
    
    def _determine_font_weight(self, font_family: str, flags: int, is_bold: bool) -> str:
        """Determine font weight from various indicators."""
        
        # Check font name for weight indicators
        font_lower = font_family.lower()
        
        if "black" in font_lower or "heavy" in font_lower:
            return "black"
        elif "extrabold" in font_lower or "ultra" in font_lower:
            return "extra-bold"
        elif "bold" in font_lower or is_bold:
            return "bold"
        elif "semibold" in font_lower or "demi" in font_lower:
            return "semi-bold"
        elif "medium" in font_lower:
            return "medium"
        elif "light" in font_lower:
            return "light"
        elif "thin" in font_lower or "hairline" in font_lower:
            return "thin"
        else:
            return "normal"
    
    def _calculate_font_statistics(self, font_data: List[Dict[str, Any]]) -> FontStatistics:
        """Calculate comprehensive font statistics."""
        
        if not font_data:
            return self._create_default_statistics()
        
        # Extract font properties
        sizes = []
        families = []
        weights = []
        
        for item in font_data:
            font_info = item["font_info"]
            char_count = item["char_count"]
            
            # Weight by character count for more accurate statistics
            sizes.extend([font_info.size] * char_count)
            families.extend([font_info.family] * char_count)
            weights.extend([font_info.weight] * char_count)
        
        # Size statistics
        sizes_array = np.array(sizes)
        unique_sizes = list(set(sizes))
        size_distribution = Counter(sizes)
        
        # Family and weight statistics
        unique_families = set(families)
        family_counter = Counter(families)
        weight_distribution = Counter(weights)
        
        # Calculate derived statistics
        avg_size = float(np.mean(sizes_array))
        median_size = float(np.median(sizes_array))
        most_common_size = size_distribution.most_common(1)[0][0]
        most_common_family = family_counter.most_common(1)[0][0]
        
        # Calculate heading thresholds
        # Use 75th percentile as body text threshold
        body_text_size = float(np.percentile(sizes_array, 75))
        heading_threshold_size = body_text_size * FONT_SIZE_THRESHOLD_RATIO
        
        return FontStatistics(
            total_fonts=len(font_data),
            unique_families=unique_families,
            size_distribution=dict(size_distribution),
            weight_distribution=dict(weight_distribution),
            most_common_size=most_common_size,
            most_common_family=most_common_family,
            size_range=(float(min(sizes)), float(max(sizes))),
            avg_size=avg_size,
            median_size=median_size,
            heading_threshold_size=heading_threshold_size,
            body_text_size=body_text_size
        )
    
    def _create_default_statistics(self) -> FontStatistics:
        """Create default statistics when analysis fails."""
        return FontStatistics(
            total_fonts=0,
            unique_families=set(),
            size_distribution={},
            weight_distribution={},
            most_common_size=12.0,
            most_common_family="unknown",
            size_range=(12.0, 12.0),
            avg_size=12.0,
            median_size=12.0,
            heading_threshold_size=14.4,
            body_text_size=12.0
        )
    
    def classify_heading_likelihood(self, font_info: FontInfo, 
                                  document_stats: FontStatistics) -> Dict[str, Any]:
        """Classify likelihood that text with given font is a heading."""
        
        score = 0.0
        reasons = []
        
        # Size-based scoring (0-40 points)
        size_ratio = font_info.size / document_stats.avg_size
        if size_ratio >= 1.5:
            score += 40
            reasons.append(f"Large font size ({font_info.size:.1f} vs avg {document_stats.avg_size:.1f})")
        elif size_ratio >= 1.2:
            score += 25
            reasons.append(f"Above average font size ({font_info.size:.1f})")
        elif size_ratio >= 1.0:
            score += 10
            reasons.append("Average font size")
        
        # Weight-based scoring (0-30 points)
        if font_info.weight in ["bold", "extra-bold", "black"]:
            score += 30
            reasons.append(f"Bold weight ({font_info.weight})")
        elif font_info.weight in ["semi-bold", "medium"]:
            score += 15
            reasons.append(f"Medium weight ({font_info.weight})")
        
        # Family-based scoring (0-15 points)
        if font_info.family != document_stats.most_common_family:
            score += 10
            reasons.append("Different font family")
        
        if font_info.is_sans_serif and document_stats.most_common_family in self.serif_patterns:
            score += 5
            reasons.append("Sans-serif in serif document")
        
        # Style-based scoring (0-10 points)
        if font_info.is_italic and font_info.is_bold:
            score += 5
            reasons.append("Bold italic style")
        
        # Rarity scoring (0-5 points)
        if font_info.size not in document_stats.size_distribution:
            score += 5
            reasons.append("Unique font size")
        
        # Normalize score to 0-1
        normalized_score = min(score / 100.0, 1.0)
        
        # Classify confidence level
        if normalized_score >= 0.8:
            confidence = "very_high"
        elif normalized_score >= 0.6:
            confidence = "high"
        elif normalized_score >= 0.4:
            confidence = "medium"
        elif normalized_score >= 0.2:
            confidence = "low"
        else:
            confidence = "very_low"
        
        return {
            "heading_score": normalized_score,
            "confidence": confidence,
            "reasons": reasons,
            "font_analysis": {
                "size_ratio": size_ratio,
                "is_larger_than_average": font_info.size > document_stats.avg_size,
                "is_bold": font_info.is_bold,
                "is_different_family": font_info.family != document_stats.most_common_family,
                "font_rarity": self._calculate_font_rarity(font_info, document_stats)
            }
        }
    
    def _calculate_font_rarity(self, font_info: FontInfo, 
                              document_stats: FontStatistics) -> float:
        """Calculate how rare this font combination is in the document."""
        
        # Size rarity
        size_frequency = document_stats.size_distribution.get(font_info.size, 0)
        total_chars = sum(document_stats.size_distribution.values())
        size_rarity = 1.0 - (size_frequency / total_chars) if total_chars > 0 else 1.0
        
        # Weight rarity
        weight_frequency = document_stats.weight_distribution.get(font_info.weight, 0)
        weight_rarity = 1.0 - (weight_frequency / total_chars) if total_chars > 0 else 1.0
        
        # Combined rarity (weighted average)
        combined_rarity = (size_rarity * 0.7) + (weight_rarity * 0.3)
        
        return combined_rarity
    
    def detect_font_patterns(self, font_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in font usage throughout the document."""
        
        patterns = {
            "title_fonts": [],
            "heading_fonts": [],
            "body_fonts": [],
            "font_transitions": [],
            "consistent_hierarchy": False,
            "mixed_font_families": False
        }
        
        if not font_data:
            return patterns
        
        # Group by page to analyze patterns
        pages = defaultdict(list)
        for item in font_data:
            pages[item["page"]].append(item)
        
        # Analyze font patterns across pages
        all_sizes = []
        all_families = []
        font_transitions = []
        
        for page_num, page_items in pages.items():
            page_sizes = [item["font_info"].size for item in page_items]
            page_families = [item["font_info"].family for item in page_items]
            
            all_sizes.extend(page_sizes)
            all_families.extend(page_families)
            
            # Detect font transitions within page
            prev_font = None
            for item in page_items:
                current_font = (item["font_info"].family, item["font_info"].size, item["font_info"].weight)
                if prev_font and prev_font != current_font:
                    font_transitions.append({
                        "page": page_num,
                        "from": prev_font,
                        "to": current_font,
                        "text": item["text"][:50]  # First 50 chars
                    })
                prev_font = current_font
        
        # Calculate font size percentiles for classification
        if all_sizes:
            size_array = np.array(all_sizes)
            p90 = np.percentile(size_array, 90)  # Title level
            p75 = np.percentile(size_array, 75)  # Heading level
            p50 = np.percentile(size_array, 50)  # Body level
            
            patterns["title_fonts"] = [
                item for item in font_data 
                if item["font_info"].size >= p90
            ]
            
            patterns["heading_fonts"] = [
                item for item in font_data 
                if p75 <= item["font_info"].size < p90
            ]
            
            patterns["body_fonts"] = [
                item for item in font_data 
                if item["font_info"].size < p75
            ]
        
        patterns["font_transitions"] = font_transitions
        patterns["mixed_font_families"] = len(set(all_families)) > 2
        patterns["consistent_hierarchy"] = self._check_hierarchy_consistency(font_data)
        
        return patterns
    
    def _check_hierarchy_consistency(self, font_data: List[Dict[str, Any]]) -> bool:
        """Check if font hierarchy is consistent throughout document."""
        
        # Group fonts by size and check if larger fonts consistently come before smaller ones
        size_positions = defaultdict(list)
        
        for i, item in enumerate(font_data):
            size = item["font_info"].size
            size_positions[size].append(i)
        
        # Sort sizes in descending order
        sorted_sizes = sorted(size_positions.keys(), reverse=True)
        
        # Check if positions are generally increasing for decreasing font sizes
        consistency_score = 0
        total_comparisons = 0
        
        for i in range(len(sorted_sizes) - 1):
            larger_size = sorted_sizes[i]
            smaller_size = sorted_sizes[i + 1]
            
            larger_positions = size_positions[larger_size]
            smaller_positions = size_positions[smaller_size]
            
            # Check if most larger fonts appear before smaller fonts
            correct_order = 0
            total_pairs = len(larger_positions) * len(smaller_positions)
            
            for large_pos in larger_positions:
                for small_pos in smaller_positions:
                    if large_pos < small_pos:
                        correct_order += 1
            
            if total_pairs > 0:
                consistency_score += correct_order / total_pairs
                total_comparisons += 1
        
        # Return True if more than 70% of comparisons show consistent hierarchy
        return (consistency_score / total_comparisons) > 0.7 if total_comparisons > 0 else False
    
    def analyze_heading_candidates(self, candidates: List, 
                                  document_stats: FontStatistics) -> List:
        """Analyze font characteristics of heading candidates."""
        
        analyzed_candidates = []
        
        for candidate in candidates:
            # Create FontInfo from candidate
            font_info = FontInfo(
                family=candidate.font_family,
                size=candidate.font_size,
                weight=candidate.font_weight,
                style="italic" if candidate.is_italic else "normal",
                flags=0,  # Not available in candidate
                is_bold=candidate.is_bold,
                is_italic=candidate.is_italic,
                is_serif=self._is_serif_font(candidate.font_family),
                is_monospace=self._is_monospace_font(candidate.font_family),
                is_sans_serif=not self._is_serif_font(candidate.font_family)
            )
            
            # Analyze heading likelihood
            font_analysis = self.classify_heading_likelihood(font_info, document_stats)
            
            # Update candidate with font analysis
            candidate.features.update({
                "font_analysis": font_analysis,
                "font_heading_score": font_analysis["heading_score"],
                "font_confidence": font_analysis["confidence"]
            })
            
            analyzed_candidates.append(candidate)
        
        return analyzed_candidates
    
    def get_font_recommendations(self, document_stats: FontStatistics) -> Dict[str, Any]:
        """Get recommendations for font-based heading detection."""
        
        recommendations = {
            "suggested_heading_threshold": document_stats.heading_threshold_size,
            "body_text_size": document_stats.body_text_size,
            "font_diversity": len(document_stats.unique_families),
            "recommended_strategy": "mixed",
            "confidence_adjustments": {}
        }
        
        # Determine recommended strategy based on font diversity
        if len(document_stats.unique_families) == 1:
            recommendations["recommended_strategy"] = "size_based"
            recommendations["confidence_adjustments"]["size_weight"] = 0.8
        elif len(document_stats.unique_families) > 5:
            recommendations["recommended_strategy"] = "conservative"
            recommendations["confidence_adjustments"]["family_weight"] = 0.3
        else:
            recommendations["recommended_strategy"] = "balanced"
            recommendations["confidence_adjustments"]["size_weight"] = 0.6
            recommendations["confidence_adjustments"]["family_weight"] = 0.4
        
        # Adjust thresholds based on size distribution
        size_variance = np.var(list(document_stats.size_distribution.keys()))
        if size_variance < 2.0:  # Low variance - similar sizes
            recommendations["confidence_adjustments"]["require_bold"] = True
        
        return recommendations
    
    def compare_fonts(self, font1: FontInfo, font2: FontInfo) -> Dict[str, Any]:
        """Compare two fonts and determine their relationship."""
        
        comparison = {
            "size_difference": font2.size - font1.size,
            "size_ratio": font2.size / font1.size if font1.size > 0 else 1.0,
            "same_family": font1.family == font2.family,
            "weight_difference": self._compare_weights(font1.weight, font2.weight),
            "style_difference": font1.style != font2.style,
            "hierarchy_relationship": "equal"
        }
        
        # Determine hierarchy relationship
        if comparison["size_ratio"] >= 1.2:
            comparison["hierarchy_relationship"] = "font2_higher"
        elif comparison["size_ratio"] <= 0.8:
            comparison["hierarchy_relationship"] = "font1_higher"
        elif font1.is_bold and not font2.is_bold:
            comparison["hierarchy_relationship"] = "font1_higher"
        elif font2.is_bold and not font1.is_bold:
            comparison["hierarchy_relationship"] = "font2_higher"
        
        return comparison
    
    def _compare_weights(self, weight1: str, weight2: str) -> int:
        """Compare font weights numerically."""
        weight_order = [
            "thin", "extra-light", "light", "normal", "medium",
            "semi-bold", "bold", "extra-bold", "black"
        ]
        
        try:
            idx1 = weight_order.index(weight1)
            idx2 = weight_order.index(weight2)
            return idx2 - idx1  # Positive if weight2 is heavier
        except ValueError:
            return 0  # Unknown weights
    
    def extract_font_hierarchy(self, font_data: List[Dict[str, Any]]) -> Dict[int, List[FontInfo]]:
        """Extract font hierarchy levels from document."""
        
        if not font_data:
            return {}
        
        # Extract all unique font combinations
        unique_fonts = {}
        for item in font_data:
            font_info = item["font_info"]
            font_key = (font_info.size, font_info.weight, font_info.family)
            
            if font_key not in unique_fonts:
                unique_fonts[font_key] = {
                    "font_info": font_info,
                    "frequency": 0,
                    "positions": []
                }
            
            unique_fonts[font_key]["frequency"] += len(item["text"])
            unique_fonts[font_key]["positions"].append(item.get("position", 0))
        
        # Sort fonts by size (descending) and frequency
        sorted_fonts = sorted(
            unique_fonts.values(),
            key=lambda x: (x["font_info"].size, x["frequency"]),
            reverse=True
        )
        
        # Assign hierarchy levels
        hierarchy = {}
        current_level = 0
        prev_size = None
        
        for font_data in sorted_fonts:
            font_info = font_data["font_info"]
            
            # Start new level if significant size difference
            if prev_size is not None and prev_size - font_info.size > 2.0:
                current_level += 1
            
            if current_level not in hierarchy:
                hierarchy[current_level] = []
            
            hierarchy[current_level].append(font_info)
            prev_size = font_info.size
        
        return hierarchy
    
    def _log_font_analysis(self, stats: FontStatistics) -> None:
        """Log detailed font analysis for debugging."""
        
        self.logger.debug("=== Font Analysis Results ===")
        self.logger.debug(f"Total fonts analyzed: {stats.total_fonts}")
        self.logger.debug(f"Unique font families: {len(stats.unique_families)}")
        self.logger.debug(f"Font families: {', '.join(list(stats.unique_families)[:5])}")
        self.logger.debug(f"Size range: {stats.size_range[0]:.1f} - {stats.size_range[1]:.1f}")
        self.logger.debug(f"Average size: {stats.avg_size:.1f}")
        self.logger.debug(f"Most common size: {stats.most_common_size:.1f}")
        self.logger.debug(f"Heading threshold: {stats.heading_threshold_size:.1f}")
        self.logger.debug(f"Body text size: {stats.body_text_size:.1f}")
        
        # Log size distribution (top 5)
        sorted_sizes = sorted(stats.size_distribution.items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        self.logger.debug("Top font sizes:")
        for size, count in sorted_sizes:
            self.logger.debug(f"  {size:.1f}pt: {count} occurrences")
        
        # Log weight distribution
        self.logger.debug("Font weights:")
        for weight, count in stats.weight_distribution.items():
            self.logger.debug(f"  {weight}: {count} occurrences")