import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from src.core.candidate_generator import HeadingCandidate
import numpy as np
from config.settings import MAX_HIERARCHY_LEVELS, TITLE_POSITION_THRESHOLD
from config.cultural_patterns import CULTURAL_PATTERNS


@dataclass
class HierarchyNode:
    """Represents a node in the heading hierarchy."""
    text: str
    level: int
    page: int
    bbox: Tuple[float, float, float, float]
    font_size: float
    confidence: float
    parent: Optional['HierarchyNode'] = None
    children: List['HierarchyNode'] = field(default_factory=list)
    numbering_pattern: Optional[str] = None
    semantic_group: Optional[str] = None


class HierarchyAssigner:
    """Assigns hierarchy levels to heading candidates using multiple strategies."""
    
    def __init__(self, language: str = 'auto', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.cultural_patterns = CULTURAL_PATTERNS
        
        # Hierarchy detection strategies
        self.strategies = [
            self._assign_by_cjk_patterns,  # New CJK-specific strategy first
            self._assign_by_font_hierarchy,
            self._assign_by_numbering_pattern,
            self._assign_by_position_and_spacing,
            self._assign_by_keywords,
            self._assign_by_indentation,
        ]
    
    def assign_hierarchy(self, candidates: List[HeadingCandidate]) -> List[Dict[str, Any]]:
        """Assign hierarchy levels to heading candidates."""
        self.logger.info(f"Assigning hierarchy to {len(candidates)} candidates")
        
        if not candidates:
            return []
        
        # Detect language from candidates if not set
        if self.language == 'auto' and candidates:
            self.language = self._detect_language_from_candidates(candidates)
        
        # Convert candidates to hierarchy nodes
        nodes = [self._candidate_to_node(candidate) for candidate in candidates]
        
        # Apply multiple strategies and combine results
        strategy_results = []
        for strategy in self.strategies:
            try:
                # Skip CJK strategy for non-CJK languages
                if (strategy.__name__ == '_assign_by_cjk_patterns' and 
                    self.language not in ['japanese', 'chinese']):
                    continue
                    
                result = strategy(nodes.copy())
                strategy_results.append(result)
                self.logger.debug(f"Strategy {strategy.__name__} completed")
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.__name__} failed: {e}")
        
        # Combine strategies using ensemble approach
        final_hierarchy = self._combine_strategies(strategy_results, nodes)
        
        # Post-process and validate hierarchy
        validated_hierarchy = self._validate_and_fix_hierarchy(final_hierarchy)
        
        # Convert back to output format
        output = self._nodes_to_output(validated_hierarchy)
        
        self.logger.info(f"Final hierarchy: {len(output)} headings across {max([h['level'] for h in output], default=0)} levels")
        return output
    
    def _detect_language_from_candidates(self, candidates: List[HeadingCandidate]) -> str:
        """Detect language from candidate text."""
        sample_text = " ".join([c.text for c in candidates[:5]])  # Sample first 5
        
        # Simple detection based on character sets
        if re.search(r'[\u4e00-\u9fff]', sample_text):
            if re.search(r'[\u3041-\u3096\u30A1-\u30FA]', sample_text):
                return 'japanese'
            else:
                return 'chinese'
        elif re.search(r'[\u0900-\u097F]', sample_text):
            return 'hindi'
        elif re.search(r'[\u0600-\u06FF]', sample_text):
            return 'arabic'
        else:
            return 'english'
    
    def _candidate_to_node(self, candidate) -> HierarchyNode:
        """Convert heading candidate to hierarchy node."""
        return HierarchyNode(
            text=candidate.text,
            level=1,  # Default level, will be updated
            page=candidate.page,
            bbox=candidate.bbox,
            font_size=candidate.font_size,
            confidence=candidate.confidence_score,
            numbering_pattern=candidate.features.get('numbering_type'),
        )
    
    def _assign_by_cjk_patterns(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on CJK-specific patterns with enhanced detection."""
        if self.language not in ['japanese', 'chinese']:
            return nodes
        
        for node in nodes:
            text = node.text.strip()
            level = self._detect_heading_level_cjk(text, node.font_size, nodes)
            node.level = level
        
        return nodes
    
    def _detect_heading_level_cjk(self, text: str, font_size: float, all_nodes: List[HierarchyNode]) -> int:
        """Detect heading level for CJK text with comprehensive pattern matching."""
        
        # Calculate average font size for reference
        avg_font_size = np.mean([n.font_size for n in all_nodes]) if all_nodes else font_size
        
        # Level 1: Chapter markers (highest priority)
        chapter_patterns = [
            r'^第[一二三四五六七八九十\d]+章',     # 第一章, 第1章
            r'^Chapter\s*[一二三四五六七八九十\d]+', # Chapter 1 (mixed)
            r'^序章',                            # Prologue chapter
            r'^終章',                            # Final chapter
            r'^付録[一二三四五六七八九十\d]*',      # Appendix
        ]
        
        for pattern in chapter_patterns:
            if re.search(pattern, text):
                return 1
        
        # Level 1: Major section markers
        major_section_patterns = [
            r'^第[一二三四五六七八九十\d]+部',     # 第一部 (Part)
            r'^第[一二三四五六七八九十\d]+編',     # 第一編 (Volume)
            r'^はじめに$',                       # Introduction (Japanese)
            r'^序論$',                          # Introduction (Japanese)
            r'^結論$',                          # Conclusion (Japanese)
            r'^まとめ$',                        # Summary (Japanese)
            r'^引言$',                          # Introduction (Chinese)
            r'^结论$',                          # Conclusion (Chinese)
            r'^总结$',                          # Summary (Chinese)
            r'^参考文献$',                       # References
            r'^謝辞$',                          # Acknowledgments (Japanese)
            r'^致谢$',                          # Acknowledgments (Chinese)
        ]
        
        for pattern in major_section_patterns:
            if re.search(pattern, text):
                return 1
        
        # Level 2: Section markers
        section_patterns = [
            r'^第[一二三四五六七八九十\d]+節',     # 第一節, 第1節
            r'^\d+\.\d+\s+[^\s]',              # 1.1 Title (with space and content)
            r'^[一二三四五六七八九十]+、',         # 一、二、三、
            r'^（[一二三四五六七八九十\d]+）',    # （一）（二）
            r'^\([一二三四五六七八九十\d]+\)',    # (一)(二)
        ]
        
        for pattern in section_patterns:
            if re.search(pattern, text):
                return 2
        
        # Level 3: Subsection markers
        subsection_patterns = [
            r'^\d+\.\d+\.\d+',                # 1.1.1
            r'^[一二三四五六七八九十]+\.',        # 一. 二. 三.
            r'^[abc一二三]\)',                 # a) b) c) or 一) 二)
            r'^[\(（][abc一二三\d]+[\)）]',      # (a) (b) or （一）（二）
        ]
        
        for pattern in subsection_patterns:
            if re.search(pattern, text):
                return 3
        
        # Level 4+: Minor subsections
        minor_patterns = [
            r'^\d+\.\d+\.\d+\.\d+',           # 1.1.1.1
            r'^・',                           # Bullet points (Japanese)
            r'^•',                            # Bullet points
        ]
        
        for pattern in minor_patterns:
            if re.search(pattern, text):
                return 4
        
        # Fallback: Use font size relative to document average
        if font_size > avg_font_size * 1.5:
            return 1
        elif font_size > avg_font_size * 1.2:
            return 2
        elif font_size > avg_font_size * 1.1:
            return 3
        else:
            return 4
    
    def _assign_by_font_hierarchy(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on font size distribution analysis."""
        if not nodes:
            return nodes
        
        # Analyze font size distribution
        font_sizes = [node.font_size for node in nodes]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        if len(unique_sizes) == 1:
            # All same size - use other factors
            return self._assign_by_non_font_factors(nodes)
        
        # Use natural breaks in font size distribution
        size_breaks = self._find_natural_size_breaks(unique_sizes)
        
        # Create level mapping based on breaks
        level_mapping = {}
        current_level = 1
        
        for i, size in enumerate(unique_sizes):
            if i in size_breaks:
                current_level += 1
            level_mapping[size] = min(current_level, 6)  # Max level 6
        
        # Apply mapping, but be more conservative for CJK languages
        for node in nodes:
            suggested_level = level_mapping.get(node.font_size, 3)
            
            # For CJK languages, prefer pattern-based levels if available
            if self.language in ['japanese', 'chinese']:
                pattern_level = self._detect_heading_level_cjk(node.text, node.font_size, nodes)
                # Use the more conservative (higher) level between pattern and font
                node.level = min(suggested_level, pattern_level)
            else:
                node.level = suggested_level
        
        return nodes

    def _find_natural_size_breaks(self, sorted_sizes: List[float]) -> Set[int]:
        """Find natural breaks in font size distribution."""
        if len(sorted_sizes) <= 2:
            return set()
        
        breaks = set()
        
        # Look for significant gaps between consecutive sizes
        for i in range(1, len(sorted_sizes)):
            size_diff = sorted_sizes[i-1] - sorted_sizes[i]
            
            # If difference is more than 2 points, it's likely a level break
            if size_diff >= 2.0:
                breaks.add(i)
            # If difference is more than 20% of the smaller size
            elif size_diff > sorted_sizes[i] * 0.2:
                breaks.add(i)
        
        return breaks

    def _assign_by_non_font_factors(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels when font sizes are similar."""
        
        # Use spacing and position as primary factors
        for node in nodes:
            level = 2  # Default
            
            # For CJK languages, prioritize pattern matching
            if self.language in ['japanese', 'chinese']:
                pattern_level = self._detect_heading_level_cjk(node.text, node.font_size, nodes)
                level = pattern_level
            else:
                # Top of page likely higher level
                if node.bbox[1] < 150:  # Top 150 points
                    level = 1
                
                # Large spacing suggests higher level
                spacing_before = getattr(node, 'spacing_before', 0)
                if spacing_before > 15:
                    level = max(1, level - 1)
                elif spacing_before > 8:
                    level = max(2, level)
                
                # Bold text suggests higher level
                if getattr(node, 'is_bold', False):
                    level = max(1, level - 1)
            
            node.level = level
        
        return nodes
    
    def _assign_by_numbering_pattern(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on numbering patterns with enhanced CJK support."""
        
        # Enhanced numbering hierarchy including CJK patterns
        numbering_hierarchy = {
            # English patterns
            'decimal': 1,                    # 1. 2. 3.
            'decimal_nested': 2,             # 1.1, 1.2
            'decimal_nested_deep': 3,        # 1.1.1
            'roman': 1,                      # I. II. III.
            'alpha': 2,                      # A. B. C.
            'chapter': 1,                    # Chapter 1
            'section': 2,                    # Section 1
            
            # CJK patterns
            'japanese_chapter': 1,           # 第1章
            'japanese_kanji_chapter': 1,     # 第一章
            'japanese_section': 2,           # 第1節
            'japanese_kanji_list': 2,        # 一、二、三、
            'chinese_chapter': 1,            # 第1章
            'chinese_kanji_chapter': 1,      # 第一章
        }
        
        for node in nodes:
            text = node.text.strip()
            detected_level = None
            
            # Enhanced CJK pattern detection
            if self.language in ['japanese', 'chinese']:
                if re.match(r'^第\d+章', text) or re.match(r'^第[一二三四五六七八九十]+章', text):
                    detected_level = 1
                    node.numbering_pattern = 'cjk_chapter'
                elif re.match(r'^第\d+節', text) or re.match(r'^第[一二三四五六七八九十]+節', text):
                    detected_level = 2
                    node.numbering_pattern = 'cjk_section'
                elif re.match(r'^\d+\.\d+\.\d+', text):  # 1.1.1
                    detected_level = 3
                    node.numbering_pattern = 'decimal_nested_deep'
                elif re.match(r'^\d+\.\d+', text):  # 1.1
                    detected_level = 2
                    node.numbering_pattern = 'decimal_nested'
                elif re.match(r'^\d+\.', text):  # 1.
                    detected_level = 1
                    node.numbering_pattern = 'decimal'
                elif re.match(r'^[一二三四五六七八九十]+、', text):  # 一、
                    detected_level = 2
                    node.numbering_pattern = 'cjk_kanji_list'
                elif re.match(r'^（[一二三四五六七八九十\d]+）', text):  # （一）
                    detected_level = 3
                    node.numbering_pattern = 'cjk_parenthetical'
            else:
                # Original English pattern detection
                if re.match(r'^\d+\.\d+\.\d+', text):  # 1.1.1
                    detected_level = 3
                    node.numbering_pattern = 'decimal_nested_deep'
                elif re.match(r'^\d+\.\d+', text):  # 1.1
                    detected_level = 2
                    node.numbering_pattern = 'decimal_nested'
                elif re.match(r'^\d+\.', text):  # 1.
                    detected_level = 1
                    node.numbering_pattern = 'decimal'
                elif re.match(r'^[IVX]+\.', text):  # I.
                    detected_level = 1
                    node.numbering_pattern = 'roman'
                elif re.match(r'^[A-Z]\.', text):  # A.
                    detected_level = 2
                    node.numbering_pattern = 'alpha'
                elif re.search(r'^Chapter \d+', text, re.IGNORECASE):
                    detected_level = 1
                    node.numbering_pattern = 'chapter'
                elif re.search(r'^Section \d+', text, re.IGNORECASE):
                    detected_level = 2
                    node.numbering_pattern = 'section'
            
            if detected_level:
                node.level = detected_level
        
        return nodes
    
    def _assign_by_position_and_spacing(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on position and spacing patterns."""
        
        # Sort nodes by page and position
        nodes_sorted = sorted(nodes, key=lambda x: (x.page, x.bbox[1]))
        
        for i, node in enumerate(nodes_sorted):
            # Check if it's at the very top of a page (likely title/chapter)
            if node.bbox[1] < 100:  # Top 100 points of page
                if i == 0 or nodes_sorted[i-1].page < node.page:
                    node.level = min(node.level, 1)
            
            # Analyze spacing context
            prev_node = nodes_sorted[i-1] if i > 0 else None
            next_node = nodes_sorted[i+1] if i < len(nodes_sorted)-1 else None
            
            # Large spacing before suggests higher level
            if prev_node and node.page == prev_node.page:
                spacing = node.bbox[1] - prev_node.bbox[3]
                if spacing > 30:  # Significant spacing
                    node.level = max(1, node.level - 1)
        
        return nodes
    
    def _assign_by_keywords(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on common heading keywords with enhanced CJK support."""
        
        # Enhanced keyword levels including CJK
        keyword_levels = {
            # Level 0 (Title)
            'title': 0, 'abstract': 0, 'summary': 0,
            
            # Level 1 (Major sections)
            'introduction': 1, 'background': 1, 'methodology': 1, 
            'results': 1, 'discussion': 1, 'conclusion': 1,
            'references': 1, 'bibliography': 1, 'appendix': 1,
            'chapter': 1,
            
            # Level 2 (Subsections)
            'section': 2, 'overview': 2, 'approach': 2,
            'analysis': 2, 'implementation': 2,
            
            # Level 3 (Sub-subsections)
            'subsection': 3, 'details': 3, 'examples': 3,
            
            # CJK keywords
            'はじめに': 1, '序論': 1, '結論': 1, 'まとめ': 1,  # Japanese
            '参考文献': 1, '付録': 1, '謝辞': 1,
            '引言': 1, '结论': 1, '总结': 1, '参考文献': 1,      # Chinese
            '附录': 1, '致谢': 1,
        }
        
        # Add cultural keywords
        if self.language in self.cultural_patterns:
            cultural_keywords = self.cultural_patterns[self.language].get('heading_keywords', [])
            for keyword in cultural_keywords:
                keyword_levels[keyword.lower()] = 1
        
        for node in nodes:
            text_lower = node.text.lower().strip()
            original_text = node.text.strip()  # Keep original for CJK
            
            # Check for exact matches first (including CJK)
            for keyword, level in keyword_levels.items():
                if (keyword in text_lower or 
                    (self.language in ['japanese', 'chinese'] and keyword in original_text)):
                    node.level = min(node.level, level)
                    node.semantic_group = keyword
                    break
        
        return nodes
    
    def _assign_by_indentation(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Assign levels based on indentation patterns."""
        
        # Group nodes by page to analyze indentation within pages
        pages = defaultdict(list)
        for node in nodes:
            pages[node.page].append(node)
        
        for page_nodes in pages.values():
            if len(page_nodes) < 2:
                continue
            
            # Sort by vertical position
            page_nodes.sort(key=lambda x: x.bbox[1])
            
            # Analyze indentation levels
            indentations = [node.bbox[0] for node in page_nodes]
            unique_indents = sorted(set(indentations))
            
            # Map indentation to hierarchy levels
            indent_to_level = {}
            for i, indent in enumerate(unique_indents):
                indent_to_level[indent] = min(i + 1, MAX_HIERARCHY_LEVELS)
            
            # Assign levels based on indentation
            for node in page_nodes:
                suggested_level = indent_to_level[node.bbox[0]]
                # Take minimum with existing level (most restrictive)
                node.level = min(node.level, suggested_level)
        
        return nodes
    
    def _combine_strategies(self, strategy_results: List[List[HierarchyNode]], 
                          original_nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Combine results from multiple strategies using ensemble approach with CJK weighting."""
        
        if not strategy_results:
            return original_nodes
        
        # Create mapping from text to level votes
        level_votes = defaultdict(list)
        strategy_weights = defaultdict(float)
        
        for i, strategy_result in enumerate(strategy_results):
            # Give higher weight to CJK pattern strategy for CJK languages
            weight = 2.0 if (i == 0 and self.language in ['japanese', 'chinese']) else 1.0
            
            for node in strategy_result:
                level_votes[node.text].append(node.level)
                strategy_weights[node.text] += weight
        
        # Assign final levels using weighted median voting
        final_nodes = []
        for original_node in original_nodes:
            votes = level_votes.get(original_node.text, [original_node.level])
            
            # Use weighted median of votes, with bias towards lower levels (higher importance)
            if self.language in ['japanese', 'chinese'] and len(votes) > 1:
                # For CJK, prefer the most confident (lowest) level
                final_level = min(votes)
            else:
                # Use median for other languages
                final_level = int(np.median(votes))
            
            # Create final node
            final_node = HierarchyNode(
                text=original_node.text,
                level=final_level,
                page=original_node.page,
                bbox=original_node.bbox,
                font_size=original_node.font_size,
                confidence=original_node.confidence,
                numbering_pattern=original_node.numbering_pattern,
                semantic_group=original_node.semantic_group,
            )
            
            final_nodes.append(final_node)
        
        return final_nodes
    
    def _validate_and_fix_hierarchy(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Validate and fix common hierarchy issues with CJK awareness."""
        
        if not nodes:
            return nodes
        
        # Sort nodes by page and position
        nodes.sort(key=lambda x: (x.page, x.bbox[1]))
        
        # Fix common issues
        fixed_nodes = []
        prev_level = 0
        
        for i, node in enumerate(nodes):
            current_level = node.level
            
            # For CJK languages, be more lenient with level jumps if clear patterns exist
            if self.language in ['japanese', 'chinese']:
                # Allow level jumps for clear chapter patterns
                if re.search(r'^第[一二三四五六七八九十\d]+章', node.text):
                    current_level = 1  # Force chapters to level 1
                elif re.search(r'^第[一二三四五六七八九十\d]+節', node.text):
                    current_level = min(current_level, 2)  # Force sections to level 2 or higher
                else:
                    # Ensure we don't skip levels (max jump of 1)
                    if current_level > prev_level + 1:
                        current_level = prev_level + 1
            else:
                # Original logic for non-CJK languages
                # Ensure we don't skip levels (max jump of 1)
                if current_level > prev_level + 1:
                    current_level = prev_level + 1
            
            # Ensure reasonable bounds
            current_level = max(1, min(current_level, MAX_HIERARCHY_LEVELS))
            
            # Special case: if first heading is not level 1, make it level 1
            if i == 0 and current_level > 1:
                current_level = 1
            
            # Update node
            node.level = current_level
            fixed_nodes.append(node)
            prev_level = current_level
        
        return fixed_nodes
    
    def _build_hierarchy_tree(self, nodes: List[HierarchyNode]) -> Dict[str, Any]:
        """Build a hierarchical tree structure."""
        if not nodes:
            return {}
        
        tree = {}
        stack = []  # Stack to track parent nodes at each level
        
        for node in nodes:
            # Adjust stack size to current level
            while len(stack) > node.level:
                stack.pop()
            
            # Add current node
            if not stack:
                # Root level
                tree[node.text] = {
                    "level": node.level,
                    "page": node.page,
                    "children": {}
                }
                stack.append((node.text, tree[node.text]))
            else:
                # Child node
                parent_name, parent_dict = stack[-1]
                parent_dict["children"][node.text] = {
                    "level": node.level,
                    "page": node.page,
                    "children": {}
                }
                stack.append((node.text, parent_dict["children"][node.text]))
        
        return tree
    
    def _nodes_to_output(self, nodes: List[HierarchyNode]) -> List[Dict[str, Any]]:
        """Convert hierarchy nodes to output format."""
        output = []
        
        for node in nodes:
            heading_dict = {
                "text": node.text,
                "level": node.level,
                "page": node.page,
                "bbox": list(node.bbox),
                "font_info": {
                    "size": node.font_size,
                    "weight": "bold" if getattr(node, "is_bold", False) else "normal",
                    "family": getattr(node, "font_family", "unknown"),
                },
                "confidence": round(node.confidence, 3),
                "features": {
                    "numbering_pattern": node.numbering_pattern,
                    "semantic_group": node.semantic_group,
                }
            }
            output.append(heading_dict)
        
        return output
    
    def generate_hierarchy_tree(self, nodes: List[HierarchyNode]) -> Dict[str, Any]:
        """Generate a hierarchical tree representation."""
        return self._build_hierarchy_tree(nodes)
    
    def get_hierarchy_statistics(self, nodes: List[HierarchyNode]) -> Dict[str, Any]:
        """Generate statistics about the detected hierarchy."""
        if not nodes:
            return {}
        
        level_counts = Counter(node.level for node in nodes)
        
        stats = {
            "total_headings": len(nodes),
            "max_level": max(node.level for node in nodes),
            "min_level": min(node.level for node in nodes),
            "level_distribution": dict(level_counts),
            "avg_confidence": np.mean([node.confidence for node in nodes]),
            "pages_with_headings": len(set(node.page for node in nodes)),
            "numbering_patterns_found": list(set(
                node.numbering_pattern for node in nodes 
                if node.numbering_pattern
            )),
            "semantic_groups_found": list(set(
                node.semantic_group for node in nodes 
                if node.semantic_group
            ))
        }
        
        return stats
    
    def _detect_document_structure(self, nodes: List[HierarchyNode]) -> str:
        """Detect the overall document structure type."""
        
        # Analyze patterns
        has_chapters = any("chapter" in node.text.lower() or "章" in node.text for node in nodes)
        has_sections = any("section" in node.text.lower() or "節" in node.text for node in nodes)
        has_numbering = any(node.numbering_pattern for node in nodes)
        level_distribution = Counter(node.level for node in nodes)
        
        # Classify document type
        if has_chapters:
            return "book_style"
        elif has_sections and has_numbering:
            return "academic_paper"
        elif len(level_distribution) <= 2:
            return "simple_document"
        elif max(level_distribution.keys()) >= 4:
            return "complex_hierarchical"
        else:
            return "standard_document"
    
    def optimize_for_document_type(self, nodes: List[HierarchyNode]) -> List[HierarchyNode]:
        """Optimize hierarchy based on detected document type."""
        
        doc_type = self._detect_document_structure(nodes)
        self.logger.debug(f"Detected document type: {doc_type}")
        
        if doc_type == "book_style":
            # Ensure chapters are level 1, sections are level 2
            for node in nodes:
                text_lower = node.text.lower()
                if "chapter" in text_lower or "章" in node.text:
                    node.level = 1
                elif "section" in text_lower or "節" in node.text:
                    node.level = 2
        
        elif doc_type == "academic_paper":
            # Standard academic structure
            academic_mapping = {
                "abstract": 0,
                "introduction": 1,
                "related work": 1,
                "methodology": 1,
                "results": 1,
                "discussion": 1,
                "conclusion": 1,
                "references": 1,
                # CJK academic terms
                "はじめに": 1, "序論": 1, "結論": 1, "まとめ": 1,
                "引言": 1, "结论": 1, "总结": 1,
            }
            
            for node in nodes:
                text_lower = node.text.lower()
                original_text = node.text  # For CJK matching
                for keyword, level in academic_mapping.items():
                    if keyword in text_lower or keyword in original_text:
                        node.level = level
                        break
        
        elif doc_type == "simple_document":
            # Flatten hierarchy for simple documents
            for node in nodes:
                if node.level > 2:
                    node.level = 2
        
        return nodes