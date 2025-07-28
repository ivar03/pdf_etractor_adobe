import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import fitz  # PyMuPDF
import numpy as np
from config.settings import (
    FONT_SIZE_THRESHOLD_RATIO, BOLD_WEIGHT_THRESHOLD,
    MIN_HEADING_LENGTH, MAX_HEADING_LENGTH,
    TITLE_POSITION_THRESHOLD, CENTER_ALIGNMENT_TOLERANCE
)
from config.cultural_patterns import CULTURAL_PATTERNS, HEADING_CONFIDENCE_BOOSTERS
from src.utils.text_utils import (
    tokenize_multilingual, 
    enhance_heading_detection_for_cjk,
    is_likely_heading,
    clean_text,
    detect_language
)


@dataclass
class HeadingCandidate:
    """Represents a potential heading extracted from PDF."""
    text: str
    page: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float
    font_weight: str
    font_family: str
    is_bold: bool
    is_italic: bool
    alignment: str  # 'left', 'center', 'right'
    position_ratio: float  # 0-1, position on page (0=top)
    line_spacing_before: float
    line_spacing_after: float
    text_length: int
    confidence_score: float = 0.0
    features: Dict[str, Any] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}


class CandidateGenerator:
    """Fast candidate generation using font and layout heuristics with advanced multilingual support."""
    
    def __init__(self, language: str = 'auto', debug: bool = False):
        self.language = language
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.cultural_patterns = CULTURAL_PATTERNS
        self.document_stats = {}
        self.detected_language = None
        
    def generate_candidates(self, pdf_path: str) -> List[HeadingCandidate]:
        """Generate heading candidates from PDF using fast heuristics with multilingual support."""
        self.logger.info(f"Generating candidates for: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        all_candidates = []
        
        try:
            self._analyze_document_stats(doc)
            page_count = len(doc)
            
            start_page = 1 if page_count > 1 else 0

            for page_num in range(start_page, page_count):
                page = doc.load_page(page_num)
                page_candidates = self._extract_page_candidates(page, page_num)
                all_candidates.extend(page_candidates)
                
        finally:
            doc.close()
            
        running_elements = self._identify_running_elements(all_candidates)
        candidates = [cand for cand in all_candidates if cand.text.strip() not in running_elements]
            
        # Filter and score candidates
        filtered_candidates = self._filter_candidates(candidates)
        scored_candidates = self._score_candidates(filtered_candidates)
        
        self.logger.info(f"Generated {len(scored_candidates)} candidates for language: {self.detected_language}")
        return scored_candidates
    
    def _analyze_document_stats(self, doc: fitz.Document) -> None:
        """Analyze document to understand typical font characteristics and detect language."""
        font_sizes = []
        font_families = set()
        text_blocks = []
        sample_text = ""
        
        for page_num in range(min(3, len(doc))):  # Sample first 3 pages
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
                            font_families.add(span["font"])
                            text_blocks.append({
                                "text": span["text"],
                                "size": span["size"],
                                "flags": span["flags"]
                            })
                            # Collect text for language detection
                            sample_text += span["text"] + " "
        
        # Detect language if set to auto
        if self.language == 'auto':
            self.detected_language = detect_language(sample_text[:1000])
            self.logger.info(f"Detected language: {self.detected_language}")
        else:
            self.detected_language = self.language
        
        # Calculate statistics
        self.document_stats = {
            "avg_font_size": np.mean(font_sizes) if font_sizes else 12,
            "median_font_size": np.median(font_sizes) if font_sizes else 12,
            "max_font_size": max(font_sizes) if font_sizes else 12,
            "min_font_size": min(font_sizes) if font_sizes else 12,
            "font_families": font_families,
            "body_text_threshold": np.percentile(font_sizes, 75) if font_sizes else 12,
            "detected_language": self.detected_language,
            "sample_text": sample_text[:500]  # Keep sample for further analysis
        }
        
        self.logger.debug(f"Document stats: {self.document_stats}")
    
    def _extract_page_candidates(self, page: fitz.Page, page_num: int) -> List[HeadingCandidate]:
        """Extract heading candidates from a single page with multilingual awareness."""
        candidates = []
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height
        
        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue
                
            # Process each line in the block
            for line_idx, line in enumerate(block["lines"]):
                line_text = ""
                line_bbox = line["bbox"]
                spans = line["spans"]
                
                if not spans:
                    continue
                
                # Combine spans in the line
                dominant_span = max(spans, key=lambda s: (s["size"], len(s["text"])))
                line_text = " ".join([span["text"].strip() for span in spans])
                
                if not self._is_potential_heading_text(line_text):
                    continue
                
                # Calculate features with language awareness
                features = self._extract_line_features(
                    line, line_text, line_bbox, page.rect.height, page.rect.width, 
                    block_idx, line_idx, blocks
                )

                # Create candidate
                candidate = HeadingCandidate(
                    text=line_text.strip(),
                    page=page_num + 1,
                    bbox=line_bbox,
                    font_size=dominant_span["size"],
                    font_weight=self._get_font_weight(dominant_span["flags"]),
                    font_family=dominant_span["font"],
                    is_bold=bool(dominant_span["flags"] & 2**4),
                    is_italic=bool(dominant_span["flags"] & 2**1),
                    alignment=self._determine_alignment(line_bbox, page.rect.width),
                    position_ratio=line_bbox[1] / page_height,
                    line_spacing_before=features["spacing_before"],
                    line_spacing_after=features["spacing_after"],
                    text_length=len(line_text.strip()),
                    features=features
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _extract_line_features(self, line: Dict, text: str, bbox: Tuple,
                           page_height: float, page_width: float,
                           block_idx: int, line_idx: int,
                           blocks: List) -> Dict[str, Any]:
        """Extract detailed features for a line with multilingual support."""
        features = {}
        
        # Spacing analysis
        features["spacing_before"] = self._calculate_spacing_before(
            block_idx, line_idx, blocks
        )
        features["spacing_after"] = self._calculate_spacing_after(
            block_idx, line_idx, blocks
        )
        
        # Enhanced text pattern analysis with language awareness
        features["has_numbering"] = self._has_numbering_pattern(text)
        features["numbering_type"] = self._detect_numbering_type(text)
        features["has_colon"] = text.strip().endswith(':')
        features["all_caps"] = text.isupper()
        features["title_case"] = text.istitle()
        
        # CJK-specific pattern detection
        if self.detected_language in ['japanese', 'chinese']:
            features["is_cjk_chapter"] = self._is_cjk_chapter_heading(text)
            features["is_cjk_section"] = self._is_cjk_section_heading(text)
            features["has_cjk_reject_pattern"] = self._has_cjk_reject_pattern(text)
        
        # Multilingual tokenization for better analysis
        if self.detected_language:
            tokens = tokenize_multilingual(text, self.detected_language)
            features["token_count"] = len(tokens)
            features["tokens"] = tokens[:10]  # Store first 10 tokens for analysis
            
            # Language-specific heading detection
            if self.detected_language in ['japanese', 'chinese']:
                cjk_analysis = enhance_heading_detection_for_cjk(text, self.detected_language)
                features["cjk_heading_analysis"] = cjk_analysis
                features["cjk_confidence"] = cjk_analysis.get("confidence", 0.0)
        
        # Enhanced position features
        features["is_top_of_page"] = bbox[1] / page_height < TITLE_POSITION_THRESHOLD
        center_pos_ratio = ((bbox[0] + bbox[2]) / 2) / page_width
        features["is_centered"] = abs(0.5 - center_pos_ratio) < CENTER_ALIGNMENT_TOLERANCE
        features["indentation"] = bbox[0]
        
        # Language-specific features
        if self.detected_language and self.detected_language in self.cultural_patterns:
            features.update(self._extract_cultural_features(text, self.detected_language))
        
        # Confidence boosters based on language
        features["confidence_boost"] = self._calculate_confidence_boost(text, self.detected_language)
        
        return features
    
    def _is_cjk_chapter_heading(self, text: str) -> bool:
        """Check if text is a CJK chapter heading."""
        chapter_patterns = [
            r'^第[一二三四五六七八九十\d]+章',  # 第一章, 第1章
            r'^第[一二三四五六七八九十\d]+節',  # 第一節, 第1節
            r'^Chapter\s*[一二三四五六七八九十\d]+',  # Chapter 1 (mixed)
        ]
        
        for pattern in chapter_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _is_cjk_section_heading(self, text: str) -> bool:
        """Check if text is a CJK section heading."""
        section_patterns = [
            r'^\d+\.\d+\s+[^\s]',             # 1.1 Title
            r'^[一二三四五六七八九十]+、',        # 一、
            r'^（[一二三四五六七八九十\d]+）',   # （一）
        ]
        
        for pattern in section_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _has_cjk_reject_pattern(self, text: str) -> bool:
        """Check if text matches CJK rejection patterns."""
        reject_patterns = [
            r'^[•·]\s',           # Bullets
            r'^行\d+',            # Table rows
            r'^ヘッダー\d+$',      # Headers
            r'です。$',           # Japanese sentence endings
            r'である。$',         # Japanese sentence endings
            r'した。$',           # Japanese sentence endings
            r'する。$',           # Japanese sentence endings
            r'^表\d+[：:]',       # Table captions (Japanese)
            r'^图\d+[：:]',       # Figure captions (Chinese)
            r'^圖\d+[：:]',       # Figure captions (Traditional Chinese)
            r'^列表\d+',          # List items (Chinese)
            r'^リスト\d+',        # List items (Japanese)
        ]
        
        for pattern in reject_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _is_potential_heading_text(self, text: str) -> bool:
        """Quick filter for potential heading text with enhanced CJK filtering."""
        text = text.strip()
        
        # Basic length check (adjusted for CJK)
        if self.detected_language in ['japanese', 'chinese']:
            # More lenient for CJK but reject very short unless clear heading
            if len(text) < 1:
                return False
            if len(text) < 3 and not re.search(r'[章節項目録]', text):
                return False
            # Reject very long text (likely paragraphs) unless clear chapter
            if len(text) > 40 and not re.search(r'^第[一二三四五六七八九十\d]+章', text):
                return False
        else:
            # Original logic for non-CJK languages
            if len(text) < MIN_HEADING_LENGTH or len(text) > MAX_HEADING_LENGTH:
                return False
        
        # Skip page numbers, footnotes, headers/footers
        if re.match(r'^\d+$', text):  # Just numbers
            return False
        
        if re.match(r'^[ivxlcdm]+$', text.lower()):  # Roman numerals only
            return False
        
        # CJK-specific rejection patterns
        if self.detected_language in ['japanese', 'chinese']:
            # Reject if it has CJK rejection patterns
            if self._has_cjk_reject_pattern(text):
                return False
            
            # Reject list items
            if re.match(r'^[•·]\s', text) or re.match(r'^\d+\.\s[^章節]', text):
                return False
            
            # Check if it's just punctuation
            if re.match(r'^[\s\.,;:!?\-\(\)\[\]{}]*$', text):
                return False
        else:
            # Enhanced filtering for non-CJK languages (original logic)
            if re.match(r'^[\s\.,;:!?\-\(\)\[\]{}]*$', text):
                return False
        
        # Skip common non-heading patterns
        skip_patterns = [
            r'^page \d+',
            r'^\d+/\d+',      
            r'^www\.',         # URLs
            r'^http',          # URLs
            r'@',              # Email patterns
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, text.lower()):
                return False
        
        return True
    
    def _get_font_weight(self, flags: int) -> str:
        """Extract font weight from flags."""
        if flags & 2**4:  # Bold flag
            return "bold"
        return "normal"
    
    def _determine_alignment(self, bbox: Tuple, page_width: float) -> str:
        """Determine text alignment based on position."""
        left_margin = bbox[0]
        right_margin = page_width - bbox[2]
        center_pos = (bbox[0] + bbox[2]) / 2
        page_center = page_width / 2
        
        # Check if centered
        if abs(center_pos - page_center) / page_width < CENTER_ALIGNMENT_TOLERANCE:
            return "center"
        
        # Check if right-aligned
        if right_margin < left_margin * 0.5:
            return "right"
        
        return "left"
    
    def _calculate_spacing_before(self, block_idx: int, line_idx: int, blocks: List) -> float:
        """Calculate spacing before current line."""
        if block_idx == 0 and line_idx == 0:
            return 0.0
        
        current_bbox = blocks[block_idx]["lines"][line_idx]["bbox"]
        
        # Look for previous line
        if line_idx > 0:
            prev_bbox = blocks[block_idx]["lines"][line_idx - 1]["bbox"]
            return current_bbox[1] - prev_bbox[3]  # Current top - previous bottom
        elif block_idx > 0:
            # Previous block's last line
            prev_block = blocks[block_idx - 1]
            if "lines" in prev_block and prev_block["lines"]:
                prev_bbox = prev_block["lines"][-1]["bbox"]
                return current_bbox[1] - prev_bbox[3]
        
        return 0.0
    
    def _calculate_spacing_after(self, block_idx: int, line_idx: int, blocks: List) -> float:
        """Calculate spacing after current line."""
        current_block = blocks[block_idx]
        current_bbox = current_block["lines"][line_idx]["bbox"]
        
        # Look for next line
        if line_idx < len(current_block["lines"]) - 1:
            next_bbox = current_block["lines"][line_idx + 1]["bbox"]
            return next_bbox[1] - current_bbox[3]
        elif block_idx < len(blocks) - 1:
            # Next block's first line
            for next_block in blocks[block_idx + 1:]:
                if "lines" in next_block and next_block["lines"]:
                    next_bbox = next_block["lines"][0]["bbox"]
                    return next_bbox[1] - current_bbox[3]
        
        return 0.0
    
    def _has_numbering_pattern(self, text: str) -> bool:
        """Check if text has numbering pattern with multilingual support."""
        # Get language-specific patterns
        if self.detected_language in self.cultural_patterns:
            patterns = self.cultural_patterns[self.detected_language].get('numbering_patterns', [])
            for pattern in patterns:
                if re.search(pattern, text):
                    return True
        
        # Default patterns
        default_patterns = [
            r'^\d+\.',           # 1. 2. 3.
            r'^\d+\.\d+',        # 1.1, 1.2
            r'^[IVX]+\.',        # I. II. III.
            r'^[A-Z]\.',         # A. B. C.
            r'^Chapter \d+',     # Chapter 1
            r'^Section \d+',     # Section 1
            r'^\(\d+\)',         # (1) (2)
        ]
        
        for pattern in default_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _detect_numbering_type(self, text: str) -> Optional[str]:
        """Detect the type of numbering used with multilingual support."""
        # Language-specific numbering detection
        if self.detected_language == 'japanese':
            if re.search(r'^第\d+章', text):
                return "japanese_chapter"
            elif re.search(r'^第[一二三四五六七八九十]+章', text):
                return "japanese_kanji_chapter"
            elif re.search(r'^[一二三四五六七八九十]+、', text):
                return "japanese_kanji_list"
        
        elif self.detected_language == 'chinese':
            if re.search(r'^第\d+章', text):
                return "chinese_chapter"
            elif re.search(r'^第[一二三四五六七八九十]+章', text):
                return "chinese_kanji_chapter"
        
        elif self.detected_language == 'hindi':
            if re.search(r'^अध्याय\s+\d+', text):
                return "hindi_chapter"
            elif re.search(r'^[१२३४५६७८९०]+\.', text):
                return "hindi_devanagari"
        
        elif self.detected_language == 'arabic':
            if re.search(r'^الفصل\s+\w+', text):
                return "arabic_chapter"
            elif re.search(r'^[\u0660-\u0669]+', text):
                return "arabic_indic"
        
        # Default detection
        if re.search(r'^\d+\.', text):
            return "decimal"
        elif re.search(r'^[IVX]+\.', text):
            return "roman"
        elif re.search(r'^[A-Z]\.', text):
            return "alpha"
        elif re.search(r'^Chapter', text, re.IGNORECASE):
            return "chapter"
        elif re.search(r'^Section', text, re.IGNORECASE):
            return "section"
        
        return None
    
    def _extract_cultural_features(self, text: str, language: str) -> Dict[str, Any]:
        """Extract language-specific features with enhanced pattern matching."""
        features = {}
        
        if language in self.cultural_patterns:
            patterns = self.cultural_patterns[language]
            
            # Check for cultural heading styles
            heading_styles = patterns.get('heading_styles', [])
            for style in heading_styles:
                if style in text:
                    features[f"has_{language}_heading_style"] = True
                    features[f"heading_style_type"] = style
                    break
            
            # Check for cultural keywords
            heading_keywords = patterns.get('heading_keywords', [])
            for keyword in heading_keywords:
                if keyword in text:
                    features[f"has_{language}_keyword"] = True
                    features[f"keyword_matched"] = keyword
                    break
            
            # Check for cultural numbering
            numbering_patterns = patterns.get('numbering_patterns', [])
            for pattern in numbering_patterns:
                if re.search(pattern, text):
                    features[f"has_{language}_numbering"] = True
                    features[f"numbering_pattern"] = pattern
                    break
            
            # Language-specific character analysis
            if language in ['japanese', 'chinese']:
                features["contains_cjk"] = bool(re.search(r'[\u4e00-\u9fff]', text))
                if language == 'japanese':
                    features["contains_hiragana"] = bool(re.search(r'[\u3041-\u3096]', text))
                    features["contains_katakana"] = bool(re.search(r'[\u30A1-\u30FA]', text))
            
            elif language == 'arabic':
                features["contains_arabic"] = bool(re.search(r'[\u0600-\u06FF]', text))
                features["is_rtl"] = True
            
            elif language == 'hindi':
                features["contains_devanagari"] = bool(re.search(r'[\u0900-\u097F]', text))
        
        return features
    
    def _calculate_confidence_boost(self, text: str, language: str) -> float:
        """Calculate confidence boost based on language-specific patterns."""
        boost = 0.0
        
        if language in HEADING_CONFIDENCE_BOOSTERS:
            boosters = HEADING_CONFIDENCE_BOOSTERS[language]
            
            # Pattern-based boosts
            for pattern, boost_value in boosters.get('patterns', []):
                if re.search(pattern, text):
                    boost += boost_value
                    break  # Only apply one pattern boost
            
            # Keyword-based boosts
            for keyword, boost_value in boosters.get('keywords', []):
                if keyword in text:
                    boost += boost_value
                    break  # Only apply one keyword boost
        
        return min(boost, 1.0)  # Cap at 1.0
    
    def _filter_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Apply filters to remove unlikely candidates with enhanced CJK filtering."""
        filtered = []
        
        for candidate in candidates:
            # Skip if has CJK reject patterns (for CJK languages)
            if (self.detected_language in ['japanese', 'chinese'] and 
                candidate.features.get("has_cjk_reject_pattern", False)):
                continue
            
            # Font size filter (adjusted for language)
            size_threshold = self.document_stats["avg_font_size"] * FONT_SIZE_THRESHOLD_RATIO
            
            # More lenient threshold for CJK languages, but stricter overall
            if self.detected_language in ['japanese', 'chinese']:
                # Stricter font requirements unless it's a clear chapter/section
                if (candidate.features.get("is_cjk_chapter", False) or 
                    candidate.features.get("is_cjk_section", False)):
                    size_threshold *= 0.7  # More lenient for clear patterns
                else:
                    size_threshold *= 1.1  # Stricter for general text
            
            if candidate.font_size < size_threshold:
                continue
            
            # Length filters (adjusted for language)
            min_length = MIN_HEADING_LENGTH
            max_length = MAX_HEADING_LENGTH
            
            # Adjust for CJK languages where characters convey more meaning
            if self.detected_language in ['japanese', 'chinese']:
                min_length = max(1, MIN_HEADING_LENGTH // 2)
                max_length = MAX_HEADING_LENGTH * 1.5  # Slightly more restrictive
                
                # Special handling for very short text
                if candidate.text_length < 3:
                    # Only allow if it has clear heading markers
                    if not (candidate.features.get("is_cjk_chapter", False) or 
                           candidate.features.get("is_cjk_section", False)):
                        continue
            
            if candidate.text_length < min_length or candidate.text_length > max_length:
                continue
            
            # Skip if mostly punctuation (adjusted for language)
            if self.detected_language in ['japanese', 'chinese']:
                # For CJK, check character ratio differently
                cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3041-\u3096\u30A1-\u30FA]', candidate.text))
                total_chars = len(candidate.text.replace(' ', ''))
                if total_chars > 0 and cjk_chars / total_chars < 0.2:
                    # Not enough meaningful characters, unless it's clearly a heading
                    if not (candidate.features.get("is_cjk_chapter", False) or 
                           candidate.features.get("is_cjk_section", False)):
                        continue
            else:
                # For non-CJK languages, use alpha ratio
                alpha_ratio = sum(c.isalpha() for c in candidate.text) / len(candidate.text)
                if alpha_ratio < 0.3:
                    continue
            
            # Apply general linguistic heading detection
            heading_analysis = is_likely_heading(candidate.text, self.detected_language)
            if heading_analysis["confidence"] < 0.1:  # Very low threshold
                continue
            
            # Store the linguistic analysis
            candidate.features["linguistic_analysis"] = heading_analysis
            
            filtered.append(candidate)
        
        self.logger.debug(f"Filtered {len(candidates)} -> {len(filtered)} candidates")
        return filtered
    
    def _score_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Score candidates based on multiple features with enhanced CJK scoring."""
        for candidate in candidates:
            score = 0.0
            
            # Font size score (0-30 points)
            size_ratio = candidate.font_size / self.document_stats["avg_font_size"]
            score += min(30, size_ratio * 10)
            
            # Bold weight (0-20 points)
            if candidate.is_bold:
                score += 20
            
            # Position score (0-15 points)
            if candidate.features.get("is_top_of_page", False):
                score += 15
            elif candidate.position_ratio < 0.3:  # Upper part of page
                score += 10
            
            # Spacing score (0-15 points)
            if candidate.line_spacing_before > 10:
                score += 8
            if candidate.line_spacing_after > 5:
                score += 7
            
            # Text pattern score (0-20 points)
            if candidate.features.get("has_numbering", False):
                score += 15
            if candidate.features.get("title_case", False):
                score += 5
            if candidate.features.get("has_colon", False):
                score += 5
            
            # Enhanced CJK-specific scoring
            if self.detected_language in ['japanese', 'chinese']:
                # Stricter font size requirements unless clear heading patterns
                if candidate.font_size < self.document_stats["avg_font_size"] * 1.2:
                    if not (candidate.features.get("is_cjk_chapter", False) or 
                           candidate.features.get("is_cjk_section", False)):
                        score *= 0.3  # Heavy penalty for small font without clear patterns
                
                # Major bonus for chapter patterns
                if candidate.features.get("is_cjk_chapter", False):
                    score += 40  # Strong bonus for chapter headings
                elif candidate.features.get("is_cjk_section", False):
                    score += 25  # Good bonus for section headings
                
                # CJK confidence from analysis
                cjk_confidence = candidate.features.get("cjk_confidence", 0.0)
                score += cjk_confidence * 15
                
                # Bonus for containing CJK characters
                if candidate.features.get("contains_cjk", False):
                    score += 10
            
            # Language-specific scoring (existing logic)
            if self.detected_language:
                # Apply confidence boost
                boost = candidate.features.get("confidence_boost", 0.0)
                score += boost * 20  # Convert to points
                
                # Cultural pattern bonuses
                if candidate.features.get(f"has_{self.detected_language}_heading_style", False):
                    score += 15
                if candidate.features.get(f"has_{self.detected_language}_numbering", False):
                    score += 10
            
            # Linguistic analysis bonus
            linguistic_confidence = candidate.features.get("linguistic_analysis", {}).get("confidence", 0.0)
            score += linguistic_confidence * 10
            
            # Normalize score to 0-1
            candidate.confidence_score = min(1.0, score / 100.0)
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        self.logger.debug(f"Top candidate scores: {[c.confidence_score for c in candidates[:5]]}")
        return candidates
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the candidate generation process."""
        return {
            "detected_language": self.detected_language,
            "document_stats": self.document_stats,
            "cultural_patterns_used": self.detected_language in self.cultural_patterns,
            "tokenization_available": self.detected_language in ['japanese', 'chinese']
        }
        
    def _identify_running_elements(self, candidates: List[HeadingCandidate], threshold: int = 2) -> set:
        text_positions = defaultdict(list)
        page_count = max([c.page for c in candidates] or [1])

        for cand in candidates:
            text_positions[cand.text.strip()].append(cand.position_ratio)

        running_elements = set()
        min_occurrences = max(threshold, int(page_count * 0.2)) if page_count > 10 else threshold

        for text, positions in text_positions.items():
            if len(positions) >= min_occurrences:
                is_header = all(p < 0.15 for p in positions)
                is_footer = all(p > 0.85 for p in positions)

                if is_header or is_footer:
                    running_elements.add(text)

        if running_elements:
            self.logger.info(f"Identified and removed running elements: {running_elements}")

        return running_elements