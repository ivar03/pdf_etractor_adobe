import re
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path

# Language detection imports
try:
    import langdetect
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available, language detection will be limited")

# NLP imports
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, some text processing features will be limited")

# SentencePiece imports for better CJK tokenization
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    logging.warning("SentencePiece not available, CJK tokenization will use fallback methods")

# Additional tokenizers for specific languages
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False


class TokenizerManager:
    """Manages different tokenizers for various languages."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tokenizers = {}
        self._initialize_tokenizers()
    
    def _initialize_tokenizers(self):
        """Initialize available tokenizers."""
        # Initialize SentencePiece models if available
        if SENTENCEPIECE_AVAILABLE:
            self._load_sentencepiece_models()
        
        # Initialize MeCab for Japanese if available
        if MECAB_AVAILABLE:
            try:
                self.tokenizers['mecab'] = MeCab.Tagger('-Owakati')
                self.logger.info("MeCab tokenizer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MeCab: {e}")
    
    def _load_sentencepiece_models(self):
        """Load SentencePiece models for different languages."""
        model_configs = {
            'japanese': 'data/models/japanese_tokenizer.model',
            'chinese': 'data/models/chinese_tokenizer.model',
            'multilingual': 'data/models/multilingual_tokenizer.model'
        }
        
        for lang, model_path in model_configs.items():
            try:
                if Path(model_path).exists():
                    sp = spm.SentencePieceProcessor()
                    sp.load(model_path)
                    self.tokenizers[f'sp_{lang}'] = sp
                    self.logger.info(f"Loaded SentencePiece model for {lang}")
                else:
                    self.logger.debug(f"SentencePiece model not found: {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load SentencePiece model for {lang}: {e}")
    
    def get_tokenizer(self, language: str, tokenizer_type: str = 'auto'):
        """Get appropriate tokenizer for language."""
        if tokenizer_type == 'sentencepiece' and f'sp_{language}' in self.tokenizers:
            return self.tokenizers[f'sp_{language}']
        elif tokenizer_type == 'mecab' and language == 'japanese' and 'mecab' in self.tokenizers:
            return self.tokenizers['mecab']
        return None


# Global tokenizer manager instance
_tokenizer_manager = TokenizerManager()


class TextUtils:
    """Advanced text processing and analysis utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK data if available
        if NLTK_AVAILABLE:
            self._ensure_nltk_data()
        
        # Text cleaning patterns
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Language-specific patterns
        self.language_patterns = {
            'english': {
                'articles': {'a', 'an', 'the'},
                'common_words': {'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'},
                'heading_indicators': {'chapter', 'section', 'part', 'introduction', 'conclusion'}
            },
            'japanese': {
                'particles': {'は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで'},
                'heading_indicators': {'章', '節', '項', '第', 'について', 'に関して'}
            },
            'chinese': {
                'particles': {'的', '了', '在', '和', '与', '对', '从', '到'},
                'heading_indicators': {'章', '节', '部分', '第', '关于', '有关'}
            },
            'arabic': {
                'articles': {'ال'},
                'heading_indicators': {'الفصل', 'القسم', 'الجزء', 'الباب', 'حول', 'عن'}
            },
            'hindi': {
                'particles': {'का', 'की', 'के', 'में', 'से', 'को', 'पर', 'और'},
                'heading_indicators': {'अध्याय', 'खंड', 'भाग', 'प्रकरण', 'के बारे में'}
            }
        }
        
        # Character encoding mappings
        self.unicode_replacements = {
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Horizontal ellipsis
            '\u00a0': ' ',  # Non-breaking space
            '\u2022': '•',  # Bullet
        }
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                except Exception as e:
                    self.logger.warning(f"Failed to download NLTK data '{data_name}': {e}")


def tokenize_japanese(text: str) -> List[str]:
    """Advanced Japanese tokenization using SentencePiece or MeCab."""
    if not text or not text.strip():
        return []
    
    text = clean_text(text)
    
    # Try SentencePiece first
    if SENTENCEPIECE_AVAILABLE:
        tokenizer = _tokenizer_manager.get_tokenizer('japanese', 'sentencepiece')
        if tokenizer:
            try:
                tokens = tokenizer.encode_as_pieces(text)
                # Filter out empty tokens and special characters
                return [token for token in tokens if token.strip() and not token.startswith('▁')]
            except Exception as e:
                logging.warning(f"SentencePiece tokenization failed: {e}")
    
    # Try MeCab as fallback
    if MECAB_AVAILABLE:
        tokenizer = _tokenizer_manager.get_tokenizer('japanese', 'mecab')
        if tokenizer:
            try:
                result = tokenizer.parse(text).strip()
                tokens = result.split()
                return [token for token in tokens if token.strip()]
            except Exception as e:
                logging.warning(f"MeCab tokenization failed: {e}")
    
    # Final fallback: character-based splitting for Japanese
    return _japanese_character_split(text)


def tokenize_chinese(text: str) -> List[str]:
    """Chinese tokenization using SentencePiece."""
    if not text or not text.strip():
        return []
    
    text = clean_text(text)
    
    # Try SentencePiece
    if SENTENCEPIECE_AVAILABLE:
        tokenizer = _tokenizer_manager.get_tokenizer('chinese', 'sentencepiece')
        if tokenizer:
            try:
                tokens = tokenizer.encode_as_pieces(text)
                return [token for token in tokens if token.strip() and not token.startswith('▁')]
            except Exception as e:
                logging.warning(f"SentencePiece Chinese tokenization failed: {e}")
    
    # Fallback: character-based splitting
    return _chinese_character_split(text)


def tokenize_multilingual(text: str, language: str = 'auto') -> List[str]:
    """Universal tokenization that handles multiple languages intelligently."""
    if not text or not text.strip():
        return []
    
    text = clean_text(text)
    
    # Detect language if auto
    if language == 'auto':
        language = detect_language(text)
    
    # Use language-specific tokenizers
    if language == 'japanese':
        return tokenize_japanese(text)
    elif language == 'chinese':
        return tokenize_chinese(text)
    elif language in ['arabic', 'hindi']:
        return _handle_rtl_languages(text, language)
    else:
        # Use NLTK for European languages
        return extract_words(text, language)


def _japanese_character_split(text: str) -> List[str]:
    """Fallback Japanese tokenization using character patterns."""
    # Simple regex-based approach for Japanese
    tokens = []
    
    # Split on hiragana/katakana word boundaries and kanji
    japanese_pattern = re.compile(r'[ひ-ゞ]+|[ァ-ヾ]+|[一-龯]+|[a-zA-Z0-9]+')
    matches = japanese_pattern.findall(text)
    
    for match in matches:
        if len(match.strip()) > 0:
            tokens.append(match)
    
    return tokens


def _chinese_character_split(text: str) -> List[str]:
    """Fallback Chinese tokenization using character patterns."""
    # Simple character-based splitting for Chinese
    chinese_chars = re.findall(r'[一-龯]+', text)
    latin_words = re.findall(r'[a-zA-Z0-9]+', text)
    
    # Combine and maintain order approximately
    tokens = []
    for char_group in chinese_chars:
        # Split long character sequences
        if len(char_group) > 4:
            for i in range(0, len(char_group), 2):
                tokens.append(char_group[i:i+2])
        else:
            tokens.append(char_group)
    
    tokens.extend(latin_words)
    return tokens


def _handle_rtl_languages(text: str, language: str) -> List[str]:
    """Handle right-to-left languages like Arabic and Hindi."""
    if language == 'arabic':
        # Arabic word tokenization
        arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
        latin_words = re.findall(r'[a-zA-Z0-9]+', text)
        return arabic_words + latin_words
    
    elif language == 'hindi':
        # Hindi/Devanagari tokenization
        hindi_words = re.findall(r'[\u0900-\u097F]+', text)
        latin_words = re.findall(r'[a-zA-Z0-9]+', text)
        return hindi_words + latin_words
    
    return text.split()


def enhance_heading_detection_for_cjk(text: str, language: str) -> Dict[str, Any]:
    """Enhanced heading detection specifically for CJK languages."""
    if not text:
        return {"is_heading": False, "confidence": 0.0, "reasons": []}
    
    reasons = []
    score = 0.0
    
    # Get language-appropriate tokens
    if language == 'japanese':
        tokens = tokenize_japanese(text)
    elif language == 'chinese':
        tokens = tokenize_chinese(text)
    else:
        tokens = text.split()
    
    # CJK-specific patterns
    if language in ['japanese', 'chinese']:
        # Check for chapter markers
        chapter_markers = ['章', '節', '第', '部', '篇']
        for marker in chapter_markers:
            if marker in text:
                score += 0.3
                reasons.append(f"contains_{marker}")
                break
        
        # Check for numbering patterns
        if re.search(r'第[一二三四五六七八九十\d]+[章節部篇]', text):
            score += 0.4
            reasons.append("numbered_chapter")
        
        # Token count analysis for CJK
        if 1 <= len(tokens) <= 6:
            score += 0.2
            reasons.append("appropriate_token_count")
    
    # Length analysis for CJK (characters, not words)
    char_count = len(text.strip())
    if 2 <= char_count <= 20:
        score += 0.2
        reasons.append("appropriate_character_length")
    
    confidence = max(0.0, min(1.0, score))
    is_heading = confidence >= 0.4
    
    return {
        "is_heading": is_heading,
        "confidence": confidence,
        "reasons": reasons,
        "token_count": len(tokens),
        "character_count": char_count,
        "language": language
    }


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if not text:
        return ""
    
    # Handle None input
    if text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Replace common Unicode characters
    utils = TextUtils()
    for unicode_char, replacement in utils.unicode_replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Remove control characters (except newlines and tabs)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    if not text:
        return ""
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_extra_spaces(text: str) -> str:
    """Remove extra spaces while preserving intentional formatting."""
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Replace multiple spaces with single space, but preserve paragraph breaks
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Clean each line
        cleaned_line = normalize_whitespace(line)
        cleaned_lines.append(cleaned_line)
    
    # Rejoin with single newlines
    text = '\n'.join(cleaned_lines)
    
    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_sentences(text: str, language: str = 'english') -> List[str]:
    """Extract sentences from text using language-aware tokenization with robust fallback."""
    if not text or not text.strip():
        return []

    text = clean_text(text)

    sentences = []

    if NLTK_AVAILABLE:
        try:
            # Sanity check: ensure 'punkt' tokenizer is available
            try:
                nltk.data.find('tokenizers/punkt/english.pickle')
            except LookupError:
                nltk.download('punkt', quiet=True)

            # Restrict language to those supported by NLTK
            supported_languages = ['english', 'spanish', 'portuguese', 'french', 'german']
            language_for_tokenize = language if language in supported_languages else 'english'

            sentences = sent_tokenize(text, language=language_for_tokenize)

        except Exception as e:
            logging.warning(f"NLTK sentence tokenization failed, falling back to simple split: {e}")
            sentences = _simple_sentence_split(text)
    else:
        sentences = _simple_sentence_split(text)

    # Clean and filter sentences
    cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    return cleaned_sentences


def _simple_sentence_split(text: str) -> List[str]:
    """Simple sentence splitting when NLTK is not available."""
    # Split on sentence endings, but be careful with abbreviations
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    
    # Clean up
    cleaned = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            cleaned.append(sentence)
    
    return cleaned


def extract_words(text: str, language: str = 'english', 
                 include_stopwords: bool = True) -> List[str]:
    """Extract words from text with advanced language-aware tokenization."""
    if not text:
        return []
    
    text = clean_text(text)
    
    # Use multilingual tokenization for better results
    if language in ['japanese', 'chinese']:
        return tokenize_multilingual(text, language)
    
    if NLTK_AVAILABLE:
        try:
            words = word_tokenize(text, language=language if language in ['english', 'spanish', 'portuguese', 'french', 'german'] else 'english')
            
            # Filter out punctuation and optionally stopwords
            if not include_stopwords and language == 'english':
                try:
                    stop_words = set(stopwords.words('english'))
                    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
                except Exception:
                    words = [word for word in words if word.isalpha()]
            else:
                words = [word for word in words if word.isalpha()]
                
        except Exception as e:
            logging.warning(f"NLTK word tokenization failed: {e}")
            words = _simple_word_split(text)
    else:
        words = _simple_word_split(text)
    
    return words


def _simple_word_split(text: str) -> List[str]:
    """Simple word splitting when NLTK is not available."""
    # Split on whitespace and punctuation
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return words


def detect_language(text: str) -> str:
    """Detect the language of the given text."""
    if not text or len(text.strip()) < 10:
        return 'en'  # Default to English for short texts
    
    if LANGDETECT_AVAILABLE:
        try:
            # Clean text for better detection
            cleaned_text = clean_text(text)[:1000]  # Use first 1000 chars for detection
            
            detected = detect(cleaned_text)
            
            # Map to our supported languages
            language_mapping = {
                'en': 'english',
                'ja': 'japanese',
                'zh': 'chinese',
                'zh-cn': 'chinese',
                'zh-tw': 'chinese',
                'ar': 'arabic',
                'hi': 'hindi',
                'es': 'spanish',
                'fr': 'french',
                'de': 'german',
                'pt': 'portuguese'
            }
            
            return language_mapping.get(detected, 'english')
            
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
    
    # Fallback: simple heuristic detection
    return _simple_language_detection(text)


def _simple_language_detection(text: str) -> str:
    """Simple language detection based on character patterns."""
    # Count different types of characters
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
    arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
    devanagari_chars = len(re.findall(r'[\u0900-\u097f]', text))
    
    total_chars = len(re.sub(r'\s', '', text))
    
    if total_chars == 0:
        return 'english'
    
    # Calculate ratios
    cjk_ratio = cjk_chars / total_chars
    arabic_ratio = arabic_chars / total_chars
    devanagari_ratio = devanagari_chars / total_chars
    
    # Determine language based on character ratios
    if cjk_ratio > 0.1:
        return 'japanese'  # Could be Chinese, but we'll default to Japanese
    elif arabic_ratio > 0.1:
        return 'arabic'
    elif devanagari_ratio > 0.1:
        return 'hindi'
    else:
        return 'english'


def is_likely_heading(text: str, language: str = 'english') -> Dict[str, Any]:
    """Analyze if text is likely to be a heading based on linguistic features with CJK support."""
    if not text:
        return {"is_heading": False, "confidence": 0.0, "reasons": []}
    
    text = text.strip()
    
    # Use enhanced CJK detection if applicable
    if language in ['japanese', 'chinese']:
        return enhance_heading_detection_for_cjk(text, language)
    
    # Original logic for other languages
    reasons = []
    score = 0.0
    
    # Length analysis
    if 3 <= len(text) <= 100:
        score += 0.2
        reasons.append("appropriate_length")
    elif len(text) > 100:
        score -= 0.1
        reasons.append("too_long")
    
    # Capitalization analysis
    if text.isupper():
        score += 0.3
        reasons.append("all_caps")
    elif text.istitle():
        score += 0.2
        reasons.append("title_case")
    elif text[0].isupper():
        score += 0.1
        reasons.append("starts_with_capital")
    
    # Punctuation analysis
    if text.endswith(':'):
        score += 0.2
        reasons.append("ends_with_colon")
    elif text.endswith('.'):
        score -= 0.1
        reasons.append("ends_with_period")
    
    # Check for numbering patterns
    numbering_patterns = [
        r'^\d+\.',          # 1. 2. 3.
        r'^\d+\.\d+',       # 1.1 1.2
        r'^[IVX]+\.',       # I. II. III.
        r'^[A-Z]\.',        # A. B. C.
        r'^Chapter \d+',    # Chapter 1
        r'^Section \d+',    # Section 1
    ]
    
    for pattern in numbering_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            score += 0.3
            reasons.append("has_numbering")
            break
    
    # Language-specific analysis
    if language in TextUtils().language_patterns:
        patterns = TextUtils().language_patterns[language]
        heading_indicators = patterns.get('heading_indicators', set())
        
        text_lower = text.lower()
        for indicator in heading_indicators:
            if indicator in text_lower:
                score += 0.2
                reasons.append(f"contains_{indicator}")
                break
    
    # Word count analysis
    word_count = len(text.split())
    if 1 <= word_count <= 8:
        score += 0.1
        reasons.append("appropriate_word_count")
    elif word_count > 15:
        score -= 0.1
        reasons.append("too_many_words")
    
    # Check for common non-heading patterns
    non_heading_patterns = [
        r'^\d+$',           # Just numbers
        r'^page \d+',       # Page numbers
        r'@',               # Email addresses
        r'http',            # URLs
        r'www\.',           # URLs
    ]
    
    for pattern in non_heading_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score -= 0.3
            reasons.append("non_heading_pattern")
            break
    
    # Normalize score
    confidence = max(0.0, min(1.0, score))
    is_heading = confidence >= 0.4
    
    return {
        "is_heading": is_heading,
        "confidence": confidence,
        "reasons": reasons,
        "word_count": word_count,
        "character_count": len(text)
    }


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[Tuple[str, float]]:
    """Extract key phrases from text using frequency and linguistic patterns."""
    if not text:
        return []
    
    text = clean_text(text)
    
    # Extract potential phrases (n-grams)
    phrases = []
    
    # Unigrams (single words)
    words = extract_words(text, include_stopwords=False)
    word_freq = Counter(words)
    
    # Add significant words
    for word, freq in word_freq.most_common(max_phrases):
        if len(word) > 3 and freq > 1:  # Filter short words and single occurrences
            phrases.append((word.lower(), freq / len(words)))
    
    # Bigrams (two-word phrases)
    if NLTK_AVAILABLE:
        try:
            from nltk import bigrams
            word_tokens = word_tokenize(text.lower())
            bigram_freq = Counter(bigrams(word_tokens))
            
            for (word1, word2), freq in bigram_freq.most_common(max_phrases // 2):
                if len(word1) > 2 and len(word2) > 2 and freq > 1:
                    phrase = f"{word1} {word2}"
                    phrases.append((phrase, freq / len(word_tokens)))
        except Exception:
            pass  # Skip bigrams if NLTK fails
    
    # Sort by score and return top phrases
    phrases.sort(key=lambda x: x[1], reverse=True)
    return phrases[:max_phrases]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using simple metrics."""
    if not text1 or not text2:
        return 0.0
    
    # Clean texts
    text1 = clean_text(text1).lower()
    text2 = clean_text(text2).lower()
    
    # Extract words
    words1 = set(extract_words(text1, include_stopwords=False))
    words2 = set(extract_words(text2, include_stopwords=False))
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for consistent comparison."""
    if not text:
        return ""
    
    # Clean and normalize
    text = clean_text(text)
    text = normalize_whitespace(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except hyphens and apostrophes
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    
    # Normalize whitespace again
    text = normalize_whitespace(text)
    
    return text


def extract_numbers(text: str) -> List[str]:
    """Extract all numbers from text, including formatted numbers."""
    if not text:
        return []
    
    # Pattern to match various number formats
    number_patterns = [
        r'\b\d+\.\d+\b',        # Decimal numbers
        r'\b\d+,\d+\b',         # Comma-separated numbers
        r'\b\d+\b',             # Integer numbers
        r'\b\d+[A-Za-z]+\b',    # Numbers with units (5km, 10th)
    ]
    
    numbers = []
    for pattern in number_patterns:
        matches = re.findall(pattern, text)
        numbers.extend(matches)
    
    return list(set(numbers))  # Remove duplicates


def is_mostly_numeric(text: str) -> bool:
    """Check if text is mostly numeric content."""
    if not text:
        return False
    
    # Remove whitespace and punctuation
    cleaned = re.sub(r'[\s\W]', '', text)
    
    if not cleaned:
        return False
    
    # Count numeric characters
    numeric_chars = sum(1 for char in cleaned if char.isdigit())
    
    # Consider mostly numeric if >60% of characters are digits
    return (numeric_chars / len(cleaned)) > 0.6


def contains_url_or_email(text: str) -> bool:
    """Check if text contains URLs or email addresses."""
    if not text:
        return False
    
    utils = TextUtils()
    return bool(utils.url_pattern.search(text) or utils.email_pattern.search(text))


def get_text_statistics(text: str) -> Dict[str, Any]:
    """Get comprehensive statistics about the text."""
    if not text:
        return {}
    
    # Basic counts
    char_count = len(text)
    char_count_no_spaces = len(text.replace(' ', ''))
    word_count = len(text.split())
    sentence_count = len(extract_sentences(text))
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    # Character analysis
    uppercase_count = sum(1 for c in text if c.isupper())
    lowercase_count = sum(1 for c in text if c.islower())
    digit_count = sum(1 for c in text if c.isdigit())
    punctuation_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    # Calculate ratios
    uppercase_ratio = uppercase_count / char_count_no_spaces if char_count_no_spaces > 0 else 0
    digit_ratio = digit_count / char_count_no_spaces if char_count_no_spaces > 0 else 0
    
    # Average lengths
    avg_word_length = char_count_no_spaces / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    return {
        "character_count": char_count,
        "character_count_no_spaces": char_count_no_spaces,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "uppercase_count": uppercase_count,
        "lowercase_count": lowercase_count,
        "digit_count": digit_count,
        "punctuation_count": punctuation_count,
        "uppercase_ratio": round(uppercase_ratio, 3),
        "digit_ratio": round(digit_ratio, 3),
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "has_url_or_email": contains_url_or_email(text),
        "is_mostly_numeric": is_mostly_numeric(text)
    }


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length while preserving word boundaries."""
    if not text or len(text) <= max_length:
        return text
    
    # Find the last space before max_length
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If space is reasonably close to end
        truncated = truncated[:last_space]
    
    return truncated + suffix


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks for processing."""
    if not text or len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            sentences = extract_sentences(text[start:end + 200])  # Look ahead
            if sentences:
                # Find the last complete sentence
                chunk_text = ""
                for sentence in sentences:
                    if len(chunk_text + sentence) <= chunk_size:
                        chunk_text += sentence + " "
                    else:
                        break
                
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                    start += len(chunk_text) - overlap
                else:
                    # Fallback to hard cut
                    chunks.append(text[start:end])
                    start = end - overlap
            else:
                chunks.append(text[start:end])
                start = end - overlap
        else:
            chunks.append(text[start:])
            break
    
    return chunks


def get_tokenizer_info() -> Dict[str, Any]:
    """Get information about available tokenizers."""
    info = {
        "sentencepiece_available": SENTENCEPIECE_AVAILABLE,
        "mecab_available": MECAB_AVAILABLE,
        "nltk_available": NLTK_AVAILABLE,
        "langdetect_available": LANGDETECT_AVAILABLE,
        "loaded_models": list(_tokenizer_manager.tokenizers.keys())
    }
    
    return info