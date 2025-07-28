CULTURAL_PATTERNS = {
    'english': {
        'heading_keywords': [
            'chapter', 'section', 'part', 'introduction', 'conclusion',
            'summary', 'abstract', 'appendix', 'preface', 'table of contents'
        ],
        'numbering_patterns': [
            r'^\d+\.',            # 1.
            r'^\d+\.\d+',         # 1.1
            r'^[IVXLC]+\.',       # I. II. IV.
            r'^[A-Z]\.',          # A. B.
            r'^Figure\s+\d+',     # Figure 1
            r'^Table\s+\d+'       # Table 1
        ],
        'layout_preferences': 'ltr',
        'tokenizer': 'nltk',
        'text_direction': 'left-to-right'
    },

    'japanese': {
        'heading_styles': ['章', '節', '項', '第', 'はじめに', '結論'],
        'heading_keywords': [
            '章', '節', '項', '第', 'について', 'に関して',
            'はじめに', '序論', '結論', '終章', '付録', '参考文献'
        ],
        'numbering_patterns': [
            r'^第\d+章',          # 第1章
            r'^第[一二三四五六七八九十]+章',  # 第一章
            r'^\d+\.\d+',         # 1.1
            r'^[一二三四五六七八九十]+、',  # 一、
            r'^（\d+）',          # （1）
            r'^[\u3041-\u3096]+、', # ひらがな、
            r'^[\u30A1-\u30FA]+、'  # カタカナ、
        ],
        # Enhanced patterns for stricter matching
        'strong_heading_patterns': [
            r'^第[一二三四五六七八九十\d]+章',  # Chapters
            r'^第[一二三四五六七八九十\d]+節',  # Sections  
            r'^\d+\.\d+\s+[^\s]',             # 1.1 Title
            r'^(はじめに|まとめ|引用|表|リスト|序論|結論|終章|付録|参考文献|謝辞)$'  # Common sections
        ],
        'reject_patterns': [
            r'^[•·]\s',           # Bullets
            r'^行\d+',            # Table rows
            r'^ヘッダー\d+$',      # Headers
            r'です。$',           # Sentence endings
            r'である。$',         # Formal sentence endings
            r'した。$',           # Past tense endings
            r'する。$',           # Present tense endings
            r'^表\d+[：:]',       # Table captions
            r'^図\d+[：:]',       # Figure captions
            r'^リスト\d+',        # List items
            r'^項目\d+',          # Item numbers
            r'^\d+\s*$',          # Just numbers
            r'^[ivxlcdm]+\s*$',   # Roman numerals only
        ],
        'layout_preferences': 'vertical_aware',
        'tokenizer': 'sentencepiece',
        'model_path': 'data/models/japanese_tokenizer.model',
        'fallback_tokenizer': 'mecab',
        'text_direction': 'top-to-bottom',
        'character_sets': {
            'hiragana': r'[\u3041-\u3096]',
            'katakana': r'[\u30A1-\u30FA]',
            'kanji': r'[\u4e00-\u9faf]'
        },
        'particles': ['は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで', 'の', 'へ'],
        'common_headings': {
            'introduction': ['はじめに', '序論', '導入', '概要'],
            'methodology': ['方法', '手法', 'アプローチ', '実験方法'],
            'results': ['結果', '実験結果', '分析結果', '成果'],
            'conclusion': ['結論', '終章', 'まとめ', '考察']
        }
    },

    'hindi': {
        'heading_styles': ['अध्याय', 'खंड', 'भाग', 'प्रकरण', 'निष्कर्ष', 'परिचय'],
        'heading_keywords': [
            'अध्याय', 'खंड', 'भाग', 'प्रकरण', 'के बारे में',
            'परिचय', 'प्रस्तावना', 'निष्कर्ष', 'सारांश', 'परिशिष्ट'
        ],
        'numbering_patterns': [
            r'^\d+\.',            # 1.
            r'^[(]?\d+[)]?',      # (1) 1)
            r'^[१२३४५६७८९०]+[.]?', # Hindi numerals with optional dot
            r'^अनुच्छेद\s+\d+',    # अनुच्छेद 1
            r'^खंड\s+[१२३४५६७८९०\d]+', # खंड १
            r'^अध्याय\s+[१२३४५६७८९०\d]+'  # अध्याय १
        ],
        'layout_preferences': 'devanagari_aware',
        'tokenizer': 'nltk',
        'text_direction': 'left-to-right',
        'character_sets': {
            'devanagari': r'[\u0900-\u097F]'
        },
        'particles': ['का', 'की', 'के', 'में', 'से', 'को', 'पर', 'और', 'या'],
        'common_headings': {
            'introduction': ['परिचय', 'प्रस्तावना', 'भूमिका'],
            'methodology': ['पद्धति', 'विधि', 'तरीका'],
            'results': ['परिणाम', 'नतीजे', 'फल'],
            'conclusion': ['निष्कर्ष', 'समापन', 'अंत']
        }
    },

    'arabic': {
        'heading_styles': ['الفصل', 'القسم', 'الجزء', 'الباب', 'مقدمة', 'خاتمة'],
        'heading_keywords': [
            'الفصل', 'القسم', 'الجزء', 'الباب', 'حول', 'عن',
            'مقدمة', 'تمهيد', 'خاتمة', 'ملخص', 'ملحق', 'مراجع'
        ],
        'numbering_patterns': [
            r'^الفصل\s+\w+',      # الفصل الأول
            r'^\d+\.',            # 1.
            r'^[\u0660-\u0669]+', # Arabic-Indic numerals ٠١٢٣٤٥٦٧٨٩
            r'^قسم\s+\d+',        # قسم 1
            r'^الباب\s+[الأولثانيثالثرابعخامسسادسسابعثامنتاسععاشر]+', # الباب الأول
            r'^[\u0660-\u0669]+[.]' # Arabic numerals with dot
        ],
        'layout_preferences': 'rtl_aware',
        'tokenizer': 'nltk',
        'text_direction': 'right-to-left',
        'character_sets': {
            'arabic': r'[\u0600-\u06FF]',
            'arabic_numerals': r'[\u0660-\u0669]'
        },
        'articles': ['ال'],
        'common_headings': {
            'introduction': ['مقدمة', 'تمهيد', 'بداية'],
            'methodology': ['منهجية', 'طريقة', 'أسلوب'],
            'results': ['نتائج', 'نتيجة', 'حصائل'],
            'conclusion': ['خاتمة', 'استنتاج', 'خلاصة']
        }
    },

    'chinese': {
        'heading_styles': ['章', '节', '部分', '第', '引言', '结论'],
        'heading_keywords': [
            '章', '节', '部分', '第', '关于', '有关',
            '引言', '前言', '结论', '总结', '附录', '参考文献'
        ],
        'numbering_patterns': [
            r'^第[一二三四五六七八九十百千]+章',   # 第一章
            r'^第\d+章',                   # 第1章
            r'^\d+\.\d+',                  # 1.1
            r'^[一二三四五六七八九十]+、',      # 一、
            r'^（\d+）',                   # （1）
            r'^[\u4e00-\u9fff]+、',        # 汉字、
            r'^第[一二三四五六七八九十]+节'      # 第一节
        ],
        # Enhanced patterns for stricter matching
        'strong_heading_patterns': [
            r'^第[一二三四五六七八九十\d]+章',     # Chapters
            r'^第[一二三四五六七八九十\d]+节',     # Sections  
            r'^\d+\.\d+\s+[^\s]',             # 1.1 Title
            r'^(引言|前言|结论|总结|附录|参考文献|致谢)$'  # Common sections
        ],
        'reject_patterns': [
            r'^[•·]\s',           # Bullets
            r'^行\d+',            # Table rows
            r'^表\d+[：:]',       # Table captions
            r'^图\d+[：:]',       # Figure captions (Simplified)
            r'^圖\d+[：:]',       # Figure captions (Traditional)
            r'^列表\d+',          # List items
            r'^项目\d+',          # Item numbers
            r'^\d+\s*$',          # Just numbers
            r'^[ivxlcdm]+\s*$',   # Roman numerals only
        ],
        'layout_preferences': 'vertical_aware',
        'tokenizer': 'sentencepiece',
        'model_path': 'data/models/chinese_tokenizer.model',
        'fallback_tokenizer': 'character_split',
        'text_direction': 'top-to-bottom',
        'character_sets': {
            'simplified': r'[\u4e00-\u9fff]',
            'traditional': r'[\u4e00-\u9fff]'
        },
        'particles': ['的', '了', '在', '和', '与', '对', '从', '到', '为', '被'],
        'common_headings': {
            'introduction': ['引言', '前言', '概述', '导论'],
            'methodology': ['方法', '方法论', '研究方法'],
            'results': ['结果', '实验结果', '分析结果'],
            'conclusion': ['结论', '总结', '结语', '小结']
        }
    },

    # Multilingual fallback for documents with mixed languages
    'multilingual': {
        'tokenizer': 'sentencepiece',
        'model_path': 'data/models/multilingual_tokenizer.model',
        'fallback_tokenizer': 'nltk',
        'supported_scripts': ['latin', 'cjk', 'arabic', 'devanagari'],
        'auto_detect': True
    },

    # Optional fallback for other languages you may add later:
    'default': {
        'heading_keywords': ['introduction', 'summary', 'abstract', 'conclusion'],
        'numbering_patterns': [r'^\d+\.', r'^\d+\.\d+'],
        'layout_preferences': 'ltr',
        'tokenizer': 'nltk',
        'text_direction': 'left-to-right'
    }
}

# Tokenizer configuration mapping
TOKENIZER_CONFIG = {
    'sentencepiece': {
        'japanese': {
            'model_path': 'data/models/japanese_tokenizer.model',
            'vocab_size': 32000,
            'model_type': 'unigram',
            'fallback': 'mecab'
        },
        'chinese': {
            'model_path': 'data/models/chinese_tokenizer.model',
            'vocab_size': 32000,
            'model_type': 'unigram',
            'fallback': 'character_split'
        },
        'multilingual': {
            'model_path': 'data/models/multilingual_tokenizer.model',
            'vocab_size': 64000,
            'model_type': 'bpe',
            'fallback': 'nltk'
        }
    },
    'mecab': {
        'japanese': {
            'dict_type': 'ipadic',
            'output_format': 'wakati'
        }
    },
    'nltk': {
        'supported_languages': ['english', 'spanish', 'portuguese', 'french', 'german'],
        'default_language': 'english'
    }
}

# Model download URLs (for automatic model download)
MODEL_URLS = {
    'japanese_tokenizer.model': 'https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/tokenizer.model',
    'chinese_tokenizer.model': 'https://huggingface.co/bert-base-chinese/resolve/main/tokenizer.model',
    'multilingual_tokenizer.model': 'https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/tokenizer.json'
}

TEST_SAMPLES = {
    'english': [
        "Chapter 1: Introduction",
        "1.1 Background",
        "II. Literature Review"
    ],
    'japanese': [
        "第1章 はじめに",
        "1.1 背景",
        "一、概要",
        "第一章 序論"
    ],
    'hindi': [
        "अध्याय 1: परिचय",
        "१.१ पृष्ठभूमि",
        "१. प्रस्तावना"
    ],
    'arabic': [
        "الفصل الأول: المقدمة",
        "1.1 الخلفية",
        "١.٢ الهدف"
    ],
    'chinese': [
        "第一章 引言",
        "1.1 背景",
        "一、概述",
        "第1章 导论"
    ]
}

# Enhanced language-specific heading confidence boosters with stricter patterns
HEADING_CONFIDENCE_BOOSTERS = {
    'japanese': {
        'patterns': [
            (r'^第\d+章', 0.9),           # Arabic numerals chapters - high confidence
            (r'^第[一二三四五六七八九十]+章', 0.9),  # Kanji chapters - high confidence
            (r'^第\d+節', 0.8),           # Arabic numerals sections
            (r'^第[一二三四五六七八九十]+節', 0.8),  # Kanji sections
            (r'^\d+\.\d+\s+[^\s]', 0.7),  # 1.1 with actual title
            (r'^[一二三四五六七八九十]+、', 0.6),  # Kanji enumeration
            (r'^（\d+）', 0.5),           # Parenthetical numbers
            (r'^(はじめに|序論|結論|まとめ|終章|付録|参考文献|謝辞)$', 0.8)  # Standard headings
        ],
        'keywords': [
            ('章', 0.7),
            ('節', 0.6),
            ('項', 0.5),
            ('第', 0.4),
            ('はじめに', 0.8),
            ('結論', 0.8),
            ('まとめ', 0.7)
        ],
        # Penalty patterns that reduce confidence
        'penalty_patterns': [
            (r'です。$', -0.8),           # Sentence endings
            (r'である。$', -0.8),         # Formal endings
            (r'^[•·]\s', -0.9),           # Bullet points
            (r'^表\d+', -0.7),           # Table captions
            (r'^図\d+', -0.7),           # Figure captions
        ]
    },
    'chinese': {
        'patterns': [
            (r'^第\d+章', 0.9),           # Arabic numerals chapters
            (r'^第[一二三四五六七八九十]+章', 0.9),  # Chinese numerals chapters
            (r'^第\d+节', 0.8),           # Arabic numerals sections
            (r'^第[一二三四五六七八九十]+节', 0.8),  # Chinese numerals sections
            (r'^\d+\.\d+\s+[^\s]', 0.7),  # 1.1 with actual title
            (r'^[一二三四五六七八九十]+、', 0.6),  # Chinese enumeration
            (r'^(引言|前言|结论|总结|附录|参考文献|致谢)$', 0.8)  # Standard headings
        ],
        'keywords': [
            ('章', 0.7),
            ('节', 0.6),
            ('部分', 0.5),
            ('引言', 0.8),
            ('结论', 0.8),
            ('总结', 0.7)
        ],
        # Penalty patterns
        'penalty_patterns': [
            (r'^[•·]\s', -0.9),           # Bullet points
            (r'^表\d+', -0.7),           # Table captions (Simplified)
            (r'^图\d+', -0.7),           # Figure captions (Simplified)
            (r'^圖\d+', -0.7),           # Figure captions (Traditional)
        ]
    },
    'hindi': {
        'patterns': [
            (r'^अध्याय\s+\d+', 0.8),
            (r'^खंड\s+[१२३४५६७८९०\d]+', 0.7),
            (r'^[१२३४५६७८९०]+\.', 0.6)
        ],
        'keywords': [
            ('अध्याय', 0.7),
            ('खंड', 0.6),
            ('भाग', 0.5)
        ]
    },
    'arabic': {
        'patterns': [
            (r'^الفصل\s+\w+', 0.8),
            (r'^القسم\s+\d+', 0.7),
            (r'^[\u0660-\u0669]+\.', 0.6)
        ],
        'keywords': [
            ('الفصل', 0.7),
            ('القسم', 0.6),
            ('الباب', 0.5)
        ]
    },
    # English patterns for comparison
    'english': {
        'patterns': [
            (r'^Chapter\s+\d+', 0.9),
            (r'^Section\s+\d+', 0.8),
            (r'^\d+\.\d+\s+[A-Z]', 0.7),  # 1.1 Title
            (r'^[IVX]+\.\s+[A-Z]', 0.7),  # Roman numerals
        ],
        'keywords': [
            ('chapter', 0.8),
            ('section', 0.6),
            ('introduction', 0.8),
            ('conclusion', 0.8)
        ]
    }
}

# Language-specific validation rules for heading detection
VALIDATION_RULES = {
    'japanese': {
        'min_confidence': 0.3,        # Minimum confidence for acceptance
        'max_length': 40,             # Maximum character length
        'min_length': 1,              # Minimum character length
        'required_cjk_ratio': 0.2,    # Minimum ratio of CJK characters
        'reject_sentence_endings': True,  # Reject obvious sentence endings
        'allow_short_cjk': True,      # Allow short text if CJK patterns match
    },
    'chinese': {
        'min_confidence': 0.3,
        'max_length': 40,
        'min_length': 1,
        'required_cjk_ratio': 0.2,
        'reject_sentence_endings': False,  # Chinese doesn't use 。 as much in headings
        'allow_short_cjk': True,
    },
    'english': {
        'min_confidence': 0.2,
        'max_length': 100,
        'min_length': 3,
        'required_alpha_ratio': 0.3,  # Minimum ratio of alphabetic characters
        'allow_short_patterns': False,
    },
    'default': {
        'min_confidence': 0.2,
        'max_length': 80,
        'min_length': 2,
        'required_alpha_ratio': 0.3,
    }
}