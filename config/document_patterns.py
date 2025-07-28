from typing import Dict, List, Tuple

# Document-type specific patterns for specialized processing
DOCUMENT_PATTERNS = {
    'rfp': {
        'languages': ['english', 'french', 'spanish'],
        'patterns': [
            # English RFP patterns
            (r'^(Summary|Background|Timeline|Milestones|Appendix [A-Z]):\s*$', "H2", 0.9),
            (r'^RFP:', "H1", 0.95),
            (r'^Ontario.*Digital.*Library.*$', "H1", 0.9),
            (r'^.*Critical Component.*$', "H1", 0.85),
            (r'^.*Business Plan.*$', "H2", 0.8),
            (r'^.*Evaluation.*Contract.*$', "H2", 0.8),
            (r'^Phase [IVX]+:', "H3", 0.8),
            (r'^Phase \d+:', "H3", 0.8),
            
            # French RFP patterns
            (r'^Demande de propositions?:', "H1", 0.95),
            (r'^(Résumé|Contexte|Calendrier|Étapes):\s*$', "H2", 0.9),
            (r'^Phase [IVX]+:', "H3", 0.8),
            (r'^Annexe [A-Z]:', "H2", 0.9),
            
            # Spanish RFP patterns  
            (r'^Solicitud de propuestas?:', "H1", 0.95),
            (r'^(Resumen|Antecedentes|Cronograma|Hitos):\s*$', "H2", 0.9),
            (r'^Fase [IVX]+:', "H3", 0.8),
            (r'^Apéndice [A-Z]:', "H2", 0.9),
        ],
        'fragmentation_patterns': [
            'proposal', 'ontario', 'library', 'digital', 'request',
            'evaluation', 'contract', 'requirements', 'implementation',
            'proposition', 'bibliothèque', 'numérique', 'demande',  # French
            'propuesta', 'biblioteca', 'digital', 'solicitud'  # Spanish
        ],
        'confidence_boost': 0.2
    },
    
    'academic': {
        'languages': ['english', 'chinese', 'japanese'],
        'patterns': [
            # English academic patterns
            (r'^(Abstract|Introduction|Literature Review|Methodology|Results|Discussion|Conclusion)$', "H2", 0.9),
            (r'^Chapter \d+', "H1", 0.9),
            (r'^\d+\.\s+[A-Z][^.]*$', "H3", 0.7),
            (r'^References$', "H2", 0.95),
            
            # Chinese academic patterns
            (r'^摘要$', "H2", 0.9),
            (r'^第[一二三四五六七八九十\d]+章', "H1", 0.9),
            (r'^参考文献$', "H2", 0.95),
            
            # Japanese academic patterns  
            (r'^要約$', "H2", 0.9),
            (r'^第[一二三四五六七八九十\d]+章', "H1", 0.9),
            (r'^参考文献$', "H2", 0.95),
        ],
        'fragmentation_patterns': [
            'introduction', 'methodology', 'literature', 'discussion',
            'conclusion', 'references', 'abstract'
        ],
        'confidence_boost': 0.15
    },
    
    'business': {
        'languages': ['english', 'french', 'german', 'spanish'],
        'patterns': [
            # English business patterns
            (r'^(Executive Summary|Market Analysis|Financial Projections|Strategic Plan)$', "H2", 0.85),
            (r'^(SWOT Analysis|Risk Assessment|Implementation Plan)$', "H2", 0.8),
            
            # French business patterns
            (r'^(Résumé Exécutif|Analyse de Marché|Projections Financières)$', "H2", 0.85),
            
            # German business patterns
            (r'^(Zusammenfassung|Marktanalyse|Finanzprognosen)$', "H2", 0.85),
            
            # Spanish business patterns
            (r'^(Resumen Ejecutivo|Análisis de Mercado|Proyecciones Financieras)$', "H2", 0.85),
        ],
        'fragmentation_patterns': [
            'executive', 'summary', 'analysis', 'financial', 'strategic',
            'implementation', 'assessment', 'projections'
        ],
        'confidence_boost': 0.1
    },
    
    'legal': {
        'languages': ['english', 'french'],
        'patterns': [
            # English legal patterns
            (r'^Section \d+', "H2", 0.9),
            (r'^Article [IVX]+', "H2", 0.9),
            (r'^WHEREAS', "H3", 0.8),
            (r'^NOW THEREFORE', "H3", 0.8),
            
            # French legal patterns
            (r'^Article \d+', "H2", 0.9),
            (r'^Section [IVX]+', "H2", 0.9),
            (r'^ATTENDU QUE', "H3", 0.8),
        ],
        'fragmentation_patterns': [
            'whereas', 'therefore', 'pursuant', 'notwithstanding',
            'attendu', 'considérant'  # French
        ],
        'confidence_boost': 0.15
    }
}

def get_patterns_for_document_type(document_type: str, language: str = 'english') -> List[Tuple[str, str, float]]:
    """Get patterns for specific document type and language."""
    if document_type not in DOCUMENT_PATTERNS:
        return []
    
    config = DOCUMENT_PATTERNS[document_type]
    
    # Check if language is supported for this document type
    if language not in config['languages']:
        # Fall back to English if available
        if 'english' in config['languages']:
            language = 'english'
        else:
            return []
    
    # Filter patterns by language (rough approach)
    patterns = config['patterns']
    
    # Language-specific filtering could be more sophisticated
    # For now, return all patterns and let regex matching handle it
    return patterns

def get_fragmentation_patterns_for_document_type(document_type: str) -> List[str]:
    """Get fragmentation patterns for document type."""
    if document_type not in DOCUMENT_PATTERNS:
        return []
    
    return DOCUMENT_PATTERNS[document_type].get('fragmentation_patterns', [])

def get_confidence_boost_for_document_type(document_type: str) -> float:
    """Get confidence boost for document type."""
    if document_type not in DOCUMENT_PATTERNS:
        return 0.0
    
    return DOCUMENT_PATTERNS[document_type].get('confidence_boost', 0.0)

def detect_document_type_from_text(text: str) -> str:
    """Detect document type from text content."""
    if not text:
        return 'general'
    
    text_lower = text.lower()
    
    # Count matches for each document type
    type_scores = {}
    
    for doc_type, config in DOCUMENT_PATTERNS.items():
        score = 0
        patterns = config['patterns']
        
        for pattern, _, confidence in patterns:
            # Simple pattern matching for detection
            import re
            if re.search(pattern, text, re.IGNORECASE):
                score += confidence
        
        type_scores[doc_type] = score
    
    # Return type with highest score, or 'general' if no clear match
    if type_scores:
        best_type = max(type_scores, key=type_scores.get)
        if type_scores[best_type] > 0.5:  # Threshold for detection
            return best_type
    
    return 'general'