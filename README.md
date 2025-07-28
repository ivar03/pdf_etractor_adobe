# PDF Heading Extractor - Hybrid Approach

An advanced PDF heading detection system that combines fast heuristic analysis with intelligent semantic verification to accurately extract document structure.

## What This Project Does

This tool automatically extracts headings (Title, H1, H2, H3) from PDF documents and outputs them in a structured JSON format. It's specifically designed for the **Round 1A Hackathon Challenge** with optimizations for speed, accuracy, and multilingual support.

### Key Features

- **Hybrid Intelligence**: Combines Adobe-style structure detection with smart heuristics
- **Semantic Verification**: Uses lightweight AI models to verify heading candidates
- **Multilingual Support**: Handles English, Japanese, Hindi, Arabic, Chinese documents
- **Fast Processing**: Processes 50-page PDFs in under 10 seconds
- **High Accuracy**: Multi-strategy approach for superior heading detection
- **Robust Fallbacks**: Multiple detection strategies ensure reliability

## How It Works

Our hybrid approach uses a **three-stage pipeline**:

1. **Adobe-Style Detection**: First checks for native PDF structure/bookmarks
2. **Heuristic Analysis**: Fast font-based candidate generation using typography patterns
3. **Semantic Filtering**: AI-powered verification using contextual analysis

This combination delivers both **speed** and **accuracy** - faster than pure AI approaches, more accurate than simple heuristics.

## Quick Start

### Prerequisites

- Python 3.8+
- 4GB RAM minimum
- No GPU required (CPU optimized)

### Installation

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd pdf-heading-extractor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create required directories
mkdir -p data/models logs outputs

# 5. Download models (happens automatically on first run)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Basic Usage

```bash
# Extract headings from a PDF
python -m src.main document.pdf

# Save results to file
python -m src.main document.pdf --output results.json

# Use Round 1A hackathon format
python -m src.main document.pdf --round1a --output results.json

# Enable debug mode for detailed logging
python -m src.main document.pdf --debug

# Specify document language for better accuracy
python -m src.main document.pdf --language ja  # Japanese
python -m src.main document.pdf --language hi  # Hindi
```

## Step-by-Step Usage Guide

### Step 1: Prepare Your PDF
- Ensure your PDF file is readable (not password-protected)
- File size should be under 100MB for optimal performance
- Text-based PDFs work best (scanned images require OCR, not included)

### Step 2: Run the Extraction
```bash
# Basic extraction
python -m src.main your_document.pdf --round1a --output result.json
```

### Step 3: Check the Output
The system generates a JSON file in the **Round 1A format**:

```json
{
  "title": "Understanding AI Systems",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "What is AI?",
      "page": 2
    },
    {
      "level": "H3",
      "text": "History of AI",
      "page": 3
    }
  ]
}
```

### Step 4: Advanced Options

```bash
# Process multiple languages
python -m src.main japanese_doc.pdf --language ja --round1a --output result_ja.json

# Enable fast mode (skip semantic filtering for speed)
FAST_MODE=true python -m src.main document.pdf --round1a --output result.json

# Get detailed processing statistics
python -m src.main document.pdf --debug --round1a --output result.json
```

## Configuration Options

### Language Support
- `auto` - Automatic detection (default)
- `en` / `english` - English documents
- `ja` / `japanese` - Japanese documents  
- `hi` / `hindi` - Hindi documents
- `ar` / `arabic` - Arabic documents
- `zh` / `chinese` - Chinese documents

### Performance Modes
- **Standard Mode**: Full hybrid pipeline with semantic verification
- **Fast Mode**: `FAST_MODE=true` - Skip semantic filtering for maximum speed
- **Debug Mode**: `--debug` - Detailed logging and processing statistics

### Output Formats
- **Round 1A Format**: `--round1a` - Competition-specific JSON format
- **Extended Format**: Default - Includes confidence scores and metadata
- **Multiple Formats**: Supports JSON, CSV, XML, HTML, Markdown

## Expected Performance

| Document Type | Processing Time | Accuracy | Multilingual Bonus |
|---------------|----------------|----------|-------------------|
| Academic Papers | 2-5 seconds | 90-95% | ✅ Full Support |
| Technical Manuals | 3-7 seconds | 85-92% | ✅ Full Support |
| Books/Reports | 4-8 seconds | 88-94% | ✅ Full Support |
| Complex Layouts | 6-10 seconds | 82-90% | ✅ Full Support |

## Hackathon Optimizations

### Round 1A Compliance
- **≤10 seconds** processing time for 50-page PDFs
- **≤200MB** total model size (MiniLM is only ~80MB)
- **CPU-only** operation (no GPU dependencies)
- **Offline mode** (all models cached locally)
- **AMD64 compatible** architecture

### Competitive Advantages
- **Multilingual Support**: Handles Japanese, Hindi, Arabic for bonus points
- **Hybrid Intelligence**: More accurate than pure heuristics, faster than pure AI
- **Robust Fallbacks**: Multiple strategies ensure reliability across document types
- **Cultural Intelligence**: Understands document patterns across languages

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# If you get import errors, use structured execution:
python -m src.main document.pdf
```

**Model Download Issues:**
```bash
# Manually download models:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Memory Issues:**
```bash
# Use fast mode to reduce memory usage:
FAST_MODE=true python src/main.py document.pdf --round1a
```

### Debug Mode
Enable debug mode for detailed diagnostics:
```bash
python -m src.main document.pdf --debug --round1a --output result.json
```

This provides:
- Processing stage timings
- Font analysis details
- Candidate generation statistics
- Semantic filtering scores
- Hierarchy assignment logic

## Architecture Overview

```
PDF Input → Structure Check → Candidate Generation → Semantic Filtering → Hierarchy Assignment → JSON Output
     ↓           ↓                    ↓                     ↓                      ↓              ↓
   Validate   Adobe TOC         Font Analysis        AI Verification      Level Assignment   Round1A Format
              Detection         Layout Patterns      Context Analysis     Cultural Rules     
```

## What Makes This System Unique

1. **Adobe-Grade Intelligence**: First checks for native PDF structure like professional tools
2. **Cultural Awareness**: Understands document patterns across different languages and cultures
3. **Adaptive Thresholds**: Adjusts detection sensitivity based on document characteristics
4. **Multi-Strategy Ensemble**: Combines 5+ detection strategies for maximum reliability
5. **Production Ready**: Comprehensive error handling, validation, and logging

## Performance Monitoring

The system provides detailed statistics:
- Processing time breakdown by stage
- Confidence scores for each detected heading
- Font analysis and layout complexity assessment
- Semantic filtering effectiveness metrics
- Memory usage and cache performance

## Ready for Production

This system is designed for:
- **Scalability**: Batch processing with parallel workers
- **Reliability**: Comprehensive error handling and fallbacks
- **Monitoring**: Detailed logging and performance metrics
- **Flexibility**: Multiple output formats and configuration options

Perfect for document processing pipelines, content management systems, and AI-powered document analysis workflows.