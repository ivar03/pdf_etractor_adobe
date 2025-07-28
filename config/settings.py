import os
from pathlib import Path

# Base Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
JSON_OUTPUT_DIR = OUTPUT_DIR / "json"
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
DEBUG_OUTPUT_DIR = OUTPUT_DIR / "debug"
MODEL_DIR = DATA_DIR / "models"

# Processing Configuration
MAX_PROCESSING_TIME = 20  # seconds
BATCH_SIZE = 32
MAX_FILE_SIZE_MB = 100

# Font Analysis Thresholds
# Make font thresholds more inclusive
FONT_SIZE_THRESHOLD_RATIO = 1.05  # No fixed ratio, use percentile-based
MIN_FONT_SIZE_FOR_HEADING = 8    # Lower absolute minimum
RELATIVE_SIZE_BONUS = 1.0        # Remove fixed bonus
BOLD_WEIGHT_THRESHOLD = 400      # Lower threshold
MIN_HEADING_LENGTH = 2           # Allow shorter headings
MAX_HEADING_LENGTH = 300         # Allow longer headings

# Semantic Filtering
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v1" #till now we can use either of the above all give similar accuracy with this one's being slightly higher cause of embedding dim
#EMBEDDING_MODEL = "sentence-transformers/nli-distilroberta-base-v2" #bigger model gives similar results so shifted to the minilml12v1

SEMANTIC_SIMILARITY_THRESHOLD = 0.5
CONTEXT_WINDOW = 3  # paragraphs before/after

# Hierarchy Assignment
MAX_HIERARCHY_LEVELS = 6
TITLE_POSITION_THRESHOLD = 0.1  # top 10% of page
CENTER_ALIGNMENT_TOLERANCE = 0.1

# Output Configuration
OUTPUT_FORMAT = "json"
INCLUDE_CONFIDENCE_SCORES = True
INCLUDE_DEBUG_INFO = False