import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, find_packages

# Package information
PACKAGE_NAME = "pdf-heading-extractor"
VERSION = "1.0.0"
AUTHOR = ""
AUTHOR_EMAIL = ""
DESCRIPTION = "Advanced PDF heading extraction using hybrid heuristic and semantic approach"
URL = ""

# Read long description from README
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return DESCRIPTION

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Core dependencies (always required)
CORE_REQUIREMENTS = [
    "PyMuPDF>=1.23.0",
    "pdfplumber>=0.9.0",
    "sentence-transformers>=2.2.2",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "click>=8.1.0",
    "tqdm>=4.65.0",
    "python-json-logger>=2.0.0",
]

# Optional dependencies for enhanced features
OPTIONAL_REQUIREMENTS = {
    "nlp": [
        "nltk>=3.8.0",
        "spacy>=3.6.0",
        "langdetect>=1.0.9",
    ],
    "validation": [
        "python-magic>=0.4.27",
        "python-magic-bin>=0.4.14; platform_system=='Windows'",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
    ]
}

# Combine all optional requirements for 'all' extra
ALL_OPTIONAL = []
for deps in OPTIONAL_REQUIREMENTS.values():
    ALL_OPTIONAL.extend(deps)
OPTIONAL_REQUIREMENTS["all"] = ALL_OPTIONAL

def get_install_requires():
    """Get core requirements with platform-specific adjustments."""
    requirements = CORE_REQUIREMENTS.copy()
    
    # Platform-specific requirements
    if platform.system() == "Windows":
        requirements.append("python-magic-bin>=0.4.14")
    else:
        requirements.append("python-magic>=0.4.27")
    
    return requirements

def check_system_requirements():
    """Check system requirements and warn about potential issues."""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("WARNING: Python 3.8+ is required. Current version:", sys.version)
        sys.exit(1)
    
    # Check available disk space (approximate)
    try:
        import shutil
        free_space_gb = shutil.disk_usage(".").free / (1024**3)
        if free_space_gb < 5:
            print(f"WARNING: Low disk space ({free_space_gb:.1f}GB). Models require ~2GB.")
    except Exception:
        pass
    
    # Check if CUDA is available (for GPU acceleration)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA detected: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ No CUDA detected - will use CPU (slower but functional)")
    except ImportError:
        pass  # torch not installed yet
    
    print("System requirements check completed.")

def post_install_setup():
    """Perform post-installation setup tasks."""
    print("\nPerforming post-installation setup...")
    
    # Create necessary directories
    directories = [
        "data",
        "data/models",
        "data/sample_pdfs",
        "data/test_cases",
        "outputs",
        "outputs/json",
        "outputs/debug",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Download NLTK data if nltk is available
    try:
        import nltk
        print("Downloading NLTK data...")
        
        nltk_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        for data_name in nltk_data:
            try:
                nltk.download(data_name, quiet=True)
                print(f"✓ Downloaded NLTK data: {data_name}")
            except Exception as e:
                print(f"⚠ Failed to download NLTK data '{data_name}': {e}")
    
    except ImportError:
        print("ℹ NLTK not installed - skipping NLTK data download")
    
    # Download spaCy model if spacy is available
    try:
        import spacy
        print("Downloading spaCy English model...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], check=True, capture_output=True)
            print("✓ Downloaded spaCy English model")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Failed to download spaCy model: {e}")
    
    except ImportError:
        print("ℹ spaCy not installed - skipping spaCy model download")
    
    # Pre-download sentence transformer model
    try:
        from sentence_transformers import SentenceTransformer
        print("Pre-downloading sentence transformer model...")
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            model = SentenceTransformer(model_name)
            print(f"✓ Downloaded model: {model_name}")
        except Exception as e:
            print(f"⚠ Failed to download model: {e}")
    
    except ImportError:
        print("ℹ sentence-transformers not installed - skipping model download")
    
    print("Post-installation setup completed!")

def create_config_file():
    """Create default configuration file."""
    config_content = '''# PDF Heading Extractor Configuration
# This file contains default settings for the PDF heading extractor

[processing]
max_processing_time = 10
max_file_size_mb = 100
batch_size = 32

[detection]
font_size_threshold_ratio = 1.2
bold_weight_threshold = 600
min_heading_length = 3
max_heading_length = 200

[semantic]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
semantic_similarity_threshold = 0.3
context_window = 2

[output]
output_format = "json"
include_confidence_scores = true
include_debug_info = false

[logging]
log_level = "INFO"
log_file = "logs/pdf_extractor.log"

[models]
cache_embeddings = true
model_cache_dir = "data/models"
'''
    
    config_path = Path("config.ini")
    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"✓ Created configuration file: {config_path}")

class PostInstallCommand:
    """Custom command to run post-installation setup."""
    
    def run(self):
        check_system_requirements()
        post_install_setup()
        create_config_file()
        
        print("\n" + "="*60)
        print("🎉 PDF Heading Extractor installation completed!")
        print("="*60)
        print("\nQuick start:")
        print("  python -m src.main document.pdf")
        print("  python -m src.main document.pdf --output results.json")
        print("  python -m src.main document.pdf --debug")
        print("\nFor help:")
        print("  python -m src.main --help")
        print("\nDocumentation:")
        print("  See README.md for detailed usage instructions")
        print("="*60)

# Custom setup commands
class DevelopCommand:
    """Setup development environment."""
    
    def run(self):
        print("Setting up development environment...")
        
        # Install pre-commit hooks
        try:
            subprocess.run([sys.executable, "-m", "pre_commit", "install"], check=True)
            print("✓ Installed pre-commit hooks")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ Failed to install pre-commit hooks")
        
        # Create development directories
        dev_dirs = [
            "tests/data",
            "tests/fixtures",
            "docs/source",
            "benchmarks",
            "examples"
        ]
        
        for directory in dev_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created dev directory: {directory}")
        
        print("Development environment setup completed!")

def main():
    """Main setup function."""
    
    # Handle custom commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "post_install":
            PostInstallCommand().run()
            return
        elif sys.argv[1] == "develop":
            DevelopCommand().run()
            return
    
    # Regular setuptools setup
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url=URL,
        packages=find_packages(include=["src", "src.*", "config", "utils"]),
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Text Processing :: Linguistic",
        ],
        python_requires=">=3.8",
        install_requires=get_install_requires(),
        extras_require=OPTIONAL_REQUIREMENTS,
        entry_points={
            "console_scripts": [
                "pdf-extract-headings=src.main:main",
                "pdf-heading-extractor=src.main:main",
            ],
        },
        include_package_data=True,
        package_data={
            "config": ["*.py", "*.json"],
            "data": ["sample_pdfs/*", "test_cases/*"],
        },
        zip_safe=False,
        keywords=[
            "pdf", "heading", "extraction", "nlp", "document-analysis", 
            "text-mining", "machine-learning", "hybrid-approach"
        ],
        project_urls={
            "Bug Reports": f"{URL}/issues",
            "Source": URL,
            "Documentation": f"{URL}/docs",
        },
    )

if __name__ == "__main__":
    main()