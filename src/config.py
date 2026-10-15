from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset paths (kept outside Git)
DATA_PATH = PROJECT_ROOT / "data" / "train"
IMAGES_PATH = DATA_PATH / "images"
ANNOTATIONS_PATH = DATA_PATH / "annotations"

# Qdrant configuration
COLLECTION_NAME = "invoice_document"
QDRANT_PATH = PROJECT_ROOT / "qdrant_storage"

# OCR evaluation
OCR_CONFIDENCE_THRESHOLD = 0.5
TEXT_SIMILARITY_THRESHOLD = 0.8
IOU_THRESHOLD = 0.5

# RAG chunking configuration
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

# Ground-truth evaluation
USE_MANUAL_GROUND_TRUTH = True
