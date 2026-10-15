from src.config import (
    PROJECT_ROOT,
    DATA_PATH,
    IMAGES_PATH,
    ANNOTATIONS_PATH,
    COLLECTION_NAME,
    QDRANT_PATH,
    OCR_CONFIDENCE_THRESHOLD,
    TEXT_SIMILARITY_THRESHOLD,
    IOU_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    USE_MANUAL_GROUND_TRUTH,
)


def test_project_root_is_correct():
    assert PROJECT_ROOT.name == "invoice-document-intelligence"


def test_dataset_paths_are_correct():
    assert DATA_PATH == PROJECT_ROOT / "data" / "train"
    assert IMAGES_PATH == DATA_PATH / "images"
    assert ANNOTATIONS_PATH == DATA_PATH / "annotations"


def test_qdrant_configuration():
    assert COLLECTION_NAME == "invoice_document"
    assert QDRANT_PATH == PROJECT_ROOT / "qdrant_storage"


def test_evaluation_thresholds():
    assert OCR_CONFIDENCE_THRESHOLD == 0.5
    assert TEXT_SIMILARITY_THRESHOLD == 0.8
    assert IOU_THRESHOLD == 0.5


def test_rag_chunking_configuration():
    assert CHUNK_SIZE == 100
    assert CHUNK_OVERLAP == 20


def test_manual_ground_truth_is_enabled():
    assert USE_MANUAL_GROUND_TRUTH is True
