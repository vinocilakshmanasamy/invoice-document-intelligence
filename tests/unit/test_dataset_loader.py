
import json
from pathlib import Path

import pytest

from src.ocr.dataset_loader import InvoiceDataset


def create_dataset(root: Path):
    """Create a small temporary dataset for testing."""
    image_dir = root / "train" / "images"
    annotation_dir = root / "train" / "annotations"

    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    # Dummy image files are sufficient for testing file discovery.
    (image_dir / "invoice_001.jpg").write_bytes(b"dummy image")

    annotation = {
        "level": [5],
        "page_num": [1],
        "block_num": [1],
        "par_num": [1],
        "line_num": [1],
        "word_num": [1],
        "left": [10],
        "top": [20],
        "width": [30],
        "height": [15],
        "conf": [95],
        "text": ["Invoice"],
    }

    (annotation_dir / "invoice_001.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    return root


def test_loads_valid_dataset(tmp_path):
    root = create_dataset(tmp_path)

    dataset = InvoiceDataset(root=root, split="train")

    assert len(dataset) == 1
    assert dataset.records[0].invoice_id == "invoice_001"
    assert dataset.records[0].image_path.exists()
    assert dataset.records[0].annotation_path.exists()


def test_rejects_unsupported_split(tmp_path):
    with pytest.raises(ValueError, match="Unsupported dataset split"):
        InvoiceDataset(root=tmp_path, split="development")


def test_detects_missing_annotation(tmp_path):
    root = create_dataset(tmp_path)

    (root / "train" / "annotations" / "invoice_001.json").unlink()

    with pytest.raises(ValueError, match="Image-annotation mismatch"):
        InvoiceDataset(root=root, split="train")


def test_rejects_missing_directories(tmp_path):
    with pytest.raises(FileNotFoundError):
        InvoiceDataset(root=tmp_path, split="train")
