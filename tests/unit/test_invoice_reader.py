
import json
from pathlib import Path

from src.ocr.dataset_loader import InvoiceRecord
from src.ocr.invoice_reader import load_invoice


def test_load_invoice_extracts_words(tmp_path: Path):
    image_path = tmp_path / "invoice_001.jpg"
    annotation_path = tmp_path / "invoice_001.json"

    image_path.write_bytes(b"dummy image")

    annotation = {
        "level": [1, 5, 5],
        "left": [0, 10, 50],
        "top": [0, 20, 20],
        "width": [100, 30, 25],
        "height": [100, 15, 15],
        "conf": [-1, 95, 90],
        "text": ["", "Invoice", "Number"],
    }

    annotation_path.write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    record = InvoiceRecord(
        invoice_id="invoice_001",
        image_path=image_path,
        annotation_path=annotation_path,
        split="train",
    )

    invoice = load_invoice(record)

    assert invoice.invoice_id == "invoice_001"
    assert invoice.split == "train"
    assert invoice.image_path == image_path
    assert len(invoice.words) == 2
    assert invoice.words[0]["text"] == "Invoice"
    assert invoice.words[0]["bbox"] == [10, 20, 40, 35]
    assert invoice.words[0]["conf"] == 0.95
