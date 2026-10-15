
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ocr.annotation_loader import (
    extract_word_entries,
    load_annotation,
)
from src.ocr.dataset_loader import InvoiceRecord


@dataclass(frozen=True)
class LoadedInvoice:
    """A loaded invoice with its source files and extracted words."""

    invoice_id: str
    split: str
    image_path: Path
    annotation_path: Path
    words: list[dict[str, Any]]


def load_invoice(record: InvoiceRecord) -> LoadedInvoice:
    """Load and parse the annotation associated with an invoice."""

    annotation = load_annotation(record.annotation_path)
    words = extract_word_entries(annotation)

    return LoadedInvoice(
        invoice_id=record.invoice_id,
        split=record.split,
        image_path=record.image_path,
        annotation_path=record.annotation_path,
        words=words,
    )
