
from pathlib import Path

from src.ocr.annotation_loader import (
    extract_word_entries,
    load_annotation,
)
from src.ocr.dataset_loader import InvoiceDataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data"

    total = 0

    for split in ("train", "val", "test"):
        dataset = InvoiceDataset(dataset_root, split)

        for record in dataset:
            annotation = load_annotation(record.annotation_path)
            words = extract_word_entries(annotation)

            if not words:
                total += 1
                print(f"\nSplit: {split}")
                print(f"Invoice ID: {record.invoice_id}")
                print(f"Image: {record.image_path}")
                print(f"Annotation: {record.annotation_path}")
                print(
                    "Annotation entries:",
                    len(annotation["level"]),
                )
                print(
                    "Word-level entries:",
                    sum(level == 5 for level in annotation["level"]),
                )

    print(f"\nTotal invoices without extracted words: {total}")


if __name__ == "__main__":
    main()
