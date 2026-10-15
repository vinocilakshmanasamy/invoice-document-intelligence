
from pathlib import Path

from src.ocr.annotation_loader import load_annotation
from src.ocr.dataset_loader import InvoiceDataset


def validate_split(dataset_root: Path, split: str) -> dict:
    """Validate image-annotation pairs and annotation contents."""

    dataset = InvoiceDataset(root=dataset_root, split=split)

    valid_count = 0
    empty_word_count = 0
    errors = []

    for record in dataset:
        try:
            annotation = load_annotation(record.annotation_path)

            if not record.image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {record.image_path}"
                )

            if not any(
                level == 5
                and isinstance(text, str)
                and text.strip()
                for level, text in zip(
                    annotation["level"],
                    annotation["text"],
                )
            ):
                empty_word_count += 1

            valid_count += 1

        except Exception as exc:
            errors.append({
                "invoice_id": record.invoice_id,
                "error": str(exc),
            })

    return {
        "split": split,
        "total": len(dataset),
        "valid": valid_count,
        "without_word_text": empty_word_count,
        "errors": errors,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data"

    overall_errors = 0

    for split in ("train", "val", "test"):
        result = validate_split(dataset_root, split)

        print(f"\n--- {result['split'].upper()} ---")
        print(f"Total invoices: {result['total']}")
        print(f"Successfully validated: {result['valid']}")
        print(f"Invoices without word text: {result['without_word_text']}")
        print(f"Errors: {len(result['errors'])}")

        for error in result["errors"][:10]:
            print(f"  {error['invoice_id']}: {error['error']}")

        overall_errors += len(result["errors"])

    print("\n--- OVERALL ---")
    print(f"Total errors: {overall_errors}")

    if overall_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
