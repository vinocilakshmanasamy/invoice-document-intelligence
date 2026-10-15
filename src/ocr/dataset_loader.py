
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}


@dataclass(frozen=True)
class InvoiceRecord:
    invoice_id: str
    image_path: Path
    annotation_path: Path
    split: str


class InvoiceDataset:
    def __init__(self, root: Path, split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset split: {split}")

        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / "images"
        self.annotation_dir = self.root / split / "annotations"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Annotation directory not found: {self.annotation_dir}"
            )

        self.records = self._build_records()

    def _build_records(self) -> list[InvoiceRecord]:
        images = {
            path.stem: path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        annotations = {
            path.stem: path
            for path in self.annotation_dir.glob("*.json")
            if path.is_file()
        }

        missing_annotations = images.keys() - annotations.keys()
        missing_images = annotations.keys() - images.keys()

        if missing_annotations or missing_images:
            raise ValueError(
                "Image-annotation mismatch. "
                f"Missing annotations: {sorted(missing_annotations)[:5]}; "
                f"missing images: {sorted(missing_images)[:5]}"
            )

        return [
            InvoiceRecord(
                invoice_id=stem,
                image_path=images[stem],
                annotation_path=annotations[stem],
                split=self.split,
            )
            for stem in sorted(images)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def get_record(self, invoice_id: str) -> InvoiceRecord:
        for record in self.records:
            if record.invoice_id == invoice_id:
                return record

        raise KeyError(f"Invoice not found: {invoice_id}")
