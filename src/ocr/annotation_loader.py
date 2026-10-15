import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "level",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)


def load_annotation(annotation_path: Path) -> dict[str, Any]:
    """Load and validate a Tesseract-style OCR annotation JSON file."""
    annotation_path = Path(annotation_path)

    with annotation_path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)

    if not isinstance(annotation, dict):
        raise ValueError("Annotation JSON must contain an object.")

    missing_fields = [
        field for field in REQUIRED_FIELDS
        if field not in annotation
    ]
    if missing_fields:
        raise ValueError(f"Missing annotation fields: {missing_fields}")

    lengths = {
        field: len(annotation[field])
        for field in REQUIRED_FIELDS
        if isinstance(annotation[field], list)
    }

    if len(lengths) != len(REQUIRED_FIELDS):
        raise ValueError("All required annotation fields must be lists.")

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Annotation array lengths do not match: {lengths}")

    return annotation


def extract_word_entries(
    annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract non-empty entries marked as word-level (level 5)."""
    words = []

    for i, level in enumerate(annotation["level"]):
        text = annotation["text"][i]

        if level == 5 and isinstance(text, str) and text.strip():
            left = annotation["left"][i]
            top = annotation["top"][i]
            width = annotation["width"][i]
            height = annotation["height"][i]

            
            raw_confidence = annotation["conf"][i]

            if (
                isinstance(raw_confidence, (int, float))
                and not isinstance(raw_confidence, bool)
                and 0 <= raw_confidence <= 100
            ):
                confidence = raw_confidence / 100.0
            else:
                confidence = None

            words.append({
                "text": text,
                "bbox": [left, top, left + width, top + height],
                "conf": confidence,
            })


    return words
