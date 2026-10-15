import pytest

from src.ocr.annotation_loader import (
    extract_word_entries,
    load_annotation,
)


def test_normalizes_confidence_to_zero_one():
    annotation = {
        "level": [5, 5, 5, 5],
        "left": [0, 10, 20, 30],
        "top": [0, 10, 20, 30],
        "width": [5, 5, 5, 5],
        "height": [5, 5, 5, 5],
        "conf": [95, 50, -1, 105],
        "text": ["Invoice", "Total", "Amount", "Tax"],
    }

    words = extract_word_entries(annotation)

    assert words[0]["conf"] == 0.95
    assert words[1]["conf"] == 0.50
    assert words[2]["conf"] is None
    assert words[3]["conf"] is None


def test_load_annotation_from_json(tmp_path):
    annotation_file = tmp_path / "annotation.json"
    annotation_file.write_text(
        """
        {
            "level": [1, 5],
            "left": [0, 10],
            "top": [0, 20],
            "width": [100, 30],
            "height": [100, 15],
            "conf": [-1, 95],
            "text": ["", "Invoice"]
        }
        """,
        encoding="utf-8",
    )

    annotation = load_annotation(annotation_file)

    assert annotation["text"] == ["", "Invoice"]


def test_reject_missing_required_field(tmp_path):
    annotation_file = tmp_path / "invalid.json"
    annotation_file.write_text(
        '{"level": [], "text": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing annotation fields"):
        load_annotation(annotation_file)


def test_reject_mismatched_array_lengths(tmp_path):
    annotation_file = tmp_path / "mismatch.json"
    annotation_file.write_text(
        """
        {
            "level": [5, 5],
            "left": [10],
            "top": [20, 30],
            "width": [30, 40],
            "height": [15, 15],
            "conf": [95, 90],
            "text": ["Invoice", "Number"]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="array lengths do not match"):
        load_annotation(annotation_file)


def test_extract_word_entries_and_convert_boxes():
    annotation = {
        "level": [1, 5, 5, 5],
        "left": [0, 10, 50, 80],
        "top": [0, 20, 20, 20],
        "width": [100, 30, 25, 10],
        "height": [100, 15, 15, 15],
        "conf": [-1, 95, 90, 85],
        "text": ["", "Invoice", "Number", "   "],
    }

    words = extract_word_entries(annotation)

    assert len(words) == 2
    assert words[0] == {
    "text": "Invoice",
    "bbox": [10, 20, 40, 35],
    "conf": 0.95,
       }
    assert words[1]["text"] == "Number"


def test_extract_word_entries_ignores_non_word_levels():
    annotation = {
        "level": [1, 4, 5],
        "left": [0, 10, 20],
        "top": [0, 10, 20],
        "width": [100, 50, 30],
        "height": [100, 20, 15],
        "conf": [-1, -1, 95],
        "text": ["", "Invoice line", "Invoice"],
    }

    words = extract_word_entries(annotation)

    assert [word["text"] for word in words] == ["Invoice"]
