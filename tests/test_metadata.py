"""Smoke tests for metadata validation."""

import json

import pytest

from childlanguagenet.ingestion.metadata_registry import validate_metadata


def _write_meta(records, tmpdir):
    path = tmpdir / "metadata.json"
    path.write_text(json.dumps(records))
    return path


def test_valid_metadata(tmp_path):
    """Valid metadata with a PDF file passes validation."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    records = [
        {
            "paper_id": "p1",
            "title": "Test Paper",
            "authors": ["Author A"],
            "year": 2024,
            "pdf_file": "test.pdf",
        }
    ]
    path = _write_meta(records, tmp_path)
    result = validate_metadata(path, data_dir=tmp_path)
    assert len(result) == 1
    assert result[0].id == "p1"


def test_duplicate_ids(tmp_path):
    """Duplicate paper_id raises ValueError."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    records = [
        {"paper_id": "dup", "title": "A", "pdf_file": "test.pdf"},
        {"paper_id": "dup", "title": "B", "pdf_file": "test.pdf"},
    ]
    path = _write_meta(records, tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        validate_metadata(path, data_dir=tmp_path)


def test_missing_required_field(tmp_path):
    records = [{"paper_id": "x"}]  # missing title
    path = _write_meta(records, tmp_path)
    with pytest.raises(ValueError, match="missing required"):
        validate_metadata(path, data_dir=tmp_path)


def test_bad_url_scheme(tmp_path):
    records = [
        {"paper_id": "u1", "title": "T", "source_url": "ftp://bad.example.com"}
    ]
    path = _write_meta(records, tmp_path)
    with pytest.raises(ValueError, match="http"):
        validate_metadata(path, data_dir=tmp_path)
