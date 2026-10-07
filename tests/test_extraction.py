"""Tests for document text extraction."""

import pymupdf
import pytest

from app.services.documents.extractor import (
    EmptyDocumentError,
    ExtractionFailureError,
    UnsupportedFileTypeError,
    extract_text,
    extract_text_from_path,
)


def _make_pdf(text: str = "") -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    return document.tobytes()


def test_txt_extraction() -> None:
    text = extract_text("notes.txt", "Hello TXT file.".encode("utf-8"))
    assert text == "Hello TXT file."


def test_txt_extraction_strips_bom() -> None:
    content = "﻿Hello with BOM.".encode("utf-8")
    assert extract_text("notes.txt", content) == "Hello with BOM."


def test_pdf_extraction() -> None:
    content = _make_pdf("Hello from PDF.")
    assert extract_text("report.pdf", content) == "Hello from PDF."


def test_unsupported_file_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("slides.docx", b"not supported")


def test_empty_content_rejected() -> None:
    with pytest.raises(EmptyDocumentError):
        extract_text("empty.txt", b"")


def test_whitespace_only_txt_rejected() -> None:
    with pytest.raises(EmptyDocumentError):
        extract_text("blank.txt", b"   \n\t  ")


def test_pdf_without_text_rejected() -> None:
    with pytest.raises(EmptyDocumentError):
        extract_text("empty_page.pdf", _make_pdf())


def test_malformed_pdf_raises_typed_error() -> None:
    with pytest.raises(ExtractionFailureError):
        extract_text("broken.pdf", b"this is not a real pdf")


def test_invalid_utf8_txt_raises_typed_error() -> None:
    with pytest.raises(ExtractionFailureError):
        extract_text("bad.txt", b"\xff\xfe\x00\x01")


def test_extract_text_from_path(tmp_path) -> None:
    file_path = tmp_path / "disk.txt"
    file_path.write_text("Read from disk.", encoding="utf-8")
    assert extract_text_from_path(file_path) == "Read from disk."
