"""Document text extraction for PDF and TXT files."""

from pathlib import Path

import pymupdf

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentExtractionError(Exception):
    """Base exception for document extraction failures."""


class UnsupportedFileTypeError(DocumentExtractionError):
    """Raised when the file type is not supported."""


class EmptyDocumentError(DocumentExtractionError):
    """Raised when a document contains no extractable text."""


class ExtractionFailureError(DocumentExtractionError):
    """Raised when extraction fails due to invalid or corrupt content."""


def extract_text(filename: str, content: bytes) -> str:
    """Extract raw text from PDF or TXT file content."""
    if not content:
        raise EmptyDocumentError(f"File '{filename}' is empty")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or 'unknown'}' for '{filename}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        return _extract_pdf(filename, content)
    return _extract_txt(filename, content)


def extract_text_from_path(path: Path | str) -> str:
    """Extract text from a file on disk."""
    file_path = Path(path)
    return extract_text(filename=file_path.name, content=file_path.read_bytes())


def _extract_pdf(filename: str, content: bytes) -> str:
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            pages = [page.get_text() for page in document]
    except Exception as error:
        raise ExtractionFailureError(
            f"Failed to parse PDF file '{filename}': {error}"
        ) from error

    text = "".join(pages).strip()
    if not text:
        raise EmptyDocumentError(
            f"PDF file '{filename}' contains no extractable text"
        )
    return text


def _extract_txt(filename: str, content: bytes) -> str:
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise ExtractionFailureError(
            f"TXT file '{filename}' is not valid UTF-8: {error}"
        ) from error

    if not text:
        raise EmptyDocumentError(f"TXT file '{filename}' contains no text")
    return text
