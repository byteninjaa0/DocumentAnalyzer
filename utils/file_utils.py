"""
File handling: extract text from uploaded PDF and text files.
Deployment-friendly: no hardcoded paths; works with file-like objects.
"""

import re
from typing import BinaryIO


def _normalize_text(text: str) -> str:
    """Collapse whitespace and strip for consistent context."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_text_from_pdf(file) -> str:
    """
    Extract text from a PDF file. Accepts file path (str/pathlib) or file-like object.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("PDF support requires: pip install pypdf")

    reader = PdfReader(file)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return _normalize_text("\n".join(parts))


def extract_text_from_upload(uploaded_file) -> str:
    """
    Extract text from a Streamlit UploadedFile or file-like object.
    Supports: .pdf, .txt, .md, .csv (first column or raw).
    """
    name = getattr(uploaded_file, "name", "") or ""
    suffix = name.lower().split(".")[-1] if "." in name else ""

    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = raw.decode("latin-1", errors="replace")
    else:
        text = str(raw)

    if suffix == "pdf":
        uploaded_file.seek(0)
        return extract_text_from_pdf(uploaded_file)
    if suffix in ("txt", "md", "text"):
        return _normalize_text(text)
    if suffix == "csv":
        # Use first column as text for simplicity
        lines = text.splitlines()
        if not lines:
            return ""
        first_col = []
        for line in lines[:500]:
            part = line.split(",")[0].strip().strip('"')
            if part:
                first_col.append(part)
        return _normalize_text("\n".join(first_col))
    # Default: treat as plain text
    return _normalize_text(text)
