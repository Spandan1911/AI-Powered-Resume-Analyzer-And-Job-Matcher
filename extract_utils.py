"""
extract_utils.py
Handles file reading (PDF/DOCX/TXT) and text cleaning.
"""

import re
import io
import PyPDF2
import docx


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file given as bytes."""
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract raw text from a DOCX file given as bytes."""
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs)

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Dispatch extraction based on file extension."""
    filename = filename.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove extra whitespace and special chars."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.\-]", " ", text)  # keep tech chars like c++, c#
    text = re.sub(r"\s+", " ", text).strip()
    return text
