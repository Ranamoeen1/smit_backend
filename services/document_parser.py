import io
from pypdf import PdfReader
from docx import Document

def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def parse_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    text = ""
    try:
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    return text

def parse_cv(file_bytes: bytes, filename: str) -> str:
    """Extract text from a CV file (PDF or DOCX)."""
    if filename.lower().endswith(".pdf"):
        return parse_pdf(file_bytes)
    elif filename.lower().endswith(".docx"):
        return parse_docx(file_bytes)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")
