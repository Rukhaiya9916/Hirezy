import io
import re
from typing import Tuple
from PyPDF2 import PdfReader

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit

def extract_text_from_pdf(file_bytes: bytes) -> Tuple[bool, str]:
    """
    Extracts plain text from PDF bytes in-memory.
    Never stores resume files permanently on disk to safeguard privacy.
    Returns: (success: bool, text_or_error: str)
    """
    if not file_bytes:
        return False, "Uploaded file is empty."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."

    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        if len(reader.pages) == 0:
            return False, "The PDF file contains no readable pages."

        extracted_text_parts = []
        for index, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text_parts.append(page_text.strip())

        full_text = "\n\n".join(extracted_text_parts)
        
        # Clean text: remove null bytes and excessive repeated whitespace
        full_text = full_text.replace("\x00", "")
        full_text = re.sub(r"[ \t]+", " ", full_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

        if not full_text:
            return False, (
                "Unable to extract text. The PDF might be scanned or image-based. "
                "Please copy and paste your resume text into the text box below."
            )

        return True, full_text

    except Exception as e:
        return False, f"PDF extraction error: {str(e)}"
