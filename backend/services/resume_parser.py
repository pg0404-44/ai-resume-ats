import io
from pathlib import Path
from typing import Tuple, Optional

import pdfplumber
from docx import Document
import PyPDF2

from backend.utils.file_utils import (
    TextExtractionError,
    FileUploadError,
    log_error,
    log_warning,
    log_info,
    with_fallback,
)

from backend.core.config import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
)


class FileParsingError(Exception):
    pass


class FileValidationError(Exception):
    pass


def validate_file(
    file_data: bytes,
    filename: str
) -> Tuple[bool, str, Optional[str]]:

    # -----------------------------
    # 1. Validate file size
    # -----------------------------
    file_size_bytes = len(file_data)

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = file_size_bytes / (1024 * 1024)

        return False, (
            f"File size ({size_mb:.2f} MB) exceeds the maximum "
            f"of {MAX_FILE_SIZE_MB} MB. "
            "Please upload a smaller file or compress your resume."
        ), None

    if file_size_bytes == 0:
        return False, (
            "Uploaded file is empty. "
            "Please check the file you have uploaded and try again."
        ), None

    # -----------------------------
    # 2. Get file extension
    # -----------------------------
    extension = Path(filename).suffix.lower()

    # -----------------------------
    # 3. Detect PDF using signature
    # -----------------------------
    if extension == ".pdf":

        # PDF files normally begin with %PDF
        if file_data[:4] == b"%PDF":
            return True, "", "pdf"

        return False, (
            "The uploaded file has a .pdf extension but does not "
            "appear to be a valid PDF file."
        ), None

    # -----------------------------
    # 4. Detect DOCX
    # -----------------------------
    if extension == ".docx":

        # DOCX is a ZIP-based Office Open XML file.
        # ZIP files begin with PK.
        if file_data[:2] == b"PK":
            return True, "", "docx"

        return False, (
            "The uploaded file has a .docx extension but does not "
            "appear to be a valid DOCX file."
        ), None

    # -----------------------------
    # 5. Legacy DOC
    # -----------------------------
    if extension == ".doc":
        return False, (
            "Legacy .doc format is not supported. "
            "Please convert your document to .docx or .pdf and try again."
        ), None

    # -----------------------------
    # 6. Unsupported extension
    # -----------------------------
    return False, (
        f"Unsupported file extension: {extension or 'none'}. "
        "Please upload a PDF or DOCX resume."
    ), None


def _extract_pdf_hyperlinks(file_data: bytes) -> str:
    urls = []

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_data))

        for page in reader.pages:

            if "/Annots" not in page:
                continue

            for annot_ref in page["/Annots"]:
                try:
                    annot = annot_ref.get_object()

                    if annot.get("/Subtype") != "/Link":
                        continue

                    action = annot.get("/A", {})
                    uri = action.get("/URI", "")

                    if uri and isinstance(uri, (str, bytes)):

                        if isinstance(uri, bytes):
                            uri = uri.decode(
                                "utf-8",
                                errors="ignore"
                            )

                        uri = uri.strip()

                        if uri.startswith("http"):
                            urls.append(uri)

                except Exception:
                    pass

    except Exception:
        pass

    return "\n".join(urls)


def _extract_pdf_with_pdfplumber(file_data: bytes) -> str:

    text = ""

    with pdfplumber.open(io.BytesIO(file_data)) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    if not text.strip():
        raise TextExtractionError(
            "pdfplumber extracted no text",
            user_message="No text could be extracted from the PDF."
        )

    hyperlinks = _extract_pdf_hyperlinks(file_data)

    if hyperlinks:
        text = text.strip() + "\n" + hyperlinks

    return text.strip()


def _extract_pdf_with_pypdf2(file_data: bytes) -> str:

    text = ""

    pdf_reader = PyPDF2.PdfReader(
        io.BytesIO(file_data)
    )

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise TextExtractionError(
            "PyPDF2 extracted no text",
            user_message="No text could be extracted from the PDF."
        )

    hyperlinks = _extract_pdf_hyperlinks(file_data)

    if hyperlinks:
        text = text.strip() + "\n" + hyperlinks

    return text.strip()


def extract_text_from_pdf(file_data: bytes) -> str:

    try:

        result, used_fallback = with_fallback(
            _extract_pdf_with_pdfplumber,
            _extract_pdf_with_pypdf2,
            file_data,
            log_fallback=True
        )

        if used_fallback:
            log_info(
                "PDF extraction succeeded using the PyPDF2 fallback",
                context="resume_parser"
            )

        return result

    except Exception as e:

        log_error(
            e,
            context="extract_text_from_pdf"
        )

        raise FileParsingError(
            "Failed to extract text from PDF using both "
            "pdfplumber and PyPDF2. "
            "The PDF may be corrupted, password-protected, "
            "or contain only scanned images. "
            "Please ensure it contains selectable text."
        ) from e


def extract_text_from_docx(file_data: bytes) -> str:

    try:

        doc = Document(
            io.BytesIO(file_data)
        )

        text_parts = []

        # Paragraphs
        for paragraph in doc.paragraphs:

            if paragraph.text.strip():
                text_parts.append(
                    paragraph.text
                )

        # Tables
        for table in doc.tables:

            for row in table.rows:

                for cell in row.cells:

                    if cell.text.strip():
                        text_parts.append(
                            cell.text
                        )

        text = "\n".join(text_parts)

        if not text.strip():
            raise FileParsingError(
                "No text could be extracted from the document. "
                "The document may be empty or corrupted."
            )

        # Extract hyperlinks
        try:

            for rel in doc.part.rels.values():

                if "hyperlink" in rel.reltype.lower():

                    url = rel._target

                    if (
                        isinstance(url, str)
                        and url.startswith("http")
                    ):
                        text += "\n" + url

        except Exception:
            pass

        log_info(
            f"Extracted {len(text)} chars from DOCX",
            context="resume_parser"
        )

        return text.strip()

    except FileParsingError:
        raise

    except Exception as e:

        log_error(
            e,
            context="extract_text_from_docx"
        )

        raise FileParsingError(
            "Failed to extract text from DOCX. "
            "The document may be corrupted or in an "
            "unsupported format. "
            "Please try re-saving or converting to PDF."
        ) from e


def extract_text_from_doc(file_data: bytes) -> str:

    raise FileParsingError(
        "Legacy .doc format is not supported. "
        "Please convert your document to .docx or .pdf and try again. "
        "You can convert using Microsoft Word or Google Docs."
    )


def extract_text(
    file_data: bytes,
    file_type: str
) -> str:

    if file_type == "pdf":

        return extract_text_from_pdf(file_data)

    elif file_type == "docx":

        return extract_text_from_docx(file_data)

    elif file_type == "doc":

        return extract_text_from_doc(file_data)

    else:

        raise FileValidationError(
            f"Invalid file type: {file_type}. "
            "Supported types are: pdf, docx and doc."
        )


def parse_resume_file(
    file_data: bytes,
    filename: str
) -> Tuple[str, dict]:

    log_info(
        f"Parsing file: {filename}",
        context="parse_resume_file"
    )

    # --------------------------------
    # PHASE 1: Validate file
    # --------------------------------

    try:

        is_valid, error_msg, file_type = validate_file(
            file_data,
            filename
        )

        if not is_valid:

            log_warning(
                f"Validation failed for file {filename}",
                context="parse_resume_file"
            )

            raise FileValidationError(
                error_msg
            )

    except FileValidationError:
        raise

    except Exception as e:

        log_error(
            e,
            context="parse_resume_file_validation"
        )

        raise FileValidationError(
            "Could not validate the uploaded file. "
            "Please ensure it is a valid PDF or DOCX."
        ) from e

    # --------------------------------
    # PHASE 2: Extract text
    # --------------------------------

    try:

        text = extract_text(
            file_data,
            file_type
        )

        log_info(
            f"Extracted {len(text)} chars from {filename}",
            context="parse_resume_file"
        )

    except FileParsingError:
        raise

    except Exception as e:

        log_error(
            e,
            context="parse_resume_file_extraction"
        )

        raise FileParsingError(
            "An unexpected error occurred while processing "
            "the file. Please try again or contact support "
            "if the problem persists."
        ) from e

    # --------------------------------
    # PHASE 3: Metadata
    # --------------------------------

    metadata = {
        "filename": filename,
        "file_type": file_type,
        "file_size_bytes": len(file_data),
        "text_length": len(text),
        "success": True,
    }

    return text, metadata