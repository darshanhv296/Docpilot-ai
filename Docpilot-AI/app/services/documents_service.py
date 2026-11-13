"""
DocPilot AI / VISION AI — Multi-Format Document Extraction & Indexing Service (documents_service.py)
Supports PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON, HTML, IPYNB, Source Code, and Images.
Never reads binary formats (PDF, DOCX, PPTX, XLSX, Images) as raw UTF-8 strings.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.services.ai_provider import ai_manager
from app.database.db import UPLOADS_DIR, db


# ============================================================================
# PURE-PYTHON & LIBRARY-BACKED EXTRACTORS
# ============================================================================

def _decode_pdf_literal_string(s: str) -> str:
    """Decodes PDF literal string escapes like \\n, \\r, \\t, \\(, \\), and octal \\ddd."""
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt in "01234567":
                oct_digits = nxt
                j = i + 2
                while j < n and j < i + 4 and s[j] in "01234567":
                    oct_digits += s[j]
                    j += 1
                try:
                    out.append(chr(int(oct_digits, 8)))
                except Exception:
                    pass
                i = j
                continue
            escape_map = {
                "n": "\n",
                "r": "\r",
                "t": "\t",
                "b": "\b",
                "f": "\f",
                "(": "(",
                ")": ")",
                "\\": "\\",
            }
            if nxt in escape_map:
                out.append(escape_map[nxt])
                i += 2
                continue
            if nxt in ("\r", "\n"):
                i += 2
                continue
            out.append(nxt)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _decode_pdf_hex_string(hex_str: str) -> str:
    cleaned = re.sub(r"[^0-9A-Fa-f]", "", hex_str)
    if len(cleaned) % 2 == 1:
        cleaned += "0"
    try:
        raw = bytes.fromhex(cleaned)
        if raw.startswith(b"\xfe\xff"):
            return raw[2:].decode("utf-16-be", errors="ignore")
        return raw.decode("latin-1", errors="ignore")
    except Exception:
        return ""


def _extract_text_from_pdf_content_stream(stream_text: str) -> str:
    """Extracts human-readable text tokens from a decompressed PDF page content stream."""
    pieces: List[str] = []
    # Match BT ... ET text blocks or entire stream if BT/ET spans across streams
    blocks = re.findall(r"BT\b([\s\S]*?)ET\b", stream_text)
    if not blocks:
        blocks = [stream_text]

    for block in blocks:
        line_tokens: List[str] = []
        # Iterate through TJ arrays, Tj literals, and hex strings in order
        for m in re.finditer(
            r"\[((?:[^\[\]]|\\\(|\\\))*)\]\s*TJ|\(((?:[^()\\]|\\.)*)\)\s*(?:Tj|'|\"|\s)|\bTd\b|\bTD\b|\bT\*\b|<([0-9A-Fa-f\s]+)>\s*Tj",
            block,
        ):
            matched_full = m.group(0)
            if matched_full in ("Td", "TD", "T*"):
                if line_tokens:
                    line_tokens.append("\n")
                continue
            tj_arr = m.group(1)
            tj_lit = m.group(2)
            tj_hex = m.group(3)
            if tj_arr is not None:
                arr_parts: List[str] = []
                for sm in re.finditer(r"\(((?:[^()\\]|\\.)*)\)|<([0-9A-Fa-f\s]+)>|(-?\d+(?:\.\d+)?)", tj_arr):
                    if sm.group(1) is not None:
                        arr_parts.append(_decode_pdf_literal_string(sm.group(1)))
                    elif sm.group(2) is not None:
                        arr_parts.append(_decode_pdf_hex_string(sm.group(2)))
                    elif sm.group(3) is not None:
                        try:
                            kern = float(sm.group(3))
                            if kern < -120:
                                arr_parts.append(" ")
                        except Exception:
                            pass
                line_tokens.append("".join(arr_parts))
            elif tj_lit is not None:
                line_tokens.append(_decode_pdf_literal_string(tj_lit))
            elif tj_hex is not None:
                line_tokens.append(_decode_pdf_hex_string(tj_hex))
        if line_tokens:
            joined = "".join(line_tokens)
            pieces.append(joined)

    result = "\n".join(pieces)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def extract_pdf(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extracts text from a PDF file using:
    1. PyMuPDF (`fitz`) if installed
    2. `pypdf` if installed
    3. Pure-Python PDF stream & FlateDecode parser
    """
    if not file_bytes.startswith(b"%PDF"):
        raise ValueError(f'File "{filename}" does not have a valid %PDF header.')

    warnings: List[str] = []
    page_texts: List[str] = []
    total_pages = 0

    # 1. Try PyMuPDF (fitz)
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        for idx, page in enumerate(doc):
            txt = (page.get_text("text") or "").strip()
            if txt:
                page_texts.append(f"[Page {idx + 1}]\n{txt}")
        doc.close()
        if page_texts:
            return {
                "text": "\n\n".join(page_texts),
                "pages": max(total_pages, len(page_texts)),
                "readable_pages": len(page_texts),
                "ocr_pages": 0,
                "warnings": warnings,
            }
    except ImportError:
        pass
    except Exception as e:
        warnings.append(f"PyMuPDF notice: {e}")

    # 2. Try pypdf
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        for idx, page in enumerate(reader.pages):
            txt = (page.extract_text() or "").strip()
            if txt:
                page_texts.append(f"[Page {idx + 1}]\n{txt}")
        if page_texts:
            return {
                "text": "\n\n".join(page_texts),
                "pages": max(total_pages, len(page_texts)),
                "readable_pages": len(page_texts),
                "ocr_pages": 0,
                "warnings": warnings,
            }
    except ImportError:
        pass
    except Exception as e:
        warnings.append(f"pypdf notice: {e}")

    # 3. Built-in Pure-Python PDF Stream & FlateDecode Extractor
    raw_latin = file_bytes.decode("latin-1", errors="ignore")
    page_count_matches = re.findall(r"/Type\s*/Page\b(?!s)", raw_latin)
    total_pages = max(len(page_count_matches), 1)

    extracted_streams: List[str] = []
    for m in re.finditer(rb"stream(?:\r\n|\n)([\s\S]*?)(?:\r\n|\n)?endstream", file_bytes):
        raw_stream = m.group(1)
        decompressed: Optional[bytes] = None
        for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
            try:
                decompressed = zlib.decompress(raw_stream, wbits)
                break
            except Exception:
                try:
                    decompressed = zlib.decompress(raw_stream.strip(), wbits)
                    break
                except Exception:
                    pass
        target_bytes = decompressed if decompressed is not None else raw_stream
        stream_str = target_bytes.decode("latin-1", errors="ignore")
        if "BT" in stream_str or "Tj" in stream_str or "TJ" in stream_str:
            txt = _extract_text_from_pdf_content_stream(stream_str)
            if txt and len(re.sub(r"\s+", "", txt)) >= 3:
                extracted_streams.append(txt)

    if extracted_streams:
        formatted_pages = [
            f"[Page {i + 1}]\n{t}" for i, t in enumerate(extracted_streams)
        ]
        return {
            "text": "\n\n".join(formatted_pages),
            "pages": max(total_pages, len(formatted_pages)),
            "readable_pages": len(formatted_pages),
            "ocr_pages": 0,
            "warnings": warnings,
        }

    raise ValueError(
        f'Could not extract selectable text from PDF "{filename}". The PDF may be encrypted or contain only scanned images without an OCR layer.'
    )


def extract_docx(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Extracts text and tables from a .docx file using python-docx or OpenXML ZIP+XML."""
    if not zipfile.is_zipfile(io.BytesIO(file_bytes)):
        # Check if legacy text/RTF saved with .doc extension
        sample = file_bytes[:512]
        if b"\x00" not in sample:
            txt = file_bytes.decode("utf-8", errors="replace").strip()
            if txt:
                return {"text": txt, "pages": 1, "readable_pages": 1, "ocr_pages": 0, "warnings": []}
        raise ValueError(f'File "{filename}" is not a valid OpenXML .docx archive.')

    # 1. Try python-docx if installed
    try:
        import docx  # type: ignore

        d = docx.Document(io.BytesIO(file_bytes))
        blocks: List[str] = []
        for p in d.paragraphs:
            if p.text and p.text.strip():
                blocks.append(p.text.strip())
        for t in d.tables:
            for row in t.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                if any(cells):
                    blocks.append(" | ".join(cells))
        joined = "\n\n".join(blocks).strip()
        if joined:
            pages = max(1, len(joined) // 2500)
            return {"text": joined, "pages": pages, "readable_pages": pages, "ocr_pages": 0, "warnings": []}
    except ImportError:
        pass
    except Exception:
        pass

    # 2. Pure-Python OpenXML word/document.xml extractor
    paragraphs: List[str] = []
    with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
        xml_targets = [
            name
            for name in zf.namelist()
            if re.match(r"^word/(document|header\d*|footer\d*)\.xml$", name)
        ]
        for xml_name in sorted(xml_targets):
            xml_bytes = zf.read(xml_name)
            root = ET.fromstring(xml_bytes)
            for elem in root.iter():
                tag = elem.tag.split("}")[-1]
                if tag == "p":
                    runs = [
                        node.text
                        for node in elem.iter()
                        if node.tag.split("}")[-1] == "t" and node.text
                    ]
                    line = "".join(runs).strip()
                    if line:
                        paragraphs.append(line)

    text_out = "\n\n".join(paragraphs).strip()
    if not text_out:
        raise ValueError(f'Word document "{filename}" contains no readable text paragraphs.')
    pages = max(1, len(text_out) // 2500)
    return {"text": text_out, "pages": pages, "readable_pages": pages, "ocr_pages": 0, "warnings": []}


def extract_pptx(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Extracts slide text and speaker notes from a .pptx file."""
    if not zipfile.is_zipfile(io.BytesIO(file_bytes)):
        raise ValueError(f'File "{filename}" is not a valid OpenXML .pptx archive.')

    slides_out: List[str] = []
    with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
        slide_files = sorted(
            [
                n
                for n in zf.namelist()
                if re.match(r"^ppt/slides/slide\d+\.xml$", n)
            ],
            key=lambda x: int(re.search(r"slide(\d+)\.xml", x).group(1)),  # type: ignore
        )
        for idx, s_file in enumerate(slide_files):
            slide_num = idx + 1
            root = ET.fromstring(zf.read(s_file))
            texts: List[str] = []
            for elem in root.iter():
                if elem.tag.split("}")[-1] == "p":
                    p_runs = [
                        n.text
                        for n in elem.iter()
                        if n.tag.split("}")[-1] == "t" and n.text
                    ]
                    line = "".join(p_runs).strip()
                    if line:
                        texts.append(line)

            # Check corresponding speaker notes
            notes_file = f"ppt/notesSlides/notesSlide{slide_num}.xml"
            if notes_file in zf.namelist():
                try:
                    n_root = ET.fromstring(zf.read(notes_file))
                    n_texts = [
                        n.text.strip()
                        for n in n_root.iter()
                        if n.tag.split("}")[-1] == "t" and n.text and n.text.strip()
                    ]
                    if n_texts:
                        texts.append("Speaker Notes: " + " ".join(n_texts))
                except Exception:
                    pass

            if texts:
                slides_out.append(f"[Page {slide_num} — Slide {slide_num}]\n" + "\n".join(texts))

    text_out = "\n\n".join(slides_out).strip()
    if not text_out:
        raise ValueError(f'PowerPoint presentation "{filename}" contains no readable slide text.')
    return {
        "text": text_out,
        "pages": max(1, len(slides_out)),
        "readable_pages": len(slides_out),
        "ocr_pages": 0,
        "warnings": [],
    }


def extract_xlsx(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Extracts sheets and rows from an .xlsx spreadsheet."""
    if not zipfile.is_zipfile(io.BytesIO(file_bytes)):
        raise ValueError(f'File "{filename}" is not a valid OpenXML .xlsx archive.')

    # 1. Try openpyxl if installed
    try:
        import openpyxl  # type: ignore

        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        sheets_text: List[str] = []
        for idx, sheet_name in enumerate(wb.sheetnames):
            ws = wb[sheet_name]
            rows_str: List[str] = []
            for row in ws.iter_rows(values_only=True):
                vals = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(vals):
                    rows_str.append(" | ".join(vals))
            if rows_str:
                sheets_text.append(f"[Page {idx + 1} — Sheet: {sheet_name}]\n" + "\n".join(rows_str))
        if sheets_text:
            return {
                "text": "\n\n".join(sheets_text),
                "pages": len(sheets_text),
                "readable_pages": len(sheets_text),
                "ocr_pages": 0,
                "warnings": [],
            }
    except ImportError:
        pass
    except Exception:
        pass

    # 2. Pure-Python OpenXML .xlsx extractor
    with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.iter():
                if si.tag.split("}")[-1] == "si":
                    parts = [
                        t.text
                        for t in si.iter()
                        if t.tag.split("}")[-1] == "t" and t.text
                    ]
                    shared_strings.append("".join(parts))

        sheet_files = sorted(
            [n for n in zf.namelist() if re.match(r"^xl/worksheets/sheet\d+\.xml$", n)]
        )
        sheets_out: List[str] = []
        for idx, s_file in enumerate(sheet_files):
            s_root = ET.fromstring(zf.read(s_file))
            rows_list: List[str] = []
            for row_el in s_root.iter():
                if row_el.tag.split("}")[-1] == "row":
                    row_vals: List[str] = []
                    for c_el in row_el:
                        if c_el.tag.split("}")[-1] != "c":
                            continue
                        c_type = c_el.attrib.get("t", "")
                        val_str = ""
                        for child in c_el.iter():
                            ctag = child.tag.split("}")[-1]
                            if ctag == "v" and child.text:
                                if c_type == "s":
                                    try:
                                        s_idx = int(child.text)
                                        val_str = shared_strings[s_idx]
                                    except Exception:
                                        val_str = child.text
                                else:
                                    val_str = child.text
                            elif ctag == "t" and c_type == "inlineStr" and child.text:
                                val_str += child.text
                        if val_str.strip():
                            row_vals.append(val_str.strip())
                    if row_vals:
                        rows_list.append(" | ".join(row_vals))
            if rows_list:
                sheets_out.append(f"[Page {idx + 1} — Sheet {idx + 1}]\n" + "\n".join(rows_list))

    text_out = "\n\n".join(sheets_out).strip()
    if not text_out:
        raise ValueError(f'Excel workbook "{filename}" contains no non-empty cells.')
    return {
        "text": text_out,
        "pages": len(sheets_out),
        "readable_pages": len(sheets_out),
        "ocr_pages": 0,
        "warnings": [],
    }


def extract_csv_text(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    raw_str = file_bytes.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(raw_str))
    rows: List[str] = []
    for r in reader:
        cleaned = [c.strip() for c in r]
        if any(cleaned):
            rows.append(" | ".join(cleaned))
    if not rows:
        raise ValueError(f'CSV file "{filename}" is empty.')
    text_out = f"[Page 1 — CSV Table ({len(rows)} rows)]\n" + "\n".join(rows)
    return {"text": text_out, "pages": 1, "readable_pages": 1, "ocr_pages": 0, "warnings": []}


def extract_ipynb(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    notebook = json.loads(file_bytes.decode("utf-8", errors="replace"))
    cells = notebook.get("cells") or []
    parts: List[str] = []
    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type", "code")
        src = cell.get("source") or []
        src_text = "".join(src) if isinstance(src, list) else str(src)
        if src_text.strip():
            parts.append(f"### Cell {idx + 1} ({ctype.upper()})\n{src_text.strip()}")
    text_out = "\n\n".join(parts).strip()
    if not text_out:
        raise ValueError(f'Notebook "{filename}" has no non-empty cells.')
    return {"text": text_out, "pages": 1, "readable_pages": 1, "ocr_pages": 0, "warnings": []}


def extract_image_document(file_bytes: bytes, filename: str, ext: str) -> Dict[str, Any]:
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/png")
    b64 = base64.b64encode(file_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    try:
        res = ai_manager.generate_text(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f'Transcribe all visible text, headings, formulas (in LaTeX), tables, and diagrams from this uploaded image "{filename}". '
                        "Provide a comprehensive structured extraction."
                    ),
                    "images": [{"dataUrl": data_url, "mimeType": mime}],
                }
            ],
            temperature=0.2,
            max_tokens=2048,
            operation="document_image_ocr",
            allow_vision_fallback=True,
            allow_model_fallback=True,
        )
        extracted = res.get("text", "").strip()
        if extracted:
            return {
                "text": f"[Page 1 — Image OCR: {filename}]\n{extracted}",
                "pages": 1,
                "readable_pages": 1,
                "ocr_pages": 1,
                "warnings": [],
            }
    except Exception as e:
        pass

    return {
        "text": f"[Page 1 — Visual Image Document: {filename} ({len(file_bytes)} bytes)]",
        "pages": 1,
        "readable_pages": 1,
        "ocr_pages": 1,
        "warnings": ["Vision OCR unavailable offline; image metadata indexed."],
    }


def extract_document_content(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError(f'Uploaded file "{filename}" is empty (0 bytes).')

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

    if ext == "pdf" or file_bytes[:4] == b"%PDF":
        return extract_pdf(file_bytes, filename)
    if ext in ("docx", "doc"):
        return extract_docx(file_bytes, filename)
    if ext in ("pptx", "ppt"):
        return extract_pptx(file_bytes, filename)
    if ext in ("xlsx", "xls"):
        return extract_xlsx(file_bytes, filename)
    if ext in ("csv", "tsv"):
        return extract_csv_text(file_bytes, filename)
    if ext == "ipynb":
        return extract_ipynb(file_bytes, filename)
    if ext in ("png", "jpg", "jpeg", "webp", "gif", "bmp"):
        return extract_image_document(file_bytes, filename, ext)

    # Reject unknown binary files before attempting text decode
    if b"\x00" in file_bytes[:1024]:
        raise ValueError(
            f'File "{filename}" appears to be an unsupported binary format. Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON, HTML, IPYNB, source code, and images.'
        )

    raw_text = file_bytes.decode("utf-8", errors="replace").strip()
    if not raw_text:
        raise ValueError(f'File "{filename}" contains no readable text.')

    if ext in ("html", "htm"):
        # Preserve readable text from HTML while stripping script/style tags
        cleaned_html = re.sub(r"<(script|style)[^>]*>[\s\S]*?</\1>", " ", raw_text, flags=re.I)
        cleaned_html = re.sub(r"<[^>]+>", " ", cleaned_html)
        cleaned_html = re.sub(r"\s+", " ", cleaned_html).strip()
        raw_text = cleaned_html or raw_text

    pages = max(1, len(raw_text) // 2500)
    return {
        "text": f"[Page 1]\n{raw_text}",
        "pages": pages,
        "readable_pages": pages,
        "ocr_pages": 0,
        "warnings": [],
    }


# ============================================================================
# CHUNKING & PERSISTENT RAG INDEXING
# ============================================================================

def split_into_structured_chunks(
    text: str, filename: str, chunk_size: int = 1200, overlap: int = 250
) -> List[Dict[str, Any]]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    paragraphs = re.split(r"\n{2,}", normalized)
    chunks: List[Dict[str, Any]] = []
    current = ""
    current_page = 1
    current_section = "Document Overview"
    idx = 0

    for para in paragraphs:
        trimmed = para.strip()
        if not trimmed:
            continue

        page_match = re.search(r"^\[Page\s+(\d+)", trimmed, re.I)
        if page_match:
            try:
                current_page = int(page_match.group(1))
            except Exception:
                pass

        lines = trimmed.split("\n")
        first_line = re.sub(r"^\[Page\s+\d+[^\]]*\]\s*", "", lines[0]).strip()
        if (
            0 < len(first_line) < 90
            and (
                first_line.startswith("#")
                or re.match(r"^(?:Chapter|Section|Part|Unit|Topic|\d+\.)\s+", first_line, re.I)
            )
        ):
            current_section = re.sub(r"^#+\s*", "", first_line).strip()

        if len(current) + len(trimmed) + 2 <= chunk_size:
            current = f"{current}\n\n{trimmed}" if current else trimmed
        else:
            if current:
                chunks.append(
                    {
                        "chunk_index": idx,
                        "page_number": current_page,
                        "section_title": current_section,
                        "content": current,
                    }
                )
                idx += 1
                tail = current[-overlap:] if len(current) > overlap else current
                current = f"{tail}\n\n{trimmed}"
            else:
                # Paragraph itself exceeds chunk_size
                pos = 0
                while pos < len(trimmed):
                    piece = trimmed[pos : pos + chunk_size]
                    chunks.append(
                        {
                            "chunk_index": idx,
                            "page_number": current_page,
                            "section_title": current_section,
                            "content": piece,
                        }
                    )
                    idx += 1
                    pos += chunk_size - overlap
                current = ""

    if current.strip():
        chunks.append(
            {
                "chunk_index": idx,
                "page_number": current_page,
                "section_title": current_section,
                "content": current.strip(),
            }
        )

    return chunks


def process_and_index_upload(
    file_bytes: bytes,
    filename: str,
    session_id: str,
    user_id: str = "usr-default",
    workspace_id: str = "ws-default",
) -> Dict[str, Any]:
    """
    Extracts text from uploaded file, splits into chunks, computes embeddings,
    and saves document + chunks persistently in `db.py` scoped to `session_id`.
    """
    extracted = extract_document_content(file_bytes, filename)
    full_text = extracted["text"]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

    raw_chunks = split_into_structured_chunks(full_text, filename)
    if not raw_chunks:
        raise ValueError(f'No text chunks could be created from "{filename}".')

    # Save a copy of the uploaded file in .data/uploads for reference
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    try:
        (UPLOADS_DIR / f"{session_id}_{safe_name}").write_bytes(file_bytes)
    except Exception:
        pass

    summary_preview = re.sub(r"\s+", " ", full_text)[:300].strip()

    doc_record = db.add_document(
        {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "filename": filename,
            "file_type": ext,
            "file_size": len(file_bytes),
            "char_count": len(full_text),
            "chunk_count": len(raw_chunks),
            "summary": summary_preview,
            "status": "ready",
        }
    )

    doc_id = doc_record["id"]
    chunk_records: List[Dict[str, Any]] = []
    for c in raw_chunks:
        emb = ai_manager.generate_embedding(c["content"])
        chunk_records.append(
            {
                "id": f"chunk-{doc_id}-{c['chunk_index']}",
                "document_id": doc_id,
                "user_id": user_id,
                "session_id": session_id,
                "filename": filename,
                "chunk_index": c["chunk_index"],
                "page_number": c["page_number"],
                "section_title": c["section_title"],
                "content": c["content"],
                "embedding": emb,
            }
        )

    db.add_chunks(chunk_records)

    return {
        "document_id": doc_id,
        "filename": filename,
        "file_type": ext,
        "chunks": len(chunk_records),
        "char_count": len(full_text),
        "pages": extracted.get("pages", 1),
        "readable_pages": extracted.get("readable_pages", 1),
        "ocr_pages": extracted.get("ocr_pages", 0),
        "warnings": extracted.get("warnings", []),
        "status": "ready",
    }
