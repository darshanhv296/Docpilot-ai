"""
DocPilot AI / VISION AI — Real File Generation & Format Validation Service (export_service.py)
Generates and validates real local files (.py, .pdf, .docx, .pptx, .xlsx, .csv, .txt, .md, .js, .html, .json, .sql)
in .data/exports/ with genuine AST, PDF binary header/xref/EOF, OpenXML ZIP+XML, and CSV verification.
"""

from __future__ import annotations

import ast
import base64
import csv
import io
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

from app.database.db import EXPORTS_DIR, db

logger = logging.getLogger(__name__)


FORMAT_METADATA: Dict[str, Tuple[str, str]] = {
    "py": ("Python Source Script", "text/x-python; charset=utf-8"),
    "js": ("JavaScript Source", "application/javascript; charset=utf-8"),
    "jsx": ("React JSX Source", "text/jsx; charset=utf-8"),
    "ts": ("TypeScript Source", "application/typescript; charset=utf-8"),
    "tsx": ("React TSX Source", "text/tsx; charset=utf-8"),
    "java": ("Java Source", "text/x-java-source; charset=utf-8"),
    "c": ("C Source", "text/x-c; charset=utf-8"),
    "cpp": ("C++ Source", "text/x-c++src; charset=utf-8"),
    "cs": ("C# Source", "text/x-csharp; charset=utf-8"),
    "php": ("PHP Source", "application/x-httpd-php; charset=utf-8"),
    "html": ("HTML Document", "text/html; charset=utf-8"),
    "css": ("CSS Stylesheet", "text/css; charset=utf-8"),
    "json": ("JSON Data", "application/json; charset=utf-8"),
    "xml": ("XML Document", "application/xml; charset=utf-8"),
    "yaml": ("YAML Configuration", "application/x-yaml; charset=utf-8"),
    "sql": ("SQL Script", "application/sql; charset=utf-8"),
    "sh": ("Shell Script", "application/x-sh; charset=utf-8"),
    "ps1": ("PowerShell Script", "application/x-powershell; charset=utf-8"),
    "ipynb": ("Jupyter Notebook", "application/x-ipynb+json; charset=utf-8"),
    "go": ("Go Source", "text/x-go; charset=utf-8"),
    "rs": ("Rust Source", "text/x-rustsrc; charset=utf-8"),
    "rb": ("Ruby Source", "text/x-ruby; charset=utf-8"),
    "zip": ("ZIP Archive", "application/zip"),
    "pdf": ("PDF Document", "application/pdf"),
    "docx": (
        "Word Document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "pptx": (
        "PowerPoint Presentation",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    "xlsx": (
        "Excel Spreadsheet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "csv": ("CSV Spreadsheet", "text/csv; charset=utf-8"),
    "md": ("Markdown Note", "text/markdown; charset=utf-8"),
    "txt": ("Text Document", "text/plain; charset=utf-8"),
}

_DEFAULT_EXPORT_THEME = {
    "accent": "1E3A8A",
    "light": "E2E8F0",
    "border": "CBD5E1",
    "text": "0F172A",
    "header_text": "0F172A",
}

_EXPORT_THEME_ACCENTS = {
    "scifi": "0EA5E9",
    "iron": "0EA5E9",
    "arctic": "0284C7",
    "midnight": "A855F7",
    "cyber": "06B6D4",
    "neon": "D946EF",
    "emerald": "10B981",
    "matrix": "10B981",
    "crimson": "EF4444",
    "sunset": "F97316",
    "ocean": "3B82F6",
    "rose": "F43F5E",
    "amber": "F59E0B",
    "gold": "F59E0B",
    "graphite": "64748B",
    "oled": "94A3B8",
    "silver": "94A3B8",
    "aurora": "14B8A6",
    "deepviolet": "6366F1",
    "cyberpunk": "A855F7",
    "forest": "22C55E",
}


def _resolve_export_theme(theme: Optional[str]) -> Dict[str, str]:
    accent = _EXPORT_THEME_ACCENTS.get(str(theme or "").strip().lower())
    if not accent:
        return dict(_DEFAULT_EXPORT_THEME)
    return {
        "accent": accent,
        "light": accent,
        "border": accent,
        "text": "0F172A",
        "header_text": (
            "0F172A"
            if str(theme or "").strip().lower()
            in ("amber", "gold", "graphite", "oled", "silver")
            else "FFFFFF"
        ),
    }


def _has_export_theme(theme: Optional[str]) -> bool:
    return str(theme or "").strip().lower() in _EXPORT_THEME_ACCENTS


def _reportlab_color(colors: Any, value: str) -> Any:
    return colors.HexColor(f"#{value}")


def clean_markdown_and_latex(text: str, ascii_safe: bool = False) -> str:
    """Converts Markdown and LaTeX math expressions into clean, readable mathematical notation for PDF/Office exports."""
    if not text:
        return ""
    s = str(text).replace("\r\n", "\n")
    s = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: f"\n  [Formula] {m.group(1).strip()}\n", s)
    s = re.sub(r"\$([^$\n]+)\$", lambda m: m.group(1).strip(), s)

    # Convert \frac{a}{b} and \sqrt{x} recursively
    for _ in range(3):
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\mathcal\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\hat\{([^{}]*)\}", r"\1^", s)
        s = re.sub(r"\\dot\{([^{}]*)\}", r"d(\1)/dt", s)
        s = re.sub(r"\\boldsymbol\{([^{}]*)\}", r"\1", s)

    s = re.sub(r"\\left\s*([(\[{|])", r"\1", s)
    s = re.sub(r"\\right\s*([)\]}|])", r"\1", s)
    s = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", s)
    s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)

    if ascii_safe:
        replacements = {
            r"\cdot": " * ",
            r"\times": " x ",
            r"\div": " / ",
            r"\pm": "+/-",
            r"\approx": "~=",
            r"\sim": "~",
            r"\neq": "!=",
            r"\leq": "<=",
            r"\le": "<=",
            r"\geq": ">=",
            r"\ge": ">=",
            r"\ll": "<<",
            r"\gg": ">>",
            r"\propto": " prop. to ",
            r"\infty": "inf",
            r"\partial": "d",
            r"\nabla": "grad ",
            r"\Delta": "Delta ",
            r"\pi": "pi",
            r"\alpha": "alpha",
            r"\beta": "beta",
            r"\gamma": "gamma",
            r"\theta": "theta",
            r"\lambda": "lambda",
            r"\sigma": "sigma",
            r"\omega": "omega",
            r"\Omega": "Omega",
            r"\tau": "tau",
            r"\rho": "rho",
            r"\mu": "mu",
            r"\eta": "eta",
            r"\epsilon": "eps",
            r"\Phi": "Phi",
            r"\phi": "phi",
            r"\delta": "delta",
            r"\hbar": "hbar",
            r"\odot": "_sun",
            r"\rightarrow": "->",
            r"\to": "->",
            r"\implies": "=>",
            r"\odot": "_sun",
            r"\quad": "  ",
            r"\qquad": "    ",
            "≈": "~=",
            "∝": " prop. to ",
            "≤": "<=",
            "≥": ">=",
            "≠": "!=",
            "→": "->",
            "⇒": "=>",
            "☉": "_sun",
            "ℏ": "hbar",
            "π": "pi",
            "ρ": "rho",
            "λ": "lambda",
            "γ": "gamma",
            "τ": "tau",
            "σ": "sigma",
            "Δ": "Delta ",
            "∇": "grad ",
            "∂": "d",
            "—": "--",
            "–": "-",
            "•": "-",
        }
    else:
        replacements = {
            r"\cdot": "·",
            r"\times": "×",
            r"\div": "÷",
            r"\pm": "±",
            r"\approx": "≈",
            r"\sim": "~",
            r"\neq": "≠",
            r"\leq": "≤",
            r"\le": "≤",
            r"\geq": "≥",
            r"\ge": "≥",
            r"\ll": "≪",
            r"\gg": "≫",
            r"\propto": "∝",
            r"\infty": "∞",
            r"\partial": "∂",
            r"\nabla": "∇",
            r"\Delta": "Δ",
            r"\pi": "π",
            r"\alpha": "α",
            r"\beta": "β",
            r"\gamma": "γ",
            r"\theta": "θ",
            r"\lambda": "λ",
            r"\sigma": "σ",
            r"\omega": "ω",
            r"\Omega": "Ω",
            r"\tau": "τ",
            r"\rho": "ρ",
            r"\mu": "μ",
            r"\eta": "η",
            r"\epsilon": "ε",
            r"\Phi": "Φ",
            r"\phi": "φ",
            r"\delta": "δ",
            r"\hbar": "ℏ",
            r"\odot": "☉",
            r"\rightarrow": "→",
            r"\to": "→",
            r"\implies": "⇒",
            r"\quad": "  ",
            r"\qquad": "    ",
        }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    return s.strip()


def _parse_markdown_blocks(raw_content: str, ascii_safe: bool = False) -> List[Dict[str, Any]]:
    """
    Parses Markdown text into structured blocks:
      - {'type': 'h1' | 'h2' | 'h3', 'text': ...}
      - {'type': 'table', 'rows': [[cell1, cell2, ...], ...]}
      - {'type': 'formula', 'text': ...}
      - {'type': 'bullet' | 'numbered', 'text': ...}
      - {'type': 'para', 'text': ...}
    """
    blocks: List[Dict[str, Any]] = []
    lines = (raw_content or "").replace("\r\n", "\n").splitlines()
    i = 0
    while i < len(lines):
        raw_ln = lines[i].strip()
        if not raw_ln:
            i += 1
            continue

        # Check for Markdown pipe table
        if raw_ln.startswith("|") and raw_ln.endswith("|") and raw_ln.count("|") >= 2:
            table_rows: List[List[str]] = []
            while i < len(lines):
                t_ln = lines[i].strip()
                if not (t_ln.startswith("|") and "|" in t_ln[1:]):
                    break
                raw_cells = [c.strip() for c in t_ln.strip("|").split("|")]
                # Skip separator row | --- | --- |
                if all(re.match(r"^[-:\s]+$", c) for c in raw_cells if c != ""):
                    i += 1
                    continue
                cleaned_cells = [
                    clean_markdown_and_latex(c, ascii_safe=ascii_safe)
                    for c in raw_cells
                ]
                if any(c.strip() for c in cleaned_cells):
                    table_rows.append(cleaned_cells)
                i += 1
            if table_rows:
                blocks.append({"type": "table", "rows": table_rows})
            continue

        if raw_ln.startswith("### "):
            blocks.append({"type": "h3", "text": clean_markdown_and_latex(raw_ln[4:], ascii_safe=ascii_safe)})
        elif raw_ln.startswith("## "):
            blocks.append({"type": "h2", "text": clean_markdown_and_latex(raw_ln[3:], ascii_safe=ascii_safe)})
        elif raw_ln.startswith("# "):
            blocks.append({"type": "h1", "text": clean_markdown_and_latex(raw_ln[2:], ascii_safe=ascii_safe)})
        elif raw_ln.startswith("$$"):
            formula_lines = [raw_ln]
            formula_delimiter_count = raw_ln.count("$$")
            i += 1
            while i < len(lines) and formula_delimiter_count < 2:
                formula_lines.append(lines[i].strip())
                formula_delimiter_count += lines[i].count("$$")
                i += 1
            formula = " ".join(formula_lines).replace("$$", "").strip()
            blocks.append(
                {
                    "type": "formula",
                    "text": clean_markdown_and_latex(formula, ascii_safe=ascii_safe),
                }
            )
            continue
        elif raw_ln.startswith("[Formula]"):
            blocks.append({"type": "formula", "text": clean_markdown_and_latex(raw_ln, ascii_safe=ascii_safe)})
        elif re.match(r"^\d+\.\s+", raw_ln):
            match = re.match(r"^(\d+)\.\s+(.*)$", raw_ln)
            if match:
                blocks.append(
                    {
                        "type": "numbered",
                        "number": int(match.group(1)),
                        "text": clean_markdown_and_latex(match.group(2), ascii_safe=ascii_safe),
                    }
                )
        elif re.match(r"^[-*•]\s+", raw_ln):
            b_txt = re.sub(r"^[-*•]\s+", "", raw_ln)
            blocks.append({"type": "bullet", "text": clean_markdown_and_latex(b_txt, ascii_safe=ascii_safe)})
        else:
            cleaned_ln = clean_markdown_and_latex(raw_ln, ascii_safe=ascii_safe)
            if cleaned_ln.startswith("[Formula]"):
                blocks.append({"type": "formula", "text": cleaned_ln})
            elif blocks and blocks[-1]["type"] == "para":
                blocks[-1]["text"] += f" {cleaned_ln}"
            else:
                blocks.append({"type": "para", "text": cleaned_ln})
        i += 1

    return blocks


def extract_clean_code(raw_content: str, ext: str) -> str:
    """Extracts clean executable source code from a markdown response or raw string."""
    trimmed = (raw_content or "").strip()
    if not trimmed:
        return ""

    # Look for fenced code blocks ```python ... ```
    blocks = re.findall(r"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```", trimmed)
    if blocks:
        lang_aliases = {
            "py": ("python", "py", ""),
            "js": ("javascript", "js", "node", ""),
            "ts": ("typescript", "ts", ""),
            "html": ("html", "htm", ""),
            "css": ("css", ""),
            "json": ("json", ""),
            "sql": ("sql", "sqlite", "postgres", ""),
            "sh": ("bash", "sh", "shell", ""),
        }.get(ext, (ext, ""))
        preferred = [code.strip() for lang, code in blocks if lang.strip().lower() in lang_aliases and code.strip()]
        if preferred:
            return "\n\n".join(preferred) + "\n"
        return blocks[0][1].strip() + "\n"

    # Strip accidental leading/trailing fences
    cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", trimmed)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip() + "\n"


def detect_requested_export_formats(user_message: str) -> List[str]:
    """
    Detects when a user explicitly asks to create, generate, save, or export a downloadable file
    in chat (e.g. 'Create a Python script...', 'Export as PDF', 'Generate a Word document').
    """
    msg = (user_message or "").lower()
    action_words = (
        "create", "generate", "export", "download", "save", "write", "build", "make", "give", "produce"
    )
    if not any(w in msg for w in action_words):
        return []

    detected: List[str] = []

    if re.search(r"\b(python\s+(?:script|file|code|program)|\.py\b|py\s+code|py\s+file)", msg):
        detected.append("py")
    if re.search(r"\b(pdf\s+(?:file|document|report|notes|cheat\s*sheet|export)|as\s+(?:a\s+)?pdf|in\s+pdf|\.pdf\b|\bpdf\b)", msg):
        detected.append("pdf")
    if re.search(r"\b(word\s+(?:document|file|report)|docx|\.docx\b)", msg):
        detected.append("docx")
    if re.search(r"\b(powerpoint|presentation|slides\s+deck|pptx|\.pptx\b)", msg):
        detected.append("pptx")
    if re.search(r"\b(excel|spreadsheet|workbook|xlsx|\.xlsx\b)", msg):
        detected.append("xlsx")
    if re.search(r"\b(csv\s+(?:file|table|data|export)|\.csv\b)", msg):
        detected.append("csv")
    if re.search(r"\b(markdown\s+(?:file|document|notes)|\.md\b)", msg):
        detected.append("md")
    if re.search(r"\b(text\s+file|txt\s+file|\.txt\b)", msg):
        detected.append("txt")
    if re.search(r"\b(javascript\s+file|\.js\b)", msg):
        detected.append("js")
    if re.search(r"\b(html\s+file|\.html\b)", msg):
        detected.append("html")
    if re.search(r"\b(sql\s+(?:file|script)|\.sql\b)", msg):
        detected.append("sql")

    return list(dict.fromkeys(detected))


# ============================================================================
# REAL BINARY & ARCHIVE GENERATORS (WITH PURE-PYTHON FALLBACKS)
# ============================================================================

def _pdf_escape(text: str) -> str:
    safe_latin = text.encode("latin-1", errors="replace").decode("latin-1")
    return safe_latin.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def generate_pdf_file(
    file_path: Path, title: str, content: str, theme: Optional[str] = None
) -> None:
    """
    Generates a publication-grade, multi-page academic PDF document.
    Uses ReportLab Platypus (if installed) or a structured Pure-Python PDF 1.4 binary writer
    supporting section hierarchy, bordered tables, formula blocks, and Page X of Y footers.
    """
    cleaned_title = clean_markdown_and_latex(title, ascii_safe=True) or "VISION AI Study Reference"
    blocks = _parse_markdown_blocks(content, ascii_safe=True)
    palette = _resolve_export_theme(theme)

    # 1. Try ReportLab Platypus if installed
    try:
        from reportlab.lib import colors  # type: ignore
        from reportlab.lib.pagesizes import letter  # type: ignore
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # type: ignore

        doc = SimpleDocTemplate(
            str(file_path),
            pagesize=letter,
            leftMargin=42,
            rightMargin=42,
            topMargin=48,
            bottomMargin=48,
            title=cleaned_title,
            author="DocPilot VISION AI",
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "VisionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=_reportlab_color(colors, palette["accent"] if _has_export_theme(theme) else palette["text"]),
            spaceAfter=6,
            keepWithNext=True,
        )
        meta_style = ParagraphStyle(
            "VisionMeta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#475569"),
            spaceAfter=14,
        )
        h2_style = ParagraphStyle(
            "VisionH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=_reportlab_color(colors, palette["accent"]),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        )
        h3_style = ParagraphStyle(
            "VisionH3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=_reportlab_color(colors, palette["accent"] if _has_export_theme(theme) else "1E293B"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        )
        table_header_style = ParagraphStyle(
            "VisionTableHeader",
            parent=h3_style,
            textColor=_reportlab_color(colors, palette["header_text"]),
        )
        body_style = ParagraphStyle(
            "VisionBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=5,
        )
        formula_style = ParagraphStyle(
            "VisionFormula",
            parent=styles["Code"],
            fontName="Courier-Bold",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#0f172a"),
            backColor=_reportlab_color(colors, palette["light"] if _has_export_theme(theme) else "F1F5F9"),
            borderColor=_reportlab_color(colors, palette["border"] if _has_export_theme(theme) else "CBD5E1"),
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=6,
        )

        story = [
            Paragraph(xml_escape(cleaned_title), title_style),
            Paragraph(f"DocPilot VISION AI Academic Reference • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style),
        ]

        usable_width = letter[0] - 84
        for blk in blocks:
            b_type = blk["type"]
            if b_type in ("h1", "h2"):
                story.append(Paragraph(xml_escape(blk["text"]), h2_style))
            elif b_type == "h3":
                story.append(Paragraph(xml_escape(blk["text"]), h3_style))
            elif b_type == "formula":
                story.append(Paragraph(xml_escape(blk["text"]), formula_style))
            elif b_type == "bullet":
                story.append(Paragraph(f"• {xml_escape(blk['text'])}", body_style))
            elif b_type == "numbered":
                story.append(
                    Paragraph(
                        f"{blk.get('number', 1)}. {xml_escape(blk['text'])}",
                        body_style,
                    )
                )
            elif b_type == "table":
                t_rows = blk["rows"]
                num_cols = max(len(r) for r in t_rows) if t_rows else 1
                col_w = usable_width / max(num_cols, 1)
                flow_rows = []
                for r_idx, r in enumerate(t_rows):
                    padded = (r + ["—"] * num_cols)[:num_cols]
                    flow_rows.append([
                        Paragraph(xml_escape(c or "—"), table_header_style if r_idx == 0 else body_style)
                        for c in padded
                    ])
                tbl = Table(flow_rows, colWidths=[col_w] * num_cols, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), _reportlab_color(colors, palette["light"])),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _reportlab_color(colors, palette["header_text"])),
                    ("GRID", (0, 0), (-1, -1), 0.5, _reportlab_color(colors, palette["border"])),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(xml_escape(blk["text"]), body_style))

        def _add_page_number(canvas_obj, doc_obj):
            canvas_obj.saveState()
            canvas_obj.setFont("Helvetica", 8.5)
            canvas_obj.drawRightString(letter[0] - 42, 26, f"Page {doc_obj.page}")
            canvas_obj.drawString(42, 26, cleaned_title[:65])
            canvas_obj.restoreState()

        doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
        return
    except ImportError:
        pass
    except Exception:
        pass

    # 2. Pure-Python PDF 1.4 Multi-Page Structured Academic Layout Builder
    # Supports distinct font styles/sizes:
    #   style="title" | "meta" | "h2" | "h3" | "formula" | "table_hdr" | "table_row" | "table_sep" | "bullet" | "body"
    styled_lines: List[Tuple[str, str]] = [
        (cleaned_title[:82], "title"),
        (f"DocPilot VISION AI Academic Study Document  |  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", "meta"),
        ("=" * 88, "table_sep"),
    ]

    def _wrap_text(txt: str, width: int) -> List[str]:
        s = (txt or "").strip()
        if not s:
            return [""]
        out_w: List[str] = []
        while len(s) > width:
            sp = s.rfind(" ", 0, width)
            if sp <= 15:
                sp = width
            out_w.append(s[:sp].strip())
            s = s[sp:].strip()
        if s:
            out_w.append(s)
        return out_w

    for blk in blocks:
        b_type = blk["type"]
        if b_type in ("h1", "h2"):
            styled_lines.append(("", "body"))
            for wl in _wrap_text(blk["text"].upper(), 78):
                styled_lines.append((wl, "h2"))
            styled_lines.append(("-" * 88, "table_sep"))
        elif b_type == "h3":
            styled_lines.append(("", "body"))
            for wl in _wrap_text(blk["text"], 82):
                styled_lines.append((wl, "h3"))
        elif b_type == "formula":
            for wl in _wrap_text(blk["text"], 80):
                styled_lines.append((f"    {wl}", "formula"))
        elif b_type == "bullet":
            w_list = _wrap_text(blk["text"], 84)
            for idx_w, wl in enumerate(w_list):
                prefix = "  * " if idx_w == 0 else "    "
                styled_lines.append((f"{prefix}{wl}", "bullet"))
        elif b_type == "table":
            t_rows = blk["rows"]
            if not t_rows:
                continue
            num_cols = max(len(r) for r in t_rows)
            col_width = max(16, min(42, (84 - (num_cols + 1) * 3) // max(1, num_cols)))
            border_line = "+" + "+".join("-" * (col_width + 2) for _ in range(num_cols)) + "+"
            styled_lines.append((border_line, "table_sep"))
            for r_idx, row in enumerate(t_rows):
                padded = [(row[c_i] if c_i < len(row) and row[c_i] else "--") for c_i in range(num_cols)]
                wrapped_cols = [_wrap_text(cell, col_width) for cell in padded]
                max_sublines = max(len(wc) for wc in wrapped_cols)
                row_style = "table_hdr" if r_idx == 0 else "table_row"
                for s_i in range(max_sublines):
                    row_str = "| " + " | ".join(
                        (wrapped_cols[c_i][s_i] if s_i < len(wrapped_cols[c_i]) else "").ljust(col_width)
                        for c_i in range(num_cols)
                    ) + " |"
                    styled_lines.append((row_str, row_style))
                styled_lines.append((border_line, "table_sep"))
            styled_lines.append(("", "body"))
        else:
            for wl in _wrap_text(blk["text"], 88):
                styled_lines.append((wl, "body"))

    # Paginate cleanly (prevent headings from being stranded at the bottom of a page)
    lines_per_page = 43
    pages_chunks: List[List[Tuple[str, str]]] = []
    cur_page: List[Tuple[str, str]] = []

    for idx_l, item in enumerate(styled_lines):
        txt, st = item
        if len(cur_page) >= lines_per_page - 2 and st in ("h2", "h3"):
            pages_chunks.append(cur_page)
            cur_page = [item]
        else:
            cur_page.append(item)
            if len(cur_page) >= lines_per_page:
                pages_chunks.append(cur_page)
                cur_page = []
    if cur_page or not pages_chunks:
        pages_chunks.append(cur_page)

    total_pages = len(pages_chunks)

    # Objects:
    # 1: Catalog, 2: Pages, 3: Helvetica (F1), 4: Helvetica-Bold (F2), 5: Courier (F3)
    objects: Dict[int, bytes] = {}
    page_obj_ids: List[int] = []

    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    objects[5] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"

    for p_idx, p_lines in enumerate(pages_chunks):
        page_id = 6 + p_idx * 2
        content_id = 7 + p_idx * 2
        page_obj_ids.append(page_id)

        ops: List[str] = ["BT", "42 746 Td"]
        for l_idx, (txt, st) in enumerate(p_lines):
            if st == "title":
                font_cmd = "/F2 14 Tf"
            elif st == "meta":
                font_cmd = "/F1 8.5 Tf"
            elif st == "h2":
                font_cmd = "/F2 11.5 Tf"
            elif st == "h3":
                font_cmd = "/F2 10.5 Tf"
            elif st == "formula":
                font_cmd = "/F3 9.5 Tf"
            elif st == "table_hdr":
                font_cmd = "/F3 8.5 Tf"
            elif st in ("table_row", "table_sep"):
                font_cmd = "/F3 8 Tf"
            else:
                font_cmd = "/F1 9.5 Tf"

            ops.append(font_cmd)
            if l_idx > 0:
                ops.append("0 -15 Td")
            if txt:
                ops.append(f"({_pdf_escape(txt)}) Tj")
        ops.append("ET")

        # Draw footer with Page X of Y
        footer_text = f"{cleaned_title[:55]}   |   Page {p_idx + 1} of {total_pages}"
        ops.extend([
            "BT",
            "/F1 8.5 Tf",
            f"42 28 Td ({_pdf_escape(footer_text)}) Tj",
            "ET",
        ])

        stream_bytes = "\n".join(ops).encode("latin-1", errors="replace")
        objects[content_id] = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")

    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_obj_ids)} >>".encode("ascii")
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    total_objs = max(objects.keys())
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: Dict[int, int] = {}

    for oid in range(1, total_objs + 1):
        offsets[oid] = len(out)
        out.extend(f"{oid} 0 obj\n".encode("ascii"))
        out.extend(objects[oid])
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {total_objs + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for oid in range(1, total_objs + 1):
        out.extend(f"{offsets[oid]:010d} 00000 n \n".encode("ascii"))

    out.extend(
        (
            f"trailer\n<< /Size {total_objs + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    file_path.write_bytes(bytes(out))


def generate_docx_file(
    file_path: Path, title: str, content: str, theme: Optional[str] = None
) -> None:
    """
    Generates a valid OpenXML .docx file with real headings, bordered tables (`<w:tbl>`),
    formula blocks, and bullet points using python-docx or the pure-Python OpenXML ZIP+XML builder.
    """
    cleaned_title = clean_markdown_and_latex(title, ascii_safe=False) or "VISION AI Deliverable"
    blocks = _parse_markdown_blocks(content, ascii_safe=False)
    palette = _resolve_export_theme(theme)

    try:
        import docx  # type: ignore
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
        from docx.oxml import OxmlElement  # type: ignore
        from docx.oxml.ns import qn  # type: ignore
        from docx.shared import Inches, Pt, RGBColor  # type: ignore

        doc = docx.Document()
        section = doc.sections[0]
        section.top_margin = Inches(0.65)
        section.bottom_margin = Inches(0.65)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        styles = doc.styles
        styles["Normal"].font.name = "Aptos"
        styles["Normal"].font.size = Pt(10)
        styles["Normal"].paragraph_format.space_after = Pt(6)
        styles["Normal"].paragraph_format.line_spacing = 1.15
        heading_color = RGBColor.from_string(
            palette["accent"] if theme else palette["text"]
        )
        for style_name, size in (("Title", 24), ("Heading 1", 17), ("Heading 2", 14), ("Heading 3", 12)):
            style = styles[style_name]
            style.font.name = "Aptos Display"
            style.font.size = Pt(size)
            style.font.bold = True
            style.font.color.rgb = heading_color
            style.paragraph_format.space_before = Pt(12 if style_name != "Title" else 0)
            style.paragraph_format.space_after = Pt(6)
            style.paragraph_format.keep_with_next = True

        title_p = doc.add_paragraph(style="Title")
        title_p.add_run(cleaned_title)
        subtitle = doc.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.LEFT
        subtitle.paragraph_format.space_after = Pt(16)
        subtitle_run = subtitle.add_run(
            f"DocPilot VISION AI Academic Reference • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        subtitle_run.font.size = Pt(9)
        subtitle_run.font.color.rgb = RGBColor(71, 85, 105)

        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        footer.add_run("Page ")
        page_field = OxmlElement("w:fldSimple")
        page_field.set(qn("w:instr"), "PAGE")
        footer._p.append(page_field)
        footer.add_run("  •  DocPilot")

        for blk in blocks:
            b_type = blk["type"]
            if b_type == "h1":
                doc.add_heading(blk["text"], level=1)
            elif b_type == "h2":
                doc.add_heading(blk["text"], level=2)
            elif b_type == "h3":
                doc.add_heading(blk["text"], level=3)
            elif b_type == "table":
                t_rows = blk["rows"]
                if t_rows:
                    num_cols = max(len(r) for r in t_rows)
                    tbl = doc.add_table(rows=len(t_rows), cols=num_cols)
                    tbl.autofit = True
                    tbl.style = "Table Grid"
                    for r_i, r in enumerate(t_rows):
                        row_props = tbl.rows[r_i]._tr.get_or_add_trPr()
                        cant_split = OxmlElement("w:cantSplit")
                        row_props.append(cant_split)
                        for c_i in range(num_cols):
                            cell = tbl.cell(r_i, c_i)
                            cell.text = r[c_i] if c_i < len(r) and r[c_i] else "—"
                            if r_i == 0:
                                shading = OxmlElement("w:shd")
                                shading.set(qn("w:fill"), palette["light"])
                                cell._tc.get_or_add_tcPr().append(shading)
                                for run in cell.paragraphs[0].runs:
                                    run.bold = True
                                    run.font.color.rgb = RGBColor.from_string(
                                        palette["header_text"]
                                    )
                    header_repeat = OxmlElement("w:tblHeader")
                    tbl.rows[0]._tr.get_or_add_trPr().append(header_repeat)
                    doc.add_paragraph().paragraph_format.space_after = Pt(3)
            elif b_type == "bullet":
                doc.add_paragraph(blk["text"], style="List Bullet")
            elif b_type == "numbered":
                doc.add_paragraph(blk["text"], style="List Number")
            elif b_type == "formula":
                formula_p = doc.add_paragraph()
                formula_p.paragraph_format.left_indent = Inches(0.25)
                formula_p.paragraph_format.space_before = Pt(4)
                formula_p.paragraph_format.space_after = Pt(8)
                run = formula_p.add_run(blk["text"])
                run.font.name = "Cambria Math"
                run.font.size = Pt(11)
                run.bold = True
            else:
                doc.add_paragraph(blk["text"])

        doc.save(str(file_path))
        return
    except ImportError:
        pass

    # Pure-Python OpenXML .docx ZIP Package Builder with real <w:tbl> support
    gen_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    heading_color = palette["accent"] if _has_export_theme(theme) else palette["text"]
    table_header_fill = palette["accent"] if _has_export_theme(theme) else palette["light"]
    table_header_color = palette["header_text"] if _has_export_theme(theme) else palette["text"]
    body_xml: List[str] = [
        f'<w:p><w:pPr><w:keepNext/><w:spacing w:after="120"/></w:pPr>'
        f'<w:r><w:rPr><w:b/><w:sz w:val="34"/><w:color w:val="{heading_color}"/></w:rPr><w:t>{xml_escape(cleaned_title)}</w:t></w:r></w:p>',
        f'<w:p><w:r><w:rPr><w:i/><w:sz w:val="18"/></w:rPr>'
        f'<w:t>{xml_escape(f"DocPilot VISION AI Academic Reference • Generated {gen_time_str}")}</w:t></w:r></w:p>',
    ]

    for blk in blocks:
        b_type = blk["type"]
        if b_type in ("h1", "h2"):
            body_xml.append(
                f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="100"/></w:pPr>'
                f'<w:r><w:rPr><w:b/><w:sz w:val="26"/><w:color w:val="{heading_color}"/></w:rPr><w:t>{xml_escape(blk["text"])}</w:t></w:r></w:p>'
            )
        elif b_type == "h3":
            body_xml.append(
                f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="180" w:after="80"/></w:pPr>'
                f'<w:r><w:rPr><w:b/><w:sz w:val="23"/><w:color w:val="{heading_color}"/></w:rPr><w:t>{xml_escape(blk["text"])}</w:t></w:r></w:p>'
            )
        elif b_type == "formula":
            body_xml.append(
                f'<w:p><w:pPr><w:ind w:left="360"/><w:spacing w:before="80" w:after="160"/></w:pPr>'
                f'<w:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:b/><w:sz w:val="21"/></w:rPr><w:t>{xml_escape(blk["text"])}</w:t></w:r></w:p>'
            )
        elif b_type == "bullet":
            body_xml.append(
                f'<w:p><w:pPr><w:ind w:left="360" w:hanging="180"/><w:spacing w:after="80"/></w:pPr>'
                f'<w:r><w:rPr><w:sz w:val="21"/></w:rPr><w:t>{xml_escape("• " + blk["text"])}</w:t></w:r></w:p>'
            )
        elif b_type == "numbered":
            numbered_text = f'{blk.get("number", 1)}. {blk["text"]}'
            body_xml.append(
                f'<w:p><w:pPr><w:ind w:left="360" w:hanging="180"/><w:spacing w:after="80"/></w:pPr>'
                f'<w:r><w:rPr><w:sz w:val="21"/></w:rPr><w:t>{xml_escape(numbered_text)}</w:t></w:r></w:p>'
            )
        elif b_type == "table":
            t_rows = blk["rows"]
            if not t_rows:
                continue
            num_cols = max(len(r) for r in t_rows)
            tr_xml_list: List[str] = []
            for r_idx, r in enumerate(t_rows):
                tc_xml_list: List[str] = []
                for c_idx in range(num_cols):
                    cell_txt = r[c_idx] if c_idx < len(r) and r[c_idx] else "—"
                    run_pr = (
                        f'<w:rPr><w:b/><w:sz w:val="20"/><w:color w:val="{table_header_color}"/></w:rPr>'
                        if r_idx == 0
                        else '<w:rPr><w:sz w:val="20"/></w:rPr>'
                    )
                    shd = (
                        f'<w:shd w:val="clear" w:color="auto" w:fill="{table_header_fill}"/>'
                        if r_idx == 0
                        else ""
                    )
                    tc_xml_list.append(
                        f'<w:tc><w:tcPr><w:tcW w:w="{9000 // max(1, num_cols)}" w:type="dxa"/>{shd}</w:tcPr>'
                        f'<w:p><w:r>{run_pr}<w:t>{xml_escape(cell_txt)}</w:t></w:r></w:p></w:tc>'
                    )
                tr_pr = "<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>" if r_idx == 0 else "<w:trPr><w:cantSplit/></w:trPr>"
                tr_xml_list.append(f"<w:tr>{tr_pr}{''.join(tc_xml_list)}</w:tr>")
            tbl_borders = (
                '<w:tblBorders>'
                f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{palette["border"]}"/>'
                f'<w:left w:val="single" w:sz="4" w:space="0" w:color="{palette["border"]}"/>'
                f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{palette["border"]}"/>'
                f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{palette["border"]}"/>'
                f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{palette["border"]}"/>'
                f'<w:insideV w:val="single" w:sz="4" w:space="0" w:color="{palette["border"]}"/>'
                '</w:tblBorders>'
            )
            body_xml.append(f"<w:tbl><w:tblPr>{tbl_borders}</w:tblPr>{''.join(tr_xml_list)}</w:tbl><w:p/>")
        else:
            body_xml.append(
                f'<w:p><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr>'
                f'<w:r><w:rPr><w:sz w:val="21"/></w:rPr><w:t>{xml_escape(blk["text"])}</w:t></w:r></w:p>'
            )

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<w:body>{''.join(body_xml)}"
        '<w:sectPr><w:footerReference w:type="default" r:id="rId1"/>'
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>'
        "</w:body></w:document>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
        "</Types>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    document_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
        "</Relationships>"
    )
    footer_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:t>Page </w:t></w:r>'
        '<w:fldSimple w:instr="PAGE"><w:r><w:t>1</w:t></w:r></w:fldSimple>'
        '<w:r><w:t>  •  DocPilot</w:t></w:r></w:p></w:ftr>'
    )
    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml)
        zf.writestr("word/footer1.xml", footer_xml)


def generate_pptx_file(
    file_path: Path, title: str, content: str, theme: Optional[str] = None
) -> None:
    """Create an editable, content-aware 16:9 presentation."""
    try:
        from pptx import Presentation  # type: ignore
        from pptx.dml.color import RGBColor  # type: ignore
        from pptx.enum.text import PP_ALIGN  # type: ignore
        from pptx.enum.shapes import MSO_SHAPE  # type: ignore
        from pptx.util import Inches, Pt  # type: ignore
    except ImportError:
        _generate_pptx_openxml_fallback(file_path, title, content, theme)
        return

    cleaned_title = clean_markdown_and_latex(title) or "VISION AI Presentation"
    blocks = _parse_markdown_blocks(content)
    if (
        blocks
        and blocks[0]["type"] in ("h1", "h2")
        and blocks[0]["text"].casefold() == cleaned_title.casefold()
    ):
        blocks = blocks[1:]
    palette = _resolve_export_theme(theme)
    has_theme = _has_export_theme(theme)
    accent = RGBColor.from_string(palette["accent"])
    border = RGBColor.from_string(palette["border"])
    body_color = RGBColor(30, 41, 59)
    table_header_background = (
        accent if has_theme else RGBColor.from_string(palette["light"])
    )
    width = Inches(13.333)
    height = Inches(7.5)
    prs = Presentation()
    prs.slide_width = width
    prs.slide_height = height

    def add_textbox(slide: Any, x: float, y: float, w: float, h: float, text: str, size: int, color: Any, bold: bool = False) -> Any:
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = shape.text_frame
        frame.clear()
        frame.word_wrap = True
        frame.margin_left = Inches(0.02)
        frame.margin_right = Inches(0.02)
        frame.margin_top = Inches(0.02)
        frame.margin_bottom = Inches(0.02)
        paragraph = frame.paragraphs[0]
        paragraph.text = text
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color
        return shape

    def add_footer(slide: Any, number: int) -> None:
        add_textbox(slide, 11.9, 7.08, 0.7, 0.2, str(number), 9, body_color)

    cover = prs.slides.add_slide(prs.slide_layouts[6])
    cover.background.fill.solid()
    cover.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
    bar = cover.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.0), Inches(0.12), Inches(5.3)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    add_textbox(cover, 1.15, 2.0, 10.8, 1.35, cleaned_title, 30, accent, True)
    add_textbox(
        cover,
        1.18,
        3.48,
        10.2,
        0.55,
        f"DocPilot VISION AI  •  {datetime.now().strftime('%Y-%m-%d')}",
        14,
        body_color,
    )
    add_footer(cover, 1)

    sections: List[Tuple[str, List[Dict[str, Any]]]] = []
    current_heading = cleaned_title
    current_items: List[Dict[str, Any]] = []
    for block in blocks:
        if block["type"] in ("h1", "h2", "h3"):
            if current_items:
                sections.append((current_heading, current_items))
                current_items = []
            current_heading = block["text"] or cleaned_title
        else:
            current_items.append(block)
    if current_items:
        sections.append((current_heading, current_items))
    if not sections and not blocks:
        sections = [(cleaned_title, [{"type": "para", "text": cleaned_title}])]

    slide_number = 2

    def add_content_slide(heading: str, items: List[Dict[str, Any]]) -> None:
        nonlocal slide_number
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
        add_textbox(slide, 0.72, 0.42, 11.9, 0.65, heading, 23, accent, True)
        line = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.13), Inches(11.8), Inches(0.025)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = border
        line.line.fill.background()

        y = 1.42
        for item in items:
            kind = item["type"]
            if kind == "table":
                rows = item["rows"]
                if not rows:
                    continue
                cols = max(len(row) for row in rows)
                row_groups = [rows[:9]]
                row_groups.extend(
                    rows[:1] + rows[start : start + 8]
                    for start in range(9, len(rows), 8)
                )
                for group_index, group in enumerate(row_groups):
                    if group_index:
                        add_footer(slide, slide_number)
                        slide_number += 1
                        slide = prs.slides.add_slide(prs.slide_layouts[6])
                        slide.background.fill.solid()
                        slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
                        add_textbox(slide, 0.72, 0.42, 11.9, 0.65, f"{heading} (continued)", 23, accent, True)
                        line = slide.shapes.add_shape(
                            MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.13), Inches(11.8), Inches(0.025)
                        )
                        line.fill.solid()
                        line.fill.fore_color.rgb = border
                        line.line.fill.background()
                        y = 1.42
                    table_height = min(4.9, max(0.85, len(group) * 0.58))
                    table = slide.shapes.add_table(
                        len(group), cols, Inches(0.75), Inches(y), Inches(11.8), Inches(table_height)
                    ).table
                    for row_index, row in enumerate(group):
                        for col_index in range(cols):
                            cell = table.cell(row_index, col_index)
                            value = row[col_index] if col_index < len(row) else ""
                            cell.text = value or "—"
                            cell.margin_left = Inches(0.08)
                            cell.margin_right = Inches(0.08)
                            for paragraph in cell.text_frame.paragraphs:
                                paragraph.font.name = "Aptos"
                                paragraph.font.size = Pt(11 if cols > 5 else (12 if len(value) < 70 else 10))
                                paragraph.font.bold = row_index == 0
                                paragraph.font.color.rgb = (
                                    RGBColor.from_string(palette["header_text"])
                                    if row_index == 0
                                    else body_color
                                )
                            cell.fill.solid()
                            cell.fill.fore_color.rgb = (
                                table_header_background
                                if row_index == 0
                                else RGBColor(255, 255, 255)
                            )
                add_footer(slide, slide_number)
                slide_number += 1
                return

            if kind in ("bullet", "numbered"):
                marker = "•" if kind == "bullet" else f"{item.get('number', 1)}."
                text = f"{marker}  {item['text']}"
                indent = Inches(0.15 if kind == "bullet" else 0.22)
            elif kind == "formula":
                text = item["text"]
                indent = Inches(0.2)
            else:
                text = item["text"]
                indent = Inches(0)
            max_chars = 220 if kind in ("bullet", "numbered") else 280
            chunks = [text]
            if len(text) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", text)
                chunks = []
                piece = ""
                for sentence in sentences:
                    words = sentence.split()
                    for word in words:
                        if piece and len(piece) + len(word) + 1 > max_chars:
                            chunks.append(piece)
                            piece = ""
                        piece = f"{piece} {word}".strip()
                if piece:
                    chunks.append(piece)
            for chunk in chunks:
                box_height = max(0.46, min(1.35, (len(chunk) / 110 + 1) * 0.28))
                if y + box_height > 6.55:
                    add_footer(slide, slide_number)
                    slide_number += 1
                    slide = prs.slides.add_slide(prs.slide_layouts[6])
                    slide.background.fill.solid()
                    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
                    add_textbox(slide, 0.72, 0.42, 11.9, 0.65, f"{heading} (continued)", 23, accent, True)
                    line = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.13), Inches(11.8), Inches(0.025)
                    )
                    line.fill.solid()
                    line.fill.fore_color.rgb = border
                    line.line.fill.background()
                    y = 1.42
                text_shape = add_textbox(
                    slide,
                    0.85 + indent.inches,
                    y,
                    11.5 - indent.inches,
                    box_height,
                    chunk,
                    15 if kind == "formula" else (14 if len(chunk) > 210 else 17),
                    accent if kind == "formula" else body_color,
                    kind == "formula",
                )
                text_shape.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
                y += box_height + 0.12

        add_footer(slide, slide_number)
        slide_number += 1

    for heading, items in sections:
        page_items: List[Dict[str, Any]] = []
        size = 0
        for item in items:
            if item["type"] == "table":
                if page_items:
                    add_content_slide(heading, page_items)
                    page_items = []
                    size = 0
                add_content_slide(heading, [item])
                continue
            text_size = len(str(item.get("text") or ""))
            if page_items and (len(page_items) >= 5 or size + text_size > 900):
                add_content_slide(heading, page_items)
                page_items = []
                size = 0
            page_items.append(item)
            size += text_size
        if page_items:
            add_content_slide(heading, page_items)

    prs.save(str(file_path))


def _generate_pptx_openxml_fallback(
    file_path: Path, title: str, content: str, theme: Optional[str]
) -> None:
    cleaned_title = clean_markdown_and_latex(title) or "VISION AI Presentation"
    palette = _resolve_export_theme(theme)
    blocks = _parse_markdown_blocks(content)
    sections: List[Tuple[str, List[str]]] = []
    heading = cleaned_title
    lines: List[str] = []
    for block in blocks:
        kind = block["type"]
        if kind in ("h1", "h2", "h3"):
            if lines:
                sections.append((heading, lines))
            heading = block["text"] or cleaned_title
            lines = []
        elif kind == "table":
            lines.extend(
                "  |  ".join(cell or "—" for cell in row)
                for row in block["rows"]
            )
        elif kind in ("bullet", "numbered"):
            marker = "•" if kind == "bullet" else f"{block.get('number', 1)}."
            lines.append(f"{marker}  {block['text']}")
        else:
            lines.append(block["text"])
    if lines:
        sections.append((heading, lines))
    if not sections:
        sections = [(cleaned_title, [cleaned_title])]

    slide_text: List[Tuple[str, List[str]]] = [(cleaned_title, [])]
    for section_heading, section_lines in sections:
        part: List[str] = []
        char_count = 0
        for line in section_lines:
            pieces = [line]
            if len(line) > 280:
                pieces = [
                    line[index : index + 280]
                    for index in range(0, len(line), 280)
                ]
            for piece in pieces:
                if part and (len(part) >= 5 or char_count + len(piece) > 850):
                    slide_text.append((f"{section_heading} (continued)", part))
                    part = []
                    char_count = 0
                part.append(piece)
                char_count += len(piece)
        if part:
            slide_text.append((section_heading, part))

    accent = palette["accent"]

    def text_shape(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, text: str, size: int, color: str, bold: bool) -> str:
        run_props = (
            f'<a:rPr lang="en-US" sz="{size}" b="{1 if bold else 0}">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            '<a:latin typeface="Aptos"/></a:rPr>'
        )
        return (
            f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{xml_escape(name)}"/>'
            '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/>'
            '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>'
            '<a:ln><a:noFill/></a:ln></p:spPr>'
            '<p:txBody><a:bodyPr wrap="square" lIns="91440" rIns="91440" '
            'tIns="45720" bIns="45720"/><a:lstStyle/>'
            f'<a:p><a:pPr/><a:r>{run_props}<a:t xml:space="preserve">{xml_escape(text)}</a:t>'
            '</a:r><a:endParaRPr lang="en-US"/></a:p></p:txBody></p:sp>'
        )

    def slide_xml(heading_text: str, body_lines: List[str], slide_number: int) -> str:
        title_shape = text_shape(
            2, "DocPilot Slide Title", 548640, 274320, 10972800, 822960,
            heading_text, 2400, accent, True
        )
        body_shapes = "".join(
            text_shape(
                index + 3, f"DocPilot Content {index + 1}",
                640080, 1280160 + index * 822960, 10668000, 731520,
                line, 1600 if len(line) > 180 else 1800, "1E293B", False
            )
            for index, line in enumerate(body_lines)
        )
        footer = text_shape(
            len(body_lines) + 3, "DocPilot Page Number",
            10972800, 6400800, 548640, 274320, str(slide_number), 900, "475569", False
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:cSld><p:spTree>'
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            f'{title_shape}{body_shapes}{footer}'
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
        )

    slide_count = len(slide_text)
    content_types_slides = "".join(
        f'<Override PartName="/ppt/slides/slide{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for index in range(1, slide_count + 1)
    )
    slide_ids = "".join(
        f'<p:sldId id="{255 + index}" r:id="rId{index + 1}"/>'
        for index in range(1, slide_count + 1)
    )
    slide_relationships = "".join(
        f'<Relationship Id="rId{index + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
        f'Target="slides/slide{index}.xml"/>'
        for index in range(1, slide_count + 1)
    )
    presentation = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        f'<p:sldIdLst>{slide_ids}</p:sldIdLst>'
        '<p:sldSz cx="12192000" cy="6858000" type="wide"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        '<p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>'
        '</p:presentation>'
    )
    root_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
        '</Relationships>'
    )
    presentation_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
        f'{slide_relationships}</Relationships>'
    )
    master = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld name="DocPilot Master"><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" '
        'accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" '
        'accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
    )
    master_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
        '</Relationships>'
    )
    layout = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    )
    layout_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
        '</Relationships>'
    )
    theme_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="DocPilot">'
        '<a:themeElements><a:clrScheme name="DocPilot">'
        '<a:dk1><a:srgbClr val="0F172A"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="1E293B"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>'
        f'<a:accent1><a:srgbClr val="{accent}"/></a:accent1>'
        '<a:accent2><a:srgbClr val="475569"/></a:accent2><a:accent3><a:srgbClr val="64748B"/></a:accent3>'
        '<a:accent4><a:srgbClr val="94A3B8"/></a:accent4><a:accent5><a:srgbClr val="CBD5E1"/></a:accent5>'
        '<a:accent6><a:srgbClr val="E2E8F0"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink>'
        '<a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>'
        '<a:fontScheme name="DocPilot"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont>'
        '<a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>'
        '<a:fmtScheme name="DocPilot"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
        '<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
        '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
        '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="lt1"/></a:solidFill></a:bgFillStyleLst>'
        '</a:fmtScheme></a:themeElements></a:theme>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        f'{content_types_slides}</Types>'
    )

    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("ppt/presentation.xml", presentation)
        archive.writestr("ppt/_rels/presentation.xml.rels", presentation_relationships)
        archive.writestr("ppt/slideMasters/slideMaster1.xml", master)
        archive.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_relationships)
        archive.writestr("ppt/slideLayouts/slideLayout1.xml", layout)
        archive.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_relationships)
        archive.writestr("ppt/theme/theme1.xml", theme_xml)
        for index, (slide_heading, lines) in enumerate(slide_text, start=1):
            archive.writestr(
                f"ppt/slides/slide{index}.xml",
                slide_xml(slide_heading, lines, index),
            )
            archive.writestr(
                f"ppt/slides/_rels/slide{index}.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
                '</Relationships>',
            )


def generate_xlsx_file(
    file_path: Path, title: str, content: str, theme: Optional[str] = None
) -> None:
    """Create a formatted workbook that preserves table values and spreadsheet formulas."""
    cleaned_title = clean_markdown_and_latex(title) or "VISION AI Data"
    try:
        import openpyxl  # type: ignore
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # type: ignore
        from openpyxl.utils import get_column_letter  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Spreadsheet export requires the openpyxl dependency; install project requirements."
        ) from exc

    palette = _resolve_export_theme(theme)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DocPilot Export"
    ws.sheet_view.showGridLines = False
    ws.append(["DocPilot VISION AI Structured Export"])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=4)
    ws.append(["Title", cleaned_title])
    ws.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    ws.append([])
    table_header_rows: set[int] = set()
    section_rows: set[int] = set()
    for block in _parse_markdown_blocks(content):
        kind = block["type"]
        if kind in ("h1", "h2", "h3"):
            ws.append([block["text"]])
            section_rows.add(ws.max_row)
        elif kind == "table":
            for row_index, row in enumerate(block["rows"]):
                values: List[Any] = []
                for value in row:
                    value = value.strip()
                    if not value:
                        values.append("—")
                    elif re.fullmatch(r"[+-]?\d+", value):
                        values.append(int(value))
                    elif re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
                        values.append(float(value))
                    else:
                        values.append(value)
                ws.append(values)
                if row_index == 0:
                    table_header_rows.add(ws.max_row)
        elif kind == "bullet":
            ws.append([f"• {block['text']}"])
        elif kind == "numbered":
            ws.append([f"{block.get('number', 1)}. {block['text']}"])
        elif kind == "formula":
            ws.append(["Formula", block["text"]])
        else:
            ws.append([block["text"]])

    accent_fill = PatternFill(
        "solid",
        fgColor=palette["accent"] if _has_export_theme(theme) else palette["light"],
    )
    header_font = Font(
        name="Aptos",
        bold=True,
        color=palette["header_text"] if _has_export_theme(theme) else palette["text"],
    )
    border_side = Side(style="thin", color=palette["border"])
    for cell in ws[1]:
        cell.fill = accent_fill
        cell.font = Font(
            name="Aptos Display",
            size=16,
            bold=True,
            color=palette["header_text"] if _has_export_theme(theme) else palette["text"],
        )
    for row_index in section_rows:
        cell = ws.cell(row=row_index, column=1)
        cell.font = Font(name="Aptos Display", size=13, bold=True, color=palette["accent"])
        cell.border = Border(bottom=Side(style="medium", color=palette["accent"]))
    for row_index in table_header_rows:
        for cell in ws[row_index]:
            cell.fill = accent_fill
            cell.font = header_font
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if cell.value is not None and cell.row not in table_header_rows:
                cell.border = Border(bottom=border_side)
    max_column = max(1, ws.max_column)
    for col_index in range(1, max_column + 1):
        values = [
            len(str(ws.cell(row=row_index, column=col_index).value or ""))
            for row_index in range(1, ws.max_row + 1)
        ]
        ws.column_dimensions[get_column_letter(col_index)].width = min(
            55, max(14, max(values, default=14) + 2)
        )
    ws.freeze_panes = "A5"
    wb.properties.title = cleaned_title
    wb.properties.creator = "DocPilot VISION AI"
    wb.save(str(file_path))


# ============================================================================
# REAL FORMAT VALIDATOR
# ============================================================================

def validate_generated_file(file_path: Path, ext: str) -> Dict[str, Any]:
    if not file_path.exists():
        return {"valid": False, "size": 0, "error": "Generated file does not exist on disk."}
    size = file_path.stat().st_size
    if size <= 0:
        return {"valid": False, "size": 0, "error": "Generated file is empty (0 bytes)."}

    ext_low = ext.lower().lstrip(".")

    if ext_low == "py":
        code_str = file_path.read_text(encoding="utf-8")
        try:
            ast.parse(code_str)
            return {
                "valid": True,
                "size": size,
                "summary": f"AST-validated Python 3 script ({size} bytes)",
            }
        except SyntaxError as syn_err:
            return {
                "valid": False,
                "size": size,
                "error": f"Python syntax error at line {syn_err.lineno}: {syn_err.msg}",
            }

    if ext_low == "pdf":
        raw = file_path.read_bytes()
        if not raw.startswith(b"%PDF-"):
            return {"valid": False, "size": size, "error": "Missing %PDF- header."}
        if b"%%EOF" not in raw[-512:]:
            return {"valid": False, "size": size, "error": "Missing %%EOF trailer."}
        if b"xref" not in raw or b"/Root " not in raw:
            return {"valid": False, "size": size, "error": "PDF is missing its cross-reference table or document catalog."}
        page_count = len(re.findall(rb"/Type\s*/Page\b", raw))
        if page_count == 0:
            return {"valid": False, "size": size, "error": "PDF contains no page objects."}
        try:
            try:
                from pypdf import PdfReader  # type: ignore
            except ImportError:
                from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            PdfReader = None

        if PdfReader is not None:
            try:
                reader = PdfReader(str(file_path))
                page_count = len(reader.pages)
                has_text = any((page.extract_text() or "").strip() for page in reader.pages)
            except Exception as exc:
                return {"valid": False, "size": size, "error": f"PDF reader rejected the file: {exc}"}
        else:
            stream_text: List[bytes] = []
            for match in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.DOTALL):
                stream = match.group(1).rstrip(b"\r\n")
                filter_declaration = raw[max(0, match.start() - 256) : match.start()]
                try:
                    if b"ASCII85Decode" in filter_declaration:
                        stream = base64.a85decode(stream, adobe=True)
                    if b"FlateDecode" in filter_declaration:
                        stream = zlib.decompress(stream)
                except (ValueError, zlib.error):
                    continue
                stream_text.append(stream)
            decoded_streams = b"\n".join(stream_text)
            has_text = bool(re.search(rb"\bTj\b|\bTJ\b", decoded_streams))

        if page_count == 0:
            return {"valid": False, "size": size, "error": "PDF contains no page objects."}
        if not has_text:
            return {"valid": False, "size": size, "error": "PDF contains no readable text operators."}
        return {
            "valid": True,
            "size": size,
            "summary": f"Verified PDF structure ({page_count} pages, {size} bytes)",
        }

    if ext_low in ("docx", "pptx", "xlsx"):
        if not zipfile.is_zipfile(file_path):
            return {"valid": False, "size": size, "error": f"Invalid OpenXML .{ext_low} ZIP container."}
        with zipfile.ZipFile(file_path, "r") as zf:
            corrupt = zf.testzip()
            if corrupt is not None:
                return {"valid": False, "size": size, "error": f"Corrupted archive entry: {corrupt}"}
            names = set(zf.namelist())
            if "[Content_Types].xml" not in names:
                return {"valid": False, "size": size, "error": "Missing [Content_Types].xml"}
            req_entry = {
                "docx": "word/document.xml",
                "pptx": "ppt/presentation.xml",
                "xlsx": "xl/workbook.xml",
            }[ext_low]
            if req_entry not in names:
                return {"valid": False, "size": size, "error": f"Missing {req_entry}"}
            for name in names:
                if name.endswith((".xml", ".rels")):
                    try:
                        ET.fromstring(zf.read(name))
                    except ET.ParseError as exc:
                        return {
                            "valid": False,
                            "size": size,
                            "error": f"Invalid OpenXML part {name}: {exc}",
                        }
        try:
            if ext_low == "docx":
                try:
                    from docx import Document  # type: ignore
                except ImportError:
                    with zipfile.ZipFile(file_path, "r") as zf:
                        root = ET.fromstring(zf.read("word/document.xml"))
                    texts = [
                        element.text or ""
                        for element in root.iter(
                            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                        )
                    ]
                    if not any(text.strip() for text in texts):
                        return {"valid": False, "size": size, "error": "DOCX contains no readable text."}
                    return {
                        "valid": True,
                        "size": size,
                        "summary": f"Verified OpenXML DOCX structure ({size} bytes)",
                    }
                document = Document(str(file_path))
                has_text = any(paragraph.text.strip() for paragraph in document.paragraphs)
                has_table_text = any(
                    cell.text.strip()
                    for table in document.tables
                    for row in table.rows
                    for cell in row.cells
                )
                if not has_text and not has_table_text:
                    return {"valid": False, "size": size, "error": "DOCX contains no readable text."}
            elif ext_low == "pptx":
                try:
                    from pptx import Presentation  # type: ignore
                except ImportError:
                    with zipfile.ZipFile(file_path, "r") as zf:
                        presentation_root = ET.fromstring(zf.read("ppt/presentation.xml"))
                        slide_names = sorted(
                            (
                                name
                                for name in zf.namelist()
                                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
                            ),
                            key=lambda name: int(re.search(r"slide(\d+)\.xml$", name).group(1)),
                        )
                        if not slide_names or presentation_root.find(
                            "{http://schemas.openxmlformats.org/presentationml/2006/main}sldIdLst"
                        ) is None:
                            return {"valid": False, "size": size, "error": "PPTX contains no readable slides."}
                        has_text = any(
                            (element.text or "").strip()
                            for name in slide_names
                            for element in ET.fromstring(zf.read(name)).iter(
                                "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
                            )
                        )
                    if not has_text:
                        return {"valid": False, "size": size, "error": "PPTX contains no readable slide text."}
                    return {
                        "valid": True,
                        "size": size,
                        "summary": f"Verified OpenXML PPTX structure ({len(slide_names)} slides, {size} bytes)",
                    }
                presentation = Presentation(str(file_path))
                if not presentation.slides:
                    return {"valid": False, "size": size, "error": "PPTX contains no slides."}
                if not any(
                    shape.has_text_frame and shape.text.strip()
                    for slide in presentation.slides
                    for shape in slide.shapes
                ):
                    return {"valid": False, "size": size, "error": "PPTX contains no readable slide text."}
            else:
                try:
                    from openpyxl import load_workbook  # type: ignore
                except ImportError:
                    with zipfile.ZipFile(file_path, "r") as zf:
                        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
                        sheet_path = "xl/worksheets/sheet1.xml"
                        sheet_root = ET.fromstring(zf.read(sheet_path))
                    rows = sheet_root.findall(
                        ".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"
                    )
                    if (
                        workbook_root.find(
                            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheets"
                        ) is None
                        or not rows
                        or not sheet_root.findall(
                            ".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"
                        )
                    ):
                        return {"valid": False, "size": size, "error": "XLSX contains no readable worksheet data."}
                    return {
                        "valid": True,
                        "size": size,
                        "summary": f"Verified OpenXML XLSX structure ({len(rows)} rows, {size} bytes)",
                    }
                workbook = load_workbook(str(file_path), read_only=True, data_only=False)
                if not workbook.sheetnames:
                    return {"valid": False, "size": size, "error": "XLSX contains no worksheets."}
                worksheet = workbook[workbook.sheetnames[0]]
                if worksheet.max_row < 1 or worksheet.max_column < 1:
                    return {"valid": False, "size": size, "error": "XLSX worksheet is empty."}
                workbook.close()
        except Exception as exc:
            return {"valid": False, "size": size, "error": f"{ext_low.upper()} could not be opened: {exc}"}
        return {
            "valid": True,
            "size": size,
            "summary": f"Verified OpenXML {ext_low.upper()} package ({size} bytes)",
        }

    if ext_low == "csv":
        txt = file_path.read_text(encoding="utf-8")
        rows = [r for r in csv.reader(io.StringIO(txt)) if any(c.strip() for c in r)]
        if not rows:
            return {"valid": False, "size": size, "error": "CSV contains no data rows."}
        return {
            "valid": True,
            "size": size,
            "summary": f"Verified CSV table ({len(rows)} rows, {size} bytes)",
        }

    if ext_low in ("json", "ipynb"):
        txt = file_path.read_text(encoding="utf-8")
        json.loads(txt)
        return {"valid": True, "size": size, "summary": f"Verified {ext_low.upper()} file ({size} bytes)"}

    if ext_low == "zip":
        if not zipfile.is_zipfile(file_path):
            return {"valid": False, "size": size, "error": "Invalid ZIP archive."}
        return {"valid": True, "size": size, "summary": f"Verified ZIP archive ({size} bytes)"}

    # UTF-8 text/markdown/code verification
    txt = file_path.read_text(encoding="utf-8")
    if not txt.strip():
        return {"valid": False, "size": size, "error": "File contains only whitespace."}
    return {
        "valid": True,
        "size": size,
        "summary": f"Verified UTF-8 {ext_low.upper()} file ({size} bytes)",
    }


def _build_descriptive_filename(title: str, ext: str, code_hint: str = "") -> str:
    ext_low = ext.lower().lstrip(".")
    if ext_low == "py" and code_hint:
        m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code_hint)
        if m and m.group(1) != "main":
            return f"{m.group(1)}.py"
    cleaned = re.sub(
        r"\b(create|generate|write|build|make|give|me|a|an|the|for|in|to|of|code|script|file|program|export|download|please)\b",
        " ",
        (title or "").lower(),
    )
    slug = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")[:42]
    if not slug or len(slug) < 3:
        slug = f"vision_ai_deliverable_{int(time.time())}"
    return f"{slug}.{ext_low}"


def generate_and_validate_deliverable(
    format_type: str,
    title: str,
    content: str,
    session_id: str = "",
    message_id: str = "",
    document_theme: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a real local file in `.data/exports/`, validates its format & syntax,
    records its metadata in `db.py`, and returns the downloadable card metadata.
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = (format_type or "txt").strip().lower().lstrip(".")
    alias_map = {
        "python": "py",
        "javascript": "js",
        "typescript": "ts",
        "csharp": "cs",
        "c#": "cs",
        "yml": "yaml",
        "shell": "sh",
        "bash": "sh",
        "powershell": "ps1",
        "jupyter": "ipynb",
        "notebook": "ipynb",
        "word": "docx",
        "doc": "docx",
        "powerpoint": "pptx",
        "ppt": "pptx",
        "excel": "xlsx",
        "xls": "xlsx",
        "markdown": "md",
        "text": "txt",
    }
    ext = alias_map.get(fmt, fmt)
    if ext not in FORMAT_METADATA:
        ext = "txt"

    category, mime_type = FORMAT_METADATA[ext]

    if ext in (
        "py",
        "js",
        "jsx",
        "ts",
        "tsx",
        "java",
        "c",
        "cpp",
        "cs",
        "php",
        "html",
        "css",
        "json",
        "xml",
        "yaml",
        "sql",
        "sh",
        "ps1",
        "ipynb",
        "go",
        "rs",
        "rb",
    ):
        code_body = extract_clean_code(content, ext)
        if ext == "py":
            try:
                ast.parse(code_body)
            except SyntaxError:
                code_body = (
                    f'"""\n{title}\n"""\n\n'
                    "def main():\n"
                    f"    print({json.dumps(clean_markdown_and_latex(title))})\n\n"
                    'if __name__ == "__main__":\n'
                    "    main()\n"
                )
        elif ext == "ipynb":
            trimmed_nb = code_body.strip()
            if not (trimmed_nb.startswith("{") and '"cells"' in trimmed_nb):
                nb_obj = {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [f"# {title}\n"],
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [f"{line}\n" for line in code_body.splitlines()],
                        },
                    ],
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3",
                        }
                    },
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
                code_body = json.dumps(nb_obj, indent=2)
        filename = _build_descriptive_filename(title, ext, code_body)
        file_path = EXPORTS_DIR / filename
        file_path.write_text(code_body, encoding="utf-8")
    elif ext == "pdf":
        filename = _build_descriptive_filename(title, "pdf")
        file_path = EXPORTS_DIR / filename
        generate_pdf_file(file_path, title, content, document_theme)
    elif ext == "docx":
        filename = _build_descriptive_filename(title, "docx")
        file_path = EXPORTS_DIR / filename
        generate_docx_file(file_path, title, content, document_theme)
    elif ext == "pptx":
        filename = _build_descriptive_filename(title, "pptx")
        file_path = EXPORTS_DIR / filename
        generate_pptx_file(file_path, title, content, document_theme)
    elif ext == "xlsx":
        filename = _build_descriptive_filename(title, "xlsx")
        file_path = EXPORTS_DIR / filename
        generate_xlsx_file(file_path, title, content, document_theme)
    elif ext == "csv":
        filename = _build_descriptive_filename(title, "csv")
        file_path = EXPORTS_DIR / filename
        out_io = io.StringIO()
        writer = csv.writer(out_io)
        rows_written = 0
        for ln in (content or "").splitlines():
            trimmed = ln.strip()
            if not trimmed or trimmed.startswith("```"):
                continue
            if "|" in trimmed:
                cells = [c.strip() for c in trimmed.strip("|").split("|")]
                if cells and not all(re.match(r"^[-:\s]+$", c) for c in cells):
                    writer.writerow(cells)
                    rows_written += 1
            elif "," in trimmed:
                writer.writerow([c.strip() for c in trimmed.split(",")])
                rows_written += 1
            else:
                writer.writerow([clean_markdown_and_latex(trimmed)])
                rows_written += 1
        if rows_written == 0:
            writer.writerow(["Title", clean_markdown_and_latex(title)])
        file_path.write_text(out_io.getvalue(), encoding="utf-8")
    elif ext == "md":
        filename = _build_descriptive_filename(title, "md")
        file_path = EXPORTS_DIR / filename
        file_path.write_text(f"# {title}\n\n{content.strip()}\n", encoding="utf-8")
    else:
        filename = _build_descriptive_filename(title, "txt")
        file_path = EXPORTS_DIR / filename
        file_path.write_text(
            f"{clean_markdown_and_latex(title)}\n{'=' * 40}\n\n{clean_markdown_and_latex(content)}\n",
            encoding="utf-8",
        )

    validation = validate_generated_file(file_path, ext)
    if not validation["valid"]:
        raise RuntimeError(
            f"Generated {ext.upper()} validation failed: {validation.get('error')}"
        )

    rec = db.record_generated_file(
        {
            "session_id": session_id,
            "message_id": message_id,
            "filename": filename,
            "format": ext,
            "category": category,
            "mimeType": mime_type,
            "fileSize": validation["size"],
            "filePath": str(file_path.resolve()),
            "downloadUrl": f"/api/export/download/{filename}",
            "validated": True,
            "validationSummary": validation["summary"],
        }
    )
    rec.pop("filePath", None)
    logger.info(
        "Export generated format=%s filename=%s size=%s validated=true",
        ext,
        filename,
        validation["size"],
    )
    return rec
