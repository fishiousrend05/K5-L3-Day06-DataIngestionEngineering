"""Read and normalize source data."""
import json
import logging
import re
import unicodedata
from pathlib import Path
from pypdf import PdfReader
from .models import DocumentPage

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding paragraph structure."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Fix common PDF ligatures and soft hyphens
    text = (
        text.replace("\u00ad", "")
        .replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    # Rejoin hyphenated line breaks
    text = re.sub(r"-\s*\n\s*", "", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Keep double newlines for paragraph boundaries, normalize single line breaks inside paragraphs
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned_paras = []
    for p in paragraphs:
        lines = [line.strip() for line in p.splitlines() if line.strip()]
        if lines:
            cleaned_paras.append(" ".join(lines))
    return "\n\n".join(cleaned_paras).strip()


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    page_range: tuple[int, int] = (37, 95),
    log_path: Path | None = None,
) -> list[DocumentPage]:
    """Extract physical pages in range, keep page number/source, and record empty pages."""
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    start_page, end_page = page_range
    pages: list[DocumentPage] = []
    ingestion_logs: list[dict] = []

    for page_num in range(start_page, min(end_page + 1, len(reader.pages) + 1)):
        raw_text = reader.pages[page_num - 1].extract_text() or ""
        norm_text = normalize_text(raw_text)

        if len(norm_text) < 100:
            msg = f"Page {page_num} skipped: insufficient text ({len(norm_text)} chars)"
            logger.warning(msg)
            ingestion_logs.append({"page": page_num, "status": "skipped", "chars": len(norm_text), "reason": msg})
            continue

        chapter = "Chapter 1" if page_num <= 71 else "Chapter 2"
        section_topic = (
            "DW/BI & Dimensional Modeling Primer"
            if chapter == "Chapter 1"
            else "Dimensional Modeling Techniques Overview"
        )

        metadata = {
            "document_id": document_id,
            "source": source,
            "page": page_num,
            "chapter": chapter,
            "topic": section_topic,
            "char_count": len(norm_text),
        }

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_num,
                text=norm_text,
                metadata=metadata,
            )
        )
        ingestion_logs.append({"page": page_num, "status": "loaded", "chars": len(norm_text), "chapter": chapter})

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(ingestion_logs, f, indent=2, ensure_ascii=False)

    return pages
