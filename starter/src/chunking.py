"""Implement fixed-size and structure-aware chunking."""
import re
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split text into overlapping character windows and retain page metadata."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")

    # Collapse newlines into spaces for pure fixed-window baseline
    flat_text = re.sub(r"\s+", " ", page.text).strip()
    if not flat_text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    idx = 0
    chunk_idx = 0

    while idx < len(flat_text):
        window = flat_text[idx : idx + size].strip()
        if window:
            chunk_id = f"{page.document_id}_p{page.page:03d}_fixed_{chunk_idx:02d}"
            meta = dict(page.metadata)
            meta.update({
                "chunk_index": chunk_idx,
                "chunk_method": "fixed_size",
                "char_length": len(window),
            })
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=window,
                    metadata=meta,
                )
            )
            chunk_idx += 1
        if idx + size >= len(flat_text):
            break
        idx += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Group paragraphs/sentences cleanly, split oversized units at sentence boundaries, and retain metadata."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", page.text) if b.strip()]
    units: list[str] = []
    for block in raw_blocks:
        if len(block) <= max_size:
            units.append(block)
        else:
            # Split long blocks on sentence boundaries so sentences stay intact
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9■•])", block)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(s) <= max_size:
                    units.append(s)
                else:
                    # Hard fallback split on word boundary if single sentence > max_size
                    words = s.split(" ")
                    cur = ""
                    for w in words:
                        if len(cur) + len(w) + 1 <= max_size:
                            cur = f"{cur} {w}".strip()
                        else:
                            if cur:
                                units.append(cur)
                            cur = w
                    if cur:
                        units.append(cur)

    chunks: list[Chunk] = []
    current_buf = ""
    chunk_idx = 0

    for unit in units:
        candidate = f"{current_buf}\n\n{unit}".strip() if current_buf else unit
        if len(candidate) <= max_size:
            current_buf = candidate
        else:
            if current_buf:
                chunk_id = f"{page.document_id}_p{page.page:03d}_para_{chunk_idx:02d}"
                meta = dict(page.metadata)
                meta.update({
                    "chunk_index": chunk_idx,
                    "chunk_method": "by_paragraph",
                    "char_length": len(current_buf),
                })
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=page.document_id,
                        source=page.source,
                        page=page.page,
                        text=current_buf,
                        metadata=meta,
                    )
                )
                chunk_idx += 1
            current_buf = unit

    if current_buf:
        chunk_id = f"{page.document_id}_p{page.page:03d}_para_{chunk_idx:02d}"
        meta = dict(page.metadata)
        meta.update({
            "chunk_index": chunk_idx,
            "chunk_method": "by_paragraph",
            "char_length": len(current_buf),
        })
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=current_buf,
                metadata=meta,
            )
        )

    return chunks
