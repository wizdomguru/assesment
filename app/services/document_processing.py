import re
from pathlib import Path

import fitz

from app.models.documents import DocumentChunk, DocumentSection


HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
PDF_HEADING_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*\.\s+(.+?)\s*$")


def parse_markdown(text: str) -> list[DocumentSection]:
    """Parse Markdown into sections while retaining the latest heading."""
    sections: list[DocumentSection] = []
    current_heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append(DocumentSection(text=content, section=current_heading))

    for line in text.splitlines():
        heading = HEADING_PATTERN.match(line.strip())
        if heading:
            flush()
            body.clear()
            current_heading = heading.group(1)
        else:
            body.append(line)
    flush()
    return sections


def parse_pdf(path: str | Path) -> list[DocumentSection]:
    """Extract PDF sections, recognizing numbered headings across the document."""
    sections: list[DocumentSection] = []
    current_heading: str | None = None
    try:
        with fitz.open(path) as document:
            for page_index, page in enumerate(document):
                body: list[str] = []

                def flush() -> None:
                    text = " ".join(body).strip()
                    if text:
                        sections.append(
                            DocumentSection(
                                text=text,
                                section=current_heading,
                                page_number=page_index + 1,
                            )
                        )

                for line in page.get_text("text").splitlines():
                    heading = PDF_HEADING_PATTERN.match(line)
                    if heading:
                        flush()
                        body.clear()
                        current_heading = f"{heading.group(1)}. {heading.group(2)}"
                    else:
                        body.append(line)
                flush()
    except (fitz.FileDataError, OSError) as exc:
        raise ValueError(f"Unable to parse PDF: {path}") from exc
    return sections


def parse_document(path: str | Path) -> list[DocumentSection]:
    """Parse a supported Markdown or PDF document from disk."""
    document_path = Path(path)
    suffix = document_path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return parse_markdown(document_path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        return parse_pdf(document_path)
    raise ValueError(f"Unsupported document type: {suffix or '<none>'}")


def chunk_sections(
    sections: list[DocumentSection],
    document_id: str,
    filename: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Split sections into bounded chunks without losing section metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    for section in sections:
        text = section.text.strip()
        words = text.split()
        start = 0
        while start < len(words):
            end = start
            length = 0
            while end < len(words):
                word_length = len(words[end])
                proposed_length = word_length if end == start else length + 1 + word_length
                if proposed_length > chunk_size and end > start:
                    break
                length = proposed_length
                end += 1
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunk_id = f"{document_id}-{len(chunks)}"
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        document_id=document_id,
                        filename=filename,
                        chunk_id=chunk_id,
                        section=section.section,
                        page_number=section.page_number,
                    )
                )
            if end >= len(words):
                break
            overlap_length = 0
            next_start = end
            while next_start > start and overlap_length < overlap:
                next_start -= 1
                overlap_length += len(words[next_start]) + (1 if overlap_length else 0)
            start = next_start
    return chunks