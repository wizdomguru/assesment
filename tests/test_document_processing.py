import fitz
import pytest

from app.models.documents import DocumentSection
from app.services.document_processing import chunk_sections, parse_markdown, parse_pdf


def test_markdown_parser_preserves_heading_context() -> None:
    sections = parse_markdown("# Authentication\nUse OAuth.\n\n## Tokens\nRotate keys.")

    assert [(item.section, item.text) for item in sections] == [
        ("Authentication", "Use OAuth."),
        ("Tokens", "Rotate keys."),
    ]


def test_pdf_parser_preserves_page_numbers(tmp_path) -> None:
    pdf_path = tmp_path / "guide.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Page one")
    document.new_page().insert_text((72, 72), "Page two")
    document.save(pdf_path)
    document.close()

    sections = parse_pdf(pdf_path)

    assert [(item.page_number, item.text) for item in sections] == [
        (1, "Page one"),
        (2, "Page two"),
    ]


def test_pdf_parser_preserves_numbered_headings(tmp_path) -> None:
    pdf_path = tmp_path / "guide.pdf"
    document = fitz.open()
    document.new_page().insert_text(
        (72, 72),
        "1. Provisioning workflow\n7. Cache and data freshness\n"
        "Order-status read cache TTL: 60 seconds\n"
        "8. RAG assistant retrieval policy\nRetrieval uses approved sources.\n"
        "9. Sample operating scenarios\nScenario details.\n"
        "10. Out-of-scope information\nExcluded details.",
    )
    document.save(pdf_path)
    document.close()

    sections = parse_pdf(pdf_path)

    assert [section.section for section in sections] == [
        "7. Cache and data freshness",
        "8. RAG assistant retrieval policy",
        "9. Sample operating scenarios",
        "10. Out-of-scope information",
    ]
    assert "60 seconds" in sections[0].text


def test_pdf_parser_carries_section_across_pages_and_chunks(tmp_path) -> None:
    pdf_path = tmp_path / "guide.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "7. Cache and data freshness\nPage one")
    document.new_page().insert_text((72, 72), "Page two")
    document.save(pdf_path)
    document.close()

    sections = parse_pdf(pdf_path)
    chunks = chunk_sections(sections, "doc-1", "guide.pdf", chunk_size=8, overlap=0)

    assert [section.section for section in sections] == [
        "7. Cache and data freshness",
        "7. Cache and data freshness",
    ]
    assert all(chunk.section == "7. Cache and data freshness" for chunk in chunks)


def test_chunking_keeps_metadata_and_overlap() -> None:
    sections = [
        DocumentSection(
            text="alpha beta gamma delta epsilon", section="Intro", page_number=3
        )
    ]

    chunks = chunk_sections(sections, "doc-1", "guide.md", chunk_size=16, overlap=6)

    assert [chunk.text for chunk in chunks] == [
        "alpha beta gamma",
        "gamma delta",
        "delta epsilon",
    ]
    assert all(chunk.section == "Intro" for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)


def test_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_sections([], "doc-1", "guide.md", chunk_size=10, overlap=10)