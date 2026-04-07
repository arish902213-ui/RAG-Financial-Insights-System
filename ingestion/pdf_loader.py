from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz


@dataclass
class PDFPage:
    file_name: str
    page_number: int
    text: str


def load_pdf_pages(pdf_path: str | Path) -> List[PDFPage]:
    path = Path(pdf_path)
    pages: List[PDFPage] = []

    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            normalized = " ".join(text.split())
            if not normalized:
                continue
            pages.append(
                PDFPage(
                    file_name=path.name,
                    page_number=index,
                    text=normalized,
                )
            )

    return pages


def load_multiple_pdfs(pdf_paths: List[str | Path]) -> List[PDFPage]:
    all_pages: List[PDFPage] = []
    for pdf_path in pdf_paths:
        all_pages.extend(load_pdf_pages(pdf_path))
    return all_pages
