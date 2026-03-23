from __future__ import annotations

from typing import List

import pdfplumber


def load_pdf(path: str) -> List[str]:
    """
    Extract text lines from a PDF in natural reading order using pdfplumber.

    Args:
        path: Absolute path to a CV PDF file.

    Returns:
        Ordered list of clean, non-empty text lines from the document.
    """
    lines: List[str] = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if not page_text:
                continue
            for line in page_text.splitlines():
                clean = line.strip()
                if not clean:
                    continue
                lines.append(clean)

    return lines
