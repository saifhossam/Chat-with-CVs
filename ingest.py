from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_cv, extract_candidate_name
from db.qdrant_client import setup_collection, upsert_chunks


def ingest_cvs(pdf_paths: List[str], reset: bool = True) -> List[Document]:
    """
    Full ingestion pipeline: load → chunk → push to Qdrant.

    Steps per CV:
      1. load_pdf()              — pdfplumber extracts lines in reading order
      2. extract_candidate_name() — NER identifies the candidate's name
      3. chunk_cv()              — Docling detects headers, lines split into
                                   one Document per section
      4. upsert_chunks()         — embed + push to Qdrant in batches

    Args:
        pdf_paths: Absolute paths to CV PDF files.
        reset:     Wipe and recreate the Qdrant collection before ingesting.
                   Set False to append to an existing collection.

    Returns:
        Flat list of all Document chunks across all CVs.
        Useful for building a BM25 index or populating the UI knowledge base.
    """
    setup_collection(reset=reset)

    all_chunks: List[Document] = []

    for path in pdf_paths:
        print(f"\n[Ingest] Processing: {os.path.basename(path)}")

        lines          = load_pdf(path)
        candidate_name = extract_candidate_name(lines)
        print(f"[Ingest] Candidate: {candidate_name}")

        chunks   = chunk_cv(pdf_path=path, lines=lines, candidate_name=candidate_name)
        sections = [c.metadata["section"] for c in chunks]
        print(f"[Ingest] {len(chunks)} chunks | sections: {sections}")

        upsert_chunks(chunks)
        all_chunks.extend(chunks)

    print(f"\n[Ingest] Done. Total chunks across all CVs: {len(all_chunks)}")
    return all_chunks
