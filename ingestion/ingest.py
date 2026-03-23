from __future__ import annotations

import os
from typing import List, Tuple

from langchain_core.documents import Document

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_cv, extract_candidate_name
from db.qdrant_client import setup_collection, upsert_chunks, is_filename_indexed


def ingest_cvs(
    pdf_paths: List[Tuple[str, str]],
    reset: bool = False,
) -> List[Document]:
    """
    Full ingestion pipeline: load -> chunk -> push to Qdrant.

    Args:
        pdf_paths: List of (temp_path, original_filename) tuples.
        reset:     Wipe and recreate the Qdrant collection before ingesting.

    Returns:
        Flat list of all Document chunks across all CVs.
    """
    setup_collection(reset=reset)

    all_chunks: List[Document] = []

    for temp_path, original_filename in pdf_paths:

        if is_filename_indexed(original_filename):
            print(f"[Ingest] Skipping '{original_filename}' — already indexed.")
            continue

        print(f"\n[Ingest] Processing: {original_filename}")

        lines          = load_pdf(temp_path)
        candidate_name = extract_candidate_name(lines)
        print(f"[Ingest] Candidate: {candidate_name}")

        chunks   = chunk_cv(
            pdf_path=temp_path,
            lines=lines,
            candidate_name=candidate_name,
            original_filename=original_filename,
        )
        sections = [c.metadata["section"] for c in chunks]
        print(f"[Ingest] {len(chunks)} chunks | sections: {sections}")

        upsert_chunks(chunks)
        all_chunks.extend(chunks)

    print(f"\n[Ingest] Done. Total chunks indexed: {len(all_chunks)}")
    return all_chunks