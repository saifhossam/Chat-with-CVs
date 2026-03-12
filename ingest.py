from __future__ import annotations


import os
import hashlib
from typing import List, BinaryIO

from langchain_core.documents import Document

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_cv, extract_candidate_name
from db.qdrant_client import setup_collection, upsert_chunks, is_indexed



def calculate_file_hash(file: BinaryIO) -> str:
    """
    Compute the SHA-256 hash of a file's raw bytes.
 
    Args:
        file: Any binary file-like object with .seek() and .read().
 
    Returns:
        64-character lowercase hex digest string.
    """
    file.seek(0)
    file_bytes = file.read()
    file.seek(0)
    return hashlib.sha256(file_bytes).hexdigest()



def calculate_path_hash(path: str) -> str:
    """
    Compute the SHA-256 hash of a PDF given its filesystem path.
 
    Args:
        path: Absolute path to the PDF file.
 
    Returns:
        64-character lowercase hex digest string.
    """
    with open(path, "rb") as f:
        return calculate_file_hash(f)



def ingest_cvs(pdf_paths: List[str], reset: bool = False) -> List[Document]:
    """
    Full ingestion pipeline: hash-check -> load -> chunk -> push to Qdrant.

    Steps per CV:
      1. calculate_path_hash()    — SHA-256 of raw PDF bytes
      2. is_indexed(hash)         — skip if already in Qdrant
      3. load_pdf()               — pdfplumber extracts lines in reading order
      4. extract_candidate_name() — NER identifies the candidate's name
      5. chunk_cv()               — Docling detects headers, lines split into
                                    one Document per section
      6. upsert_chunks(hash)      — embed + push to Qdrant with hash in payload

    Args:
        pdf_paths: Absolute paths to CV PDF files.
        reset:     Wipe and recreate the Qdrant collection before ingesting.
                   Defaults to False — deduplication makes a full reset
                   unnecessary on re-upload. Pass True only when you want
                   to wipe all indexed CVs and start clean.

    Returns:
        Flat list of all Document chunks across all CVs.
        Useful for building a BM25 index or populating the UI knowledge base.
    """
    setup_collection(reset=reset)
 
    all_chunks: List[Document] = []
 
    for path in pdf_paths:
        filename  = os.path.basename(path)
        file_hash = calculate_path_hash(path)
 

        if is_indexed(file_hash):
            print(f"[Ingest] Skipping '{filename}' — already indexed (hash: {file_hash[:12]}...)")
            continue
 
        print(f"\n[Ingest] Processing: {filename}  (hash: {file_hash[:12]}...)")
 

        lines          = load_pdf(path)
        candidate_name = extract_candidate_name(lines)
        print(f"[Ingest] Candidate: {candidate_name}")
 
        chunks   = chunk_cv(pdf_path=path, lines=lines, candidate_name=candidate_name, file_hash=file_hash)
        sections = [c.metadata["section"] for c in chunks]
        print(f"[Ingest] {len(chunks)} chunks | sections: {sections}")
 
        upsert_chunks(chunks, file_hash=file_hash)
        all_chunks.extend(chunks)
 
    print(f"\n[Ingest] Done. Newly indexed chunks: {len(all_chunks)}")
    return all_chunks
