from __future__ import annotations

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi



def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace(".", " ")
    text = text.replace('"', " ")
    text = text.replace("'", " ")
    text = re.sub(r"\s+", " ", text)
    return text

def _tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"[A-Za-z0-9_+#]+", text)


def build_bm25_index(docs: List[Document]) -> BM25Okapi:
    corpus = []
    id_map = []

    for i, doc in enumerate(docs):
        candidate_name = str(doc.metadata.get("candidate_name", "")).strip()
        section = str(doc.metadata.get("section", "")).strip()
        text = str(doc.page_content or "").strip()

        searchable_text = f"{candidate_name} {section} {text}".strip()
        corpus.append(_tokenize(searchable_text))

        chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
        if not chunk_id:
            source_cv = str(doc.metadata.get("source_cv", "unknown")).strip() or "unknown"
            chunk_id = f"{source_cv}:{section or 'general'}:c{i}"

        id_map.append(chunk_id)

    bm25 = BM25Okapi(corpus)
    bm25.id_map = id_map
    return bm25


def search_bm25(
    bm25_index: BM25Okapi,
    docs: List[Document],
    query: str,
    k: int,
    allowed_indices: Optional[set] = None,
) -> List[Tuple[Document, int]]:
    tokens = _tokenize(query)
    scores = bm25_index.get_scores(tokens)
    id_map = getattr(bm25_index, "id_map", [])

    candidates = [
        (scores[i], i)
        for i in range(len(docs))
        if allowed_indices is None or i in allowed_indices
    ]
    candidates.sort(reverse=True)

    results = []
    for score, i in candidates[:k]:
        doc = docs[i]
        results.append((
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "chunk_id": id_map[i] if i < len(id_map) else "",
                    "bm25_score": float(score),
                },
            ),
            i,
        ))

    return results