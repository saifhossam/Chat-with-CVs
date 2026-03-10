from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

from retrieval.config import QDRANT_COLLECTION, RAGConfig
from retrieval.embeddings import get_embeddings
from retrieval.bm25 import search_bm25


def search_qdrant(
    client: QdrantClient,
    docs: List[Document],
    query: str,
    k: int,
    allowed_indices: Optional[set] = None,
) -> List[Document]:
    query_vector = get_embeddings().embed_query(query)

    qdrant_filter = None
    if allowed_indices is not None:
        candidate_ids = list({
            str(docs[i].metadata.get("candidate_id", "")).strip()
            for i in allowed_indices
            if i < len(docs) and str(docs[i].metadata.get("candidate_id", "")).strip()
        })
        if candidate_ids:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="candidate_id",
                        match=MatchAny(any=candidate_ids),
                    )
                ]
            )

    points = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=k,
        query_filter=qdrant_filter,
        with_payload=True,
        with_vectors=False,
    ).points

    results = []
    for rank, p in enumerate(points):
        payload = p.payload or {}
        chunk_id = str(payload.get("chunk_id") or p.id or f"dense_{rank}")

        results.append(
            Document(
                page_content=payload.get("page_content", "") or payload.get("text", ""),
                metadata={
                    "candidate_id": payload.get("candidate_id", ""),
                    "candidate_name": payload.get("candidate_name", "Unknown"),
                    "section": payload.get("section", ""),
                    "source_cv": payload.get("source_cv", ""),
                    "chunk_id": chunk_id,
                    "dense_score": float(p.score),
                },
            )
        )

    return results


def rrf_fuse(
    dense_hits: List[Document],
    bm25_hits: List[Tuple[Document, int]],
    rrf_k: int = 60,
) -> Dict[str, float]:
    fused = {}

    for rank, doc in enumerate(dense_hits, start=1):
        chunk_id = str(doc.metadata.get("chunk_id", f"dense_{rank}"))
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (doc, idx) in enumerate(bm25_hits, start=1):
        chunk_id = str(doc.metadata.get("chunk_id", f"bm25_{idx}"))
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    return fused


def hybrid_search(
    client: QdrantClient,
    
    bm25_index,
    docs: List[Document],
    query: str,
    k: int,
    allowed_indices: Optional[set] = None,
) -> List[dict]:
    dense_hits = search_qdrant(
        client,
        docs,
        query,
        max(20, k * 3),
        allowed_indices,
    )
    bm25_hits = search_bm25(
        bm25_index,
        docs,
        query,
        max(20, k * 3),
        allowed_indices,
    )

    fused_scores = rrf_fuse(dense_hits, bm25_hits)

    dense_by_id = {str(doc.metadata.get("chunk_id", "")): doc for doc in dense_hits}
    bm25_by_id = {str(doc.metadata.get("chunk_id", "")): doc for doc, _ in bm25_hits}

    results = []
    for chunk_id, hybrid_score in sorted(
    fused_scores.items(),
    key=lambda x: (-x[1], x[0]),
    )[:k]:  
        dense_doc = dense_by_id.get(chunk_id)
        bm25_doc = bm25_by_id.get(chunk_id)

        doc = dense_doc or bm25_doc
        if doc is None:
            continue

        results.append({
            "document": doc,
            "chunk_id": chunk_id,
            "dense_score": dense_doc.metadata.get("dense_score") if dense_doc else None,
            "bm25_score": bm25_doc.metadata.get("bm25_score") if bm25_doc else None,
            "hybrid_score": float(hybrid_score),
        })

    return results