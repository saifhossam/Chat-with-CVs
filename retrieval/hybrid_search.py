from typing import Any
from qdrant_client import QdrantClient

from retrieval.bm25 import bm25_search
from retrieval.config import RAGConfig


def dense_search(
    client: QdrantClient,
    cfg: RAGConfig,
    query_vec: list[float],
    topn: int,
) -> list[tuple[str, float, dict[str, Any]]]:  # (chunk_id, score, payload)

    if not query_vec:
        return []

    response = client.query_points(
        collection_name=cfg.qdrant_collection,
        query=query_vec,
        limit=topn,
        with_payload=True,
        with_vectors=False,
    )

    points = getattr(response, "points", None) or []
    results = []

    for point in points:
        payload = point.payload or {}

        chunk_id = payload.get("chunk_id") or str(point.id)
        if not chunk_id:
            continue

        results.append((chunk_id, float(point.score), payload))

    return results


def rrf_fuse(
    dense_hits: list[tuple[str, float, dict[str, Any]]],
    bm25_hits: list[tuple[str, float]],
    rrf_k: int,
) -> dict[str, float]:

    fused = {}

    for rank, (chunk_id, _, _) in enumerate(dense_hits, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (chunk_id, _) in enumerate(bm25_hits, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    return fused


def hybrid_search(
    client: QdrantClient,
    cfg: RAGConfig,
    bm25_index: dict[str, Any],
    query_text: str,
    query_vec: list[float],
    k: int,
) -> list[dict[str, Any]]:

    if not query_text and not query_vec:
        return []

    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 8

    if k <= 0:
        k = 8

    try:
        dense_topn = int(getattr(cfg, "dense_topn", max(20, k * 3)))
    except (TypeError, ValueError):
        dense_topn = max(20, k * 3)

    try:
        bm25_topn = int(getattr(cfg, "bm25_topn", max(20, k * 3)))
    except (TypeError, ValueError):
        bm25_topn = max(20, k * 3)

    try:
        rrf_k = int(getattr(cfg, "rrf_k", 60))
    except (TypeError, ValueError):
        rrf_k = 60

    dense_hits = dense_search(
        client=client,
        cfg=cfg,
        query_vec=query_vec,
        topn=dense_topn,
    )

    bm25_hits = bm25_search(
        index=bm25_index,
        query=query_text,
        topn=bm25_topn,
    )

    if not dense_hits and not bm25_hits:
        return []

    fused_scores = rrf_fuse(
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        rrf_k=rrf_k,
    )

    ranked_ids = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:k]

    dense_payload_by_id = {chunk_id: payload for chunk_id, _, payload in dense_hits}
    dense_score_by_id = {chunk_id: float(score) for chunk_id, score, _ in dense_hits}
    bm25_score_by_id = {chunk_id: float(score) for chunk_id, score in bm25_hits}

    chunk_by_id = bm25_index.get("chunk_by_id", {})

    results = []

    for chunk_id, hybrid_score in ranked_ids:

        payload = dense_payload_by_id.get(chunk_id)
        raw_chunk = payload if payload is not None else chunk_by_id.get(chunk_id)

        if raw_chunk is None:
            continue

        if isinstance(raw_chunk, dict):

            try:
                page_number = int(raw_chunk.get("page_number", raw_chunk.get("page", 0)))
            except (TypeError, ValueError):
                page_number = 0

            chunk = {
                "chunk_id": raw_chunk.get("chunk_id", chunk_id),
                "text": str(raw_chunk.get("text", "")).strip(),
                "candidate_id": str(raw_chunk.get("candidate_id", "")).strip(),
                "candidate_name": str(
                    raw_chunk.get("candidate_name")
                    or raw_chunk.get("candidate")
                    or ""
                ).strip(),
                "section": str(raw_chunk.get("section", "")).strip(),
                "page_number": page_number,
            }

        else:

            try:
                page_number = int(
                    getattr(raw_chunk, "page_number", getattr(raw_chunk, "page", 0))
                )
            except (TypeError, ValueError):
                page_number = 0

            chunk = {
                "chunk_id": getattr(raw_chunk, "chunk_id", chunk_id),
                "text": str(getattr(raw_chunk, "text", "")).strip(),
                "candidate_id": str(getattr(raw_chunk, "candidate_id", "")).strip(),
                "candidate_name": str(
                    getattr(raw_chunk, "candidate_name", "")
                    or getattr(raw_chunk, "candidate", "")
                ).strip(),
                "section": str(getattr(raw_chunk, "section", "")).strip(),
                "page_number": page_number,
            }

        results.append(
            {
                "chunk": chunk,
                "dense_score": dense_score_by_id.get(chunk_id),
                "bm25_score": bm25_score_by_id.get(chunk_id),
                "hybrid_score": float(hybrid_score),
            }
        )

    return results