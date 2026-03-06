from typing import Any

from qdrant_client import QdrantClient

from retrieval.config import RAGConfig
from retrieval.bm25 import bm25_search


def _get_cfg_int(cfg: RAGConfig, field_name: str, default: int) -> int:
    """
    Read integer config safely with fallback.
    """
    value = getattr(cfg, field_name, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _payload_to_chunk(payload: dict[str, Any], fallback_chunk_id: str = "") -> dict[str, Any]:
    """
    Normalize Qdrant payload into a generic chunk dict.
    """
    return {
        "chunk_id": payload.get("chunk_id", fallback_chunk_id),
        "text": payload.get("text", ""),
        "candidate_id": payload.get("candidate_id", ""),
        "candidate_name": payload.get("candidate_name", ""),
        "section": payload.get("section", ""),
        "page_number": _safe_int(payload.get("page_number", 0), 0),
    }


def _normalize_chunk(item: Any, fallback_chunk_id: str = "") -> dict[str, Any]:
    """
    Normalize either dict-chunk or object-chunk into one stable shape.
    """
    if isinstance(item, dict):
        return {
            "chunk_id": item.get("chunk_id", fallback_chunk_id),
            "text": item.get("text", ""),
            "candidate_id": item.get("candidate_id", ""),
            "candidate_name": item.get("candidate_name", ""),
            "section": item.get("section", ""),
            "page_number": _safe_int(item.get("page_number", 0), 0),
        }

    return {
        "chunk_id": getattr(item, "chunk_id", fallback_chunk_id),
        "text": getattr(item, "text", ""),
        "candidate_id": getattr(item, "candidate_id", ""),
        "candidate_name": getattr(item, "candidate_name", ""),
        "section": getattr(item, "section", ""),
        "page_number": _safe_int(getattr(item, "page_number", 0), 0),
    }


def dense_search(
    client: QdrantClient,
    cfg: RAGConfig,
    query_vec: list[float],
    topn: int,
) -> list[tuple[str, float, dict[str, Any]]]:
    """
    Dense vector search from Qdrant.

    Returns:
        list of (chunk_id, dense_score, payload)
    """
    if not query_vec:
        return []

    response = client.query_points(
        collection_name=cfg.qdrant_collection,
        query=query_vec,
        using="dense",
        limit=topn,
        with_payload=True,
        with_vectors=False,
    )

    points = getattr(response, "points", None) or []

    hits: list[tuple[str, float, dict[str, Any]]] = []
    for point in points:
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id") or str(point.id)
        score = float(point.score)

        if not chunk_id:
            continue

        hits.append((chunk_id, score, payload))

    return hits


def rrf_fuse(
    dense_hits: list[tuple[str, float, dict[str, Any]]],
    bm25_hits: list[tuple[str, float]],
    rrf_k: int = 60,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion over dense and BM25 rankings.

    Uses rank only, not raw scores.
    """
    fused_scores: dict[str, float] = {}

    for rank, (chunk_id, _, _) in enumerate(dense_hits, start=1):
        if chunk_id:
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (chunk_id, _) in enumerate(bm25_hits, start=1):
        if chunk_id:
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    return fused_scores


def hybrid_search(
    client: QdrantClient,
    cfg: RAGConfig,
    bm25_index: dict[str, Any],
    query_text: str,
    query_vec: list[float],
    k: int,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval pipeline:
    1) Dense search from Qdrant
    2) BM25 search
    3) RRF fusion
    4) Reconstruct chunk from Qdrant payload when possible
    5) Fallback to bm25_index['chunk_by_id']

    Returns:
        [
            {
                "chunk": {
                    "chunk_id": str,
                    "text": str,
                    "candidate_id": str,
                    "candidate_name": str,
                    "section": str,
                    "page_number": int,
                },
                "dense_score": float | None,
                "bm25_score": float | None,
                "hybrid_score": float,
            }
        ]
    """
    if not query_text and not query_vec:
        return []

    try:
        final_k = int(k)
    except (TypeError, ValueError):
        final_k = 5

    if final_k <= 0:
        final_k = 5

    dense_topn = _get_cfg_int(cfg, "dense_topn", 20)
    bm25_topn = _get_cfg_int(cfg, "bm25_topn", 20)
    rrf_k = _get_cfg_int(cfg, "rrf_k", 60)

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

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:final_k]

    dense_payload_by_id = {
        chunk_id: payload
        for chunk_id, _, payload in dense_hits
        if chunk_id and payload
    }

    dense_score_by_id = {
        chunk_id: float(score)
        for chunk_id, score, _ in dense_hits
        if chunk_id
    }

    bm25_score_by_id = {
        chunk_id: float(score)
        for chunk_id, score in bm25_hits
        if chunk_id
    }

    chunk_by_id = bm25_index.get("chunk_by_id", {})

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk_id, hybrid_score in ranked:
        if not chunk_id or chunk_id in seen:
            continue

        payload = dense_payload_by_id.get(chunk_id)

        if payload:
            chunk = _payload_to_chunk(payload, fallback_chunk_id=chunk_id)
        else:
            raw_chunk = chunk_by_id.get(chunk_id)
            if raw_chunk is None:
                continue
            chunk = _normalize_chunk(raw_chunk, fallback_chunk_id=chunk_id)

        results.append(
            {
                "chunk": chunk,
                "dense_score": dense_score_by_id.get(chunk_id),
                "bm25_score": bm25_score_by_id.get(chunk_id),
                "hybrid_score": float(hybrid_score),
            }
        )
        seen.add(chunk_id)

    return results