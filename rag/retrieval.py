import re
from typing import Any
from rank_bm25 import BM25Okapi
from config import COLLECTION, qdrant_client
from embedding import embed_query


_WORD_RE = re.compile(r"[A-Za-z0-9_+#]+")


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace(".", " ")
    text = text.replace('"', " ")
    text = text.replace("'", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_text(text))


def load_chunks_from_qdrant(limit: int = 1000) -> list[dict[str, Any]]:
    chunks = []
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id") or str(point.id)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": str(payload.get("text", "")).strip(),
                    "candidate_name": str(payload.get("candidate_name", "")).strip(),
                    "section": str(payload.get("section", "")).strip(),
                    "source_cv": str(payload.get("source_cv", "")).strip(),
                    "file_hash": str(payload.get("file_hash", "")).strip(),
                }
            )

        if offset is None:
            break

    return chunks


def build_bm25_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    corpus = []
    id_map = []
    chunk_by_id = {}

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        text = str(chunk.get("text", "")).strip()
        candidate_name = str(chunk.get("candidate_name", "")).strip()
        section = str(chunk.get("section", "")).strip()

        searchable_text = f"{candidate_name} {section}  {text}".strip()
        tokens = tokenize(searchable_text)

        if not chunk_id or not tokens:
            continue

        corpus.append(tokens)
        id_map.append(chunk_id)
        chunk_by_id[chunk_id] = chunk

    if not corpus:
        return {
            "bm25": None,
            "id_map": [],
            "chunk_by_id": {},
        }

    return {
        "bm25": BM25Okapi(corpus),
        "id_map": id_map,
        "chunk_by_id": chunk_by_id,
    }


def bm25_search(
    index: dict[str, Any],
    query: str,
    topn: int = 20,
    candidate_name: str | None = None,
) -> list[tuple[str, float]]:
    if not index or not query:
        return []

    bm25 = index.get("bm25")
    id_map = index.get("id_map", [])
    chunk_by_id = index.get("chunk_by_id", {})

    if bm25 is None or not id_map:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for idx in ranked_indices:
        score = float(scores[idx])
        if score <= 0:
            continue

        chunk_id = id_map[idx]
        chunk = chunk_by_id.get(chunk_id, {})

        if candidate_name:
            stored_name = normalize_text(chunk.get("candidate_name", ""))
            target_name = normalize_text(candidate_name)
            if stored_name != target_name:
                continue

        results.append((chunk_id, score))

        if len(results) >= topn:
            break

    return results


def dense_search(
    query_vec: list[float],
    topn: int = 20,
    candidate_name: str | None = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    if not query_vec:
        return []

    query_filter = None

    if candidate_name:
        from qdrant_client.http import models

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="candidate_name",
                    match=models.MatchValue(value=candidate_name),
                )
            ]
        )

    response = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        using="dense",
        limit=topn,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter,)
    

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
    rrf_k: int = 60,
) -> dict[str, float]:
    fused = {}

    for rank, (chunk_id, _, _) in enumerate(dense_hits, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (chunk_id, _) in enumerate(bm25_hits, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    return fused


def hybrid_search(
    query_text: str,
    bm25_index: dict[str, Any],
    k: int = 8,
    dense_topn: int = 20,
    bm25_topn: int = 20,
    rrf_k: int = 60,
    candidate_name: str | None = None,
) -> list[dict[str, Any]]:
    if not query_text or not query_text.strip():
        return []

    query_vec = embed_query(query_text)

    dense_hits = dense_search(
        query_vec=query_vec,
        topn=dense_topn,
        candidate_name=candidate_name,)

    bm25_hits = bm25_search(
        index=bm25_index,
        query=query_text,
        topn=bm25_topn,
        candidate_name=candidate_name,)

    if not dense_hits and not bm25_hits:
        return []

    fused_scores = rrf_fuse(
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        rrf_k=rrf_k,)

    ranked_ids = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,)[:k]

    dense_payload_by_id = {chunk_id: payload for chunk_id, _, payload in dense_hits}
    dense_score_by_id = {chunk_id: float(score) for chunk_id, score, _ in dense_hits}
    bm25_score_by_id = {chunk_id: float(score) for chunk_id, score in bm25_hits}
    chunk_by_id = bm25_index.get("chunk_by_id", {})

    results = []

    for chunk_id, hybrid_score in ranked_ids:
        payload = dense_payload_by_id.get(chunk_id) or chunk_by_id.get(chunk_id)
        if not payload:
            continue

        chunk = {
            "chunk_id": str(payload.get("chunk_id", chunk_id)).strip(),
            "text": str(payload.get("text", "")).strip(),
            "candidate_name": str(payload.get("candidate_name", "")).strip(),
            "section": str(payload.get("section", "")).strip(),
            "source_cv": str(payload.get("source_cv", "")).strip(),
            "file_hash": str(payload.get("file_hash", "")).strip(),
        }

        if candidate_name:
            if normalize_text(chunk["candidate_name"]) != normalize_text(candidate_name):
                continue

        results.append(
            {
                "chunk": chunk,
                "dense_score": dense_score_by_id.get(chunk_id),
                "bm25_score": bm25_score_by_id.get(chunk_id),
                "hybrid_score": float(hybrid_score),
            })

    return results