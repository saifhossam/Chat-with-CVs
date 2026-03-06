import re
from typing import Any

from rank_bm25 import BM25Okapi

_WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """
    Lowercase word tokenizer for BM25.
    """
    return _WORD_RE.findall((text or "").lower())


def _get_field(item: Any, field_name: str, default: Any = None) -> Any:
    """
    Safely read a field from either:
    - dict-like chunk
    - object-like chunk
    """
    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)


def build_bm25_index(chunks: list[Any], text_field: str = "text") -> dict[str, Any]:
    """
    Build a BM25 index from generic chunk objects.

    Minimal required fields per chunk:
    - chunk_id
    - text (or custom text_field)

    Returns:
        {
            "bm25": BM25Okapi | None,
            "id_map": list[str],
            "chunk_by_id": dict[str, Any],
        }
    """
    if not chunks:
        return {
            "bm25": None,
            "id_map": [],
            "chunk_by_id": {},
        }

    corpus: list[list[str]] = []
    id_map: list[str] = []
    chunk_by_id: dict[str, Any] = {}

    for chunk in chunks:
        chunk_id = _get_field(chunk, "chunk_id", "")
        text = _get_field(chunk, text_field, "")

        if not chunk_id:
            continue

        tokens = tokenize(text)
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


def bm25_search(index: dict[str, Any], query: str, topn: int = 20) -> list[tuple[str, float]]:
    """
    Search a pre-built BM25 index.

    Returns:
        list of (chunk_id, score)
    """
    if not index or not query:
        return []

    bm25 = index.get("bm25")
    id_map = index.get("id_map", [])

    if bm25 is None or not id_map:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    try:
        limit = int(topn)
    except (TypeError, ValueError):
        limit = 20

    if limit <= 0:
        limit = 20

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results: list[tuple[str, float]] = []
    for idx in ranked_indices:
        chunk_id = id_map[idx]
        if not chunk_id:
            continue

        score = float(scores[idx])

        # optional: skip fully non-relevant zero/negative scores
        if score <= 0:
            continue

        results.append((chunk_id, score))

        if len(results) >= limit:
            break

    return results