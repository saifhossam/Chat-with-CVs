from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
)


def rerank_results(question: str, results: list[dict], top_k: int = 5) -> list[dict]:
    if not question or not results:
        return []

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5

    if top_k <= 0:
        top_k = 5

    pairs = []
    valid_results = []

    for item in results:
        doc = item.get("document")
        if doc is None:
            continue

        text = str(getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue

        pairs.append((question, text[:1500]))
        valid_results.append(item)

    if not pairs:
        return []

    scores = model.predict(pairs)

    reranked = []
    for item, score in zip(valid_results, scores):
        row = dict(item)
        row["rerank_score"] = float(score)
        reranked.append(row)

    reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
    return reranked[:top_k]