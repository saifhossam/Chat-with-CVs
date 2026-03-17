from sentence_transformers import CrossEncoder
from .query_expansion import generate_multi_queries
from .retrieval import load_chunks_from_qdrant, build_bm25_index, hybrid_search

reranker = CrossEncoder("BAAI/bge-reranker-base",  device="cpu")

def rerank(query, results, top_k=5):
    if not results:
        return []

    pairs = [[query, item["chunk"]["text"]] for item in results]
    scores = reranker.predict(pairs)
    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)

    return [item for item, _ in scored_results[:top_k]]


def unique_union(results: list[dict]) -> list[dict]:
    best_by_chunk = {}

    for item in results:
        chunk_id = item.get("chunk", {}).get("chunk_id")
        if not chunk_id:
            continue

        current_score = float(item.get("hybrid_score", 0))
        existing = best_by_chunk.get(chunk_id)

        if existing is None or current_score > float(existing.get("hybrid_score", 0)):
            best_by_chunk[chunk_id] = item

    return list(best_by_chunk.values())


def retrieve_pipeline(
    question: str,
    bm25_index: dict,
    candidate_name: str | None = None,
    num_queries: int = 2,
    per_query_k: int = 8,
    final_top_k: int = 5,
    rerank_pool_size: int = 20,
) -> list[dict]:
    queries = generate_multi_queries(question, num_queries=num_queries)
    print("Generated queries:", queries)

    all_results = []
    for query in queries:
        all_results.extend(
            hybrid_search(
                query_text=query,
                bm25_index=bm25_index,
                k=per_query_k,
                candidate_name=candidate_name,
            )
        )

    merged_results = unique_union(all_results)
    merged_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
    merged_results = merged_results[:rerank_pool_size]

    return rerank(question, merged_results, top_k=final_top_k)


# Example usage for testing the pipeline. In the app.py, you would call retrieve_pipeline() with the user's question and the pre-built BM25 index.
if __name__ == "__main__":
    chunks = load_chunks_from_qdrant()
    bm25_index = build_bm25_index(chunks)

    question = input("Ask a question: ").strip()
    #candidate_name = input("Candidate name (optional): ").strip()

    #if not candidate_name:
        #candidate_name = None

    results = retrieve_pipeline(
        question=question,
        bm25_index=bm25_index,
        #candidate_name=candidate_name,
    )

    for item in results:
        print(item["chunk"]["candidate_name"])
        print(item["chunk"]["section"])
        print(item["chunk"]["text"][:300])
        print("-" * 30)

