from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from retrieval.bm25 import build_bm25_index
from retrieval.config import RAGConfig, QDRANT_COLLECTION
from retrieval.hybrid import hybrid_search
from retrieval.multi_query import generate_multi_queries

load_dotenv()

cfg = RAGConfig()

qdrant_client = QdrantClient(
    url=cfg.qdrant_url,
    api_key=cfg.qdrant_api_key,
)


def load_docs() -> list[Document]:
    docs = []
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for i, point in enumerate(points):
            payload = point.payload or {}

            page_content = str(
                payload.get("page_content")
                or payload.get("text")
                or ""
            ).strip()

            if not page_content:
                continue

            chunk_id = str(payload.get("chunk_id") or point.id or "").strip()
            candidate_name = str(
                payload.get("candidate_name")
                or payload.get("candidate")
                or ""
            ).strip()
            section = str(payload.get("section", "")).strip()
            source_cv = str(
                payload.get("source_cv")
                or payload.get("cv_source")
                or ""
            ).strip()

            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "chunk_id": chunk_id,
                        "candidate_id": str(payload.get("candidate_id", "")).strip(),
                        "candidate_name": candidate_name,
                        "section": section,
                        "source_cv": source_cv,
                    },
                )
            )

        if offset is None:
            break

    return docs


def run_pipeline():
    question = input("Ask a question about the CVs:\n").strip()
    if not question:
        print("Empty question.")
        return []

    docs = load_docs()
    if not docs:
        print("No chunks found in Qdrant.")
        return []

    queries = generate_multi_queries(question)

    print("\nGenerated Queries:\n")
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")

    bm25_index = build_bm25_index(docs)
    merged = {}

    retrieve_k = int(getattr(cfg, "retrieve_k", 12))
    shortlist_k = int(getattr(cfg, "shortlist_k", 10))
    final_topk = int(getattr(cfg, "final_topk", 6))

    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")

        retrieved = hybrid_search(
            client=qdrant_client,
            bm25_index=bm25_index,
            docs=docs,
            query=query,
            k=retrieve_k,
        )

        for item in retrieved:
            doc = item.get("document")
            if doc is None:
                continue

            chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
            if not chunk_id:
                continue

            old = merged.get(chunk_id)
            if old is None or item.get("hybrid_score", 0.0) > old.get("hybrid_score", 0.0):
                merged[chunk_id] = item

    merged_results = sorted(
        merged.values(),
        key=lambda item: item.get("hybrid_score", 0.0),
        reverse=True,
    )

    if not merged_results:
        print("\nNo results found.")
        return []

    top_pool = merged_results[: min(len(merged_results), shortlist_k)]

    selected = []
    used_ids = set()
    candidate_counts = {}
    section_counts = {}

    for item in top_pool:
        doc = item.get("document")
        if doc is None:
            continue

        chunk_id = str(doc.metadata.get("chunk_id", "")).strip()
        candidate_name = str(doc.metadata.get("candidate_name", "")).strip()
        section = str(doc.metadata.get("section", "")).strip().lower()

        if not chunk_id or chunk_id in used_ids:
            continue

        candidate_penalty = candidate_counts.get(candidate_name, 0) * 0.05
        section_penalty = section_counts.get(section, 0) * 0.02

        hybrid_part = float(item.get("hybrid_score", 0.0))
        base_score = hybrid_part * 100.0

        row = dict(item)
        row["final_score"] = base_score - candidate_penalty - section_penalty
        selected.append(row)

        used_ids.add(chunk_id)
        candidate_counts[candidate_name] = candidate_counts.get(candidate_name, 0) + 1
        section_counts[section] = section_counts.get(section, 0) + 1

        if len(selected) >= final_topk:
            break

    final_results = sorted(
        selected,
        key=lambda item: item.get("final_score", 0.0),
        reverse=True,
    )

    print("\nFinal Results:\n")

    for item in final_results:
        doc = item.get("document")
        if doc is None:
            continue

        print(
            f"- candidate: {doc.metadata.get('candidate_name', '')} | "
            f"section: {doc.metadata.get('section', '')} | "
            f"dense: {item.get('dense_score')} | "
            f"bm25: {item.get('bm25_score')} | "
            f"hybrid: {item.get('hybrid_score')} | "
            f"final: {item.get('final_score')}"
        )

    return final_results


if __name__ == "__main__":
    run_pipeline()