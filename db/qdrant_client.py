import uuid
from config import qdrant_client, COLLECTION, EMBED_DIM
from embedding import get_embeddings


def setup_collection(reset: bool = False):
    """
    Prepare the Qdrant collection.
    - reset=True  → drop and recreate (wipes all data).
    - reset=False → create only if it doesn't already exist (safe default).
    """
    from qdrant_client.http import models

    existing = [c.name for c in qdrant_client.get_collections().collections]

    if reset and COLLECTION in existing:
        qdrant_client.delete_collection(collection_name=COLLECTION)
        existing = []

    if COLLECTION not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        qdrant_client.create_payload_index(
            collection_name=COLLECTION,
            field_name="candidate_name",
            field_schema="keyword",
        )
        qdrant_client.create_payload_index(
            collection_name=COLLECTION,
            field_name="source_cv",
            field_schema="keyword",
        )


def is_filename_indexed(filename: str) -> bool:
    """Check if a CV file has already been indexed by its filename."""
    from qdrant_client.http import models

    try:
        result = qdrant_client.count(
            collection_name=COLLECTION,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_cv",
                        match=models.MatchValue(value=filename),
                    )
                ]
            ),
            exact=True,
        )
        return result.count > 0
    except Exception:
        return False


def upsert_chunks(chunks):
    """
    Embed chunks and upsert them into Qdrant.
    Each point gets a random UUID as its ID.
    """
    if not chunks:
        return

    texts   = [chunk.page_content for chunk in chunks]
    vectors = get_embeddings(texts)

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append({
            "id":      str(uuid.uuid4()),
            "vector":  vector,
            "payload": {
                "text": chunk.page_content,
                **chunk.metadata,
            },
        })

    qdrant_client.upsert(collection_name=COLLECTION, points=points)