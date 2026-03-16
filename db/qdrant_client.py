import hashlib
import os
from config import qdrant_client, COLLECTION, EMBED_DIM

def setup_collection():
    qdrant_client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            "size": EMBED_DIM,
            "distance": "Cosine"
        }
    )
    qdrant_client.create_payload_index(
        collection_name=COLLECTION,
        field_name="candidate_name",
        field_schema="keyword"  
    )

def upsert_chunks(chunks, vectors, file_hash):
    points = []
    
    # We zip the chunks and the pre-calculated vectors together
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        # Generate a stable ID for the chunk
        point_id = hashlib.md5(f"{file_hash}_{i}".encode()).hexdigest()
        
        points.append({
            "id": point_id,
            "vector": vector, # Pass the vector here
            "payload": {
                "text": chunk.page_content,
                **chunk.metadata
            }
        })

    qdrant_client.upsert(collection_name=COLLECTION, points=points)
# Inside db/qdrant_client.py

def is_indexed(file_hash: str) -> bool:
    from qdrant_client.http import models
    
    # 'count' is much faster than 'scroll' for existence checks
    result = qdrant_client.count(
        collection_name=COLLECTION,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_hash",
                    match=models.MatchValue(value=file_hash),
                )
            ]
        ),
        exact=True,
    )
    # result.count returns the number of points matching the filter
    return result.count > 0
