from qdrant_client import QdrantClient, models

# DENSE RETRIEVAL -> SPARSE RERANKING
client = QdrantClient(url="http://localhost:6333")

query = "What is the capital of France?"

client = QdrantClient(...)
client.query_points(
    collection_name="my_collection",
    prefetch=[
        models.Prefetch(
            query=models.Document(
                text=query,
                model="sentence-transformers/all-MiniLM-L6-v2",
            ),
            using="dense",
            limit=20,
        ),
    ],
    query=models.Document(
        text=query,
        model="Qdrant/bm25",
    ),
    using="sparse",
    limit=20,
)

# SPARSE RETRIEVAL -> DENSE RERANKING
client.query_points(
    collection_name="my_collection",
    prefetch=[
        models.Prefetch(
            query=models.Document(
                text=query,
                model="Qdrant/bm25",
            ),
            using="sparse",
            limit=20,
        ),
    ],
    query=models.Document(
        text=query,
        model="sentence-transformers/all-MiniLM-L6-v2",
    ),
    using="dense",
    limit=20,
)

#NOTE: the initial retrieval step (dense or sparse) can be used to quickly narrow down the candidate set of documents, while the reranking step (sparse or dense) can provide a more accurate ranking of those candidates based on the query. 