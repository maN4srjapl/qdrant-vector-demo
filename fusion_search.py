'''
Reciprocal Rank Fusion (RRF) is a method used in information retrieval to combine the results of multiple search algorithms or models. 
It assigns a score to each document based on its rank in the individual search results, 
and then combines these scores to produce a final ranking. The formula for RRF is:

RRF(d) = 1 / (k + rank1(d)) + 1 / (k + rank2(d)) + ... + 1 / (k + rankN(d))

Where:
- RRF(d) is the reciprocal rank fusion score for document d.
- k is a constant that controls the influence of the ranks (commonly set to 60).
- rank1(d), rank2(d), ..., rankN(d) are the ranks of document
''' 

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

query = "What is the capital of France?"

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
        models.Prefetch(
            query=models.Document(
                text=query,
                model="Qdrant/bm25",
            ),
            using="sparse",
            limit=20,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
)