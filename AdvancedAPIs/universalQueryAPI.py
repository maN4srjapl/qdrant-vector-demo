# Dense Retrieval and ColBERT Reranking with Multivectors

from qdrant_client import QdrantClient, models
import os
import numpy as np
from fastembed import LateInteractionTextEmbedding, TextEmbedding

client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

if client.collection_exists("articles"):
    client.delete_collection(collection_name="articles")

client.create_collection(
    collection_name="articles",
    vectors_config={
        # Fast HNSW-indexed dense retrieval
        "bge-dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
        # Precise multivector reranking (HNSW disabled to save RAM)
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
        )
    }
)

# Encode with both models
dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

documents = [
    "The 2024 healthcare policy update includes several key changes to coverage and accessibility.",
    "Qdrant provides high-performance vector search and retrieval for large-scale datasets.",
    "BGE and ColBERT models are widely used for dense and late-interaction retrieval respectively.",
    "Vector databases are essential for building RAG applications and semantic search engines."
]

# Create point objects for the collection
points = []
for i, doc in enumerate(documents):
    dense_vector = next(dense_model.embed([doc])).tolist()
    colbert_multivector = next(colbert_model.embed([doc])).tolist()
    
    points.append(
        models.PointStruct(
            id=i,
            vector={
                "bge-dense": dense_vector,
                "colbert": colbert_multivector
            },
            payload={"text": doc}
        )
    )

# Upsert the documents into the collection
client.upsert(
    collection_name="articles",
    points=points
)

query_text = "what is the policy?"

# Encode the query
dense_query_vector = next(dense_model.query_embed([query_text])).tolist()
colbert_query_multivector = next(colbert_model.query_embed([query_text])).tolist()

# Execute query using Reciprocal Rank Fusion (RRF)
# This combines the rankings from two parallel searches: 
# 1. A dense vector search using BGE
# 2. A multivector search using ColBERT (here simplified to parallel retrieval)
response = client.query_points(
    collection_name="articles",
    prefetch=[
        models.Prefetch(
            query=dense_query_vector,
            using="bge-dense",
            limit=100
        ),
        models.Prefetch(
            query=colbert_query_multivector,
            using="colbert",
            limit=100
        )
    ],
    query=models.FusionQuery(
        fusion=models.Fusion.RRF
    ),
    limit=10
)

# Print the results
print(f"Query: {query_text} (Using RRF Fusion)\n")
for i, point in enumerate(response.points):
    print(f"Rank {i+1}: Score: {point.score:.4f} | Text: {point.payload['text']}")


'''
The Universal Query API enables complex search patterns through a simple declarative interface:

Prefetch Stage: Execute multiple searches in parallel against different vector fields. Each prefetch can have its own filters, limits, and vector types used.
Fusion Stage: Combine results from multiple prefetches using algorithms like Reciprocal Rank Fusion (RRF) or Distribution-Based Score Fusion (DBSF).
Reranking Stage: Or rerank candidates from a single prefetch with a stronger scorer such as ColBERT. Fusion and reranking are alternative final steps in most pipelines.
Filtering: Apply filters globally at the query level (propagated to all prefetches) or add prefetch-specific filters for additional constraints on individual searches.


'''