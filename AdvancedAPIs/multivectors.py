'''
Late interaction models like ColBERT retain per document tokens in the index, allowing for more accurate scoring at query time. 
This token-level precision preserves local semantic matches and delivers superior relevance, especially for complex queries and documents.

Implemented through multivector representations
basically token level semantic matching
'''

from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"))

# generate token level embeddings for a document
encoder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
doc_multivectors = list(encoder.embed(["A long document about AI in medicine."]))



# setting up collection with multivector support
client.create_collection(
    collection_name="my_colbert_collection",
    vectors_config={
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM # Compare query tokens to document tokens using max similarity
            ),
            # Disable HNSW to save RAM - it won't typically be used with multivectors
            hnsw_config=models.HnswConfigDiff(m=0),
        )
    }
)

'''
. Without HNSW, queries use brute-force MaxSim scoring across all points, which provides maximum precision but may be slower on large collections
MaxSim means for each query token, we find the document token with the highest similarity, and then aggregate those max similarities across all query tokens to get the final score for the document.
'''

'''
But with multivectors:

each document already has many vectors

HNSW memory cost becomes large

So this disables HNSW to save RAM.

'''

# Encode the query
colbert = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
colbert_query = next(colbert.query_embed(["what is the policy?"])).tolist()

# Search using ColBERT multivector
hits = client.query_points(
    collection_name="my_colbert_collection",
    query=colbert_query,
    using="colbert",
    limit=20,
)