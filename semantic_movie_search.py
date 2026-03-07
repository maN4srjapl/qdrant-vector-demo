from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# Initialize components
encoder = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name='movie_search',
    vectors_config={
        'fixed': models.VectorParams(size=384, distance=models.Distance.COSINE),
        'sentence': models.VectorParams(size=384, distance=models.Distance.COSINE),
        'semantic': models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
)