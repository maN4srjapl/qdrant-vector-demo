from qdrant_client import QdrantClient, models
import time
from sentence_transformers import SentenceTransformer
import numpy as np
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"))

# Initialize the sentence transformer model for encoding queries
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Define different HNSW configurations to test
configs = [
    {"hnsw_ef": 10, "description": "Low ef (faster, less accurate)"},
    {"hnsw_ef": 50, "description": "Medium ef (balanced)"},
    {"hnsw_ef": 200, "description": "High ef (slower, more accurate)"},
]   

for config in configs:  
    collection_name = f"benchmark_collection_ef_{config['hnsw_ef']}"
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=200,
                full_scan_threshold=1000,
            ),
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=10,
        ),
    )

    # Upload synthetic data
    print(f"Uploading points to {collection_name}...")
    points = []
    for i in range(10000):
        vector = np.random.rand(384).tolist()
        points.append(models.PointStruct(id=i, vector=vector, payload={"text": f"Document {i}"}))
    client.upload_points(collection_name=collection_name, points=points)

    # Prepare test queries
    test_queries = [
        encoder.encode(f"Find documents similar to Document {i}").tolist() for i in range(10)
    ]

    # Benchmark search performance
    print(f"Benchmarking search performance for {collection_name}...")
    start_time = time.time()
    for query in test_queries:
        client.query_points(
            collection_name=collection_name,
            query=query,
            limit=5,
            search_params=models.SearchParams(hnsw_ef=config["hnsw_ef"])
        )
    end_time = time.time()
    avg_time_per_query = (end_time - start_time) / len(test_queries)
    print(f"{config['description']}: Average time per query: {avg_time_per_query} seconds")
                                                              