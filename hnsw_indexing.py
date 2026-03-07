'''
HNSW works similarly by building a multi-layered graph where each vector is a node. 
The idea is that the graph has a hierarchical structure, where the top layer contains a smaller number of nodes that are broadly connected, 
and each lower layer has more nodes with increasingly specific connections.
'''

'''
Graph Connectivity (m) :- 
This parameter controls the number of connections each node has in the graph.
Higher values of m can lead to better recall but may increase memory usage and search time.
'''

'''
Build Thoroughness: ef_construct 
It controls how many candidate neighbors are considered during the index construction phase.
Higher values of ef_construct can lead to better recall but will increase the time taken to build the index.
between 100 and 500. Complex data can require higher values to maintain reliable connections.
'''

'''
Search Throughness: hnsw_ef
determines the number of candidates evaluated during a search query.
Higher values of hnsw_ef can improve recall but will increase search latency.
A common starting point is to set hnsw_ef to a value between 100 and 200, and then adjust based on the desired balance between recall and latency.

'''

import time
from http import client

from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="http://localhost:6333",
)

# For Colab:
# from google.colab import userdata
# client = QdrantClient(url=userdata.get("QDRANT_URL"), api_key=userdata.get("QDRANT_API_KEY"))

collection_name = "my_collection"

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)


# Development / testing: faster builds
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=4,
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfigDiff(
            m=8,  # Fewer connections
            ef_construct=100,  # Faster builds
            full_scan_threshold=100,  # Use brute force below this size (default)
        ),
    ),
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=100,  # Use brute force below this size (default)
    ),
)

# upload data
import random

points = []
for i in range(20000):
    points.append(
        models.PointStruct(id=i, vector=[random.random() for _ in range(4)], payload={})
    )
client.upload_points(
    collection_name=collection_name,
    points=points,
)

def benchmark_search_performance(collection_name, test_queries, ef_values):
    """Compare latency across hnsw_ef values"""

    results = {}
    for hnsw_ef in ef_values:
        start_time = time.time()
        for query in test_queries:
            client.query_points(
                collection_name=collection_name,
                query=query,
                limit=10,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            )

        avg_time = (time.time() - start_time) / len(test_queries)
        results[hnsw_ef] = avg_time
        print(f"hnsw_ef={hnsw_ef}: {avg_time:.3f}s per query")

    return results


# Test different hnsw_ef values
test_queries = [
    [30, 60, 90, 120],
    [150, 180, 210, 240],
    [270, 300, 330, 360],
    [390, 420, 450, 480],
    [510, 540, 570, 600],
]

ef_values = [32, 64, 128, 256]
performance = benchmark_search_performance(collection_name, test_queries, ef_values)

# Inspect collection status
info = client.get_collection(collection_name)

vectors_per_point = 1  # set per your vectors_config
vectors_count = info.points_count * vectors_per_point

print(f"Collection status: {info.status}") 
print(f"Total points: {info.points_count}")
print(f"Indexed vectors: {info.indexed_vectors_count}")

if vectors_count:
    proportion_unindexed = 1 - (info.indexed_vectors_count / vectors_count)
else:
    proportion_unindexed = 0

print(f"Proportion unindexed: {proportion_unindexed:.2%}")

if info.status == models.CollectionStatus.GREEN:
    print("\n✅ Collection is indexed and ready!")
elif info.status == models.CollectionStatus.YELLOW:
    print("\n⚠️ Collection is still being indexed (optimizing).")
else:
    print(f"\n❌ Collection status is {info.status}.")