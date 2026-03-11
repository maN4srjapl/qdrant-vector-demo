from qdrant_client import QdrantClient, models
import os
import numpy as np

client = QdrantClient(url=os.getenv("QDRANT_URL"))

client.recreate_collection(
    collection_name="laion400m_collection",
    vectors_config=models.VectorParams(
        size=512,  # CLIP embedding dimensions
        distance=models.Distance.COSINE,
        on_disk=True,  # Store original vectors on disk
    ),
    quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            always_ram=True,  # Keep quantized vectors in RAM
        )
    ),
    optimizers_config=models.OptimizersConfigDiff(
        max_segment_size=5_000_000, # Create larger segments for faster search
    ),
    hnsw_config=models.HnswConfigDiff(
        m=6,  # Lower m to reduce memory usage
        on_disk=False  # Keep the HNSW index graph in RAM
    ),
)

def upload_data_to_qdrant(client, embeddings, metadata, parallel=4):
    """
    Uploads data to Qdrant using the upsert method.
    """
    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i],
            payload=metadata[i]
        )
        for i in range(len(metadata))
    ]
    
    client.upsert(
        collection_name="laion400m_collection",
        points=points,
    )

# Generate sample data points
num_samples = 1000
embeddings = np.random.randn(num_samples, 512).astype(np.float32)
metadata = [
    {"text": f"Image description {i}", "source": "laion400m", "id": i}
    for i in range(num_samples)
]

# Upload data to Qdrant
upload_data_to_qdrant(client, embeddings, metadata)
print(f"Successfully uploaded {num_samples} data points to Qdrant")
