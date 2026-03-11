'''
Scalar Quantization
It is a technique used to compress high-dimensional vectors by quantizing each dimension independently.
This method reduces the memory footprint of the vectors while still allowing for efficient similarity search.
In Qdrant, you can configure scalar quantization for your collections to optimize storage and search.

'''

from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"))

client.create_collection(
    collection_name="scalar_collection",
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
        on_disk=True,  
    ),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,  # only quantize values up to the 99th percentile to reduce the impact of outliers
            always_ram=True,  # keep quantized vectors in RAM for faster access, even if on_disk is True
        )
    ),
)

