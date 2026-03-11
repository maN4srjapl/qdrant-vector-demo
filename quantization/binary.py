'''
Binary Quantization
It is an extreme form of quantization where each dimension of the vector is represented by a single bit (0 or 1).
This method is particularly useful for very large datasets where memory constraints are a concern, but it may lead to a significant loss of information and reduced search accuracy compared to other quantization methods.
In Qdrant, you can configure binary quantization for your collections to optimize storage and search.

'''
from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"))

client.create_collection(
    collection_name="binary_collection",
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
        on_disk=True,
    ),
    quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            encoding=models.BinaryQuantizationEncoding.ONE_BIT,  # each dimension is represented by a single bit (0 or 1)
            always_ram=True,  # keep quantized vectors in RAM for faster access, even if on_disk is True
        )
    ),
)   