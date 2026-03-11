'''
Product Quantization
It is a technique that divides the vector space into smaller subspaces and quantizes each subspace independently.
This method allows for a more efficient representation of high-dimensional vectors while still maintaining a good level of accuracy in similarity search.
In Qdrant, you can configure product quantization for your collections to optimize storage and search
'''

from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"))

client.create_collection(
    collection_name="product_collection",
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
        on_disk=True,  # store vectors on disk to save RAM, but they will be loaded into RAM for search operations
    ),
    quantization_config=models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X32,  # compress vectors by a factor of 32, which means each vector will be represented by 48 bytes instead of 1536 bytes
            always_ram=True,  # keep quantized vectors in RAM for faster access, even if on_disk is True
        )
    ),
)