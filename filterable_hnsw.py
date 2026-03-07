from qdrant_client import QdrantClient
from qdrant_client import models
import os
import random

client = QdrantClient(
    url="http://localhost:6333",
)

collection_name = "store"
vector_size = 384

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

# Create collection with filterable HNSW index
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
    ),
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=100, # Set indexing threshold to 100 to trigger HNSW index creation after 100 points are added
    ),
)

# Create payload index for the "category" field to enable filtering
client.create_payload_index(
    collection_name=collection_name,
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD, 
)

# Create payload index for the "brand" field to enable filtering
client.create_payload_index(
    collection_name=collection_name,
    field_name="brand",
    field_schema=models.PayloadSchemaType.KEYWORD,
)


points = []
for i in range(1000):
    points.append(
        models.PointStruct(
            id=i,
            vector=[random.random() for _ in range(vector_size)],
            payload={
                "category": random.choice(["laptop", "phone", "tablet"]),
                "price": random.randint(0, 1000),
                "brand": random.choice(
                    ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "Samsung"]
                ),
            },
        )
    )
client.upload_points(
    collection_name=collection_name,
    points=points,
)

# Create filter combining multiple conditions
filter_conditions = models.Filter(
    must=[
        models.FieldCondition(key="category", match=models.MatchValue(value="laptop")),
        models.FieldCondition(key="price", range=models.Range(lte=1000)),
        models.FieldCondition(key="brand", match=models.MatchAny(any=["Apple", "Dell", "HP"])),
    ]
)

query_vector = [random.random() for _ in range(vector_size)]

# Execute filtered search
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=filter_conditions,
    limit=10,
    search_params=models.SearchParams(hnsw_ef=128),
)

'''
Connects to a local Qdrant vector database
Creates a collection with 384-dimensional vectors indexed for fast similarity search
Generates & uploads 1,000 random product records (laptops, phones, tablets) with vectors and metadata
Executes a filtered search that finds the 10 most similar products to a query vector while filtering for laptops under $1000 from Apple, Dell, or HP


'''