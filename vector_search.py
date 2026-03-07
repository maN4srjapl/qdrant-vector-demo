from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(url="http://localhost:6333")

collection_name = "day0_first_system"

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
)

qdrant_client.create_payload_index(
    collection_name=collection_name,
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

points = [
    models.PointStruct(
        id=1,
        vector=[0.9, 0.1, 0.1, 0.8],
        payload={"name": "Budget Smartphone", "category": "electronics", "price": 299},
    ),
    models.PointStruct(
        id=2,
        vector=[0.2, 0.9, 0.8, 0.5],
        payload={"name": "Bestselling Novel", "category": "books", "price": 19},
    ),
    models.PointStruct(
        id=3,
        vector=[0.8, 0.3, 0.2, 0.9],
        payload={"name": "Smart Home Hub", "category": "electronics", "price": 89},
    ),
]

qdrant_client.upsert(collection_name=collection_name, points=points)

query_vector = [0.85, 0.2, 0.1, 0.9]

basic_results = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3
)

filtered_results = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value="electronics")
            )
        ]
    ),
    limit=3
)

print("Basic results:", basic_results)
print("Filtered results:", filtered_results)
