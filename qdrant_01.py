from qdrant_client import QdrantClient, models


qdrant_client = QdrantClient(
    url="http://localhost:6333",
)

# now create a collection

'''
a collection is a container for your data, you can have multiple collections in your Qdrant instance.
each collection has a name and a configuration, the configuration defines the type of vectors you want to store in the collection and the distance metric you want to use for similarity search.
'''

collection_name = "my_first_collection"

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
)

#verify the creation
collections  = qdrant_client.get_collections()
print("Existing collections:", collections)

# now insert some data into the collection
'''
points are the basic unit of data in Qdrant, each point has an id, a vector and optional payload (metadata).
the vector is a list of floats that represents the data you want to store, the payload is a dictionary that can contain any additional information you want to associate with the point.
'''

# Define the vectors to be inserted
points = [
    models.PointStruct(
        id=1,
        vector=[0.1, 0.2, 0.3, 0.4],  # 4D vector
        payload={"category": "example"}  # Metadata (optional)
    ),
    models.PointStruct(
        id=2,
        vector=[0.2, 0.3, 0.4, 0.5],
        payload={"category": "demo"}
    )
]

# Insert vectors into the collection
qdrant_client.upsert(
    collection_name=collection_name,
    points=points
)

#retrieve the collection details
collection_info = qdrant_client.get_collection(collection_name=collection_name)
print("Collection info:", collection_info)

query_vector = [0.08, 0.14, 0.33, 0.28]

search_results = qdrant_client.query_points(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=1  # Return the top 1 most similar vector
)

print("Search results:", search_results)