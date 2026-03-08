from datasets import load_dataset
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import openai
import time
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))


try:
    collections = client.get_collections()
    print(f"Connected to Qdrant Cloud successfully!")
    print(f"Current collections: {len(collections.collections)}")
except Exception as e:
    print(f"Connection failed: {e}")
    print("Check your QDRANT_URL and QDRANT_API_KEY in .env file")

# Load the dataset (this may take a few minutes for first download)
print("Loading DBpedia 100K dataset...")
ds = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K")
collection_name = "dbpedia_100K"

print("Dataset loaded successfully!")
print(f"Dataset size: {len(ds['train'])} articles")

# Explore the dataset structure
print("\nDataset structure:")
print("Available columns:", ds["train"].column_names)

# Look at a sample entry
sample = ds["train"][0]
print(f"\nSample article:")
print(f"Title: {sample['title']}")
print(f"Text preview: {sample['text'][:200]}...")
print(f"Embedding dimensions: {len(sample['text-embedding-3-large-1536-embedding'])}")

try:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection '{collection_name}'")
except Exception as e:
    print(f"Failed to delete collection '{collection_name}': {e}")


print(f"\nCreating collection {collection_name}")

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
    ),
    hnsw_config=models.HnswConfigDiff( 
        # this is bulk loading optimization, which will be used when we upload points in batches
        m=0,  
        ef_construct=200,  #index construction quality
        full_scan_threshold=1000 # if collection has less than 1000 points, it will use brute-force search instead of HNSW index
    ),
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=10, # Set indexing threshold to 10 to trigger HNSW index creation after 10 points are added
    ),
    strict_mode_config=models.StrictModeConfig(
        enabled=False    # Disable strict mode to allow creating collection without waiting for index to be fully built, which is useful for performance tuning and testing
    ),
)

print(f"Collection '{collection_name}' created successfully!")
collection_info = client.get_collection(collection_name)
print(f"Vector size: {collection_info.config.params.vectors.size}")
print(f"Distance metric: {collection_info.config.params.vectors.distance}")
print(f"HNSW M: {collection_info.config.hnsw_config.m}")

# bulk upload with metadata
# VECTOR INGESTION PIPELINE:
def upload_batch(start_idx, end_idx):
    points = [] # a point consists of an id, a vector, and an optional payload (metadata)
    for i in range(start_idx, min(end_idx, total_points)):
        example = ds["train"][i]

        embedding = example["text-embedding-3-large-1536-embedding"]

        payload = {
            "text": example["text"],
            "title": example["title"],
            "_id": example["_id"],
            "length": len(example["text"]),
            "has_numbers": any(char.isdigit() for char in example["text"]),
        }

        points.append(models.PointStruct(id=i, vector=embedding, payload=payload))

    if points:
        client.upload_points(collection_name=collection_name, points=points)
        return len(points)
    return 0


batch_size = 64 * 10
total_points = len(ds["train"])
print(f"Uploading {total_points} points in batches of {batch_size}")

total_uploaded = 0
for i in tqdm(range(0, total_points, batch_size), desc="Uploading points"):
    uploaded = upload_batch(i, i + batch_size)
    total_uploaded += uploaded

print(f"Upload completed! Total points uploaded: {total_uploaded}")

# enabling hnsw indexing

client.update_collection(
    collection_name=collection_name,
    hnsw_config=models.HnswConfigDiff(
        m=16  
    ),
)

print("HNSW indexing enabled with m=16")