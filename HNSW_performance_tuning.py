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
    hnsw_config=models.HnswConfig( 
        # this is bulk loading optimization, which will be used when we upload points in batches
        m=0,  
        ef_construction=200,  #index construction quality
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
print(f"Vector size: {collection_info.vectors.size}")
print(f"Distance metric: {collection_info.vectors.distance}")
print(f"HNSW M: {collection_info.hnsw_config.m}")