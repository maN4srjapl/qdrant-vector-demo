from datetime import datetime, timedelta
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
import os

client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

collection_name = "research_papers"

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False)
        )
    },
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="research_area",
    field_schema="keyword",  
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="open_access",
    field_schema="bool",  
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="published_date",
    field_schema="datetime",
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="impact_score",
    field_schema="float",
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="citation_count",
    field_schema="integer",
)


sample_data = [
    {
        "title": "Zero-Shot Retrieval for Scalable Visual Search in a Two-Sided Marketplace",
        "authors": ["Andre Rusli", "Shoma Ishimoto", "Sho Akiyama", "Aman Kumar Singh"],
        "abstract": "Visual search offers an intuitive way for customers to explore diverse product catalogs, particularly in consumer-to-consumer (C2C) marketplaces where listings are often unstructured and visually driven. This paper presents a scalable visual search system deployed in Mercari's C2C marketplace...",
        "research_area": "computer_vision",
        "published_date": "2025-07-31",
        "impact_score": 0.78,
        "citation_count": 12,
        "open_access": True,
    },
    {
        "title": "TALI: Towards A Lightweight Information Retrieval Framework for Neural Search",
        "authors": ["Chaoqun Liu", "Yuanming Zhang", "Jianmin Zhang", "Jiawei Han"],
        "abstract": "Neural search systems have emerged as a promising approach to enhance user engagement in information retrieval. However, their high computational costs and memory usage have limited their widespread adoption. In this paper, we present TALI, a lightweight information retrieval framework for neural search that efficiently addresses these challenges...",
        "research_area": "machine_learning",
        "published_date": "2025-07-31",
        "impact_score": 0.78,
        "citation_count": 12,
        "open_access": True,
    },
    {
        "title": "Zero-Shot Retrieval for Scalable Visual Search in a Two-Sided Marketplace",
        "authors": ["Andre Rusli", "Shoma Ishimoto", "Sho Akiyama", "Aman Kumar Singh"],
        "abstract": "Visual search offers an intuitive way for customers to explore diverse product catalogs, particularly in consumer-to-consumer (C2C) marketplaces where listings are often unstructured and visually driven. This paper presents a scalable visual search system deployed in Mercari's C2C marketplace...",
        "research_area": "computer_vision",
        "published_date": "2025-07-31",
        "impact_score": 0.78,
        "citation_count": 12,
        "open_access": True,
    },
]

texts = [it["abstract"] for it in sample_data]


DENSE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
SPARSE_MODEL_ID = "prithivida/Splade_PP_en_v1"  # SPLADE sparse
COLBERT_MODEL_ID = "colbert-ir/colbertv2.0"  # 128-dim multivector

dense_model = TextEmbedding(DENSE_MODEL_ID)
sparse_model = SparseTextEmbedding(SPARSE_MODEL_ID)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_ID)

dense_embeds = list(dense_model.embed(texts, parallel=0))
sparse_embeds = list(sparse_model.embed(texts, parallel=0))
colbert_multivectors = list(colbert_model.embed(texts, parallel=0))

points = []
for i, text in enumerate(texts):
    sparse_embed = sparse_embeds[i].as_object()
    dense_embed = dense_embeds[i]
    colbert_embed = colbert_multivectors[i]

    points.append(
        models.PointStruct(
            id=i,
            vector={
                "dense": dense_embed,
                "sparse": sparse_embed,
                "colbert": colbert_embed,
            },
            payload=sample_data[i],
        )
    )

client.upload_points(
    collection_name=collection_name,
    points=points,
)

