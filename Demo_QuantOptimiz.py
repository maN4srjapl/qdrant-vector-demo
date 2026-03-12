'''

Implementing:
1. Performance Comparison
2. Method Evaluation
3. Accuracy Recovery
4. Production Deployment

'''

import time
import numpy as np
from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"))

VECTOR_SIZE = 384  # Adjust to your embedding size
NUM_POINTS = 10000  # Number of points to upsert for testing


# generating the sample dataset

def generate_dataset(n_points=NUM_POINTS, dim=VECTOR_SIZE):
    vectors = np.random.rand(n_points, dim).astype(np.float32)

    points = []
    for i, vec in enumerate(vectors):
        points.append(
            models.PointStruct(
                id=i,
                vector=vec.tolist(),
                payload={"category": "demo"}
            )
        )
    return points

dataset = generate_dataset()

# uploading the dataset

def upload_data(collection_name, dataset, batch_size=500):
    for i in range(0, len(dataset), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=dataset[i:i+batch_size]
        )
    print(f"Uploaded data to {collection_name}")


# create the baseline collection
client.recreate_collection(
    collection_name="store",
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE
    )
)

upload_data("store", dataset)


def measure_search_performance(collection_name, test_queries, label ="Baseline"):
    latencies = []
    for query in test_queries:
        start_time = time.time()

        response = client.query_points(
            collection_name=collection_name,
            query=query,
            limit=10,)
        
        latencies.append(time.time() - start_time)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"{label} - Average Latency: {avg_latency:.4f} seconds, P95 Latency: {p95_latency:.4f} seconds")

        return {"average_latency": avg_latency, "p95_latency": p95_latency}
    

test_queries = np.random.rand(100, VECTOR_SIZE).tolist()

baseline_metrics = measure_search_performance(
    "store",
    test_queries,
    "Baseline"
)
# Now testing with quantization methods

# Define quantization configurations
quantization_configs = {
    "scalar": {
        "config": models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        ),
        "expected_speedup": "2x",
        "expected_compression": "4x"
    },
    "binary": {
        "config": models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                encoding=models.BinaryQuantizationEncoding.ONE_BIT,
                always_ram=True,
            )
        ),
        "expected_speedup": "40x",
        "expected_compression": "32x" # 1-bit quantization can achieve significant compression and speedup, but it may come with a substantial loss in accuracy, especially for high-dimensional vectors. It's best suited for applications where speed is critical and some loss in accuracy is acceptable.
    },
    "binary_2bit": {
        "config": models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                encoding=models.BinaryQuantizationEncoding.TWO_BITS,
                always_ram=True,
            )
        ),
        "expected_speedup": "20x", # 2-bit quantization is a middle ground between 1-bit and 8-bit, offering better accuracy than 1-bit while still providing significant speedup and compression benefits.
        "expected_compression": "16x"
    }
}

# Create quantized collections
for method_name, config_info in quantization_configs.items():
    collection_name = f"quantized_{method_name}"
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,  # Adjust to your embedding size
            distance=models.Distance.COSINE,
            on_disk=True,  # Store originals on disk
        ),
        quantization_config=config_info["config"]
    )
    
    upload_data(collection_name, dataset)
    print(f"Created {method_name} quantized collection: {collection_name}")


# Upload the same data to each quantized collection (omitted for brevity, but you would repeat the upsert process for each collection)

def benchmark(collection_name, queries, method_name):
    """Measure quantized search performance"""
    
    # Test without oversampling/rescoring first
    no_rescoring_metrics = measure_search_performance(
        collection_name, 
        queries, 
        f"{method_name} (No Rescoring)"
    )

    latencies = []

    for q in queries:

        start = time.time()

        client.query_points(
            collection_name=collection_name,
            query=q,
            limit=10,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=3
                )
            )
        )

        latencies.append((time.time() - start) * 1000)

    avg = np.mean(latencies)
    p95 = np.percentile(latencies, 95)

    print(f"{method_name} (Rescore) -> Avg: {avg:.2f}ms | P95: {p95:.2f}ms")

    return {
        "no_rescoring": no_rescoring_metrics,
        "with_rescoring": {"avg": avg, "p95": p95}
    }



quantization_results = {}

for method_name in quantization_configs.keys():
        collection_name = f"quantized_{method_name}"
        quantization_results[method_name] = benchmark(
            collection_name, test_queries, method_name
        )

def measure_accuracy_retention(original, quantized, queries, factors):

    results = {}

    for factor in factors:

        scores = []

        for q in queries:

            baseline = client.query_points(
                collection_name=original,
                query=q,
                limit=10
            )

            quantized_result = client.query_points(
                collection_name=quantized,
                query=q,
                limit=10,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        rescore=True,
                        oversampling=factor
                    )
                )
            )

            base_ids = {p.id for p in baseline.points}
            quant_ids = {p.id for p in quantized_result.points}

            overlap = len(base_ids & quant_ids)

            scores.append(overlap / 10)

        results[factor] = np.mean(scores)

    return results


def tune_oversampling(collection, queries, factors):

    results = {}

    for factor in factors:

        latencies = []

        for q in queries:

            start = time.time()

            client.query_points(
                collection_name=collection,
                query=q,
                limit=10,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        rescore=True,
                        oversampling=factor
                    )
                )
            )

            latencies.append((time.time() - start) * 1000)

        results[factor] = {
            "avg": np.mean(latencies),
            "p95": np.percentile(latencies, 95)
        }

    return results



best_method = "binary"

factors = [2, 3, 5, 8, 10]

lat_results = tune_oversampling(
    f"quantized_{best_method}",
    test_queries,
    factors
)

acc_results = measure_accuracy_retention(
    "store",
    f"quantized_{best_method}",
    test_queries,
    factors
)


print("\n===== QUANTIZATION PERFORMANCE ANALYSIS =====")

print("\nBaseline:")
print(f"Avg: {baseline_metrics['avg']:.2f}ms")
print(f"P95: {baseline_metrics['p95']:.2f}ms")

for method, result in quantization_results.items():

    no = result["no_rescoring"]
    res = result["with_rescoring"]

    speed_no = baseline_metrics["avg"] / no["avg"]
    speed_res = baseline_metrics["avg"] / res["avg"]

    print(f"\n{method.upper()}")

    print(f"Without Rescore: {no['avg']:.2f}ms ({speed_no:.2f}x speedup)")
    print(f"With Rescore: {res['avg']:.2f}ms ({speed_res:.2f}x speedup)")

print("\nOversampling Optimization")

for f in factors:

    print(
        f"{f}x -> "
        f"Latency {lat_results[f]['avg']:.2f}ms | "
        f"Accuracy {acc_results[f]:.2f}"
    )