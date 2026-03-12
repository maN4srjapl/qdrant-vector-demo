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
    

baseline_metrics = measure_search_performance("store", [[0.12] * 384] * 100, label="Baseline")

# Define test queries for benchmarking quantized collections
your_test_queries = [[0.12] * 384] * 100

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
'''
# Create quantized collections
for method_name, config_info in quantization_configs.items():
    collection_name = f"quantized_{method_name}"
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,  # Adjust to your embedding size
            distance=models.Distance.COSINE,
            on_disk=True,  # Store originals on disk
        ),
        quantization_config=config_info["config"]
    )
    
    print(f"Created {method_name} quantized collection: {collection_name}")

'''
# Upload the same data to each quantized collection (omitted for brevity, but you would repeat the upsert process for each collection)

def benchmark(collection_name, your_test_queries, method_name):
    """Measure quantized search performance"""
    
    # Test without oversampling/rescoring first
    no_rescoring_metrics = measure_search_performance(
        collection_name, 
        your_test_queries, 
        f"{method_name} (No Rescoring)"
    )
    
    # Test with oversampling and rescoring
    def search_with_rescoring(collection_name, query, oversampling_factor=3.0):
        start_time = time.time()
        
        response = client.query_points(
            collection_name=collection_name,
            query=query,
            limit=10,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=oversampling_factor,
                )
            ),
        )
        
        return (time.time() - start_time) * 1000, response
    
    # Measure with rescoring
    rescoring_latencies = []
    for query in your_test_queries:
        latency, response = search_with_rescoring(collection_name, query)
        rescoring_latencies.append(latency)
    
    avg_rescoring = np.mean(rescoring_latencies)
    p95_rescoring = np.percentile(rescoring_latencies, 95)
    
    print(f"{method_name} (With Rescoring):")
    print(f"  Average latency: {avg_rescoring:.2f}ms")
    print(f"  P95 latency: {p95_rescoring:.2f}ms")
    
    return {
        "no_rescoring": no_rescoring_metrics,
        "with_rescoring": {"avg": avg_rescoring, "p95": p95_rescoring}
    }


quantization_results = {}
for method_name in quantization_configs.keys():
    collection_name = f"quantized_{method_name}"
    quantization_results[method_name] = benchmark(
        collection_name, your_test_queries, method_name
    )

def measure_accuracy_retention(original_collection, quantized_collection, test_queries, factors=[2, 3, 5, 8, 10]):
    """Compare search results between original and quantized collections"""

    results = {}

    for factor in factors:
        accuracy_scores = []
        
        for query in test_queries:
            # Get baseline results
            baseline_results = client.query_points(
                collection_name=original_collection,
                query=query,
                limit=10
            )
            baseline_ids = [point.id for point in baseline_results.points]

            # Get quantized results with rescoring
            quantized_results = client.query_points(
                collection_name=quantized_collection,
                query=query,
                limit=10,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        rescore=True,
                        oversampling=factor,
                    )
                ),
            )
            quantized_ids = [point.id for point in quantized_results.points]
            
            # Calculate overlap (simple accuracy measure)
            overlap = len(set(baseline_ids) & set(quantized_ids))
            accuracy = overlap / len(baseline_ids)
            accuracy_scores.append(accuracy)
        
        results[factor] = {
            "avg_accuracy": np.mean(accuracy_scores)
        }
    
    return results


def tune_oversampling(collection_name, test_queries, factors=[2, 3, 5, 8, 10]):
    """Find optimal oversampling factor"""
    results = {}
    
    for factor in factors:
        latencies = []
        
        for query in test_queries:
            start_time = time.time()
            
            response = client.query_points(
                collection_name=collection_name,
                query=query,
                limit=10,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        rescore=True,
                        oversampling=factor,
                    )
                ),
            )
            
            latencies.append((time.time() - start_time) * 1000)
        
        results[factor] = {
            "avg_latency": np.mean(latencies),
            "p95_latency": np.percentile(latencies, 95)
        }
    
    return results

# Tune oversampling for your method of choice
best_method = "binary"  # Choose based on your results
oversampling_factors = [2, 3, 5, 8, 10]

oversampling_results_latency = tune_oversampling(
    f"quantized_{best_method}", 
    your_test_queries,
    oversampling_factors
)

oversampling_results_accuracy = measure_accuracy_retention(
    "your_domain_collection",
    f"quantized_{best_method}", 
    your_test_queries,
    oversampling_factors
)


print("Oversampling Factor Optimization:")
for factor in oversampling_factors:
    print(f"  {factor}x:")
    print(f"  {oversampling_results_latency[factor]['avg_latency']:.2f}ms avg latency, {oversampling_results_latency[factor]['p95_latency']:.2f}ms P95 latency")
    print(f"  {oversampling_results_accuracy[factor]['avg_accuracy']:.2f} avg accuracy retention")

print("=" * 60)
print("QUANTIZATION PERFORMANCE ANALYSIS")
print("=" * 60)

print(f"\nBaseline Performance:")
print(f"  Average latency: {baseline_metrics['avg']:.2f}ms")
print(f"  P95 latency: {baseline_metrics['p95']:.2f}ms")

print(f"\nQuantization Results:")
for method, results in quantization_results.items():
    no_rescoring = results['no_rescoring']
    with_rescoring = results['with_rescoring']
    
    speedup_no_rescoring = baseline_metrics['avg'] / no_rescoring['avg']
    speedup_with_rescoring = baseline_metrics['avg'] / with_rescoring['avg']
    
    print(f"\n{method.upper()}:")
    print(f"  Without rescoring: {no_rescoring['avg']:.2f}ms ({speedup_no_rescoring:.1f}x speedup)")
    print(f"  With rescoring: {with_rescoring['avg']:.2f}ms ({speedup_with_rescoring:.1f}x speedup)")