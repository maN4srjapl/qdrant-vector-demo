'''
Oversampling + Rescoring
- Oversampling: This technique involves generating synthetic data points to balance the dataset. It can help improve the performance of machine learning models by providing more examples for underrepresented classes.
- Rescoring: After generating synthetic data points, you can use a rescoring mechanism to evaluate the relevance of the generated points to the original data. This can help filter out less relevant synthetic points and improve the overall accuracy of the model.

If quantization is impacting performance in an application that requires high accuracy, 
combining oversampling with rescoring is a great choice. However, if you need faster searches and can tolerate some loss in accuracy, 
you might choose to use oversampling without rescoring, or adjust the oversampling factor to a lower value.


'''

from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"))
                      

response = client.query_points(
    collection_name="store",
    query=[0.12] * 384,
    limit=10,
    search_params=models.SearchParams(
        hnsw_ef=128, # Higher ef values can improve recall but may increase search time
        quantization=models.QuantizationSearchParams(
            ignore=False, # Consider quantized vectors during search
            rescore=True,   # Enable original vectors-based rescoring
            oversampling=3.0,  # Retrieve 3x candidates for rescoring
        ),
    ),
    with_payload=True,
)