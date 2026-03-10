from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

docs = [
    "machine learning is fun",
    "machine learning is powerful",
    "machine learning is the future",
]


# Calculate term frequency for each document and score based on the query
query = "machine learning".split()

for i, doc in enumerate(docs):
    word = doc.split()
    tf = Counter(word)

    score = sum(tf[q] for q in query)
    print(f"Document {i} score: {score}")


# Calculate TF-IDF vectors for the documents and the query
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)
query_vector = vectorizer.transform(["machine learning"])
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
print("Query Vector:\n", query_vector.toarray())

# BM25 scoring
# IT IS A RANKING FUNCTION USED BY SEARCH ENGINES TO RANK DOCUMENTS BASED ON THE QUERY
# IT CONSIDERS TERM FREQUENCY, INVERSE DOCUMENT FREQUENCY, AND DOCUMENT LENGTH TO CALCULATE A RELEVANCE SCORE FOR EACH DOCUMENT IN RELATION TO THE QUERY

tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

query = "machine learning".split()
scores = bm25.get_scores(query)

for i, score in enumerate(scores):
    print(f"Document {i} BM25 score: {score}")
