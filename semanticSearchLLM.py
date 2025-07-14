#Semantic Search with LLM Embeddings

import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
openai.api_key = "your-api-key"

# Sample documents
documents = [
    "How to train a neural network using PyTorch.",
    "Best practices for deploying machine learning models.",
    "Understanding semantic similarity in NLP.",
    "How to cook a perfect steak."
]

# Query
query = "What is semantic similarity in language models?"

# Function to get embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return np.array(response['data'][0]['embedding'])

# Get embeddings
doc_embeddings = [get_embedding(doc) for doc in documents]
query_embedding = get_embedding(query)

# Compute cosine similarities
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the most similar document
most_similar_idx = np.argmax(similarities)
print("Most relevant document:")
print(documents[most_similar_idx])
