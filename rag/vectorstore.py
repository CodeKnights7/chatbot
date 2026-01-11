import faiss
import json
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/faiss.index"
CHUNKS_PATH = "data/chunks.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

def search(query, k=4):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results
