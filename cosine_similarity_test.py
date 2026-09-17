from ollama import embed
import numpy as np

sentences = [
    "SUID binaries can be abused for privilege escalation",
    "Check for files with the SUID bit set to escalate privileges",
    "My favorite pizza topping is pepperoni",
]

vectors = [embed(model='nomic-embed-text', input=s)['embeddings'][0] for s in sentences]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Sentence 1 vs 2 (both about SUID):", cosine_similarity(vectors[0], vectors[1]))
print("Sentence 1 vs 3 (SUID vs pizza):   ", cosine_similarity(vectors[0], vectors[2]))
