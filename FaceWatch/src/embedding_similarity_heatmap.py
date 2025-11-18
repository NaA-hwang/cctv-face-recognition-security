import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))

EMB_DIR = Path("outputs/embeddings")
persons = ["danielle", "hani", "harin", "minji"]

embs = {p: np.load(EMB_DIR / f"{p}.npy") for p in persons}

# Similarity Matrix 생성
n = len(persons)
M = np.zeros((n, n))

for i, p1 in enumerate(persons):
    for j, p2 in enumerate(persons):
        M[i, j] = cosine(embs[p1], embs[p2])

plt.figure(figsize=(6, 5))
sns.heatmap(M, annot=True, xticklabels=persons, yticklabels=persons, cmap="coolwarm", vmin=0, vmax=1)
plt.title("Person-to-Person Embedding Similarity")
plt.show()
