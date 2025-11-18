import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EMB_DIR = Path("outputs/embeddings")

persons = ["danielle", "hani", "harin", "minji"]

embs = {p: np.load(EMB_DIR / f"{p}.npy") for p in persons}

# 정규화
embs = {p: v / (np.linalg.norm(v) + 1e-6) for p, v in embs.items()}

plt.figure(figsize=(12, 6))

for p in persons:
    plt.hist(embs[p], bins=50, alpha=0.5, label=p)

plt.title("Embedding Distribution Comparison")
plt.xlabel("value")
plt.ylabel("count")
plt.legend()
plt.grid(True)
plt.show()

# 간단 요약 로그
print("=== Embedding Statistics ===")
for p in persons:
    v = embs[p]
    print(f"{p:10s} | mean={v.mean():.4f}  std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}")
