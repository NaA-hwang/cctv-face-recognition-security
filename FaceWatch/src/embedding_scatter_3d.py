import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

EMB_DIR = Path("outputs/embeddings")
persons = ["danielle", "hani", "harin", "minji"]

embs = []
labels = []

for p in persons:
    e = np.load(EMB_DIR / f"{p}.npy")
    embs.append(e)
    labels.append(p)

embs = np.array(embs)  # (4, 512)

# PCA 3D
pca = PCA(n_components=3)
xyz = pca.fit_transform(embs)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for i, p in enumerate(persons):
    ax.scatter(xyz[i, 0], xyz[i, 1], xyz[i, 2], s=80, label=p)

ax.set_title("PCA 3D Projection")
ax.legend()
plt.show()
