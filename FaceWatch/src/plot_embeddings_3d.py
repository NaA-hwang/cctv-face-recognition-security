import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

def load_embeddings(dir_path="outputs/embeddings"):
    emb_dir = Path(dir_path)
    embeddings = []
    labels = []

    for npy_file in emb_dir.glob("*.npy"):
        emb = np.load(npy_file)
        name = npy_file.stem     # 파일명 → 사람 이름
        embeddings.append(emb)
        labels.append(name)

    return np.array(embeddings), labels

def main():
    embeddings, labels = load_embeddings()

    print("임베딩 shape:", embeddings.shape)

    # PCA로 512 → 3차원 축소
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(embeddings)

    xs = emb_3d[:, 0]
    ys = emb_3d[:, 1]
    zs = emb_3d[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x, y, z, label in zip(xs, ys, zs, labels):
        ax.scatter(x, y, z, s=60)
        ax.text(x, y, z, label)

    plt.title("Face Embeddings (PCA 3D Projection)")
    plt.show()

if __name__ == "__main__":
    main()
