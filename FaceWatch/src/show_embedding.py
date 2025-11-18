import numpy as np
from pathlib import Path

def main():
    emb_path = Path("outputs/embeddings/hani.npy")

    emb = np.load(emb_path)
    print("=== Hani Embedding Info ===")
    print(f"shape : {emb.shape}")
    print(f"dtype : {emb.dtype}")
    print(f"min   : {emb.min():.4f}")
    print(f"max   : {emb.max():.4f}")
    print(f"mean  : {emb.mean():.4f}")
    print(f"std   : {emb.std():.4f}")
    print()
    print("first 10 values:")
    print(emb[:10])

    # 혹시 벡터 norm이 알고 싶으면:
    norm = np.linalg.norm(emb)
    print(f"\nL2 norm: {norm:.4f}")

if __name__ == "__main__":
    main()