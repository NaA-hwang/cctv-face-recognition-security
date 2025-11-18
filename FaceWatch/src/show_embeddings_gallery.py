# src/show_embeddings_gallery.py
from pathlib import Path
import numpy as np

EMB_DIR = Path("outputs") / "embeddings"

def show_stats(name: str, emb: np.ndarray):
    print(f"\n=== {name} ===")
    print(f"shape : {emb.shape}")
    print(f"dtype : {emb.dtype}")
    print(f"min   : {emb.min():.4f}")
    print(f"max   : {emb.max():.4f}")
    print(f"mean  : {emb.mean():.4f}")
    print(f"std   : {emb.std():.4f}")
    norm = float(np.linalg.norm(emb))
    print(f"L2 norm: {norm:.4f}")

def main():
    if not EMB_DIR.exists():
        print(f"❌ 임베딩 폴더를 찾을 수 없음: {EMB_DIR}")
        return

    npy_files = sorted(EMB_DIR.glob("*.npy"))
    if not npy_files:
        print(f"❌ {EMB_DIR} 안에 .npy 파일이 없음")
        return

    print(f"📂 임베딩 폴더: {EMB_DIR}")
    print(f"📄 발견된 임베딩 파일: {[f.name for f in npy_files]}")

    for path in npy_files:
        emb = np.load(path)
        name = path.stem  # hani, minji, ...
        show_stats(name, emb)

if __name__ == "__main__":
    main()
