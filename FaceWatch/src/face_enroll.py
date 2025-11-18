# src/face_enroll.py
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
from utils.device_config import get_device_id

def get_main_face_embedding(app, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {img_path}")

    faces = app.get(img)
    if len(faces) == 0:
        raise RuntimeError(f"얼굴을 찾지 못했어: {img_path}")

    # 가장 큰 얼굴 하나 선택
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True
    )
    main_face = faces_sorted[0]
    return main_face.embedding

def main():
    # GPU 우선, 없으면 CPU
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=device_id, det_size=(640, 640))

    # ⭐ 등록할 사진 지정
    enroll_img = Path("images") / "newjeans_hani.jpg"
    emb = get_main_face_embedding(app, enroll_img)

    # ⭐ 저장될 임베딩 파일 이름
    out_dir = Path("outputs") / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hani.npy"
    np.save(out_path, emb)

    print("✅ 등록 완료")
    print(f"  이미지: {enroll_img}")
    print(f"  임베딩 저장: {out_path}")
    print(f"  벡터 shape: {emb.shape}")

if __name__ == "__main__":
    main()
