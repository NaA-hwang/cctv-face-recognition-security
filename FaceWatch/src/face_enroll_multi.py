# src/face_enroll_multi.py
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
from utils.device_config import get_device_id

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def iter_enroll_targets(root: Path):
    """
    images/enroll/ 구조를 다음 두 가지 모두 지원:
    1) images/enroll/hani.jpg        -> person_id = 'hani'
    2) images/enroll/hani/*.jpg ...  -> person_id = 'hani'
    """
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p.stem, [p]
        elif p.is_dir():
            imgs = [x for x in p.glob("**/*") if x.suffix.lower() in IMG_EXTS]
            if imgs:
                yield p.name, imgs

def get_main_face_embedding(app: FaceAnalysis, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {img_path}")

    faces = app.get(img)
    if len(faces) == 0:
        raise RuntimeError(f"얼굴을 찾지 못했어: {img_path}")

    # 가장 큰 얼굴 선택
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True
    )
    main_face = faces_sorted[0]
    return main_face.embedding

def main():
    enroll_root = Path("images") / "enroll"
    if not enroll_root.exists():
        raise FileNotFoundError(f"등록용 폴더가 없음: {enroll_root}")

    # 모델 로드 (GPU 우선, 없으면 CPU)
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=device_id, det_size=(640, 640))

    out_dir = Path("outputs") / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 등록 대상 폴더: {enroll_root}")
    for person_id, img_list in iter_enroll_targets(enroll_root):
        print(f"\n=== {person_id} 등록 시작 ({len(img_list)}장) ===")
        embs = []
        for img_path in img_list:
            try:
                emb = get_main_face_embedding(app, img_path)
                embs.append(emb)
                print(f"  ✅ {img_path.name} -> ok")
            except Exception as e:
                print(f"  ⚠️ {img_path.name} -> 실패: {e}")

        if not embs:
            print(f"  ❌ {person_id}: 사용 가능한 얼굴 없음, 스킵")
            continue

        embs = np.stack(embs, axis=0)     # [N, 512]
        mean_emb = embs.mean(axis=0)      # [512]
        # L2 정규화(나중 매칭 계산 안정)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        out_path = out_dir / f"{person_id}.npy"
        np.save(out_path, mean_emb)
        print(f"  💾 저장 완료: {out_path} (이미지 {len(embs)}장 평균)")

    print("\n✅ 전체 등록 완료")

if __name__ == "__main__":
    main()
