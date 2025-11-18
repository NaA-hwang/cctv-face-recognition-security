# src/face_enroll_bank.py

from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
from utils.device_config import get_device_id

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화 (norm=1)"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def get_main_face_embedding(app: FaceAnalysis, img_path: Path) -> np.ndarray | None:
    """이미지에서 가장 큰 얼굴 한 개의 임베딩을 반환 (없으면 None)"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️ 이미지 읽기 실패: {img_path}")
        return None

    faces = app.get(img)
    if len(faces) == 0:
        print(f"  ⚠️ 얼굴 미검출: {img_path}")
        return None

    # 가장 큰 얼굴 선택
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    main_face = faces_sorted[0]
    emb = main_face.embedding.astype("float32")

    # 개별 임베딩도 먼저 L2 정규화
    emb = l2_normalize(emb)
    return emb

def process_person_folder(app: FaceAnalysis, person_dir: Path, out_dir: Path):
    """특정 사람 폴더(예: images/enroll/hani)에서 모든 이미지 임베딩 → bank 및 centroid 저장"""
    person_id = person_dir.name
    print(f"\n===== {person_id} 등록 시작 =====")

    emb_list: list[np.ndarray] = []

    # 이미지 파일 순회
    for img_path in sorted(person_dir.glob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        print(f"  ▶ 이미지 처리: {img_path.name}")
        emb = get_main_face_embedding(app, img_path)
        if emb is None:
            continue
        emb_list.append(emb)

    if not emb_list:
        print(f"  ❌ 유효한 얼굴 임베딩 없음 → {person_id} 스킵")
        return

    embs = np.stack(emb_list, axis=0)   # (N, 512)
    centroid = embs.mean(axis=0)        # (512,)
    centroid = l2_normalize(centroid)   # 최종 centroid도 L2 정규화

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Bank 저장 (N, 512)
    bank_path = out_dir / f"{person_id}_bank.npy"
    np.save(bank_path, embs)
    
    # Centroid 저장 (512,)
    centroid_path = out_dir / f"{person_id}_centroid.npy"
    np.save(centroid_path, centroid)
    
    # 기존 호환성을 위해 person_id.npy도 저장
    legacy_path = out_dir / f"{person_id}.npy"
    np.save(legacy_path, centroid)

    print(f"  ✅ {person_id} 등록 완료")
    print(f"     사용된 이미지 수 : {len(emb_list)}장")
    print(f"     Bank shape       : {embs.shape}")
    print(f"     Bank 저장 경로   : {bank_path}")
    print(f"     Centroid 저장 경로: {centroid_path}")
    print(f"     Legacy 저장 경로 : {legacy_path}")
    print(f"     L2 norm          : {np.linalg.norm(centroid):.4f}")

def main():
    # 1) InsightFace 모델 준비 (GPU 우선, 없으면 CPU)
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=device_id, det_size=(640, 640))

    # 2) 경로 설정
    project_root = Path(".")  # C:\FaceWatch 에서 실행한다고 가정
    enroll_root = project_root / "images" / "enroll"
    out_root = project_root / "outputs" / "embeddings"

    if not enroll_root.exists():
        raise FileNotFoundError(f"enroll 폴더를 찾을 수 없음: {enroll_root}")

    # 3) 사람별 폴더 순회
    person_dirs = [p for p in enroll_root.iterdir() if p.is_dir()]
    if not person_dirs:
        print(f"⚠️ {enroll_root} 안에 사람별 폴더가 없습니다. (예: images/enroll/hani)")
        return

    print("👥 등록 대상 사람 목록:")
    for d in person_dirs:
        print(f"  - {d.name}")

    for person_dir in person_dirs:
        process_person_folder(app, person_dir, out_root)

    print("\n🎉 모든 사람에 대한 bank 및 centroid 임베딩 생성 완료!")

if __name__ == "__main__":
    main()

