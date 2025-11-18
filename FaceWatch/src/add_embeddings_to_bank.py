# src/add_embeddings_to_bank.py
# 기존 bank에 새로운 임베딩을 추가하는 도구
# 옆얼굴 등 다양한 각도의 얼굴 이미지를 추가하여 인식 성능 향상

from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
from utils.device_config import get_device_id, safe_prepare_insightface, _ensure_cuda_in_path

_ensure_cuda_in_path()

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
    emb = l2_normalize(emb)
    return emb


def add_images_to_bank(app: FaceAnalysis, person_id: str, image_paths: list[Path], 
                       emb_dir: Path, similarity_threshold: float = 0.95):
    """
    기존 bank에 새로운 이미지들의 임베딩을 추가
    
    Args:
        app: FaceAnalysis 인스턴스
        person_id: 사람 ID
        image_paths: 추가할 이미지 경로 리스트
        emb_dir: 임베딩 저장 디렉토리
        similarity_threshold: 중복 체크 임계값 (이 값 이상이면 중복으로 간주)
    
    Returns:
        추가된 임베딩 개수
    """
    bank_path = emb_dir / f"{person_id}_bank.npy"
    centroid_path = emb_dir / f"{person_id}_centroid.npy"
    
    # 기존 bank 로드
    if bank_path.exists():
        bank = np.load(bank_path)  # (N, 512)
        print(f"📚 기존 bank 로드: {bank_path.name} ({bank.shape[0]}개 임베딩)")
    else:
        bank = np.empty((0, 512), dtype=np.float32)
        print(f"📚 새 bank 생성: {person_id}")
    
    # 각 이미지에서 임베딩 추출
    new_embeddings = []
    skipped_count = 0
    
    for img_path in image_paths:
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        
        print(f"  ▶ 처리 중: {img_path.name}")
        emb = get_main_face_embedding(app, img_path)
        
        if emb is None:
            skipped_count += 1
            continue
        
        # 중복 체크: 기존 bank와 너무 유사한 임베딩이 있으면 스킵
        if bank.shape[0] > 0:
            # 모든 기존 임베딩과 유사도 계산
            sims = bank @ emb  # (N,)
            max_sim = float(np.max(sims))
            
            if max_sim >= similarity_threshold:
                print(f"     ⏭ 스킵 (기존 임베딩과 유사도 {max_sim:.3f} >= {similarity_threshold})")
                skipped_count += 1
                continue
        
        new_embeddings.append(emb)
        print(f"     ✅ 추가 (기존 bank와 최대 유사도: {max_sim:.3f if bank.shape[0] > 0 else 'N/A'})")
    
    if not new_embeddings:
        print(f"\n⚠️ 추가할 새로운 임베딩이 없습니다. (스킵: {skipped_count}개)")
        return 0
    
    # Bank에 추가
    new_embs_array = np.stack(new_embeddings, axis=0)  # (M, 512)
    updated_bank = np.vstack([bank, new_embs_array])  # (N+M, 512)
    
    # Centroid 재계산
    updated_centroid = updated_bank.mean(axis=0)
    updated_centroid = l2_normalize(updated_centroid)
    
    # 저장
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(bank_path, updated_bank)
    np.save(centroid_path, updated_centroid)
    
    # 기존 호환성을 위해 person_id.npy도 업데이트
    legacy_path = emb_dir / f"{person_id}.npy"
    np.save(legacy_path, updated_centroid)
    
    print(f"\n✅ Bank 업데이트 완료!")
    print(f"   추가된 임베딩: {len(new_embeddings)}개")
    print(f"   총 임베딩 수: {updated_bank.shape[0]}개 (기존 {bank.shape[0]}개 + 신규 {len(new_embeddings)}개)")
    print(f"   Bank 저장: {bank_path}")
    print(f"   Centroid 저장: {centroid_path}")
    print(f"   Legacy 저장: {legacy_path}")
    
    return len(new_embeddings)


def add_from_folder(app: FaceAnalysis, person_id: str, folder_path: Path, 
                    emb_dir: Path, similarity_threshold: float = 0.95):
    """
    폴더 내의 모든 이미지를 bank에 추가
    
    Args:
        app: FaceAnalysis 인스턴스
        person_id: 사람 ID
        folder_path: 이미지가 있는 폴더 경로
        emb_dir: 임베딩 저장 디렉토리
        similarity_threshold: 중복 체크 임계값
    """
    image_paths = [p for p in sorted(folder_path.glob("*")) 
                   if p.suffix.lower() in IMG_EXTS]
    
    if not image_paths:
        print(f"⚠️ {folder_path} 안에 이미지 파일이 없습니다.")
        return
    
    print(f"📁 폴더에서 이미지 찾음: {len(image_paths)}개")
    
    return add_images_to_bank(
        app=app,
        person_id=person_id,
        image_paths=image_paths,
        emb_dir=emb_dir,
        similarity_threshold=similarity_threshold
    )


def main():
    """사용 예시"""
    # 설정
    person_id = "hani"  # 업데이트할 사람 ID
    
    # 방법 1: 특정 폴더의 이미지들을 추가
    # 예: extracted_frames/hani 폴더에 옆얼굴 이미지들이 있다면
    image_folder = Path("images") / "extracted_frames" / person_id
    
    # 방법 2: enroll 폴더에 새로 추가한 이미지들을 bank에 반영
    # enroll_folder = Path("images") / "enroll" / person_id
    
    emb_dir = Path("outputs") / "embeddings"
    
    print("=" * 60)
    print("Bank에 임베딩 추가 도구")
    print("=" * 60)
    print(f"대상: {person_id}")
    print(f"이미지 폴더: {image_folder}")
    print()
    
    # InsightFace 초기화
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    
    # Bank에 추가
    added_count = add_from_folder(
        app=app,
        person_id=person_id,
        folder_path=image_folder,
        emb_dir=emb_dir,
        similarity_threshold=0.95  # 0.95 이상이면 중복으로 간주
    )
    
    if added_count > 0:
        print(f"\n💡 다음 단계:")
        print(f"   python src/face_match_video_multi.py 실행하여 인식 성능 확인")


if __name__ == "__main__":
    main()



