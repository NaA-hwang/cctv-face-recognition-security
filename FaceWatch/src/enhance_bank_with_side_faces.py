# src/enhance_bank_with_side_faces.py
# 영상에서 옆얼굴 등 다양한 각도의 얼굴을 자동으로 찾아 bank에 추가하는 통합 도구

from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
from utils.device_config import get_device_id, safe_prepare_insightface, _ensure_cuda_in_path
from utils.gallery_loader import load_gallery, match_with_bank

_ensure_cuda_in_path()


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def estimate_face_angle(face) -> str:
    """
    얼굴 각도를 대략적으로 추정 (랜드마크 기반)
    
    Returns:
        "front", "left", "right", "profile" 등
    """
    if not hasattr(face, 'kps') or face.kps is None:
        return "unknown"
    
    # 간단한 추정: 코와 눈의 위치로 판단
    # 실제로는 더 정교한 계산이 필요하지만, 여기서는 기본적인 예시만 제공
    return "front"  # 기본값


def find_diverse_faces_in_video(video_path: Path, person_id: str, gallery: dict,
                                emb_dir: Path, app: FaceAnalysis,
                                match_threshold: float = 0.30,
                                similarity_threshold: float = 0.90,
                                max_faces_per_person: int = 10):
    """
    영상에서 특정 인물의 다양한 각도 얼굴을 찾아 bank에 추가
    
    Args:
        video_path: 분석할 영상 경로
        person_id: 찾을 사람 ID
        gallery: 갤러리 딕셔너리
        emb_dir: 임베딩 저장 디렉토리
        app: FaceAnalysis 인스턴스
        match_threshold: 매칭 임계값 (이 값 이상이면 해당 인물로 인식)
        similarity_threshold: bank에 추가할 때 중복 체크 임계값
        max_faces_per_person: 인물당 최대 추가할 얼굴 수
    
    Returns:
        추가된 얼굴 개수
    """
    if person_id not in gallery:
        print(f"❌ 갤러리에 {person_id}가 없습니다.")
        return 0
    
    print(f"🎥 영상 분석 시작: {video_path.name}")
    print(f"   대상 인물: {person_id}")
    print(f"   매칭 임계값: {match_threshold}")
    print(f"   중복 체크 임계값: {similarity_threshold}")
    print()
    
    # 영상 로드
    frames = imageio.mimread(str(video_path))
    total_frames = len(frames)
    print(f"   총 프레임 수: {total_frames}")
    
    # 기존 bank 로드
    bank_path = emb_dir / f"{person_id}_bank.npy"
    if bank_path.exists():
        bank = np.load(bank_path)
        print(f"📚 기존 bank: {bank.shape[0]}개 임베딩")
    else:
        bank = np.empty((0, 512), dtype=np.float32)
        print(f"📚 새 bank 생성")
    
    # 수집된 얼굴 임베딩
    collected_embeddings = []
    frame_info = []
    
    # 각 프레임 분석
    for f_idx, frame in enumerate(frames):
        # RGB → BGR 변환
        if frame.ndim == 2:
            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            img = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        faces = app.get(img)
        
        for face in faces:
            face_emb = face.embedding.astype("float32")
            face_emb = l2_normalize(face_emb)
            
            # 갤러리와 매칭
            best_person, best_sim = match_with_bank(face_emb, gallery)
            
            # 해당 인물이고 임계값 이상이면 수집
            if best_person == person_id and best_sim >= match_threshold:
                # 중복 체크: 기존 bank와 비교
                is_duplicate = False
                if bank.shape[0] > 0:
                    max_existing_sim = float(np.max(bank @ face_emb))
                    if max_existing_sim >= similarity_threshold:
                        is_duplicate = True
                
                # 수집된 임베딩과도 비교
                if not is_duplicate and collected_embeddings:
                    collected_array = np.stack(collected_embeddings, axis=0)
                    max_collected_sim = float(np.max(collected_array @ face_emb))
                    if max_collected_sim >= similarity_threshold:
                        is_duplicate = True
                
                if not is_duplicate:
                    collected_embeddings.append(face_emb)
                    frame_info.append({
                        "frame": f_idx,
                        "similarity": best_sim,
                        "angle": estimate_face_angle(face)
                    })
                    print(f"  ✅ 프레임 {f_idx}: 수집 (sim={best_sim:.3f}, 각도={estimate_face_angle(face)})")
                    
                    if len(collected_embeddings) >= max_faces_per_person:
                        print(f"  ⏹ 최대 수집 개수 도달 ({max_faces_per_person}개)")
                        break
        
        if len(collected_embeddings) >= max_faces_per_person:
            break
    
    if not collected_embeddings:
        print(f"\n⚠️ {person_id}의 새로운 얼굴을 찾지 못했습니다.")
        return 0
    
    # Bank에 추가
    new_embs_array = np.stack(collected_embeddings, axis=0)
    updated_bank = np.vstack([bank, new_embs_array])
    
    # Centroid 재계산
    updated_centroid = updated_bank.mean(axis=0)
    updated_centroid = l2_normalize(updated_centroid)
    
    # 저장
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(bank_path, updated_bank)
    
    centroid_path = emb_dir / f"{person_id}_centroid.npy"
    np.save(centroid_path, updated_centroid)
    
    legacy_path = emb_dir / f"{person_id}.npy"
    np.save(legacy_path, updated_centroid)
    
    print(f"\n✅ Bank 업데이트 완료!")
    print(f"   추가된 임베딩: {len(collected_embeddings)}개")
    print(f"   총 임베딩 수: {updated_bank.shape[0]}개")
    print(f"   수집된 프레임: {[info['frame'] for info in frame_info]}")
    
    return len(collected_embeddings)


def main():
    """사용 예시"""
    # 설정
    video_path = Path("images") / "newjeans_dance.gif"
    person_id = "hani"  # 옆얼굴 임베딩을 추가할 사람
    emb_dir = Path("outputs") / "embeddings"
    
    print("=" * 60)
    print("영상에서 다양한 각도 얼굴 찾아 Bank에 추가")
    print("=" * 60)
    
    # 갤러리 로드
    gallery = load_gallery(emb_dir, use_bank=True)
    if not gallery:
        raise RuntimeError(f"갤러리 비어 있음: {emb_dir}")
    
    print("👥 갤러리 로드 완료:", list(gallery.keys()))
    
    # InsightFace 초기화
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    
    # 다양한 각도 얼굴 찾아 추가
    added_count = find_diverse_faces_in_video(
        video_path=video_path,
        person_id=person_id,
        gallery=gallery,
        emb_dir=emb_dir,
        app=app,
        match_threshold=0.30,  # 이 값 이상이면 해당 인물로 인식
        similarity_threshold=0.90,  # 이 값 이상이면 중복으로 간주
        max_faces_per_person=10  # 최대 10개까지 추가
    )
    
    if added_count > 0:
        print(f"\n💡 다음 단계:")
        print(f"   python src/face_match_video_multi.py 실행하여 인식 성능 확인")


if __name__ == "__main__":
    main()



