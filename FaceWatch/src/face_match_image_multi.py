# src/face_match_image_multi.py
# CUDA 경로를 먼저 설정 (가장 먼저 import)
from utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
from utils.gallery_loader import load_gallery, match_with_bank
from utils.device_config import get_device_id, safe_prepare_insightface
from utils.mask_detector import estimate_mask_from_similarity, get_adjusted_threshold


def main():
    # 0. 사용할 이미지 지정 (여기만 바꿔주면 됨)
    img_path = Path("images") / "ive_mask.jpg"   # ← 테스트할 사진 파일 이름 (마스크된 얼굴)

    # 1. 갤러리(등록된 사람들) 로드 (bank 우선)
    emb_dir = Path("outputs") / "embeddings"
    gallery = load_gallery(emb_dir, use_bank=True)
    if not gallery:
        raise RuntimeError(f"갤러리 비어 있음: {emb_dir} 안에 .npy가 없습니다.")
    print("👥 갤러리 로드 완료:", list(gallery.keys()))
    # Bank 사용 여부 확인
    for pid, data in gallery.items():
        if data.ndim == 2:
            print(f"  - {pid}: bank ({data.shape[0]}개 임베딩)")
        else:
            print(f"  - {pid}: centroid")

    # 2. InsightFace 준비 (GPU 우선, 없으면 CPU)
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    if actual_device_id != device_id:
        print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")

    # 3. 이미지 로드
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {img_path}")
    print(f"🖼 이미지 로드: {img_path}, shape={img.shape}")

    # 4. 얼굴 검출
    faces = app.get(img)
    print(f"감지된 얼굴 개수: {len(faces)}")

    if len(faces) == 0:
        print("⚠ 얼굴을 하나도 찾지 못했어요.")
        return

    # 5. 각 얼굴마다 → 갤러리와 비교해서 가장 가까운 사람 찾기
    BASE_THRESH = 0.30  # 기본 임계값 (마스크 없는 일반 얼굴용)

    for i, face in enumerate(faces):
        face_emb = face.embedding.astype("float32")
        face_emb_normalized = face_emb / (np.linalg.norm(face_emb) + 1e-6)
        
        # Bank 기반 매칭 (또는 centroid)
        best_id, best_sim = match_with_bank(face_emb, gallery)
        
        # 실제 매칭 결과의 유사도 사용
        actual_sim = best_sim
        
        # 실제 유사도로 마스크 착용 가능성 추정
        mask_prob = estimate_mask_from_similarity(actual_sim)
        
        # 마스크 가능성과 유사도에 따라 적응형 임계값 계산
        use_thresh = get_adjusted_threshold(BASE_THRESH, mask_prob, actual_sim)
        
        # 마스크 정보 표시
        if mask_prob > 0.3:
            mask_info = f" [마스크 가능성: {mask_prob:.1f}, 임계값: {use_thresh:.2f}]"
        else:
            mask_info = ""

        x1, y1, x2, y2 = map(int, face.bbox)
        is_match = actual_sim >= use_thresh

        if is_match:
            label = f"{best_id} {actual_sim:.2f}"
            color = (0, 255, 0)  # 초록
        else:
            label = f"unknown {actual_sim:.2f}"
            color = (0, 0, 255)  # 빨강

        print(f"[face {i}] best={best_id}, sim={actual_sim:.3f}, thresh={use_thresh:.3f}, match={is_match}{mask_info}")

        # 박스 + 라벨 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # 6. 결과 이미지 저장
    out_dir = Path("outputs") / "matches_multi"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem          # 예: newjeans_group
    out_path = out_dir / f"{stem}_multi_result.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"✅ 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
