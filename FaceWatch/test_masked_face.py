"""
마스크를 쓴 얼굴 인식 테스트 스크립트
얼굴의 일부가 가려진 상태에서도 인식 가능한지 테스트합니다.
"""
import sys
from pathlib import Path

# src 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

# CUDA 경로 설정
from utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
import cv2
import numpy as np
from utils.gallery_loader import load_gallery, match_with_bank
from utils.device_config import get_device_id, safe_prepare_insightface

def test_masked_face():
    """마스크를 쓴 얼굴 인식 테스트"""
    print("=" * 60)
    print("마스크 얼굴 인식 테스트")
    print("=" * 60)
    
    # 1. 갤러리 로드
    emb_dir = Path("outputs") / "embeddings"
    gallery = load_gallery(emb_dir, use_bank=True)
    if not gallery:
        raise RuntimeError(f"갤러리 비어 있음: {emb_dir} 안에 .npy가 없습니다.")
    
    print("\n👥 갤러리 로드 완료:", list(gallery.keys()))
    for pid, data in gallery.items():
        if data.ndim == 2:
            print(f"  - {pid}: bank ({data.shape[0]}개 임베딩)")
        else:
            print(f"  - {pid}: centroid")
    
    # 2. InsightFace 준비
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"\n🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    if actual_device_id != device_id:
        print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
    
    # 3. 마스크 이미지 로드
    mask_img_path = Path("images") / "hani_mask.jpg"
    if not mask_img_path.exists():
        print(f"\n❌ 이미지를 찾을 수 없습니다: {mask_img_path}")
        return
    
    img = cv2.imread(str(mask_img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {mask_img_path}")
    
    print(f"\n🖼 이미지 로드: {mask_img_path}")
    print(f"   이미지 크기: {img.shape}")
    
    # 4. 얼굴 검출
    faces = app.get(img)
    print(f"\n감지된 얼굴 개수: {len(faces)}")
    
    if len(faces) == 0:
        print("⚠️ 얼굴을 하나도 찾지 못했습니다.")
        return
    
    # 5. 각 얼굴마다 매칭 테스트
    THRESH = 0.30
    
    print("\n" + "=" * 60)
    print("매칭 결과")
    print("=" * 60)
    
    for i, face in enumerate(faces):
        face_emb = face.embedding.astype("float32")
        
        # Bank 기반 매칭
        best_id, best_sim = match_with_bank(face_emb, gallery)
        
        x1, y1, x2, y2 = map(int, face.bbox)
        is_match = best_sim >= THRESH
        
        print(f"\n[얼굴 {i+1}]")
        print(f"  위치: ({x1}, {y1}) ~ ({x2}, {y2})")
        print(f"  크기: {x2-x1} x {y2-y1} 픽셀")
        print(f"  매칭 결과: {best_id}")
        print(f"  유사도: {best_sim:.4f}")
        print(f"  임계값 ({THRESH}) 이상: {'✅ 매칭됨' if is_match else '❌ 매칭 안됨'}")
        
        # 결과 이미지에 표시
        if is_match:
            label = f"{best_id} {best_sim:.2f}"
            color = (0, 255, 0)  # 초록
        else:
            label = f"unknown {best_sim:.2f}"
            color = (0, 0, 255)  # 빨강
        
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
    
    out_path = out_dir / "hani_mask_result.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"\n✅ 결과 저장: {out_path}")
    
    # 7. 요약
    print("\n" + "=" * 60)
    print("테스트 요약")
    print("=" * 60)
    
    matched_faces = [f for f in faces if match_with_bank(f.embedding.astype("float32"), gallery)[1] >= THRESH]
    
    print(f"\n총 얼굴 개수: {len(faces)}")
    print(f"매칭 성공: {len(matched_faces)}개")
    print(f"매칭 실패: {len(faces) - len(matched_faces)}개")
    
    if len(matched_faces) > 0:
        print("\n✅ 마스크를 쓴 상태에서도 얼굴 인식이 가능합니다!")
    else:
        print("\n⚠️ 마스크로 인해 인식이 어려울 수 있습니다.")
        print("   - 임계값을 낮춰보거나")
        print("   - 마스크를 쓴 이미지도 갤러리에 추가해보세요")

if __name__ == "__main__":
    test_masked_face()



