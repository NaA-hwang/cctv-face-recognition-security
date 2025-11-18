# src/face_match_crowd.py
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
from utils.device_config import get_device_id

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def main():
    # GPU 우선, 없으면 CPU
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=device_id, det_size=(640, 640))

    # 1) 등록된 임베딩 불러오기
    emb_path = Path("outputs") / "embeddings" / "hani.npy"
    enroll_emb = np.load(emb_path)
    print(f"✅ 등록 임베딩 로드: {emb_path}, shape={enroll_emb.shape}")

    # 2) 군중 이미지에서 얼굴들 찾기
    crowd_path = Path("images") / "newjeans_group_ditto.jpg"
    crowd_img = cv2.imread(str(crowd_path))
    if crowd_img is None:
        raise FileNotFoundError(f"군중 이미지를 찾을 수 없음: {crowd_path}")

    faces = app.get(crowd_img)
    print(f"군중 이미지에서 감지된 얼굴 수: {len(faces)}")

    if len(faces) == 0:
        print("⚠️ 군중 이미지에서 얼굴을 하나도 찾지 못했어.")
        return

    best_sim = -1.0
    best_face = None

    for i, face in enumerate(faces):
        sim = cosine_similarity(enroll_emb, face.embedding)
        print(f"Face {i}: similarity = {sim:.3f}")
        if sim > best_sim:
            best_sim = sim
            best_face = face

    # 3) 임계값 설정
    THRESH = 0.35  # 필요하면 나중에 조절

    # 제일 비슷한 얼굴은 항상 표시
    x1, y1, x2, y2 = map(int, best_face.bbox)
    is_match = best_sim >= THRESH

    # 임계값 넘었으면 초록, 아니면 빨강
    color = (0, 255, 0) if is_match else (0, 0, 255)
    label = f"{'MATCH' if is_match else 'maybe'} {best_sim:.2f}"

    cv2.rectangle(crowd_img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        crowd_img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    if is_match:
        print(f"\n✅ 임계값 {THRESH} 이상! 같은 사람일 가능성이 높음 (sim={best_sim:.3f})")
    else:
        print(f"\n❌ best_sim={best_sim:.3f} < THRESH={THRESH}. 같은 사람이라고 보기 애매함.")

    # 4) 결과 이미지 저장 (창 안 띄우고 파일로만)
    out_dir = Path("outputs") / "matches"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 입력 파일명 기반으로 결과 파일 이름 생성
    stem = crowd_path.stem        # newjeans_group_ditto
    out_path = out_dir / f"{stem}_result.jpg"
    cv2.imwrite(str(out_path), crowd_img)
    print(f"🖼 결과 이미지 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
