# src/face_match_gif.py
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
import imageio.v2 as imageio  # pip install imageio
from utils.device_config import get_device_id

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def main():
    # InsightFace 로드 (GPU 우선, 없으면 CPU)
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=device_id, det_size=(640, 640))

    # 등록한 사람의 임베딩 파일
    emb_path = Path("outputs") / "embeddings" / "hani.npy"
    enroll_emb = np.load(emb_path)
    print(f"✅ 등록 임베딩 로드: {emb_path}, shape={enroll_emb.shape}")

    # GIF 파일
    gif_path = Path("images") / "newjeans_dance.gif"
    frames = imageio.mimread(str(gif_path))  # RGB 프레임 리스트
    print(f"총 프레임 수: {len(frames)}")

    best_sim = -1.0
    best_frame_idx = -1
    best_face = None
    best_frame_bgr = None

    # GIF 모든 프레임 검사
    for idx, frame_rgb in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(frame_bgr)

        if len(faces) == 0:
            continue

        for face in faces:
            sim = cosine_similarity(enroll_emb, face.embedding)
            print(f"[frame {idx}] similarity = {sim:.3f}")

            if sim > best_sim:
                best_sim = sim
                best_face = face
                best_frame_idx = idx
                best_frame_bgr = frame_bgr.copy()

    # 얼굴을 하나도 못 찾은 경우
    if best_face is None:
        print("❌ 어떤 프레임에서도 얼굴을 찾지 못했어.")
        return

    # 임계값 판단
    THRESH = 0.35
    match = best_sim >= THRESH

    # bbox 그리기
    x1, y1, x2, y2 = map(int, best_face.bbox)
    color = (0, 255, 0) if match else (0, 0, 255)
    label = f"{'MATCH' if match else 'maybe'} {best_sim:.2f}"

    cv2.rectangle(best_frame_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        best_frame_bgr, label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        color, 2
    )

    # 결과 저장
    out_dir = Path("outputs") / "matches"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{gif_path.stem}_bestframe_result.jpg"
    cv2.imwrite(str(out_path), best_frame_bgr)

    print("\n🎯 === 결과 ===")
    print(f"프레임: {best_frame_idx} / sim={best_sim:.3f}")
    print(f"🖼 저장 완료: {out_path}")
    print("====================")

if __name__ == "__main__":
    main()
