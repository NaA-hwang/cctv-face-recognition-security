# CUDA 경로를 먼저 설정 (가장 먼저 import)
from utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
import csv
import time
from utils.gallery_loader import load_gallery, match_with_bank
from utils.device_config import get_device_id, safe_prepare_insightface


def main():
    # ===== 설정 =====
    video_path = Path("images") / "newjeans_dance.gif"   # 분석할 영상/GIF
    emb_dir = Path("outputs") / "embeddings"             # 등록 임베딩 폴더
    THRESH = 0.30                                        # 임계값(일단 조금 낮게)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    # 갤러리 로드 (bank 우선)
    gallery = load_gallery(emb_dir, use_bank=True)
    # Bank 사용 여부 확인
    for pid, data in gallery.items():
        if data.ndim == 2:
            print(f"  - {pid}: bank ({data.shape[0]}개 임베딩)")
        else:
            print(f"  - {pid}: centroid")

    # FaceAnalysis 로드 (GPU 우선, 없으면 CPU)
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    if actual_device_id != device_id:
        print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
    print("set det-size: (640, 640)")
    print(f"🎥 영상 분석 시작: {video_path}")

    # 출력 폴더
    matches_dir = Path("outputs") / "matches_multi"
    matches_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("outputs") / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    log_path = logs_dir / f"{stem}_matches.csv"

    # CSV: 이제는 모든 얼굴 기록 + is_match 컬럼 포함
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["frame", "person_id", "similarity", "x1", "y1", "x2", "y2", "is_match"])

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # GIF에서 frame count 못 가져오는 경우도 있어서, 그냥 진행하면서 카운트
        total_frames = None

    frame_idx = 0
    hit_count = 0
    max_sim_ever = -1.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        # 디버깅용 출력: 프레임별 얼굴 개수
        print(f"[frame {frame_idx}] faces: {len(faces)}")

        if not faces:
            frame_idx += 1
            continue

        for face in faces:
            face_emb = face.embedding.astype("float32")
            
            # Bank 기반 매칭 (또는 centroid)
            best_person, best_sim = match_with_bank(face_emb, gallery)

            # 전체 중 최대 similarity 기록
            if best_sim > max_sim_ever:
                max_sim_ever = best_sim

            x1, y1, x2, y2 = map(int, face.bbox)
            is_match = 1 if best_sim >= THRESH else 0

            # ★ 이제는 매치 여부와 상관없이 CSV에 다 기록
            log_writer.writerow([frame_idx, best_person, best_sim, x1, y1, x2, y2, is_match])

            # 콘솔에 상위 결과만 간단히 출력
            print(f"  -> best: {best_person}, sim={best_sim:.3f}, match={bool(is_match)}")

            # 임계값 넘는 경우에만 이미지 스냅샷 저장
            if is_match:
                hit_count += 1
                label = f"{best_person} {best_sim:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                out_name = f"{stem}_f{frame_idx}_{best_person}_{best_sim:.2f}.jpg"
                cv2.imwrite(str(matches_dir / out_name), frame)

        frame_idx += 1

    cap.release()
    log_f.close()
    elapsed = time.time() - start_time

    print("\n✅ 분석 완료")
    print(f"  📄 로그: {log_path}")
    print(f"  🖼 스냅샷 수: {hit_count}장 (폴더: {matches_dir})")
    print(f"  🔎 관측된 최대 similarity: {max_sim_ever:.3f}")
    print(f"  ⏱  소요시간: {elapsed:.2f}초, 프레임 수: {frame_idx}")

if __name__ == "__main__":
    main()
