# src/face_match_video_tracks.py
# CUDA 경로를 먼저 설정 (가장 먼저 import)
from utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
import time
from collections import defaultdict
from utils.gallery_loader import load_gallery, match_with_bank
from utils.device_config import get_device_id, safe_prepare_insightface

# -------------------------------
# 유틸 함수들
# -------------------------------

def iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter + 1e-6)

# -------------------------------
# 메인 로직
# -------------------------------

def main():
    # 1) 갤러리 로드 (bank 우선)
    emb_dir = Path("outputs") / "embeddings"
    gallery = load_gallery(emb_dir, use_bank=True)
    print("👥 갤러리 로드 완료:", list(gallery.keys()))
    # Bank 사용 여부 확인
    for pid, data in gallery.items():
        if data.ndim == 2:
            print(f"  - {pid}: bank ({data.shape[0]}개 임베딩)")
        else:
            print(f"  - {pid}: centroid")

    # 2) InsightFace 초기화 (GPU 우선, 없으면 CPU)
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    if actual_device_id != device_id:
        print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
    print("set det-size: (640, 640)")

    # 3) 입력 영상 (GIF / mp4 상관 없음)
    video_path = Path("images") / "newjeans_dance.gif"
    frames = imageio.mimread(str(video_path))
    total_frames = len(frames)
    print(f"🎥 영상 분석 시작: {video_path}  (프레임 수: {total_frames})")

    # ---------------------------
    # Track 구조:
    # track_id: {
    #   'person': 추정 인물명,
    #   'detections': [ {frame, bbox, sim}, ... ],
    #   'last_bbox': [...],
    #   'last_frame': int
    # }
    # ---------------------------
    tracks = {}
    next_track_id = 0    
    
    # ---------------------------
    # 하이퍼파라미터 (자동 모드)
    # ---------------------------
    MODE = "test"   # 🔧 여기서 "test" <-> "prod" 바꿔 쓰면 됨

    if MODE == "test":
        # 👉 실험용: 조금 느슨하게
        BASE_THRESH   = 0.25   # 이 값 이상이면 "이 사람일 가능성 있음"
        STRONG_THRESH = 0.35   # 트랙 확정 임계값
        MIN_TRACK_LEN = 3      # 최소 감지 횟수
        IOU_THRESH    = 0.3    # 트래킹 IoU 기준
        MAX_SKIP      = 5      # 몇 프레임까지 끊기지 않은 것으로 볼지
    else:  # MODE == "prod"
        # 👉 실제 CCTV에 가깝게: 더 엄격하게
        BASE_THRESH   = 0.30
        STRONG_THRESH = 0.45
        MIN_TRACK_LEN = 5
        IOU_THRESH    = 0.4
        MAX_SKIP      = 3

    print(f"\n⚙ MODE = {MODE}")
    print(f"   BASE_THRESH   = {BASE_THRESH}")
    print(f"   STRONG_THRESH = {STRONG_THRESH}")
    print(f"   MIN_TRACK_LEN = {MIN_TRACK_LEN}")
    print(f"   IOU_THRESH    = {IOU_THRESH}")
    print(f"   MAX_SKIP      = {MAX_SKIP}\n")
    

    t0 = time.time()

    for f_idx, frame in enumerate(frames):
        # imageio가 넘겨준 frame을 numpy 배열로 보장
        frame = np.array(frame)

        # 채널 수에 따라 BGR 3채널로 변환
        if frame.ndim == 2:
            # 흑백 → BGR
            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            # RGBA → BGR
            img = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            # RGB → BGR
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        faces = app.get(img)
        print(f"[frame {f_idx}] faces: {len(faces)}")

        detections = []
        for face in faces:
            emb = face.embedding.astype("float32")
            
            # Bank 기반 매칭 (또는 centroid)
            best_person, best_sim = match_with_bank(emb, gallery)

            bbox = list(map(int, face.bbox))
            detections.append({
                "person": best_person,
                "sim": best_sim,
                "bbox": bbox,
                "embedding": emb / (np.linalg.norm(emb) + 1e-6)  # Online Learning을 위해 저장
            })
            print(f"  -> det person={best_person}, sim={best_sim:.3f}")

        # ---------------------------
        # 데이터 연계 (Tracking)
        # ---------------------------
        # 각 detection을 기존 track에 붙이거나 새 트랙 생성
        for det in detections:
            if det["sim"] < BASE_THRESH:
                # 너무 낮으면 아예 트랙에 안 붙임 (unknown 취급)
                continue

            assigned_tid = None
            best_iou = 0.0

            for tid, tr in tracks.items():
                # 같은 사람이고, 프레임 차이가 너무 크지 않을 때만 후보
                if tr["person"] != det["person"]:
                    continue
                if f_idx - tr["last_frame"] > MAX_SKIP:
                    continue

                iou_val = iou(tr["last_bbox"], det["bbox"])
                if iou_val > IOU_THRESH and iou_val > best_iou:
                    best_iou = iou_val
                    assigned_tid = tid

            if assigned_tid is None:
                # 새 트랙 생성
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = {
                    "person": det["person"],
                    "detections": [],
                    "last_bbox": det["bbox"],
                    "last_frame": f_idx
                }
                assigned_tid = tid
            else:
                # 기존 트랙 갱신
                tracks[assigned_tid]["last_bbox"] = det["bbox"]
                tracks[assigned_tid]["last_frame"] = f_idx

            # 공통: detection 기록 추가
            tracks[assigned_tid]["detections"].append({
                "frame": f_idx,
                "bbox": det["bbox"],
                "sim": det["sim"]
            })

    # ---------------------------
    # 트랙별 요약 & 스냅샷 저장
    # ---------------------------
    matches_dir = Path("outputs") / "tracks"
    matches_dir.mkdir(parents=True, exist_ok=True)

    # 스냅샷을 위해 프레임 다시 로드 (메모리 아끼려면 처음부터 저장해도 됨)
    frames = imageio.mimread(str(video_path))

    print("\n===== 트랙 요약 =====")
    for tid, tr in tracks.items():
        person = tr["person"]
        sims = [d["sim"] for d in tr["detections"]]
        max_sim = max(sims)
        avg_sim = sum(sims) / len(sims)
        length = len(tr["detections"])

        print(f"[track {tid}] person={person}, length={length}, "
              f"avg_sim={avg_sim:.3f}, max_sim={max_sim:.3f}")

        # 충분히 길고, max sim이 STRONG_THRESH 이상이면 "확실한 트랙"으로 간주
        if length >= MIN_TRACK_LEN and max_sim >= STRONG_THRESH:
            # 최고 sim이 나온 프레임의 bbox로 스냅샷 저장
            best_det = max(tr["detections"], key=lambda d: d["sim"])
            f_idx = best_det["frame"]
            x1, y1, x2, y2 = best_det["bbox"]

            img = frames[f_idx].copy()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{person} {max_sim:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            out_name = f"{person}_track{tid}_best.jpg"
            cv2.imwrite(str(matches_dir / out_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  -> ✅ 확정 트랙, 스냅샷 저장: {out_name}")
            
            # Online Learning: Bank에 임베딩 추가
            bank_path = emb_dir / f"{person}_bank.npy"
            if bank_path.exists() and "embedding" in best_det:
                best_emb = best_det["embedding"]
                
                # Bank 로드
                bank = np.load(bank_path)  # (N, 512)
                
                # 중복 체크: 기존 bank와 너무 유사한 임베딩이 있으면 스킵
                if bank.ndim == 2 and bank.shape[0] > 0:
                    max_existing_sim = float(np.max(bank @ best_emb))
                    if max_existing_sim < 0.95:  # 거의 동일한 임베딩이 아니면 추가
                        bank = np.vstack([bank, best_emb.reshape(1, -1)])
                        np.save(bank_path, bank)
                        print(f"  -> 📚 Bank 업데이트: {person}_bank.npy ({bank.shape[0]}개 임베딩)")
                    else:
                        print(f"  -> (Bank 업데이트 스킵: 유사 임베딩 존재, sim={max_existing_sim:.3f})")
                else:
                    # Bank가 비어있거나 형식이 이상한 경우
                    bank = best_emb.reshape(1, -1)
                    np.save(bank_path, bank)
                    print(f"  -> 📚 Bank 생성/업데이트: {person}_bank.npy")
        else:
            print("  -> (스냅샷 저장 안 함: 길이 or sim 부족)")

    t1 = time.time()
    print(f"\n✅ 전체 분석 완료, 소요시간: {t1 - t0:.2f}초, 총 트랙 수: {len(tracks)}")

if __name__ == "__main__":
    main()
