# src/extract_frames_from_video.py
# 영상에서 특정 인물(hani 등)을 식별하여 얼굴을 추출하는 도구
# 옆얼굴 등 다양한 각도의 얼굴 이미지를 수집하여 bank에 추가하기 위해 사용

import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from utils.device_config import get_device_id, safe_prepare_insightface, _ensure_cuda_in_path
from utils.gallery_loader import load_gallery, match_with_bank

_ensure_cuda_in_path()


def extract_frames(video_path: Path, output_dir: Path, frame_indices: list[int] = None, 
                   extract_all: bool = False, interval: int = 1):
    """
    영상에서 프레임을 추출하여 저장
    
    Args:
        video_path: 입력 영상 경로 (GIF, MP4 등)
        output_dir: 프레임 저장 디렉토리
        frame_indices: 추출할 특정 프레임 번호 리스트 (예: [10, 20, 30])
        extract_all: True면 모든 프레임 추출
        interval: extract_all=True일 때 프레임 간격
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없음: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 영상 정보: {video_path.name}")
    print(f"   총 프레임 수: {total_frames}")
    
    saved_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        should_save = False
        
        if extract_all:
            if frame_idx % interval == 0:
                should_save = True
        elif frame_indices and frame_idx in frame_indices:
            should_save = True
        
        if should_save:
            out_path = output_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
            print(f"  💾 프레임 {frame_idx} 저장: {out_path.name}")
        
        frame_idx += 1
    
    cap.release()
    print(f"\n✅ 총 {saved_count}개 프레임 저장 완료: {output_dir}")


def extract_faces_from_frames(video_path: Path, output_dir: Path, person_id: str,
                              frame_indices: list[int] = None, min_face_size: int = 50,
                              use_imageio: bool = True, match_threshold: float = 0.30,
                              emb_dir: Path = None):
    """
    영상에서 특정 인물을 식별하여 얼굴을 추출하여 저장
    옆얼굴 등 다양한 각도의 얼굴을 수집하기 위해 사용
    
    Args:
        video_path: 입력 영상 경로
        output_dir: 얼굴 이미지 저장 디렉토리 (person_id 폴더 생성)
        person_id: 식별할 사람 ID (갤러리에서 매칭)
        frame_indices: 추출할 프레임 번호 리스트 (None이면 모든 프레임)
        min_face_size: 최소 얼굴 크기 (픽셀)
        use_imageio: True면 imageio 사용 (GIF에 권장), False면 cv2.VideoCapture 사용
        match_threshold: 매칭 임계값 (이 값 이상이면 해당 인물로 인식)
        emb_dir: 임베딩 디렉토리 (None이면 outputs/embeddings 사용)
    """
    # 갤러리 로드 (인물 식별용)
    if emb_dir is None:
        emb_dir = Path("outputs") / "embeddings"
    gallery = load_gallery(emb_dir, use_bank=True)
    
    if person_id not in gallery:
        raise RuntimeError(f"❌ 갤러리에 {person_id}가 없습니다. 먼저 등록해주세요.")
    
    print(f"👤 대상 인물: {person_id}")
    if gallery[person_id].ndim == 2:
        print(f"   Bank 임베딩: {gallery[person_id].shape[0]}개")
    else:
        print(f"   Centroid 임베딩 사용")
    print(f"   매칭 임계값: {match_threshold}")
    
    # InsightFace 초기화
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    
    # 출력 디렉토리
    person_dir = output_dir / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    
    # GIF는 imageio 사용 권장
    if use_imageio and video_path.suffix.lower() in ['.gif', '.gifv']:
        import imageio.v2 as imageio
        frames = imageio.mimread(str(video_path))
        total_frames = len(frames)
        print(f"📹 영상 분석 시작: {video_path.name} (imageio 사용)")
        print(f"   총 프레임 수: {total_frames}")
        print(f"   저장 경로: {person_dir}")
        
        if frame_indices:
            # 범위 체크
            valid_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]
            invalid_indices = [idx for idx in frame_indices if idx < 0 or idx >= total_frames]
            if invalid_indices:
                print(f"   ⚠️ 범위를 벗어난 프레임 번호: {invalid_indices}")
            if valid_indices:
                print(f"   처리할 프레임: {valid_indices}")
            else:
                print(f"   ❌ 유효한 프레임 번호가 없습니다!")
                return
        else:
            valid_indices = list(range(total_frames))
        
        saved_count = 0
        
        for frame_idx in valid_indices:
            frame_rgb = frames[frame_idx]
            # RGB → BGR 변환
            if frame_rgb.ndim == 2:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
            elif frame_rgb.shape[2] == 4:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            faces = app.get(frame)
            print(f"[프레임 {frame_idx}] 감지된 얼굴: {len(faces)}개")
            
            if len(faces) > 0:
                # 가장 큰 얼굴 선택
                faces_sorted = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True
                )
                main_face = faces_sorted[0]
                
                # 얼굴 크기 체크
                face_w = main_face.bbox[2] - main_face.bbox[0]
                face_h = main_face.bbox[3] - main_face.bbox[1]
                
                print(f"  → 가장 큰 얼굴 크기: {face_w:.0f}x{face_h:.0f} (최소: {min_face_size})")
                
                if face_w >= min_face_size and face_h >= min_face_size:
                    # 얼굴 영역 추출 (약간의 여유 공간 추가)
                    x1, y1, x2, y2 = map(int, main_face.bbox)
                    margin = 20
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    # 저장
                    out_path = person_dir / f"{person_id}_f{frame_idx:05d}.jpg"
                    cv2.imwrite(str(out_path), face_img)
                    saved_count += 1
                    
                    print(f"  ✅ 얼굴 추출 완료 → {out_path.name}")
                else:
                    print(f"  ⏭ 스킵: 얼굴 크기가 너무 작음 ({face_w:.0f}x{face_h:.0f} < {min_face_size})")
            else:
                print(f"  ⚠️ 얼굴을 찾지 못함")
        
        print(f"\n✅ 총 {saved_count}개 얼굴 이미지 저장 완료: {person_dir}")
        
    else:
        # cv2.VideoCapture 사용 (MP4 등)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없음: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 영상 분석 시작: {video_path.name}")
        print(f"   총 프레임 수: {total_frames}")
        print(f"   저장 경로: {person_dir}")
        
        saved_count = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 특정 프레임만 처리
            if frame_indices and frame_idx not in frame_indices:
                frame_idx += 1
                continue
            
            faces = app.get(frame)
            print(f"[프레임 {frame_idx}] 감지된 얼굴: {len(faces)}개")
            
            if len(faces) > 0:
                # 각 얼굴을 갤러리와 비교하여 해당 인물인지 확인
                matched_faces = []
                
                for face in faces:
                    face_emb = face.embedding.astype("float32")
                    best_person, best_sim = match_with_bank(face_emb, gallery)
                    
                    # 해당 인물이고 임계값 이상이면 수집 대상
                    if best_person == person_id and best_sim >= match_threshold:
                        face_w = face.bbox[2] - face.bbox[0]
                        face_h = face.bbox[3] - face.bbox[1]
                        
                        if face_w >= min_face_size and face_h >= min_face_size:
                            matched_faces.append({
                                "face": face,
                                "sim": best_sim,
                                "size": (face_w, face_h)
                            })
                
                # 매칭된 얼굴이 있으면 가장 큰 얼굴 선택
                if matched_faces:
                    # 유사도가 높고 크기도 큰 순으로 정렬
                    matched_faces.sort(key=lambda x: (x["sim"], x["size"][0] * x["size"][1]), reverse=True)
                    best_match = matched_faces[0]
                    main_face = best_match["face"]
                    best_sim = best_match["sim"]
                    face_w, face_h = best_match["size"]
                    
                    print(f"  → {person_id} 매칭! sim={best_sim:.3f}, 크기={face_w:.0f}x{face_h:.0f}")
                    
                    # 얼굴 영역 추출 (약간의 여유 공간 추가)
                    x1, y1, x2, y2 = map(int, main_face.bbox)
                    margin = 20
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    # 저장
                    out_path = person_dir / f"{person_id}_f{frame_idx:05d}_sim{best_sim:.2f}.jpg"
                    cv2.imwrite(str(out_path), face_img)
                    saved_count += 1
                    
                    print(f"  ✅ 얼굴 추출 완료 → {out_path.name}")
                else:
                    # 매칭된 얼굴이 없으면 모든 얼굴의 매칭 결과 출력
                    for face in faces:
                        face_emb = face.embedding.astype("float32")
                        best_person, best_sim = match_with_bank(face_emb, gallery)
                        print(f"  → 매칭: {best_person} (sim={best_sim:.3f}) - {person_id} 아님")
            else:
                print(f"  ⚠️ 얼굴을 찾지 못함")
            
            frame_idx += 1
        
        cap.release()
        print(f"\n✅ 총 {saved_count}개 얼굴 이미지 저장 완료: {person_dir}")


def main():
    """사용 예시"""
    # 설정
    video_path = Path("images") / "newjeans_dance.gif"
    output_dir = Path("images") / "extracted_frames"
    person_id = "hani"  # 추출할 사람 ID
    
    # 방법 1: 특정 프레임 번호 지정 (옆얼굴이 보이는 프레임)
    # 예: 프레임 50, 60, 70에서 옆얼굴이 보인다면
    # frame_indices = [50, 60, 70, 80, 90]  # 여기에 옆얼굴 프레임 번호 입력
    
    # 방법 2: None으로 설정하면 모든 프레임에서 얼굴 추출 (더 많은 옵션)
    frame_indices = None  # 모든 프레임 처리
    
    # 또는 특정 범위만 처리
    # frame_indices = list(range(0, 73))  # 0~72 프레임 모두
    
    print("=" * 60)
    print("영상에서 얼굴 프레임 추출 도구")
    print("=" * 60)
    print(f"영상: {video_path}")
    print(f"대상: {person_id}")
    if frame_indices:
        print(f"프레임 번호: {frame_indices}")
    else:
        print(f"프레임 번호: 모든 프레임")
    print()
    
    # 얼굴 추출 실행 (GIF는 imageio 사용)
    extract_faces_from_frames(
        video_path=video_path,
        output_dir=output_dir,
        person_id=person_id,
        frame_indices=frame_indices,
        min_face_size=30,  # 최소 크기를 낮춰서 더 많은 얼굴 수집
        use_imageio=True,  # GIF는 imageio 사용 권장
        match_threshold=0.30,  # 이 값 이상이면 해당 인물로 인식
        emb_dir=Path("outputs") / "embeddings"  # 임베딩 디렉토리
    )
    
    print("\n💡 다음 단계:")
    print(f"1. {output_dir / person_id} 폴더에서 추출된 이미지 확인")
    print("2. 옆얼굴 이미지들을 선택하여 images/enroll/{person_id}/ 폴더로 복사")
    print("3. python src/face_enroll_bank.py 실행하여 bank 업데이트")
    print("   또는 python src/add_embeddings_to_bank.py 실행하여 직접 bank에 추가")


if __name__ == "__main__":
    main()

