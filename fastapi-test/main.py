from __future__ import annotations

# 표준 라이브러리: 이미지 인코딩, 파일 I/O, 경로 처리, 임시 파일 생성 등에 사용
import base64
import io
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

# InsightFace GPU 설정
# ONNX Runtime이 GPU를 사용할 수 있도록 설정
try:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        print(f"✓ GPU 사용 가능 (ONNX Runtime)")
        GPU_AVAILABLE = True
        DEVICE_ID = 0  # GPU 사용
    else:
        print("⚠ GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다.")
        GPU_AVAILABLE = False
        DEVICE_ID = -1  # CPU 사용
except ImportError:
    print("⚠ ONNX Runtime을 임포트할 수 없습니다. CPU 모드로 실행됩니다.")
    GPU_AVAILABLE = False
    DEVICE_ID = -1
except Exception as e:
    print(f"⚠ GPU 확인 중 오류 발생: {e}. CPU 모드로 실행됩니다.")
    GPU_AVAILABLE = False
    DEVICE_ID = -1

# FastAPI 관련: 웹 프레임워크, 파일 업로드, HTML 응답, 템플릿 엔진
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
# PIL: 이미지 처리 및 바운딩 박스 그리기
from PIL import Image, ImageDraw, ImageFont
# InsightFace: 얼굴 감지 및 인식 라이브러리
import insightface
from insightface.app import FaceAnalysis
# OpenCV: 영상 처리 및 프레임 추출
import cv2
import numpy as np

# 프로젝트 루트 디렉토리 경로 설정
BASE_DIR = Path(__file__).resolve().parent
# HTML 템플릿 파일이 있는 디렉토리 경로
TEMPLATES_DIR = BASE_DIR / "templates"
# 예시 파일 경로 설정
EXAMPLE_IMAGE_PATH = BASE_DIR.parent / "data" / "newjeans.jpg"
EXAMPLE_VIDEO_PATH = BASE_DIR.parent / "data" / "video.mp4"

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="InsightFace 얼굴 인식 데모",
    description="InsightFace를 사용하여 특정 얼굴을 이미지/영상에서 찾는 웹 데모.",
)

# InsightFace FaceAnalysis 전역 인스턴스 (애플리케이션 시작 시 초기화)
face_app: Optional[FaceAnalysis] = None


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 InsightFace 모델을 로드합니다."""
    global face_app
    
    print("\n" + "=" * 50)
    print("🚀 FastAPI 애플리케이션 시작")
    print(f"🔧 디바이스: {'GPU' if GPU_AVAILABLE else 'CPU'} (ctx_id={DEVICE_ID})")
    
    try:
        # InsightFace FaceAnalysis 초기화 (buffalo_l 모델 사용)
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=DEVICE_ID, det_size=(640, 640))
        print("✅ InsightFace 모델 'buffalo_l' 로드 완료")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"❌ InsightFace 모델 로드 실패: {e}")
        print("=" * 50 + "\n")
        raise


# Jinja2 템플릿 엔진 초기화 (HTML 템플릿 렌더링용)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 임베딩 벡터 간의 코사인 유사도를 계산합니다.
    
    Args:
        a: 첫 번째 임베딩 벡터
        b: 두 번째 임베딩 벡터
    
    Returns:
        코사인 유사도 (0~1 사이 값, 1에 가까울수록 유사)
    """
    # 정규화
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def _encode_image_with_face_matches(
    image_bytes: bytes, faces: list, target_embedding: np.ndarray, threshold: float = 0.3
) -> str:
    """
    감지된 얼굴에 바운딩 박스를 그려서 base64 인코딩된 문자열로 변환하는 함수.
    찾은 얼굴(매칭된 얼굴)은 빨간색, 일반 얼굴은 초록색으로 표시합니다.
    
    Args:
        image_bytes: 원본 이미지의 바이트 데이터
        faces: InsightFace가 감지한 얼굴 리스트
        target_embedding: 찾을 얼굴의 임베딩 벡터
        threshold: 매칭 임계값 (기본값: 0.3)
    
    Returns:
        base64로 인코딩된 이미지 문자열 (HTML에서 직접 표시 가능)
    """
    # 바이트 데이터를 PIL Image 객체로 변환 (RGB 형식으로 통일)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # 이미지에 그림을 그리기 위한 Draw 객체 생성
    draw = ImageDraw.Draw(image)

    matched_count = 0
    
    # 감지된 각 얼굴에 대해 바운딩 박스와 매칭 정보 표시
    for face in faces:
        # 바운딩 박스 좌표 추출 (x1, y1, x2, y2)
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # 임베딩 비교
        similarity = cosine_similarity(target_embedding, face.embedding)
        is_matched = similarity >= threshold
        
        if is_matched:
            matched_count += 1
            # 찾은 얼굴: 빨간색 바운딩 박스
            color = "#FF0000"
            label = f"Matched! {similarity:.2f}"
        else:
            # 일반 얼굴: 초록색 바운딩 박스
            color = "#00FF00"
            label = f"{similarity:.2f}"
        
        # 바운딩 박스 그리기 (두께 3픽셀)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        
        # 텍스트 배경을 위한 좌표 계산
        label_x = x1 + 4
        label_y = y1 - 25 if y1 > 25 else y1 + 4
        
        # 텍스트 배경 그리기 (가독성 향상)
        try:
            # 텍스트 크기 추정
            bbox_text = draw.textbbox((label_x, label_y), label)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            draw.rectangle(
                [(label_x - 2, label_y - text_height - 2), (label_x + text_width + 2, label_y + 2)],
                fill=color,
            )
            # 텍스트를 흰색으로 표시
            draw.text((label_x, label_y - text_height), label, fill="#FFFFFF")
        except:
            # 폰트 문제 시 간단하게 표시
            draw.text((label_x, label_y), label, fill=color)

    # 이미지를 JPEG 형식으로 메모리 버퍼에 저장
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    # 버퍼의 바이트 데이터를 base64 문자열로 인코딩 (HTML img 태그에서 사용 가능)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded, matched_count


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """
    메인 페이지 엔드포인트.
    사용자가 처음 접속하거나 페이지를 새로고침할 때 호출됩니다.
    빈 상태의 업로드 폼을 보여줍니다.
    """
    # index.html 템플릿을 렌더링하여 반환
    # 초기 상태이므로 결과 이미지와 얼굴 개수는 None으로 설정
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result_image": None,
            "message": "찾을 얼굴 이미지와 대상 이미지/영상을 업로드해서 특정 얼굴을 찾아보세요.",
            "faces_found": None,
            "matched_faces": None,
            "result_video": None,
            "total_frames": None,
            "frames_with_faces": None,
            "frames_with_matches": None,
            "processing_time": None,
        },
    )


@app.post("/detect", response_class=HTMLResponse)
async def detect_faces(
    request: Request,
    target_face: UploadFile = File(..., description="찾을 얼굴 이미지"),
    search_image: UploadFile = File(..., description="여러 얼굴이 포함된 이미지"),
) -> HTMLResponse:
    """
    특정 얼굴 감지 엔드포인트.
    사용자가 업로드한 찾을 얼굴 이미지와 대상 이미지를 받아서 InsightFace로 특정 얼굴을 찾고,
    결과 이미지와 함께 HTML 페이지를 반환합니다.
    """
    global face_app
    
    # 결과 메시지, 인코딩된 이미지, 감지된 얼굴 개수를 저장할 변수 초기화
    message: Optional[str] = None
    result_image: Optional[str] = None
    faces_found: Optional[int] = None
    matched_faces: Optional[int] = None

    if face_app is None:
        message = "InsightFace 모델이 로드되지 않았습니다. 서버를 재시작해주세요."
    else:
        target_path = None
        search_path = None
        try:
            # 찾을 얼굴 이미지 저장
            target_contents = await target_face.read()
            if not target_contents:
                message = "찾을 얼굴 이미지를 찾을 수 없습니다."
            else:
                with NamedTemporaryFile(delete=False, suffix=Path(target_face.filename).suffix) as tmp:
                    tmp.write(target_contents)
                    target_path = tmp.name

                # 대상 이미지 저장
                search_contents = await search_image.read()
                if not search_contents:
                    message = "대상 이미지를 찾을 수 없습니다."
                else:
                    with NamedTemporaryFile(delete=False, suffix=Path(search_image.filename).suffix) as tmp:
                        tmp.write(search_contents)
                        search_path = tmp.name

                    # 찾을 얼굴 이미지에서 얼굴 임베딩 추출
                    target_img = cv2.imread(target_path)
                    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                    target_faces = face_app.get(target_img_rgb)
                    
                    if len(target_faces) == 0:
                        message = "찾을 얼굴 이미지에서 얼굴을 찾을 수 없습니다."
                    else:
                        # 첫 번째 얼굴의 임베딩 사용
                        target_embedding = target_faces[0].embedding
                        
                        # 대상 이미지에서 모든 얼굴 감지
                        search_img = cv2.imread(search_path)
                        search_img_rgb = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
                        search_faces = face_app.get(search_img_rgb)
                        
                        if len(search_faces) == 0:
                            message = "대상 이미지에서 얼굴을 찾을 수 없습니다."
                        else:
                            faces_found = len(search_faces)
                            # 바운딩 박스가 그려진 이미지를 base64로 인코딩
                            result_image, matched_faces = _encode_image_with_face_matches(
                                search_contents, search_faces, target_embedding, threshold=0.3
                            )
                            message = f"총 {faces_found}개의 얼굴 중 {matched_faces}개의 얼굴이 매칭되었습니다."
                            
        except Exception as exc:
            message = f"처리 도중 오류가 발생했습니다: {exc}"
        finally:
            # 임시 파일 정리
            if target_path and os.path.exists(target_path):
                os.remove(target_path)
            if search_path and os.path.exists(search_path):
                os.remove(search_path)

    # 결과를 포함한 HTML 페이지 반환
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result_image": result_image,
            "message": message,
            "faces_found": faces_found,
            "matched_faces": matched_faces,
            "result_video": None,
        },
    )




def _draw_boxes_on_frame(
    frame: np.ndarray, faces: list, target_embedding: np.ndarray, threshold: float = 0.3
) -> tuple[np.ndarray, int]:
    """
    프레임에 감지된 얼굴의 바운딩 박스를 그리는 함수.
    찾은 얼굴(매칭된 얼굴)은 빨간색, 일반 얼굴은 초록색으로 표시합니다.
    
    Args:
        frame: OpenCV로 읽은 프레임 (numpy 배열, BGR 형식)
        faces: InsightFace가 감지한 얼굴 리스트
        target_embedding: 찾을 얼굴의 임베딩 벡터
        threshold: 매칭 임계값 (기본값: 0.3)
    
    Returns:
        (바운딩 박스가 그려진 프레임, 매칭된 얼굴 수) 튜플
    """
    # 프레임 복사 (원본 보존)
    result_frame = frame.copy()
    matched_count = 0
    
    # 감지된 각 얼굴에 대해 바운딩 박스와 매칭 정보 표시
    for face in faces:
        # 바운딩 박스 좌표 추출 (x1, y1, x2, y2)
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # 임베딩 비교
        similarity = cosine_similarity(target_embedding, face.embedding)
        is_matched = similarity >= threshold
        
        if is_matched:
            matched_count += 1
            # 찾은 얼굴: 빨간색 바운딩 박스 (BGR 형식: (0, 0, 255))
            color = (0, 0, 255)
            label = f"매칭! {similarity:.2f}"
        else:
            # 일반 얼굴: 초록색 바운딩 박스 (BGR 형식: (0, 255, 0))
            color = (0, 255, 0)
            label = f"{similarity:.2f}"
        
        # 바운딩 박스 그리기 (두께 3픽셀)
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)
        
        # 텍스트 배경을 위한 좌표 계산
        label_x = x1 + 4
        label_y = y1 - 20 if y1 > 20 else y1 + 20
        
        # 텍스트 배경 그리기 (가독성 향상)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            result_frame,
            (label_x - 2, label_y - text_height - 2),
            (label_x + text_width + 2, label_y + 2),
            color,
            -1,
        )
        # 텍스트를 흰색으로 표시
        cv2.putText(
            result_frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    
    return result_frame, matched_count


def _process_video(
    video_path: str, output_path: str, target_embedding: np.ndarray, threshold: float = 0.3
) -> tuple[int, int, int, float]:
    """
    영상의 각 프레임에 대해 특정 얼굴을 찾고 결과 영상을 생성하는 함수.
    
    Args:
        video_path: 입력 영상 파일 경로
        output_path: 출력 영상 파일 경로
        target_embedding: 찾을 얼굴의 임베딩 벡터
        threshold: 매칭 임계값 (기본값: 0.3)
    
    Returns:
        (총 프레임 수, 감지된 얼굴이 있는 프레임 수, 매칭된 얼굴이 있는 프레임 수, 처리 시간(초)) 튜플
    """
    global face_app
    
    # 영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("영상 파일을 열 수 없습니다.")
    
    # 영상 정보 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 출력 영상 작성자 초기화
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_with_faces = 0
    frames_with_matches = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 프레임을 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # InsightFace를 사용하여 얼굴 감지 수행
            faces = face_app.get(frame_rgb)
            
            if len(faces) > 0:
                frames_with_faces += 1
                # 바운딩 박스 그리기
                frame, matched_count = _draw_boxes_on_frame(frame, faces, target_embedding, threshold)
                if matched_count > 0:
                    frames_with_matches += 1
            
            # 처리된 프레임을 출력 영상에 작성
            out.write(frame)
    
    finally:
        # 리소스 해제
        cap.release()
        out.release()
    
    return total_frames, frames_with_faces, frames_with_matches, frame_count / fps if fps > 0 else 0


@app.post("/detect_video", response_class=HTMLResponse)
async def detect_faces_in_video(
    request: Request,
    target_face: UploadFile = File(..., description="찾을 얼굴 이미지"),
    search_video: UploadFile = File(..., description="여러 얼굴이 포함된 영상"),
) -> HTMLResponse:
    """
    영상에서 특정 얼굴 감지 엔드포인트.
    사용자가 업로드한 찾을 얼굴 이미지와 대상 영상을 받아서 각 프레임에 대해 InsightFace로 특정 얼굴을 찾고,
    결과 영상을 생성하여 다운로드 링크를 제공합니다.
    """
    global face_app
    
    message: Optional[str] = None
    result_video: Optional[str] = None
    total_frames: Optional[int] = None
    frames_with_faces: Optional[int] = None
    frames_with_matches: Optional[int] = None
    processing_time: Optional[float] = None

    if face_app is None:
        message = "InsightFace 모델이 로드되지 않았습니다. 서버를 재시작해주세요."
    else:
        target_path = None
        input_path = None
        output_path = None
        try:
            # 찾을 얼굴 이미지 저장
            target_contents = await target_face.read()
            if not target_contents:
                message = "찾을 얼굴 이미지를 찾을 수 없습니다."
            else:
                with NamedTemporaryFile(delete=False, suffix=Path(target_face.filename).suffix) as tmp:
                    tmp.write(target_contents)
                    target_path = tmp.name

                # 찾을 얼굴 이미지에서 얼굴 임베딩 추출
                target_img = cv2.imread(target_path)
                target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                target_faces = face_app.get(target_img_rgb)
                
                if len(target_faces) == 0:
                    message = "찾을 얼굴 이미지에서 얼굴을 찾을 수 없습니다."
                else:
                    # 첫 번째 얼굴의 임베딩 사용
                    target_embedding = target_faces[0].embedding
                    
                    # 대상 영상 저장
                    video_contents = await search_video.read()
                    if not video_contents:
                        message = "영상 파일을 찾을 수 없습니다."
                    else:
                        input_suffix = Path(search_video.filename).suffix or ".mp4"
                        with NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp:
                            tmp.write(video_contents)
                            input_path = tmp.name
                        
                        # 출력 영상 임시 파일 생성
                        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            output_path = tmp.name
                        
                        # 영상 처리 수행
                        total_frames, frames_with_faces, frames_with_matches, processing_time = _process_video(
                            input_path, output_path, target_embedding, threshold=0.3
                        )
                        
                        # 결과 영상을 base64로 인코딩
                        with open(output_path, "rb") as f:
                            video_bytes = f.read()
                            result_video = base64.b64encode(video_bytes).decode("utf-8")
                        
                        message = (
                            f"처리 완료! 총 {total_frames}개 프레임 중 "
                            f"{frames_with_faces}개 프레임에서 얼굴을 감지했고, "
                            f"{frames_with_matches}개 프레임에서 매칭된 얼굴을 찾았습니다. "
                            f"(처리 시간: {processing_time:.2f}초)"
                        )
                        
        except Exception as exc:
            message = f"영상 처리 도중 오류가 발생했습니다: {exc}"
        finally:
            # 임시 파일 정리
            if target_path and os.path.exists(target_path):
                os.remove(target_path)
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
    
    # 결과를 포함한 HTML 페이지 반환
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result_image": None,
            "message": message,
            "faces_found": None,
            "matched_faces": None,
            "result_video": result_video,
            "total_frames": total_frames,
            "frames_with_faces": frames_with_faces,
            "frames_with_matches": frames_with_matches,
            "processing_time": processing_time,
        },
    )


