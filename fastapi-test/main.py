from __future__ import annotations

# 표준 라이브러리: 이미지 인코딩, 파일 I/O, 경로 처리, 임시 파일 생성 등에 사용
import base64
import io
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

# TensorFlow GPU 설정 (RetinaFace가 사용하므로 먼저 설정)
# RetinaFace가 TensorFlow를 사용하므로 TensorFlow GPU 설정을 먼저 수행
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow 로그 레벨 설정 (에러만 표시)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # GPU 메모리 동적 증가 허용

# TensorFlow를 임포트하여 GPU 확인 및 설정
try:
    import tensorflow as tf
    
    # GPU 사용 가능 여부 확인
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # GPU가 있는 경우 메모리 증가 허용 설정
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU 감지됨: {len(gpus)}개의 GPU 사용 가능")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
        except RuntimeError as e:
            # GPU 메모리 설정 중 오류 발생 (이미 설정된 경우 등)
            print(f"GPU 메모리 설정 경고: {e}")
        GPU_AVAILABLE = True
    else:
        print("⚠ GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다.")
        GPU_AVAILABLE = False
except ImportError:
    print("⚠ TensorFlow를 임포트할 수 없습니다. GPU 확인을 건너뜁니다.")
    GPU_AVAILABLE = False
except Exception as e:
    print(f"⚠ GPU 확인 중 오류 발생: {e}. CPU 모드로 실행됩니다.")
    GPU_AVAILABLE = False

# FastAPI 관련: 웹 프레임워크, 파일 업로드, HTML 응답, 템플릿 엔진
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
# PIL: 이미지 처리 및 바운딩 박스 그리기
from PIL import Image, ImageDraw
# RetinaFace: 얼굴 감지 라이브러리
from retinaface import RetinaFace
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
    title="RetinaFace Demo",
    description="Web demo that detects faces on uploaded images with RetinaFace.",
)


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 GPU 정보를 출력합니다."""
    if GPU_AVAILABLE:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            print("\n" + "=" * 50)
            print("🚀 FastAPI 애플리케이션 시작")
            print(f"✓ GPU 모드: {len(gpus)}개의 GPU 사용 중")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
            print("=" * 50 + "\n")
        except Exception:
            pass
    else:
        print("\n" + "=" * 50)
        print("🚀 FastAPI 애플리케이션 시작")
        print("⚠ CPU 모드로 실행 중 (GPU 미사용)")
        print("=" * 50 + "\n")


# Jinja2 템플릿 엔진 초기화 (HTML 템플릿 렌더링용)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _encode_image_with_boxes(
    image_bytes: bytes, detections: Dict[str, Dict[str, object]]
) -> str:
    """
    감지된 얼굴에 바운딩 박스를 그려서 base64 인코딩된 문자열로 변환하는 함수.
    
    Args:
        image_bytes: 원본 이미지의 바이트 데이터
        detections: RetinaFace가 감지한 얼굴 정보 딕셔너리
    
    Returns:
        base64로 인코딩된 이미지 문자열 (HTML에서 직접 표시 가능)
    """
    # 바이트 데이터를 PIL Image 객체로 변환 (RGB 형식으로 통일)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # 이미지에 그림을 그리기 위한 Draw 객체 생성
    draw = ImageDraw.Draw(image)

    # 감지된 각 얼굴에 대해 바운딩 박스와 점수 표시
    for face in detections.values():
        # 얼굴 영역 좌표 추출 (x1, y1, x2, y2)
        area = face.get("facial_area") if isinstance(face, dict) else None
        if not area or len(area) != 4:
            continue
        x1, y1, x2, y2 = area
        # 초록색 바운딩 박스 그리기 (두께 3픽셀)
        draw.rectangle([(x1, y1), (x2, y2)], outline="#00FF00", width=3)
        # 얼굴 감지 신뢰도 점수 추출
        score = face.get("score") if isinstance(face, dict) else None
        if score:
            # 점수를 소수점 둘째 자리까지 표시
            label = f"{float(score):.2f}"
            label_x = x1 + 4  # 박스 왼쪽 상단에서 약간 오른쪽으로
            label_y = y1 + 4  # 박스 왼쪽 상단에서 약간 아래로
            # 점수 텍스트를 초록색으로 표시
            draw.text((label_x, label_y), label, fill="#00FF00")

    # 이미지를 JPEG 형식으로 메모리 버퍼에 저장
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    # 버퍼의 바이트 데이터를 base64 문자열로 인코딩 (HTML img 태그에서 사용 가능)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


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
            "message": "이미지 또는 영상을 업로드해서 얼굴을 감지해 보세요.",
            "faces_found": None,
            "result_video": None,
            "total_frames": None,
            "frames_with_faces": None,
            "processing_time": None,
        },
    )


@app.post("/detect", response_class=HTMLResponse)
async def detect_faces(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    """
    얼굴 감지 엔드포인트.
    사용자가 업로드한 이미지 파일을 받아서 RetinaFace로 얼굴을 감지하고,
    결과 이미지와 함께 HTML 페이지를 반환합니다.
    """
    # 결과 메시지, 인코딩된 이미지, 감지된 얼굴 개수를 저장할 변수 초기화
    message: Optional[str] = None
    result_image: Optional[str] = None
    faces_found: Optional[int] = None

    # 업로드된 파일의 바이트 데이터 읽기
    contents = await file.read()
    if not contents:
        # 파일이 비어있으면 에러 메시지 설정
        message = "이미지를 찾을 수 없습니다. 다시 시도해 주세요."
    else:
        temp_path = None
        try:
            # RetinaFace는 파일 경로를 필요로 하므로 임시 파일 생성
            # 원본 파일 확장자를 유지하여 올바른 형식으로 처리
            with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                tmp.write(contents)
                temp_path = tmp.name

            # RetinaFace를 사용하여 얼굴 감지 수행
            detections = RetinaFace.detect_faces(temp_path)
            if isinstance(detections, dict) and detections:
                # 얼굴이 감지된 경우
                faces_found = len(detections)
                # 바운딩 박스가 그려진 이미지를 base64로 인코딩
                result_image = _encode_image_with_boxes(contents, detections)
                message = f"총 {faces_found}개의 얼굴을 찾았습니다."
            else:
                # 얼굴이 감지되지 않은 경우
                message = "얼굴을 찾지 못했습니다. 다른 이미지를 사용해 주세요."
        except Exception as exc:  # pragma: no cover - runtime safeguard
            # 예외 발생 시 에러 메시지 설정
            message = f"감지 도중 오류가 발생했습니다: {exc}"
        finally:
            # 임시 파일 정리 (메모리 누수 방지)
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # 결과를 포함한 HTML 페이지 반환
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result_image": result_image,
            "message": message,
            "faces_found": faces_found,
            "result_video": None,
        },
    )


@app.get("/detect_example_image", response_class=HTMLResponse)
async def detect_example_image(request: Request) -> HTMLResponse:
    """
    예시 이미지를 사용한 얼굴 감지 엔드포인트.
    data/newjeans.jpg 파일을 사용하여 얼굴 감지를 수행합니다.
    """
    # 결과 메시지, 인코딩된 이미지, 감지된 얼굴 개수를 저장할 변수 초기화
    message: Optional[str] = None
    result_image: Optional[str] = None
    faces_found: Optional[int] = None

    # 예시 이미지 파일 존재 여부 확인
    if not EXAMPLE_IMAGE_PATH.exists():
        message = f"예시 이미지 파일을 찾을 수 없습니다: {EXAMPLE_IMAGE_PATH}"
    else:
        try:
            # 예시 이미지 파일 읽기
            with open(EXAMPLE_IMAGE_PATH, "rb") as f:
                contents = f.read()

            # RetinaFace를 사용하여 얼굴 감지 수행
            detections = RetinaFace.detect_faces(str(EXAMPLE_IMAGE_PATH))
            if isinstance(detections, dict) and detections:
                # 얼굴이 감지된 경우
                faces_found = len(detections)
                # 바운딩 박스가 그려진 이미지를 base64로 인코딩
                result_image = _encode_image_with_boxes(contents, detections)
                message = f"예시 이미지에서 총 {faces_found}개의 얼굴을 찾았습니다."
            else:
                # 얼굴이 감지되지 않은 경우
                message = "예시 이미지에서 얼굴을 찾지 못했습니다."
        except Exception as exc:
            # 예외 발생 시 에러 메시지 설정
            message = f"예시 이미지 처리 도중 오류가 발생했습니다: {exc}"

    # 결과를 포함한 HTML 페이지 반환
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result_image": result_image,
            "message": message,
            "faces_found": faces_found,
            "result_video": None,
        },
    )


def _draw_boxes_on_frame(
    frame: np.ndarray, detections: Dict[str, Dict[str, object]]
) -> np.ndarray:
    """
    프레임에 감지된 얼굴의 바운딩 박스를 그리는 함수.
    
    Args:
        frame: OpenCV로 읽은 프레임 (numpy 배열, BGR 형식)
        detections: RetinaFace가 감지한 얼굴 정보 딕셔너리
    
    Returns:
        바운딩 박스가 그려진 프레임
    """
    # 프레임 복사 (원본 보존)
    result_frame = frame.copy()
    
    # 감지된 각 얼굴에 대해 바운딩 박스와 점수 표시
    for face in detections.values():
        # 얼굴 영역 좌표 추출 (x1, y1, x2, y2)
        area = face.get("facial_area") if isinstance(face, dict) else None
        if not area or len(area) != 4:
            continue
        x1, y1, x2, y2 = map(int, area)
        
        # 초록색 바운딩 박스 그리기 (BGR 형식: (0, 255, 0))
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 얼굴 감지 신뢰도 점수 추출
        score = face.get("score") if isinstance(face, dict) else None
        if score:
            # 점수를 소수점 둘째 자리까지 표시
            label = f"{float(score):.2f}"
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
                (0, 255, 0),
                -1,
            )
            # 점수 텍스트를 검은색으로 표시
            cv2.putText(
                result_frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
    
    return result_frame


def _process_video(
    video_path: str, output_path: str
) -> tuple[int, int, float]:
    """
    영상의 각 프레임에 대해 얼굴 감지를 수행하고 결과 영상을 생성하는 함수.
    
    Args:
        video_path: 입력 영상 파일 경로
        output_path: 출력 영상 파일 경로
    
    Returns:
        (총 프레임 수, 감지된 얼굴이 있는 프레임 수, 처리 시간(초)) 튜플
    """
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
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 임시 파일에 프레임 저장 (RetinaFace는 파일 경로를 필요로 함)
            temp_frame_path = None
            try:
                with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    temp_frame_path = tmp.name
                    cv2.imwrite(temp_frame_path, frame)
                
                # RetinaFace를 사용하여 얼굴 감지 수행
                detections = RetinaFace.detect_faces(temp_frame_path)
                
                if isinstance(detections, dict) and detections:
                    # 얼굴이 감지된 경우 바운딩 박스 그리기
                    frame = _draw_boxes_on_frame(frame, detections)
                    frames_with_faces += 1
                
                # 처리된 프레임을 출력 영상에 작성
                out.write(frame)
                
            finally:
                # 임시 프레임 파일 정리
                if temp_frame_path and os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
    
    finally:
        # 리소스 해제
        cap.release()
        out.release()
    
    return total_frames, frames_with_faces, frame_count / fps if fps > 0 else 0


@app.post("/detect_video", response_class=HTMLResponse)
async def detect_faces_in_video(
    request: Request, file: UploadFile = File(...)
) -> HTMLResponse:
    """
    영상 얼굴 감지 엔드포인트.
    사용자가 업로드한 영상 파일을 받아서 각 프레임에 대해 RetinaFace로 얼굴을 감지하고,
    결과 영상을 생성하여 다운로드 링크를 제공합니다.
    """
    message: Optional[str] = None
    result_video: Optional[str] = None
    total_frames: Optional[int] = None
    frames_with_faces: Optional[int] = None
    processing_time: Optional[float] = None
    
    # 업로드된 파일의 바이트 데이터 읽기
    contents = await file.read()
    if not contents:
        message = "영상 파일을 찾을 수 없습니다. 다시 시도해 주세요."
    else:
        input_path = None
        output_path = None
        try:
            # 입력 영상 임시 파일 생성
            input_suffix = Path(file.filename).suffix or ".mp4"
            with NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp:
                tmp.write(contents)
                input_path = tmp.name
            
            # 출력 영상 임시 파일 생성
            with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                output_path = tmp.name
            
            # 영상 처리 수행
            total_frames, frames_with_faces, processing_time = _process_video(
                input_path, output_path
            )
            
            # 결과 영상을 base64로 인코딩
            with open(output_path, "rb") as f:
                video_bytes = f.read()
                result_video = base64.b64encode(video_bytes).decode("utf-8")
            
            message = (
                f"처리 완료! 총 {total_frames}개 프레임 중 "
                f"{frames_with_faces}개 프레임에서 얼굴을 감지했습니다. "
                f"(처리 시간: {processing_time:.2f}초)"
            )
            
        except Exception as exc:
            message = f"영상 처리 도중 오류가 발생했습니다: {exc}"
        finally:
            # 임시 파일 정리
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
            "result_video": result_video,
            "total_frames": total_frames,
            "frames_with_faces": frames_with_faces,
            "processing_time": processing_time,
        },
    )


@app.get("/detect_example_video", response_class=HTMLResponse)
async def detect_example_video(request: Request) -> HTMLResponse:
    """
    예시 영상을 사용한 얼굴 감지 엔드포인트.
    data/video.mp4 파일을 사용하여 각 프레임에 대해 얼굴 감지를 수행합니다.
    """
    message: Optional[str] = None
    result_video: Optional[str] = None
    total_frames: Optional[int] = None
    frames_with_faces: Optional[int] = None
    processing_time: Optional[float] = None

    # 예시 영상 파일 존재 여부 확인
    if not EXAMPLE_VIDEO_PATH.exists():
        message = f"예시 영상 파일을 찾을 수 없습니다: {EXAMPLE_VIDEO_PATH}"
    else:
        output_path = None
        try:
            # 출력 영상 임시 파일 생성
            with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                output_path = tmp.name

            # 영상 처리 수행
            total_frames, frames_with_faces, processing_time = _process_video(
                str(EXAMPLE_VIDEO_PATH), output_path
            )

            # 결과 영상을 base64로 인코딩
            with open(output_path, "rb") as f:
                video_bytes = f.read()
                result_video = base64.b64encode(video_bytes).decode("utf-8")

            message = (
                f"예시 영상 처리 완료! 총 {total_frames}개 프레임 중 "
                f"{frames_with_faces}개 프레임에서 얼굴을 감지했습니다. "
                f"(처리 시간: {processing_time:.2f}초)"
            )

        except Exception as exc:
            message = f"예시 영상 처리 도중 오류가 발생했습니다: {exc}"
        finally:
            # 임시 파일 정리
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
            "result_video": result_video,
            "total_frames": total_frames,
            "frames_with_faces": frames_with_faces,
            "processing_time": processing_time,
        },
    )
