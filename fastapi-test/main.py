from __future__ import annotations

# 표준 라이브러리: 이미지 인코딩, 파일 I/O, 경로 처리, 임시 파일 생성 등에 사용
import base64
import io
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

# FastAPI 관련: 웹 프레임워크, 파일 업로드, HTML 응답, 템플릿 엔진
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# PIL: 이미지 처리 및 바운딩 박스 그리기
from PIL import Image, ImageDraw
# RetinaFace: 얼굴 감지 라이브러리
from retinaface import RetinaFace

# 프로젝트 루트 디렉토리 경로 설정
BASE_DIR = Path(__file__).resolve().parent
# HTML 템플릿 파일이 있는 디렉토리 경로
TEMPLATES_DIR = BASE_DIR / "templates"

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="RetinaFace Demo",
    description="Web demo that detects faces on uploaded images with RetinaFace.",
)

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
            "message": "이미지를 업로드해서 얼굴을 감지해 보세요.",
            "faces_found": None,
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
        },
    )
