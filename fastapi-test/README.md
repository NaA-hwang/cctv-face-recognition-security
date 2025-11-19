# FastAPI RetinaFace Demo

RetinaFace를 활용해 업로드한 이미지에서 얼굴을 감지하고, 감지된 영역을 FastAPI + HTML UI로 보여주는 간단한 데모입니다.

## Tech Stack

- **FastAPI**: 업로드 요청을 받고 RetinaFace 추론을 실행하는 백엔드 프레임워크
- **RetinaFace** (retinaface-py311 가상환경): 얼굴 검출 모델
- **Uvicorn**: FastAPI를 실행하는 ASGI 서버
- **Pillow / numpy**: 업로드된 이미지를 열고 박스를 그리기 위한 도구
- **HTML (Jinja2 템플릿)**: 사용자가 접근할 수 있는 단일 페이지 업로드 UI

## 준비 사항

1. `retina-face-0.0.17.tar.gz`를 다운받습니다.
2. `requirements.txt`의 line 97의 `///Users/yunha_hwa_ng/Document...`에 다운받은 retina-
3. RetinaFace가 설치된 `retinaface-py311` 가상환경을 활성화합니다.
   ```bash
   conda activate retinaface-py311
   # 또는 해당 환경을 활성화하는 명령을 사용하세요.
   ```
4. FastAPI와 웹 서버 실행에 필요한 라이브러리를 설치합니다.
   ```bash
   pip install fastapi uvicorn python-multipart pillow numpy
   ```

   RetinaFace는 이미 가상환경에 설치되어 있다고 가정합니다.

## 실행 방법

1. 프로젝트 디렉터리로 이동합니다.
   ```bash
   cd fastapi-test
   ```
2. FastAPI 앱을 실행합니다.
   ```bash
   uvicorn main:app --reload
   ```
3. 브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000)를 열고, 하나뿐인 업로드 HTML 페이지에 접속합니다.
4. 이미지를 업로드하면 RetinaFace가 얼굴을 감지하고, 결과 이미지를 박스로 표시해줍니다.

## 동작 방식

- `/` (GET): 업로드 폼을 가진 HTML 페이지를 렌더링합니다.
- `/detect` (POST): 사용자가 업로드한 이미지를 임시 파일로 저장한 뒤 `RetinaFace.detect_faces`로 얼굴을 검출합니다.
  - 감지된 영역을 기반으로 Pillow로 이미지에 박스를 그립니다.
  - 변환된 이미지를 Base64로 인코딩해 HTML 페이지에 즉시 표시합니다.

필요에 따라 HTML/CSS를 수정하거나, JSON API 응답을 추가해 다른 서비스와 연동할 수도 있습니다.
