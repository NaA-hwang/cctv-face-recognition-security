# FaceWatch - 얼굴 인식 시스템

CCTV 영상에서 등록된 얼굴을 검출하는 시스템입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 얼굴 등록

`images/enroll/` 폴더 아래에 사람별 폴더를 만들고 정면 사진을 넣으세요.

예:
```
images/enroll/ja/ja.jpeg
images/enroll/js/js.jpeg
images/enroll/jw/jw.jpeg
images/enroll/yh/yh.jpeg
```

등록 실행:
```bash
python src/face_enroll_bank.py
```

### 2. 영상에서 얼굴 검출

`images/cctv.MOV` 파일을 준비하고 실행:
```bash
python src/face_match_video_multi.py
```

## 결과

- `outputs/embeddings/`: 등록된 얼굴 임베딩 파일 (실행 후 생성됨)
- `outputs/matches_multi/`: 매칭된 프레임 스냅샷
  - 샘플 결과 이미지 8개 포함 (각 인물별 대표 이미지)
  - 파일명 형식: `cctv_f{프레임번호}_{인물ID}_{유사도}.jpg`
- `outputs/logs/`: 매칭 로그 (CSV)
  - 샘플 로그 파일 `cctv_matches.csv` 포함 (100줄 샘플)

## 주요 파일

- `src/face_enroll_bank.py`: 얼굴 등록 스크립트
- `src/face_match_video_multi.py`: 영상 얼굴 검출 스크립트
- `src/utils/gallery_loader.py`: 갤러리 로드 및 매칭 유틸리티
- `src/utils/device_config.py`: GPU/CPU 디바이스 설정
