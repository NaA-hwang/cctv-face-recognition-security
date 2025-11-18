# 🕵️‍♂️ FaceWatch — 실시간 얼굴 식별·추적 시스템 (InsightFace 기반)

FaceWatch는 CCTV/영상/이미지 콘텐츠에서 특정 인물(지명수배자 등)을 자동으로 식별하고 추적하는 Python 기반 AI 시스템입니다.
InsightFace의 고성능 얼굴 인식 모델(buffalo_l)을 기반으로 임베딩 비교, 트래킹, 신뢰도 누적, 스냅샷 저장, 시각화 도구까지 포함하고 있습니다.

본 프로젝트는 다음을 목표로 합니다:

-  정확한 얼굴 인식 및 인물 식별
-  영상(프레임 단위)에서 지속적으로 동일 인물 추적
-  신뢰도 기반 스냅샷 저장(중복 방지)
-  임베딩 분석(히스토그램, heatmap, PCA 등)
-  범죄자 또는 특정 대상자 등록 및 갤러리 관리

## 📂 프로젝트 구조

```
FaceWatch/
├─ images/
│  ├─ enroll/         # 인물 등록 이미지 (인물별 폴더)
│  │   ├─ hani/
│  │   │   └─ *.jpg
│  │   ├─ danielle/
│  │   ├─ minji/
│  │   └─ harin/
│  └─ test/           # 테스트 이미지/영상
│      └─ newjeans_dance.gif
│
├─ outputs/
│  ├─ embeddings/     # 인물별 임베딩 (*.npy, *_centroid.npy, *_bank.npy)
│  ├─ matches/        # 이미지 매칭 결과
│  ├─ matches_multi/   # 멀티 얼굴 매칭 결과
│  ├─ tracks/         # 영상 트래킹 결과 스냅샷
│  ├─ faces/          # 얼굴 추출 결과
│  └─ logs/           # 매칭 로그 (CSV)
│
├─ src/
│  ├─ face_enroll.py              # 단일 이미지 등록
│  ├─ face_enroll_multi.py         # 다중 이미지 등록 (유연한 폴더 구조)
│  ├─ face_enroll_centroid.py     # 사람등록 + centroid 생성
│  ├─ face_enroll_bank.py          # 사람등록 + bank 생성
│  ├─ detect_face_image.py        # YOLO 기반 이미지 얼굴 검출
│  ├─ detect_face_webcam.py        # YOLO 기반 웹캠 실시간 검출
│  ├─ face_match_image_multi.py    # 단일 이미지 인물 식별 (multi-face)
│  ├─ face_match_gif.py            # GIF 파일 매칭
│  ├─ face_match_crowd.py          # 군중 이미지에서 특정 인물 찾기
│  ├─ face_match_video_multi.py   # 영상 + multi-face 매칭
│  ├─ face_match_video_tracks.py  # 영상 tracking + 신뢰도 누적
│  ├─ embedding_distribution_compare.py  # 임베딩 분포 비교
│  ├─ embedding_scatter_3d.py     # 3D scatter plot
│  ├─ embedding_similarity_heatmap.py  # 유사도 heatmap
│  ├─ plot_embeddings_3d.py       # 3D 임베딩 시각화
│  ├─ vis_embedding_heatmap.py    # 임베딩 heatmap 시각화
│  ├─ show_embedding.py           # 단일 임베딩 정보 표시
│  ├─ show_embeddings_gallery.py  # 갤러리 임베딩 통계 표시
│  └─ utils/                      # 유틸리티 모듈
│     ├─ device_config.py         # GPU/CPU 설정
│     └─ gallery_loader.py        # 갤러리 로더
│
└─ README.md
```

## 🚀 주요 기능

### 1. 사람 등록 (Face Enrollment)

여러 방법으로 인물을 등록할 수 있습니다:

#### 1-1. Centroid 기반 등록
여러 장의 얼굴 이미지를 기반으로 임베딩 벡터의 평균값(centroid)을 계산하여 각 인물 별로 안정적인 대표 벡터(*.npy)를 생성합니다.

```bash
python src/face_enroll_centroid.py
```

**출력:**
- `outputs/embeddings/<person>.npy` 생성
- L2-normalized 512-dimensional vector

#### 1-2. Bank 기반 등록
여러 이미지의 임베딩을 모두 저장하여 더 정확한 매칭을 수행합니다.

```bash
python src/face_enroll_bank.py
```

**출력:**
- `outputs/embeddings/<person>_bank.npy` (N×512 배열)
- `outputs/embeddings/<person>_centroid.npy` (512 벡터)

#### 1-3. 단일/다중 이미지 등록
```bash
python src/face_enroll.py          # 단일 이미지
python src/face_enroll_multi.py    # 다중 이미지 (유연한 폴더 구조)
```


### 2. 얼굴 검출 및 식별

#### 2-1. YOLO 기반 얼굴 검출
YOLOv12n-face 모델을 사용한 빠른 얼굴 검출 (인식 없이 검출만)

```bash
python src/detect_face_image.py    # 이미지에서 얼굴 검출
python src/detect_face_webcam.py   # 웹캠 실시간 얼굴 검출
```

**출력:**
- 얼굴 bounding box만 표시 (인물 식별 없음)
- `outputs/faces/` 폴더에 결과 저장

#### 2-2. 이미지 얼굴 식별 (Single Image Recognition)
단일 이미지에서 여러 얼굴을 찾아 갤러리에 등록된 인물들과 similarity 비교 후 가장 유사한 사람을 판단합니다.
**마스크 쓴 얼굴 자동 감지 및 적응형 임계값 적용 기능 포함**

```bash
python src/face_match_image_multi.py
```

**출력:**
- 이미지에 bounding box + 인물명 표시
- similarity score 출력
- 마스크 착용 가능성 추정 및 자동 임계값 조정
- `outputs/matches_multi/` 폴더에 결과 이미지 저장

**마스크 인식 개선:**
- 기존 등록 이미지(마스크 없음)만으로 마스크 쓴 얼굴도 인식 가능
- 유사도가 낮을 때(0.25~0.35) 마스크 가능성으로 판단
- 마스크 가능성이 높으면 임계값을 자동으로 낮춰 인식 (0.30 → 0.22~0.25)
- 오탐 방지를 위해 최소 임계값(0.20) 보장

#### 2-3. GIF 파일 매칭
```bash
python src/face_match_gif.py
```

#### 2-4. 군중 이미지에서 특정 인물 찾기
```bash
python src/face_match_crowd.py
```

### 3. 영상 얼굴 탐지 (Multi-face Matching)

각 프레임마다 얼굴을 감지하고 등록된 갤러리와 cosine similarity를 비교해 가장 유사한 사람을 찾습니다.

```bash
python src/face_match_video_multi.py
```

**출력:**
- 프레임 단위 로깅
- 매칭된 얼굴 bbox/이름 확인
- low-quality, rotating frames 등에서도 robust

### 4. Tracking 기반 "신뢰도 누적" 알고리즘

영상 프레임에서는:
- 같은 사람일 가능성이 있음 (sim ≥ BASE_THRESH)
- IoU가 일정 수준 이상이면 동일 트랙으로 연결
- 일정 길이 + 높은 sim의 트랙만 "확실한 트랙"으로 인정
- 최고 similarity 프레임을 스냅샷으로 저장

```bash
python src/face_match_video_tracks.py
```

**출력:**
- 확정된 인물 스냅샷 (`outputs/tracks/`)
- 트랙 길이, avg sim, max sim
- false positive 최소화한 신뢰도 상승 방식

### 5. 임베딩 분석 도구 (Visualization Toolkit)

사람별 임베딩의 분포가 잘 분리되어 있는지 시각화할 수 있습니다.

**지원:**
- 히스토그램(embedding distribution) - `embedding_distribution_compare.py`
- 유사도 heatmap - `embedding_similarity_heatmap.py`, `vis_embedding_heatmap.py`
- 3D scatter plot - `embedding_scatter_3d.py`, `plot_embeddings_3d.py`
- 임베딩 정보 표시 - `show_embedding.py`, `show_embeddings_gallery.py`

```bash
# 임베딩 분포 비교
python src/embedding_distribution_compare.py

# 유사도 heatmap
python src/embedding_similarity_heatmap.py

# 3D scatter plot
python src/plot_embeddings_3d.py

# 임베딩 통계 확인
python src/show_embedding.py              # 단일 임베딩 정보
python src/show_embeddings_gallery.py     # 갤러리 전체 통계
```

**결과 예시:**
- 멤버 간 임베딩 분포 차이
- hani vs danielle vs minji similarity matrix
- PCA 공간에서의 클러스터링 확인

## ⚙️ 자동 튜닝 기능 (Test / Prod Mode)

영상 추적 스크립트에 자동 설정을 추가함:

```python
MODE = "test"  # 또는 "prod"

if MODE == "test":
    BASE_THRESH = 0.25
    STRONG_THRESH = 0.35
    MIN_TRACK_LEN = 3
else:
    BASE_THRESH = 0.30
    STRONG_THRESH = 0.45
    MIN_TRACK_LEN = 5
```

- **test 모드**: 실험 중 매칭 잘 나오는지 확인용
- **prod 모드**: 실제 CCTV 품질을 가정한 엄격한 설정

## 📌 기술 스택

| 분야 | 기술 |
|------|------|
| Face Recognition | InsightFace (buffalo_l), ONNX Runtime |
| Detection | RetinaFace (InsightFace), YOLOv12n-face |
| Tracking | IoU-based lightweight tracking |
| Embedding Analysis | NumPy, Matplotlib, Seaborn, Scikit-learn |
| Video Handling | OpenCV, imageio |
| Language | Python 3.9+ |

## 💡 FaceWatch의 핵심 알고리즘

### 🔍 1. 임베딩 기반 Similarity Matching

- 얼굴 → 512-d vector (InsightFace)
- L2-normalized
- cosine similarity로 비교
- 등록된 centroid embedding과 매칭

### 🚶 2. IoU 기반 Tracking

- bbox 간 IoU 판단
- 특정 사람의 "움직임"을 하나의 track으로 묶음

### 📈 3. 신뢰도 누적

- 트랙에 포함된 similarity들의 평균/최대값 계산
- length & STRONG_THRESH 기준으로 진짜 사람만 "확정"

### 📷 4. Snapshot 저장

- 최고 similarity 프레임을 저장하여 false positive를 줄임

## 🧪 실험 결과 예시

- ✔ GIF·영상 속 빠르게 움직이는 인물도 정확히 식별
- ✔ 여러 인물이 등장해도 개별 tracking 유지
- ✔ hani 임베딩 max similarity: 0.388
- ✔ track 기반 신뢰도 판단으로 오탐 감소

## 📘 향후 개발 로드맵

- ✓ 업로드된 CCTV 영상에서 실시간 스트리밍 인식
- ✓ 다중 카메라 기반 multi-view tracking
- YOLOv9 기반 fast detector 추가
- Face anti-spoofing (딥페이크 방지)
- Web dashboard (Flask/React)
- GPU 옵션 지원 (CUDA)

## 🧑‍💻 개발자

**Jongwoo Shin**

InsightFace + Computer Vision 기반 얼굴 인식 시스템 개발  
Cloud · AI Engineering · MLOps

## 📄 라이선스

MIT License
