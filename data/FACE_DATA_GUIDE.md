# 팀원 얼굴 데이터 저장 가이드 📸

## 📁 디렉터리 구조

```
data/
├── suspects/
│   ├── images/                 # 원본 얼굴 이미지들
│   │   ├── hong_gildong/       # 홍길동 (절도범)
│   │   │   ├── front_1.jpg     # 정면 사진 1
│   │   │   ├── front_2.jpg     # 정면 사진 2
│   │   │   ├── left_side.jpg   # 왼쪽 측면
│   │   │   ├── right_side.jpg  # 오른쪽 측면
│   │   │   └── slight_angle.jpg # 약간 비스듬한 각도
│   │   ├── kim_cheolsu/        # 김철수 (일반인)
│   │   │   ├── front_1.jpg
│   │   │   ├── front_2.jpg
│   │   │   └── ...
│   │   ├── park_younghee/      # 박영희 (일반인)
│   │   │   └── ...
│   │   └── lee_minsu/          # 이민수 (일반인)
│   │       └── ...
│   ├── metadata/               # 메타데이터 파일들
│   │   └── suspect_profiles.json
│   └── processed/              # 처리된 데이터
│       ├── aligned_faces/      # 정렬된 얼굴
│       └── embeddings.pkl      # 추출된 임베딩
└── videos/                     # CCTV 영상들
    └── test_videos/
```

## 📋 사진 촬영 가이드라인

### 🎯 **각 팀원당 5-10장 권장**

#### **필수 각도 (최소 5장):**
1. **정면 1**: 눈을 똑바로 보고 무표정
2. **정면 2**: 약간 미소를 짓거나 다른 표정
3. **왼쪽 측면**: 45도 각도
4. **오른쪽 측면**: 45도 각도  
5. **약간 위/아래**: 카메라보다 높거나 낮은 각도

#### **추가 권장 (총 10장):**
6. **정면 3**: 다른 조명 조건
7. **왼쪽 측면 2**: 90도 측면
8. **오른쪽 측면 2**: 90도 측면
9. **위에서 내려다본 각도**: CCTV 시각
10. **자연스러운 표정**: 말하는 모습 등

### 📷 **촬영 기술 요구사항**

#### **이미지 품질:**
- **해상도**: 최소 640x480, 권장 1280x720 이상
- **파일 형식**: JPG 또는 PNG
- **파일 크기**: 500KB ~ 5MB
- **얼굴 크기**: 전체 이미지의 30-70% 차지

#### **조명 조건:**
- **자연광**: 창가에서 촬영 (가장 이상적)
- **실내조명**: 충분히 밝고 균등한 조명
- **그림자 최소화**: 얼굴에 그림자가 지지 않도록
- **역광 금지**: 배경이 밝으면 안됨

#### **배경:**
- **단색 배경** 권장 (흰색, 회색, 파란색)
- **복잡한 배경 피하기**
- **사람이나 물체가 배경에 없도록**

### 🏷️ **파일 명명 규칙**

```
{이름}_{각도}_{번호}.jpg

예시:
hong_gildong_front_1.jpg
hong_gildong_front_2.jpg  
hong_gildong_left_45_1.jpg
hong_gildong_right_45_1.jpg
hong_gildong_up_angle_1.jpg
```

### 📝 **메타데이터 파일 (JSON)**

```json
{
  "suspects": [
    {
      "id": "1",
      "name": "홍길동",
      "name_en": "hong_gildong", 
      "age": 42,
      "gender": "male",
      "role": "thief",
      "criminal_record": ["절도", "상해"],
      "risk_level": "high",
      "features": {
        "hair": "검은색",
        "accessories": "모자 착용",
        "distinguishing_marks": "왼쪽 뺨 흉터",
        "clothing": "어두운 색 옷"
      },
      "images": [
        "hong_gildong/front_1.jpg",
        "hong_gildong/front_2.jpg",
        "hong_gildong/left_45_1.jpg",
        "hong_gildong/right_45_1.jpg",
        "hong_gildong/up_angle_1.jpg"
      ]
    },
    {
      "id": "2", 
      "name": "김철수",
      "name_en": "kim_cheolsu",
      "age": 35,
      "gender": "male",
      "role": "civilian",
      "criminal_record": [],
      "risk_level": "low",
      "features": {
        "hair": "검은색 단발",
        "accessories": "안경",
        "clothing": "정장"
      },
      "images": [
        "kim_cheolsu/front_1.jpg",
        "kim_cheolsu/front_2.jpg",
        "kim_cheolsu/left_45_1.jpg",
        "kim_cheolsu/right_45_1.jpg",
        "kim_cheolsu/up_angle_1.jpg"
      ]
    },
    {
      "id": "3",
      "name": "박영희", 
      "name_en": "park_younghee",
      "age": 28,
      "gender": "female",
      "role": "civilian",
      "criminal_record": [],
      "risk_level": "low", 
      "features": {
        "hair": "긴 머리",
        "clothing": "간호사복 또는 밝은 색 옷"
      },
      "images": [
        "park_younghee/front_1.jpg",
        "park_younghee/front_2.jpg",
        "park_younghee/left_45_1.jpg",
        "park_younghee/right_45_1.jpg",
        "park_younghee/up_angle_1.jpg"
      ]
    },
    {
      "id": "4",
      "name": "이민수",
      "name_en": "lee_minsu", 
      "age": 39,
      "gender": "male",
      "role": "civilian",
      "criminal_record": [],
      "risk_level": "low",
      "features": {
        "hair": "검은색 짧은 머리",
        "facial_hair": "수염",
        "clothing": "요리사 모자 또는 캐주얼"
      },
      "images": [
        "lee_minsu/front_1.jpg",
        "lee_minsu/front_2.jpg", 
        "lee_minsu/left_45_1.jpg",
        "lee_minsu/right_45_1.jpg",
        "lee_minsu/up_angle_1.jpg"
      ]
    }
  ]
}
```

## 🔧 **데이터 처리 파이프라인**

### 1단계: 이미지 전처리
```python
# 얼굴 정렬 및 크기 정규화
- 얼굴 검출 (RetinaFace)
- 랜드마크 기반 정렬
- 112x112 크기로 리사이즈
- 히스토그램 평활화
```

### 2단계: 임베딩 추출
```python
# ArcFace 모델을 통한 특징 벡터 생성
- 각 이미지마다 512차원 벡터 추출
- 동일 인물의 여러 임베딩 평균화 (선택사항)
- 정규화 및 최적화
```

### 3단계: 데이터베이스 저장
```python
# SQLite DB에 저장
- 개인정보: suspects 테이블
- 임베딩: embeddings 테이블  
- 이미지 메타데이터: image_metadata 테이블
```

## 📱 **실제 촬영 팁**

### ✅ **해야 할 것:**
- 📱 스마트폰으로도 충분 (최신 폰 권장)
- 🔆 자연광 또는 밝은 실내조명 활용
- 😐 다양한 표정으로 촬영
- 👤 얼굴이 화면의 50% 이상 차지하도록
- 🎯 눈이 선명하게 보이도록 초점 맞추기

### ❌ **피해야 할 것:**
- 🌃 어두운 환경에서 촬영
- 😎 선글라스, 마스크 착용
- 📐 극도로 기울어진 각도
- 👥 다른 사람이 함께 나오는 사진
- 🌫️ 흐릿하거나 움직임이 있는 사진

## 🚀 **빠른 시작 가이드**

1. **디렉터리 생성**: `data/suspects/images/{이름}/` 
2. **사진 촬영**: 최소 5장 (정면 2장, 측면 각각 1장, 특수각도 1장)
3. **파일 저장**: 명명 규칙에 따라 저장
4. **메타데이터 작성**: `suspect_profiles.json` 업데이트
5. **AI 모델 실행**: 자동으로 임베딩 생성 및 DB 저장

이렇게 준비하시면 실제 AI 모델이 팀원들을 정확하게 인식할 수 있습니다! 🎯