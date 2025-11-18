import os
from ultralytics import YOLO
import cv2

# --------- 설정 ---------
MODEL_PATH = r"models\yolov12n-face.pt"
INPUT_DIR = r"images"          # 원본 이미지 폴더
OUTPUT_DIR = r"outputs\faces"  # 결과 저장 폴더


# 여기만 바꾸면서 테스트하면 됨
IMAGE_NAME = "korean_crowd.jpg"
# ------------------------

def main():
    # 경로 준비
    image_path = os.path.join(INPUT_DIR, IMAGE_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, IMAGE_NAME)

    # 모델 로드
    model = YOLO(MODEL_PATH)

    # 추론 (YOLO 자체 저장 기능은 끔: save=False)
    results = model(image_path, conf=0.25, save=False, verbose=True)

    # 첫 번째 결과에 그려진 이미지 얻기
    annotated = results[0].plot()

    # 우리가 정한 OUTPUT_DIR에 저장
    cv2.imwrite(save_path, annotated)

    print(f"[완료] {image_path} 처리 → {save_path} 로 저장")

if __name__ == "__main__":
    main()