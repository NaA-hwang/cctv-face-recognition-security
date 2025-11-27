from typing import Tuple, Union
# Python의 타입 힌트(Type Hint)를 위해 Tuple과 Union 타입을 가져옵니다. (함수의 입/출력 타입을 명확히 하기 위함)
import math
# 수학 관련 함수 모듈
import cv2
# 이미지 처리 및 시각화에 널리 사용되는 OpenCV 라이브러리
import numpy as np
# 배열 및 행렬 계산을 위한 numpy 라이브러리
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


MARGIN = 10  
# 텍스트 위치 지정 시 여백 값(10픽셀)
ROW_SIZE = 10  
# 텍스트 줄 간격 값(10픽셀)
FONT_SIZE = 1
# 텍스트(레이블)의 글꼴 크기
FONT_THICKNESS = 1
# 텍스트의 글꼴 두께
TEXT_COLOR = (255, 0, 0)  # red
# 바운딩 박스와 텍스트에 사용할 색상 BGR 형식

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
# 매개변수: 정규화된 X, Y 좌표 (float), 이미지의 너비(int), 높이(int) 
# 반환값: 실패 시 None, 성공 시 정수형 튜플(x, y 픽셀 좌표)을 반환
    
# Checks if the float value is between 0 and 1. 
# 정규화된 값이 유효한지(0과 1 사이인지) 확인
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape
# 이미지의 높이(height)와 너비(width)를 가져옴
# _는 채널 수(색상)를 무시한다는 의미

  for detection in detection_result.detections:    
    # detection_result) 내의 각각의 감지된 객체(detection)에 대해 반복
    # Draw bounding_box
        bbox = detection.bounding_box
    # 현재 감지된 객체의 바운딩 박스 정보
        start_point = bbox.origin_x, bbox.origin_y
    # 바운딩 박스의 좌측 상단 시작 픽셀 좌표 설정
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    # 바운딩 박스의 우측 하단 끝 픽셀 좌표를 설정
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 1)
    # OpenCV를 사용하여 복사된 이미지(annotated_image)에 직사각형(바운딩 박스)을 그림
    
    # Draw keypoints(관절점) 그리기
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
    # 키포인트의 정규화된 좌표를 앞서 정의한 함수를 이용해 픽셀 좌표로 변환
            color, thickness, radius = (0, 255, 0), 1, 1
    # 키포인트에 사용할 색상(녹색), 두께(1), 반지름(1)을 정의
            if keypoint_px is not None:
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
    # OpenCV를 사용하여 해당 픽셀 좌표에 원(키포인트)을 그림
        if detection.categories:
    # Draw label and score(레이블(이름)과 확률(점수)을 그리는 부분)
           category = detection.categories[0]
    # 감지된 객체의 가장 높은 점수를 가진 첫 번째 카테고리 정보
           category_name = category.category_name if category.category_name else ''
    # 카테고리 이름이 None이면 빈 문자열로 설정하여 오류를 방지
           probability = round(category.score, 2)
    # 카테고리의 점수(확률)를 가져와 소수점 둘째 자리에서 반올림
           result_text = category_name + ' (' + str(probability) + ')'
    # 최종적으로 바운딩 박스 위에 표시할 텍스트 # OK추가해보기
           text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
    # 텍스트가 바운딩 박스의 좌측 상단 모서리(origin)에 
    # 약간의 여백(MARGIN, ROW_SIZE)을 두고 표시될 위치를 계산
           cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    # OpenCV를 사용하여 계산된 위치에 결과 텍스트 그리기

           return annotated_image


# 모델과 파일 경로 설정
MODEL_PATH = r'C:\dev\NIPA_google_jam_2511\detector.tflite'
IMAGE_FILE = r'C:\dev\NIPA_google_jam_2511\180503.jpg'

# 얼굴 감지 로직
try:
   print(f"모델 로드 중 : {MODEL_PATH}")
   if not os.path.exists(MODEL_PATH):
      raise FileNotFoundError(f'모델파일 없음:{MODEL_PATH}. 파일다운로드 및 경로확인 필요')
# STEP 2: Create an FaceDetector object.모델 기본 설정
   base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
# model_asset_path 인수에 감지에 사용할 모델 파일(detector.tflite)의 경로 지정
   options = vision.FaceDetectorOptions(base_options=base_options)
# FaceDetector에 특화된 옵션 객체 생성(base_options 사용으로 정의함)
   detector = vision.FaceDetector.create_from_options(options)

# 객체로 생성한 detector를 사용하여 이미지에서 얼굴을 찾음

   print(f"이미지 로드 중 : {IMAGE_FILE}")
   if not os.path.exists(IMAGE_FILE):
      raise FileNotFoundError(f'이미지 파일 없음:{IMAGE_FILE}. 파일다운로드 및 경로확인 필요')
   image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect faces in the input image.
   detection_result = detector.detect(image)
   print(f"감지 완료: {len(detection_result.detections)}")
# detector 객체의 detect() 메서드를 사용하여 로드된 image에서 얼굴을 감지


# STEP 5: Process the detection result. In this case, visualize it.
   image_copy = np.copy(image.numpy_view())
# 시각화 작업이 원본 이미지를 변경하지 않도록 원본 mp.Image의 NumPy 뷰를 가져와 그 복사본을 만들어 처리
   annotated_image = visualize(image_copy, detection_result)
   rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
# MediaPipe나 OpenCV는 이미지를 BGR(Blue-Green-Red) 순서로 처리하는 경우가 많으므로, 이를 일반적인 웹/디스플레이 표준인 RGB(Red-Green-Blue) 순서로 색상 채널을 변환
   cv2.imshow('Face Detection Result (Press any key to close)', rgb_annotated_image)
# 변환된 시각화 결과 이미지를 화면에 출력
   cv2.waitKey(0) 
   cv2.destroyAllWindows()
    
except FileNotFoundError as e:
    print(f"\n[오류 발생] {e}")
except Exception as e:
    print(f"\n[예상치 못한 오류] {e}")

