"""
업데이트된 FaceDetector 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.models.face_detector import test_face_detector, create_face_detector_for_bentoml
import base64
import numpy as np
import cv2

def test_updated_face_detector():
    """업데이트된 FaceDetector 테스트"""
    print("🧪 업데이트된 FaceDetector 테스트 시작...\n")
    
    # 1. 기본 테스트
    print("1️⃣ 기본 FaceDetector 테스트:")
    detector = test_face_detector()
    print()
    
    # 2. BentoML용 FaceDetector 테스트
    print("2️⃣ BentoML용 FaceDetector 테스트:")
    bento_detector = create_face_detector_for_bentoml()
    print(f"모델 정보: {bento_detector.get_model_info()}")
    print()
    
    # 3. Base64 이미지 처리 테스트
    print("3️⃣ Base64 이미지 처리 테스트:")
    
    # 더미 이미지 생성
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 이미지를 Base64로 인코딩
    _, buffer = cv2.imencode('.jpg', dummy_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Base64 이미지로 얼굴 검출 테스트
    result = bento_detector.detect_faces_from_base64(image_base64, confidence_threshold=0.5)
    
    print(f"Base64 검출 결과:")
    print(f"  - 성공: {result.get('success')}")
    print(f"  - 검출된 얼굴: {result.get('total_faces', 0)}개")
    if result.get('success') and result.get('detected_faces'):
        face = result['detected_faces'][0]
        print(f"  - 첫 번째 얼굴 바운딩박스: {face.get('bbox')}")
        print(f"  - 첫 번째 얼굴 신뢰도: {face.get('confidence')}")
    
    print(f"  - 모델 정보: {result.get('model_info', {}).get('stub_mode')}")
    print()
    
    print("🎉 모든 테스트 완료!")
    return True

if __name__ == "__main__":
    try:
        test_updated_face_detector()
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()