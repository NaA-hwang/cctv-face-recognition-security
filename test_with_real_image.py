"""
실제 이미지 파일로 FaceDetector 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.models.face_detector import create_face_detector_for_bentoml
import base64
import cv2

def test_with_real_image():
    """실제 이미지 파일로 테스트"""
    print("🖼️ 실제 이미지 파일로 테스트 시작...\n")
    
    # 이미지 파일 경로 설정 (예시)
    image_paths = [
        "normal01/person1.jpg",  # 실제 이미지 경로로 변경
        "normal02/person2.jpg",
        "criminal/suspect1.jpg"
    ]
    
    detector = create_face_detector_for_bentoml()
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            print(f"📷 테스트 이미지: {image_path}")
            
            # 이미지 읽기
            img = cv2.imread(image_path)
            if img is not None:
                # Base64로 인코딩
                _, buffer = cv2.imencode('.jpg', img)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 얼굴 검출
                result = detector.detect_faces_from_base64(image_base64)
                
                print(f"  - 검출 성공: {result.get('success')}")
                print(f"  - 검출된 얼굴: {result.get('total_faces', 0)}개")
                print(f"  - 스텁 모드: {result.get('model_info', {}).get('stub_mode')}")
            else:
                print(f"  ❌ 이미지 읽기 실패")
        else:
            print(f"📷 {image_path}: 파일이 존재하지 않음 (건너뜀)")
        print()

if __name__ == "__main__":
    test_with_real_image()