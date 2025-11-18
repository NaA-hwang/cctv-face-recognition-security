"""
Flask 백엔드 API 엔드포인트 테스트
"""

import requests
import base64
import cv2
import numpy as np
import json

def test_api_endpoints():
    """Flask API 엔드포인트들 테스트"""
    print("🌐 Flask API 엔드포인트 테스트 시작...\n")
    
    base_url = "http://localhost:5000"
    
    # 1. 상태 확인 API 테스트
    print("1️⃣ 상태 확인 API 테스트:")
    try:
        response = requests.get(f"{base_url}/api/status")
        print(f"  - 상태코드: {response.status_code}")
        print(f"  - 응답: {response.json()}")
    except Exception as e:
        print(f"  ❌ 오류: {e}")
    print()
    
    # 2. 얼굴 검출 API 테스트
    print("2️⃣ 얼굴 검출 API 테스트:")
    try:
        # 더미 이미지 생성
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # API 요청
        data = {
            "image": f"data:image/jpeg;base64,{image_base64}",
            "threshold": 0.5
        }
        
        response = requests.post(f"{base_url}/api/detect", json=data)
        print(f"  - 상태코드: {response.status_code}")
        result = response.json()
        print(f"  - 검출 성공: {result.get('success')}")
        print(f"  - 검출된 얼굴: {result.get('total_faces', 0)}개")
        print(f"  - 처리시간: {result.get('processing_time', 0):.3f}초")
        
    except Exception as e:
        print(f"  ❌ 오류: {e}")
    print()
    
    # 3. Swagger UI 접근 테스트
    print("3️⃣ Swagger UI 접근 테스트:")
    try:
        response = requests.get(f"{base_url}/apidocs/")
        print(f"  - Swagger UI 상태코드: {response.status_code}")
        if response.status_code == 200:
            print("  ✅ Swagger UI 접근 가능")
        else:
            print("  ❌ Swagger UI 접근 실패")
    except Exception as e:
        print(f"  ❌ 오류: {e}")

if __name__ == "__main__":
    test_api_endpoints()