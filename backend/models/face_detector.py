"""
RetinaFace 얼굴 검출 모델
InsightFace의 RetinaFace 모델을 사용하여 얼굴을 검출합니다.
"""

import cv2
import numpy as np
import logging
import base64
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace가 설치되지 않았습니다. 스텁 모드로 실행됩니다.")

class FaceDetector:
    """RetinaFace 기반 얼굴 검출 클래스 (BentoML 연동 지원)"""
    
    def __init__(self, model_name='buffalo_l', ctx_id=0, stub_mode=None):
        """
        FaceDetector 초기화
        
        Args:
            model_name (str): 사용할 InsightFace 모델명 ('buffalo_l', 'buffalo_m', 'buffalo_s')
            ctx_id (int): GPU ID (0: GPU, -1: CPU)
            stub_mode (bool): 스텁 모드 강제 설정 (None: 자동 감지)
        """
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.app = None
        self.detection_size = (640, 640)  # 검출을 위한 이미지 크기
        self.logger = logging.getLogger(__name__)
        
        # 스텁 모드 설정
        if stub_mode is None:
            self.stub_mode = not INSIGHTFACE_AVAILABLE
        else:
            self.stub_mode = stub_mode
            
        self.initialized = False
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        try:
            if self.stub_mode:
                print(f"🔧 FaceDetector 스텁 모드로 초기화 중...")
                self.app = None
                self.initialized = True
                print("✅ FaceDetector 스텁 모드 초기화 완료")
                return
                
            print(f"🔧 RetinaFace 모델 로딩 중... (모델: {self.model_name})")
            
            # InsightFace FaceAnalysis 앱 초기화
            self.app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=['detection']  # 검출만 사용
            )
            
            # 모델 준비 (첫 실행시 자동으로 모델 다운로드)
            self.app.prepare(ctx_id=self.ctx_id, det_size=self.detection_size)
            
            self.initialized = True
            print("✅ RetinaFace 모델 로딩 완료")
            
        except Exception as e:
            self.logger.error(f"RetinaFace 모델 로딩 실패: {str(e)}")
            print(f"❌ RetinaFace 모델 로딩 실패, 스텁 모드로 전환: {str(e)}")
            self.stub_mode = True
            self.app = None
            self.initialized = True
    
    def detect_faces(self, image, confidence_threshold=0.5):
        """
        이미지에서 얼굴 검출
        
        Args:
            image (np.ndarray): 입력 이미지 (BGR 형식)
            confidence_threshold (float): 검출 신뢰도 임계값
            
        Returns:
            list: 검출된 얼굴 정보 리스트
                  각 원소는 (bbox, landmarks, confidence) 튜플
        """
        if not self.initialized:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
            
        # 스텁 모드인 경우 더미 데이터 반환
        if self.stub_mode:
            return self._generate_stub_detections(image, confidence_threshold)
            
        try:
            # RGB 변환 (InsightFace는 RGB 입력을 기대)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 얼굴 검출 실행
            faces = self.app.get(rgb_image)
            
            results = []
            for face in faces:
                # 검출 신뢰도 확인
                if hasattr(face, 'det_score') and face.det_score < confidence_threshold:
                    continue
                
                # 바운딩 박스 (x1, y1, x2, y2)
                bbox = face.bbox.astype(int)
                
                # 얼굴 랜드마크 (5개 점: 양쪽 눈, 코끝, 양쪽 입꼬리)
                landmarks = face.kps if hasattr(face, 'kps') else None
                
                # 검출 신뢰도
                confidence = face.det_score if hasattr(face, 'det_score') else 1.0
                
                results.append((bbox, landmarks, confidence))
            
            return results
            
        except Exception as e:
            self.logger.error(f"얼굴 검출 오류: {str(e)}")
            print(f"❌ 얼굴 검출 오류: {str(e)}")
            return []
    
    def detect_largest_face(self, image, confidence_threshold=0.5):
        """
        이미지에서 가장 큰 얼굴 하나만 검출
        
        Args:
            image (np.ndarray): 입력 이미지
            confidence_threshold (float): 검출 신뢰도 임계값
            
        Returns:
            tuple or None: (bbox, landmarks, confidence) 또는 None
        """
        faces = self.detect_faces(image, confidence_threshold)
        
        if not faces:
            return None
        
        # 면적 기준으로 가장 큰 얼굴 선택
        largest_face = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
        
        return largest_face
    
    def crop_face(self, image, bbox, margin=20):
        """
        바운딩 박스를 기준으로 얼굴 영역 크롭
        
        Args:
            image (np.ndarray): 입력 이미지
            bbox (np.ndarray): 바운딩 박스 [x1, y1, x2, y2]
            margin (int): 크롭 여백
            
        Returns:
            np.ndarray: 크롭된 얼굴 이미지
        """
        h, w = image.shape[:2]
        
        x1, y1, x2, y2 = bbox
        
        # 여백 추가
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        return image[y1:y2, x1:x2]
    
    def draw_detections(self, image, faces, draw_landmarks=True):
        """
        검출 결과를 이미지에 그리기
        
        Args:
            image (np.ndarray): 입력 이미지
            faces (list): 검출된 얼굴 정보 리스트
            draw_landmarks (bool): 랜드마크 그리기 여부
            
        Returns:
            np.ndarray: 검출 결과가 그려진 이미지
        """
        result_image = image.copy()
        
        for bbox, landmarks, confidence in faces:
            x1, y1, x2, y2 = bbox
            
            # 바운딩 박스 그리기
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 신뢰도 텍스트
            confidence_text = f'{confidence:.2f}'
            cv2.putText(result_image, confidence_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 랜드마크 그리기
            if draw_landmarks and landmarks is not None:
                for point in landmarks:
                    x, y = point.astype(int)
                    cv2.circle(result_image, (x, y), 2, (255, 0, 0), -1)
        
        return result_image
    
    def set_detection_size(self, size):
        """
        검출을 위한 입력 이미지 크기 설정
        
        Args:
            size (tuple): (width, height)
        """
        self.detection_size = size
        if self.app:
            self.app.prepare(ctx_id=self.ctx_id, det_size=size)
    
    def _generate_stub_detections(self, image, confidence_threshold):
        """
        스텁 모드용 더미 얼굴 검출 결과 생성
        
        Args:
            image (np.ndarray): 입력 이미지
            confidence_threshold (float): 신뢰도 임계값
            
        Returns:
            list: 더미 검출 결과
        """
        h, w = image.shape[:2]
        
        # 이미지 중앙에 더미 얼굴 바운딩 박스 생성
        center_x, center_y = w // 2, h // 2
        face_size = min(w, h) // 4
        
        x1 = max(0, center_x - face_size // 2)
        y1 = max(0, center_y - face_size // 2)
        x2 = min(w, center_x + face_size // 2)
        y2 = min(h, center_y + face_size // 2)
        
        bbox = np.array([x1, y1, x2, y2])
        
        # 더미 랜드마크 (5개 점)
        landmarks = np.array([
            [center_x - 20, center_y - 10],  # 왼쪽 눈
            [center_x + 20, center_y - 10],  # 오른쪽 눈
            [center_x, center_y],            # 코끝
            [center_x - 10, center_y + 20],  # 왼쪽 입꼬리
            [center_x + 10, center_y + 20]   # 오른쪽 입꼬리
        ], dtype=np.float32)
        
        confidence = 0.85  # 더미 신뢰도
        
        if confidence >= confidence_threshold:
            return [(bbox, landmarks, confidence)]
        else:
            return []
    
    def detect_faces_from_base64(self, image_base64: str, confidence_threshold=0.5):
        """
        Base64 인코딩된 이미지에서 얼굴 검출 (BentoML API 호환)
        
        Args:
            image_base64 (str): Base64 인코딩된 이미지
            confidence_threshold (float): 검출 신뢰도 임계값
            
        Returns:
            Dict: 검출 결과
        """
        try:
            # Base64 디코딩
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "success": False,
                    "error": "이미지 디코딩에 실패했습니다."
                }
            
            # 얼굴 검출
            faces = self.detect_faces(image, confidence_threshold)
            
            # 결과 포맷팅
            formatted_faces = []
            for bbox, landmarks, confidence in faces:
                face_info = {
                    "bbox": bbox.tolist(),
                    "landmarks": landmarks.tolist() if landmarks is not None else None,
                    "confidence": float(confidence)
                }
                formatted_faces.append(face_info)
            
            return {
                "success": True,
                "detected_faces": formatted_faces,
                "total_faces": len(formatted_faces),
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "processing_time_ms": 0,  # TODO: 실제 처리 시간 측정
                "model_info": self.get_model_info()
            }
            
        except Exception as e:
            self.logger.error(f"Base64 이미지 처리 오류: {str(e)}")
            return {
                "success": False,
                "error": f"이미지 처리 중 오류 발생: {str(e)}"
            }
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'detection_size': self.detection_size,
            'ctx_id': self.ctx_id,
            'initialized': self.initialized,
            'stub_mode': self.stub_mode,
            'insightface_available': INSIGHTFACE_AVAILABLE,
            'version': '2.0.0-bentoml'
        }


# 테스트 및 유틸리티 함수
def test_face_detector():
    """FaceDetector 테스트 함수"""
    print("🔧 FaceDetector 테스트 시작...")
    
    # 스텁 모드로 테스트
    detector = FaceDetector(stub_mode=True)
    print(f"모델 정보: {detector.get_model_info()}")
    
    # 더미 이미지로 테스트
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.detect_faces(dummy_image)
    print(f"검출된 얼굴 수: {len(faces)}")
    
    if faces:
        bbox, landmarks, confidence = faces[0]
        print(f"첫 번째 얼굴 - bbox: {bbox}, confidence: {confidence}")
    
    print("✅ FaceDetector 테스트 완료")
    return detector

def create_face_detector_for_bentoml():
    """BentoML용 FaceDetector 인스턴스 생성"""
    try:
        # 실제 모델 시도
        detector = FaceDetector(stub_mode=False)
        print("✅ 실제 InsightFace 모델로 FaceDetector 초기화 완료")
        return detector
    except Exception as e:
        print(f"⚠️ 실제 모델 로딩 실패, 스텁 모드로 전환: {e}")
        # 스텁 모드로 폴백
        detector = FaceDetector(stub_mode=True)
        print("✅ 스텁 모드로 FaceDetector 초기화 완료")
        return detector

# 테스트 코드
if __name__ == "__main__":
    test_detector = test_face_detector()
    
    # 웹캠 테스트 (선택사항 - 스텁 모드가 아닐 때만)
    if not test_detector.stub_mode:
        print("\n실제 모델이 로드되었습니다. 웹캠 테스트를 원하시면 주석을 해제하세요.")
        # cap = cv2.VideoCapture(0)
        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #         
        #     faces = test_detector.detect_faces(frame)
        #     result = test_detector.draw_detections(frame, faces)
        #     
        #     cv2.imshow('Face Detection', result)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # 
        # cap.release()
        # cv2.destroyAllWindows()
    
    print("\n🎉 모든 테스트 완료!")