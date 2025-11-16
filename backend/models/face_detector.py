"""
RetinaFace 얼굴 검출 모델
InsightFace의 RetinaFace 모델을 사용하여 얼굴을 검출합니다.
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import logging

class FaceDetector:
    """RetinaFace 기반 얼굴 검출 클래스"""
    
    def __init__(self, model_name='buffalo_l', ctx_id=0):
        """
        FaceDetector 초기화
        
        Args:
            model_name (str): 사용할 InsightFace 모델명 ('buffalo_l', 'buffalo_m', 'buffalo_s')
            ctx_id (int): GPU ID (0: GPU, -1: CPU)
        """
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.app = None
        self.detection_size = (640, 640)  # 검출을 위한 이미지 크기
        
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        try:
            print(f"🔧 RetinaFace 모델 로딩 중... (모델: {self.model_name})")
            
            # InsightFace FaceAnalysis 앱 초기화
            self.app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=['detection']  # 검출만 사용
            )
            
            # 모델 준비 (첫 실행시 자동으로 모델 다운로드)
            self.app.prepare(ctx_id=self.ctx_id, det_size=self.detection_size)
            
            print("✅ RetinaFace 모델 로딩 완료")
            
        except Exception as e:
            print(f"❌ RetinaFace 모델 로딩 실패: {str(e)}")
            raise e
    
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
        if self.app is None:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
            
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
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'detection_size': self.detection_size,
            'ctx_id': self.ctx_id,
            'initialized': self.app is not None
        }


# 테스트 코드
if __name__ == "__main__":
    # 테스트용 코드
    detector = FaceDetector()
    
    # 웹캠 테스트 (선택사항)
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #         
    #     faces = detector.detect_faces(frame)
    #     result = detector.draw_detections(frame, faces)
    #     
    #     cv2.imshow('Face Detection', result)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # 
    # cap.release()
    # cv2.destroyAllWindows()
    
    print("✅ FaceDetector 테스트 완료")