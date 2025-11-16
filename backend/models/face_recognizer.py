"""
ArcFace 얼굴 인식 모델
InsightFace의 ArcFace 모델을 사용하여 얼굴 특징 벡터를 추출합니다.
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import logging

class FaceRecognizer:
    """ArcFace 기반 얼굴 인식 클래스"""
    
    def __init__(self, model_name='buffalo_l', ctx_id=0):
        """
        FaceRecognizer 초기화
        
        Args:
            model_name (str): 사용할 InsightFace 모델명
            ctx_id (int): GPU ID (0: GPU, -1: CPU)
        """
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.app = None
        self.embedding_dim = 512  # ArcFace 임베딩 차원
        
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        try:
            print(f"🔧 ArcFace 모델 로딩 중... (모델: {self.model_name})")
            
            # InsightFace FaceAnalysis 앱 초기화
            self.app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=['recognition']  # 인식만 사용
            )
            
            # 모델 준비
            self.app.prepare(ctx_id=self.ctx_id)
            
            print("✅ ArcFace 모델 로딩 완료")
            
        except Exception as e:
            print(f"❌ ArcFace 모델 로딩 실패: {str(e)}")
            raise e
    
    def extract_embedding(self, face_image, normalize=True):
        """
        얼굴 이미지에서 특징 벡터(임베딩) 추출
        
        Args:
            face_image (np.ndarray): 얼굴 이미지 (BGR 형식)
            normalize (bool): 벡터 정규화 여부
            
        Returns:
            np.ndarray: 512차원 특징 벡터 또는 None
        """
        if self.app is None:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
        
        try:
            # 이미지 크기 확인
            if face_image.shape[0] < 10 or face_image.shape[1] < 10:
                print("❌ 얼굴 이미지가 너무 작습니다.")
                return None
            
            # RGB 변환
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # 얼굴 검출 및 특징 추출
            faces = self.app.get(rgb_image)
            
            if len(faces) == 0:
                print("⚠️ 얼굴을 찾을 수 없습니다.")
                return None
            
            # 첫 번째 (가장 큰) 얼굴의 임베딩 사용
            face = faces[0]
            embedding = face.normed_embedding if hasattr(face, 'normed_embedding') else face.embedding
            
            # 정규화
            if normalize and not hasattr(face, 'normed_embedding'):
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"❌ 특징 추출 오류: {str(e)}")
            return None
    
    def extract_embeddings_batch(self, face_images, normalize=True):
        """
        여러 얼굴 이미지에서 배치로 특징 벡터 추출
        
        Args:
            face_images (list): 얼굴 이미지 리스트
            normalize (bool): 벡터 정규화 여부
            
        Returns:
            list: 특징 벡터 리스트
        """
        embeddings = []
        
        for face_image in face_images:
            embedding = self.extract_embedding(face_image, normalize)
            embeddings.append(embedding)
        
        return embeddings
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        두 임베딩 벡터 간의 유사도 계산 (코사인 유사도)
        
        Args:
            embedding1 (np.ndarray): 첫 번째 임베딩 벡터
            embedding2 (np.ndarray): 두 번째 임베딩 벡터
            
        Returns:
            float: 유사도 점수 (0~1, 높을수록 유사)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # 벡터 차원 확인
            if len(embedding1.shape) == 1:
                embedding1 = embedding1.reshape(1, -1)
            if len(embedding2.shape) == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
            # 0~1 범위로 정규화 (코사인 유사도는 -1~1 범위)
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            print(f"❌ 유사도 계산 오류: {str(e)}")
            return 0.0
    
    def is_same_person(self, embedding1, embedding2, threshold=0.6):
        """
        두 임베딩이 같은 사람인지 판단
        
        Args:
            embedding1 (np.ndarray): 첫 번째 임베딩
            embedding2 (np.ndarray): 두 번째 임베딩
            threshold (float): 같은 사람 판단 임계값
            
        Returns:
            tuple: (is_same, similarity_score)
        """
        similarity = self.calculate_similarity(embedding1, embedding2)
        is_same = similarity >= threshold
        
        return is_same, similarity
    
    def find_best_match(self, query_embedding, candidate_embeddings, threshold=0.6):
        """
        후보 임베딩들 중에서 가장 유사한 것 찾기
        
        Args:
            query_embedding (np.ndarray): 검색할 임베딩
            candidate_embeddings (list): 후보 임베딩 리스트
            threshold (float): 매칭 임계값
            
        Returns:
            tuple: (best_index, best_similarity) 또는 (None, None)
        """
        if not candidate_embeddings:
            return None, None
        
        best_similarity = 0.0
        best_index = None
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_index = i
        
        return best_index, best_similarity
    
    def preprocess_face_for_recognition(self, face_image, target_size=(112, 112)):
        """
        얼굴 이미지를 인식용으로 전처리
        
        Args:
            face_image (np.ndarray): 얼굴 이미지
            target_size (tuple): 목표 크기 (width, height)
            
        Returns:
            np.ndarray: 전처리된 얼굴 이미지
        """
        try:
            # 크기 조정
            resized = cv2.resize(face_image, target_size)
            
            # 히스토그램 평활화 (조명 정규화)
            if len(resized.shape) == 3:
                # 컬러 이미지인 경우 각 채널별로 적용
                for i in range(3):
                    resized[:, :, i] = cv2.equalizeHist(resized[:, :, i])
            else:
                resized = cv2.equalizeHist(resized)
            
            return resized
            
        except Exception as e:
            print(f"❌ 얼굴 전처리 오류: {str(e)}")
            return face_image
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'ctx_id': self.ctx_id,
            'initialized': self.app is not None
        }


class EmbeddingMatcher:
    """임베딩 매칭을 위한 헬퍼 클래스"""
    
    def __init__(self, face_recognizer):
        self.face_recognizer = face_recognizer
        self.registered_embeddings = {}  # {person_id: embedding}
        
    def register_person(self, person_id, face_image):
        """사람 등록"""
        embedding = self.face_recognizer.extract_embedding(face_image)
        if embedding is not None:
            self.registered_embeddings[person_id] = embedding
            return True
        return False
    
    def identify_person(self, face_image, threshold=0.6):
        """사람 식별"""
        query_embedding = self.face_recognizer.extract_embedding(face_image)
        if query_embedding is None:
            return None, 0.0
        
        best_person_id = None
        best_similarity = 0.0
        
        for person_id, registered_embedding in self.registered_embeddings.items():
            similarity = self.face_recognizer.calculate_similarity(
                query_embedding, registered_embedding
            )
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_person_id = person_id
        
        return best_person_id, best_similarity


# 테스트 코드
if __name__ == "__main__":
    # 테스트용 코드
    recognizer = FaceRecognizer()
    
    print("✅ FaceRecognizer 테스트 완료")
    print(f"모델 정보: {recognizer.get_model_info()}")