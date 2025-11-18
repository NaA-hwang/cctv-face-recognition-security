"""
BentoML Face Recognition Service
CCTV 용의자 식별 시스템을 위한 AI 서비스

이 서비스는 얼굴 감지와 인식 기능을 제공합니다:
- RetinaFace를 사용한 얼굴 검출
- ArcFace를 사용한 얼굴 특징 추출
- 임베딩 데이터베이스와의 매칭
"""

import bentoml
import numpy as np
import cv2
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import base64
import json
from pathlib import Path

# 백엔드 모델 임포트 (상대 경로)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# 임베딩 데이터베이스 로더
class EmbeddingLoader:
    """실제 임베딩 데이터를 로드하는 클래스"""
    
    def __init__(self):
        # 프로젝트 루트 경로를 더 정확하게 찾기
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        # 여러 가능한 경로 시도
        possible_paths = [
            project_root / "data" / "embeddings",
            Path("c:/Users/PC/Desktop/google_study/data/embeddings"),
            current_dir / ".." / "data" / "embeddings"
        ]
        
        self.embeddings_dir = None
        for path in possible_paths:
            if path.exists():
                self.embeddings_dir = path
                break
        
        if self.embeddings_dir is None:
            print(f"❌ 임베딩 디렉토리를 찾을 수 없음. 시도한 경로들:")
            for path in possible_paths:
                print(f"  - {path}")
            self.embeddings_data = {}
            return
            
        self.embeddings_data = {}
        self.load_embeddings()
    
    def load_embeddings(self):
        """저장된 임베딩 데이터 로드"""
        try:
            embeddings_file = self.embeddings_dir / "all_embeddings.json"
            
            if embeddings_file.exists():
                with open(embeddings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for person in data.get('persons', []):
                    person_id = person['person_id']
                    self.embeddings_data[person_id] = {
                        'name': person['name'],
                        'info': person['info'],
                        'mean_embedding': np.array(person['mean_embedding']),
                        'embeddings': person['embeddings']
                    }
                
                print(f"✅ 임베딩 로드 완료: {len(self.embeddings_data)}명")
            else:
                print(f"⚠️ 임베딩 파일을 찾을 수 없음: {embeddings_file}")
                
        except Exception as e:
            print(f"❌ 임베딩 로드 실패: {e}")
    
    def find_matches(self, query_embedding, threshold=0.6):
        """쿼리 임베딩과 매칭되는 인물 찾기"""
        matches = []
        query_emb = np.array(query_embedding)
        
        for person_id, data in self.embeddings_data.items():
            # 코사인 유사도 계산
            stored_emb = data['mean_embedding']
            similarity = np.dot(query_emb, stored_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(stored_emb)
            )
            
            if similarity > threshold:
                matches.append({
                    'person_id': person_id,
                    'name': data['name'],
                    'similarity': float(similarity),
                    'confidence': min(float(similarity * 100), 99.9)
                })
        
        # 유사도 기준 내림차순 정렬
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches

# 전역 임베딩 로더 인스턴스
embedding_loader = EmbeddingLoader()

try:
    from models.face_detector import FaceDetector
    from models.face_recognizer import FaceRecognizer
    from models.embedding_db import ModernEmbeddingDB
except ImportError:
    # TODO: 실제 구현이 완료되면 이 부분을 제거
    print("⚠️ AI 모델 모듈을 찾을 수 없습니다. 스텁 모드로 실행됩니다.")
    FaceDetector = None
    FaceRecognizer = None
    ModernEmbeddingDB = None

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@bentoml.service(
    name="cctv-face-recognition",
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={"timeout": 60}
)
class CCTVFaceRecognitionService:
    """CCTV 얼굴 인식 서비스"""
    
    def __init__(self):
        """서비스 초기화"""
        logger.info("🚀 CCTV Face Recognition Service 초기화 중...")
        
        # 모델 초기화
        self.face_detector = None
        self.face_recognizer = None
        self.embedding_db = None
        self.is_initialized = False
        
        # 초기화 시도
        try:
            self._initialize_models()
            self.is_initialized = True
            logger.info("✅ 모든 모델이 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {e}")
            logger.info("⚠️ 스텁 모드로 실행됩니다.")
    
    def _initialize_models(self):
        """AI 모델들을 초기화 - 업데이트된 스텁 모드 지원"""
        try:
            # 업데이트된 FaceDetector 사용
            from backend.models.face_detector import create_face_detector_for_bentoml
            
            # 1. 얼굴 검출 모델 초기화
            logger.info("RetinaFace 모델 로딩 중...")
            self.face_detector = create_face_detector_for_bentoml()
            
            # 2. 얼굴 인식 모델 초기화 (TODO: 실제 구현 시 업데이트)
            logger.info("ArcFace 모델 - 스텁 모드로 실행")
            self.face_recognizer = None  # 스텁 모드
            
            # 3. 임베딩 데이터베이스 초기화 (TODO: 실제 구현 시 업데이트)
            logger.info("임베딩 데이터베이스 - 스텁 모드로 실행")
            self.embedding_db = None  # 스텁 모드
            
        except Exception as e:
            logger.error(f"모델 초기화 실패: {e}")
            # 완전한 스텁 모드로 전환
            self.face_detector = None
            self.face_recognizer = None
            self.embedding_db = None
            raise ImportError("AI 모델 모듈을 사용할 수 없습니다.")
        logger.info("임베딩 데이터베이스 연결 중...")
        self.embedding_db = ModernEmbeddingDB()
        
        logger.info("모든 모델 초기화 완료!")
    
    @bentoml.api
    def detect_faces(self, 
                     image_data: str,
                     confidence_threshold: float = 0.8) -> Dict:
        """
        이미지에서 얼굴 감지
        
        Args:
            image_data: Base64 인코딩된 이미지 데이터
            confidence_threshold: 감지 신뢰도 임계값
            
        Returns:
            감지된 얼굴들의 정보
        """
        try:
            start_time = datetime.now()
            
            # Base64 이미지 디코딩
            image = self._decode_base64_image(image_data)
            
            if not self.is_initialized:
                # 스텁 응답
                return self._generate_stub_detection_response(image.shape[:2])
            
            # 스텁 모드 확인
            if self.face_detector is None:
                return self._generate_stub_face_detection()
            
            # 실제 얼굴 검출 수행
            result = self.face_detector.detect_faces_from_base64(
                image_data, 
                confidence_threshold
            )
            
            # 처리 시간 계산
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # 결과 업데이트
            if result.get("success"):
                result["processing_time_ms"] = processing_time
                result["timestamp"] = datetime.now().isoformat()
                return result
            else:
                return {
                    "success": False,
                    "error": result.get("error", "얼굴 검출 실패"),
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"얼굴 감지 중 오류 발생: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @bentoml.api
    def recognize_suspects(self, 
                          image_data: str,
                          detection_threshold: float = 0.8,
                          matching_threshold: float = 0.7) -> Dict:
        """
        이미지에서 용의자 인식 및 매칭
        
        Args:
            image_data: Base64 인코딩된 이미지 데이터
            detection_threshold: 얼굴 감지 임계값
            matching_threshold: 용의자 매칭 임계값
            
        Returns:
            용의자 매칭 결과
        """
        try:
            start_time = datetime.now()
            
            # 스텁 모드 확인
            if not self.is_initialized or self.face_detector is None:
                return self._generate_stub_recognition()
            
            # 실제 얼굴 검출 먼저 수행
            detection_result = self.face_detector.detect_faces_from_base64(
                image_data, 
                detection_threshold
            )
            
            if not detection_result.get("success"):
                return detection_result
            
            detected_faces = detection_result.get("detected_faces", [])
            
            results = []
            for face in detected_faces:
                # 얼굴 임베딩 생성 (시뮬레이션)
                # 실제로는 InsightFace 모델로 생성
                face_embedding = self._generate_face_embedding_from_bbox(image_data, face.get('bbox'))
                
                # 임베딩 매칭
                matches = embedding_loader.find_matches(face_embedding, threshold=matching_threshold)
                
                if matches:
                    # 가장 유사한 매치 사용
                    best_match = matches[0]
                    results.append({
                        "face_bbox": face.get('bbox'),
                        "detection_confidence": face.get('confidence'),
                        "suspect_match": {
                            "suspect_id": best_match['person_id'],
                            "name": best_match['name'],
                            "similarity": best_match['similarity'],
                            "confidence": best_match['confidence'],
                            "is_criminal": best_match['person_id'].startswith('criminal'),
                            "risk_level": "high" if best_match['person_id'].startswith('criminal') else "low",
                            "category": "criminal" if best_match['person_id'].startswith('criminal') else "normal"
                        }
                    })
                else:
                    # 매칭되지 않은 경우
                    results.append({
                        "face_bbox": face.get('bbox'),
                        "detection_confidence": face.get('confidence'),
                        "suspect_match": {
                            "suspect_id": "unknown_person",
                            "name": "알 수 없는 인물", 
                            "similarity": 0.0,
                            "confidence": 0.0,
                            "is_criminal": False,
                            "risk_level": "low",
                            "category": "unknown"
                        }
                    })
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "recognition_results": results,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"용의자 인식 중 오류 발생: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_face_embedding_from_bbox(self, image_data: str, bbox: List[float]) -> np.ndarray:
        """바운딩 박스에서 얼굴 임베딩 생성 (시뮬레이션)"""
        try:
            # 실제로는 얼굴 영역을 잘라서 InsightFace 모델로 임베딩 생성
            # 현재는 시뮬레이션 임베딩 반환
            x, y, w, h = bbox
            
            # bbox 좌표를 기반으로 고유한 시드 생성
            seed = int((x + y + w + h) * 1000) % (2**32)
            np.random.seed(seed)
            
            # 512차원 정규화된 임베딩 생성
            embedding = np.random.normal(0, 1, 512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            # 기본 임베딩 반환
            return np.random.normal(0, 1, 512).astype(np.float32)
    
    @bentoml.api
    def add_suspect(self, 
                   suspect_id: str,
                   name: str,
                   image_data: str,
                   metadata: Optional[Dict] = None) -> Dict:
        """
        새로운 용의자를 데이터베이스에 추가
        
        Args:
            suspect_id: 용의자 ID
            name: 용의자 이름
            image_data: Base64 인코딩된 얼굴 이미지
            metadata: 추가 메타데이터
            
        Returns:
            추가 결과
        """
        try:
            if not self.is_initialized:
                return {
                    "success": False,
                    "error": "AI 모델이 초기화되지 않았습니다.",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 이미지 디코딩
            image = self._decode_base64_image(image_data)
            
            # 얼굴 검출
            faces = self.face_detector.detect(image)
            if not faces:
                return {
                    "success": False,
                    "error": "이미지에서 얼굴을 찾을 수 없습니다.",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 첫 번째 얼굴의 특징 추출
            main_face = faces[0]
            embedding = self.face_recognizer.extract_features(image, main_face['bbox'])
            
            # 데이터베이스에 추가
            result = self.embedding_db.add_suspect(suspect_id, name, embedding, metadata)
            
            return {
                "success": True,
                "suspect_id": suspect_id,
                "embedding_id": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"용의자 추가 중 오류 발생: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @bentoml.api
    def get_service_status(self) -> Dict:
        """서비스 상태 확인"""
        return {
            "service_name": "cctv-face-recognition",
            "version": "1.0.0",
            "status": "ready" if self.is_initialized else "initializing",
            "models": {
                "face_detector": self.face_detector is not None,
                "face_recognizer": self.face_recognizer is not None,
                "embedding_db": self.embedding_db is not None
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_stub_face_detection(self) -> Dict:
        """스텁 모드용 얼굴 검출 결과"""
        return {
            "success": True,
            "detected_faces": [
                {
                    "bbox": [150, 100, 350, 300],
                    "landmarks": [
                        [200, 150], [300, 150], [250, 180], [220, 220], [280, 220]
                    ],
                    "confidence": 0.85
                }
            ],
            "total_faces": 1,
            "image_size": {"width": 640, "height": 480},
            "processing_time_ms": 50,
            "model_info": {
                "model_name": "stub_mode",
                "stub_mode": True,
                "version": "2.0.0-bentoml"
            },
            "note": "스텁 모드로 실행 중 - 실제 AI 모델 없이 더미 데이터 반환"
        }
    
    def _generate_stub_recognition(self) -> Dict:
        """스텁 모드용 용의자 인식 결과"""
        return {
            "success": True,
            "recognition_results": [
                {
                    "face_bbox": [150, 100, 350, 300],
                    "detection_confidence": 0.85,
                    "suspect_match": {
                        "suspect_id": "unknown_person",
                        "name": "알 수 없는 인물",
                        "similarity": 0.0,
                        "is_criminal": False,
                        "risk_level": "low",
                        "category": "normal"
                    }
                }
            ],
            "processing_time_ms": 100,
            "note": "스텁 모드로 실행 중 - 실제 AI 모델 없이 더미 데이터 반환"
        }
    
    def _decode_base64_image(self, image_data: str) -> np.ndarray:
        """Base64 이미지를 OpenCV 형식으로 디코딩"""
        try:
            # data:image/jpeg;base64, 접두사 제거
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Base64 디코딩
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("이미지 디코딩에 실패했습니다.")
            
            return image
            
        except Exception as e:
            raise ValueError(f"이미지 디코딩 오류: {e}")
    
    def _generate_stub_detection_response(self, image_shape: tuple) -> Dict:
        """스텁 모드에서 사용할 얼굴 감지 응답 생성"""
        h, w = image_shape
        return {
            "success": True,
            "detected_faces": [
                {
                    "bbox": [w//4, h//4, 3*w//4, 3*h//4],
                    "confidence": 0.95,
                    "landmarks": []
                }
            ],
            "processing_time_ms": 50,
            "timestamp": datetime.now().isoformat(),
            "note": "스텁 모드 - 실제 AI 모델이 아닙니다."
        }
    
    def _generate_stub_recognition_response(self) -> Dict:
        """스텁 모드에서 사용할 용의자 인식 응답 생성"""
        return {
            "success": True,
            "recognition_results": [
                {
                    "face_bbox": [100, 100, 300, 300],
                    "detection_confidence": 0.95,
                    "suspect_match": {
                        "suspect_id": "demo",
                        "name": "데모 용의자",
                        "similarity": 0.85,
                        "is_criminal": True,
                        "risk_level": "high"
                    },
                    "embedding_extracted": True
                }
            ],
            "processing_time_ms": 120,
            "timestamp": datetime.now().isoformat(),
            "note": "스텁 모드 - 실제 AI 모델이 아닙니다."
        }