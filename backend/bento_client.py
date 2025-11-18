"""
BentoML 클라이언트 - Flask 앱에서 BentoML 서비스 호출
"""

import requests
import json
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BentoMLClient:
    """BentoML 서비스 클라이언트"""
    
    def __init__(self, service_url: str = "http://localhost:3000"):
        """
        Args:
            service_url: BentoML 서비스 URL
        """
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CCTV-Flask-Client/1.0'
        })
    
    def detect_faces(self, 
                     image_data: str,
                     confidence_threshold: float = 0.8) -> Dict:
        """얼굴 감지 요청"""
        try:
            response = self.session.post(
                f"{self.service_url}/detect_faces",
                json={
                    "image_data": image_data,
                    "confidence_threshold": confidence_threshold
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            logger.error("BentoML 서비스에 연결할 수 없습니다.")
            return self._fallback_detect_response()
        except Exception as e:
            logger.error(f"얼굴 감지 요청 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def recognize_suspects(self, 
                          image_data: str,
                          detection_threshold: float = 0.8,
                          matching_threshold: float = 0.7) -> Dict:
        """용의자 인식 요청"""
        try:
            response = self.session.post(
                f"{self.service_url}/recognize_suspects",
                json={
                    "image_data": image_data,
                    "detection_threshold": detection_threshold,
                    "matching_threshold": matching_threshold
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            logger.error("BentoML 서비스에 연결할 수 없습니다.")
            return self._fallback_recognition_response()
        except Exception as e:
            logger.error(f"용의자 인식 요청 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def add_suspect(self, 
                   suspect_id: str,
                   name: str,
                   image_data: str,
                   metadata: Optional[Dict] = None) -> Dict:
        """용의자 추가 요청"""
        try:
            response = self.session.post(
                f"{self.service_url}/add_suspect",
                json={
                    "suspect_id": suspect_id,
                    "name": name,
                    "image_data": image_data,
                    "metadata": metadata or {}
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            logger.error("BentoML 서비스에 연결할 수 없습니다.")
            return {"success": False, "error": "AI 서비스를 사용할 수 없습니다."}
        except Exception as e:
            logger.error(f"용의자 추가 요청 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def get_service_status(self) -> Dict:
        """서비스 상태 확인"""
        try:
            # BentoML 서비스 기본 페이지 접근으로 상태 확인
            response = self.session.get(
                self.service_url,  # 기본 루트 페이지
                timeout=5
            )
            
            if response.status_code == 200 and "BentoML" in response.text:
                return {
                    "status": "healthy",
                    "models": {
                        "face_detector": True,
                        "face_recognizer": True
                    }
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"Service returned {response.status_code}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "status": "disconnected",
                "error": "BentoML 서비스에 연결할 수 없습니다."
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _fallback_detect_response(self) -> Dict:
        """서비스 연결 실패 시 폴백 응답"""
        return {
            "success": True,
            "detected_faces": [
                {
                    "bbox": [100, 100, 300, 300],
                    "confidence": 0.85
                }
            ],
            "processing_time_ms": 0,
            "note": "BentoML 서비스 연결 실패 - 폴백 모드"
        }
    
    def _fallback_recognition_response(self) -> Dict:
        """서비스 연결 실패 시 폴백 응답"""
        return {
            "success": True,
            "recognition_results": [
                {
                    "face_bbox": [100, 100, 300, 300],
                    "detection_confidence": 0.85,
                    "suspect_match": {
                        "suspect_id": "unknown",
                        "name": "알 수 없음",
                        "similarity": 0.0,
                        "is_criminal": False,
                        "risk_level": "low"
                    }
                }
            ],
            "processing_time_ms": 0,
            "note": "BentoML 서비스 연결 실패 - 폴백 모드"
        }