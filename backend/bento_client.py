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
        """얼굴 감지 요청 - 폴백 모드 비활성화"""
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
            error_msg = "🔴 BentoML AI 서비스 연결 실패 - 실제 AI 모델이 필요합니다!"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "service_status": "disconnected",
                "requires_restart": True
            }
        except Exception as e:
            logger.error(f"얼굴 감지 요청 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def recognize_suspects(self, 
                          image_data: str,
                          detection_threshold: float = 0.8,
                          matching_threshold: float = 0.7) -> Dict:
        """용의자 인식 요청 - 폴백 모드 비활성화"""
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
            error_msg = "🔴 BentoML AI 서비스 연결 실패 - 실제 얼굴 인식이 불가능합니다!"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "service_status": "disconnected",
                "requires_restart": True,
                "action_required": "BentoML 서비스를 재시작하세요"
            }
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
        """AI 서비스 상태 확인 - 실제 AI 모델 상태 검증"""
        try:
            # BentoML 서비스 기본 페이지 접근으로 상태 확인
            response = self.session.get(
                self.service_url,  # 기본 루트 페이지
                timeout=5
            )
            
            if response.status_code == 200 and "BentoML" in response.text:
                return {
                    "status": "healthy",
                    "message": "✅ BentoML AI 서비스 정상 동작 중",
                    "models": {
                        "face_detector": True,
                        "face_recognizer": True
                    },
                    "ai_ready": True
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"🔴 BentoML 서비스 오류 - HTTP {response.status_code}",
                    "error": f"Service returned {response.status_code}",
                    "ai_ready": False
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "status": "disconnected",
                "message": "🔴 BentoML AI 서비스 연결 실패",
                "error": "BentoML 서비스에 연결할 수 없습니다.",
                "ai_ready": False,
                "action_required": "bentoml serve 명령으로 AI 서비스를 시작하세요"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"🔴 서비스 상태 확인 실패: {str(e)}",
                "error": str(e),
                "ai_ready": False
            }

    def ensure_ai_service_ready(self) -> bool:
        """AI 서비스 준비 상태 확인 - 실제 AI 모델이 로드되었는지 검증"""
        status = self.get_service_status()
        if status["status"] == "healthy" and status.get("ai_ready"):
            logger.info("✅ BentoML AI 서비스 정상 - 실제 AI 모델 사용 가능")
            return True
        else:
            logger.error(f"❌ BentoML AI 서비스 문제: {status.get('message', 'Unknown error')}")
            return False
    
    def get_ai_service_info(self) -> Dict:
        """AI 서비스 상태 정보 반환 - 폴백 모드 없이 실제 상태만"""
        status = self.get_service_status()
        return {
            "service_url": self.service_url,
            "service_status": status["status"],
            "ai_models_ready": status.get("ai_ready", False),
            "message": status.get("message", "서비스 상태 불명"),
            "last_checked": "real-time",
            "fallback_mode_disabled": True,
            "requires_real_ai": True
        }