"""
CCTV 용의자 식별 시스템 - Vercel 배포용 엔트리포인트
"""

import sys
import os

# 백엔드 디렉토리를 Python 경로에 추가
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    # 백엔드 앱 임포트
    from app import app
    
    # Vercel용 애플리케이션 객체 노출
    application = app
    
    print("✅ Flask 앱 로드 성공 - Vercel 배포 준비 완료")
    
except ImportError as e:
    print(f"❌ 백엔드 앱 임포트 실패: {e}")
    
    # 최소한의 Flask 앱 생성 (폴백)
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route('/')
    def health_check():
        return jsonify({
            "status": "OK",
            "message": "CCTV 용의자 식별 시스템",
            "note": "백엔드 모듈 로드 실패 - 개발 중"
        })
    
    application = app
    print("⚠️ 폴백 앱 생성됨")

if __name__ == '__main__':
    application.run(debug=True)