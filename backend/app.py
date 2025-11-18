"""
CCTV 용의자 식별 시스템 - Flask 백엔드 서버
InsightFace (RetinaFace + ArcFace) 모델 통합
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flasgger import Swagger, swag_from
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
import json
from datetime import datetime

# 모델 관련 imports - BentoML 클라이언트 사용
from bento_client import BentoMLClient

# API 엔드포인트
from api.upload import upload_bp
from api.detect import detect_bp
from api.suspects import suspects_bp

app = Flask(__name__)
CORS(app)

# 전역 BentoML 클라이언트 변수
bento_client = None

# 전역 모델 변수들 (호환성을 위해 유지)
face_detector = None
face_recognizer = None
embedding_db = None

# Swagger UI 설정
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "CCTV 용의자 식별 시스템 API",
        "description": "RetinaFace + ArcFace 기반 얼굴 인식 시스템",
        "version": "1.0.0",
        "contact": {
            "name": "CCTV 프로젝트 팀",
            "email": "project@example.com"
        }
    },
    "host": "localhost:5000",
    "basePath": "/api",
    "schemes": ["http", "https"],
    "tags": [
        {"name": "detection", "description": "얼굴 감지 및 인식"},
        {"name": "suspects", "description": "용의자 관리"},
        {"name": "upload", "description": "파일 업로드"},
        {"name": "system", "description": "시스템 상태"}
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# 설정
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'videos')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 제한
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# 전역 변수 - 모델 인스턴스
face_detector = None
face_recognizer = None
embedding_db = None

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_bento_client():
    """BentoML 클라이언트 초기화"""
    global bento_client
    
    print("🔧 BentoML 클라이언트 초기화 중...")
    
    try:
        # BentoML 서비스 URL 설정 (환경변수 또는 기본값)
        service_url = os.getenv('BENTOML_SERVICE_URL') or 'http://localhost:3000'
        bento_client = BentoMLClient(service_url)
        
        # 서비스 연결 테스트
        status = bento_client.get_service_status()
        if status.get("status") == "healthy":
            print(f"✅ BentoML 서비스 연결 성공: {service_url}")
        else:
            print(f"⚠️ BentoML 서비스 연결 실패, 폴백 모드로 동작: {service_url}")
            print(f"   상태: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ BentoML 클라이언트 초기화 실패: {str(e)}")
        service_url = 'http://localhost:3000'
        bento_client = BentoMLClient(service_url)  # 기본 설정으로 생성
        return False

@app.route('/')
def index():
    """메인 페이지 - HTML 파일 서빙"""
    try:
        # 프로젝트 루트에서 HTML 파일 찾기
        html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cctv_suspect_identification.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>CCTV 용의자 식별 시스템</h1>
        <p>HTML 파일을 찾을 수 없습니다. cctv_suspect_identification.html 파일을 확인해주세요.</p>
        <p><a href="/api/status">시스템 상태 확인</a></p>
        <p><a href="/docs/">API 문서</a></p>
        """

@app.route('/api/status')
def status():
    """
    시스템 상태 확인 API
    ---
    tags:
      - system
    summary: 시스템 상태 및 모델 준비 상태 확인
    description: AI 모델의 로드 상태와 데이터베이스 연결 상태를 확인합니다.
    responses:
      200:
        description: 시스템 상태 정보
        schema:
          type: object
          properties:
            timestamp:
              type: string
              format: date-time
              description: 요청 시간
            models:
              type: object
              properties:
                face_detector:
                  type: boolean
                  description: 얼굴 검출 모델 준비 상태
                face_recognizer:
                  type: boolean
                  description: 얼굴 인식 모델 준비 상태
                embedding_db:
                  type: boolean
                  description: 임베딩 데이터베이스 준비 상태
            database:
              type: object
              properties:
                suspects_count:
                  type: integer
                  description: 등록된 용의자 수
                embeddings_loaded:
                  type: boolean
                  description: 임베딩 데이터 로드 상태
            system:
              type: object
              properties:
                status:
                  type: string
                  enum: ["ready", "initializing", "error"]
                  description: 전체 시스템 상태
    """
    global bento_client
    
    # BentoML 클라이언트를 통해 AI 서비스 상태 확인
    ai_service_ready = False
    ai_service_info = {}
    if bento_client:
        ai_service_info = bento_client.get_ai_service_info()
        ai_service_ready = ai_service_info.get("ai_models_ready", False)
    
    # 전체 시스템 상태 결정
    if ai_service_ready:
        overall_status = "ready"
    elif bento_client:
        overall_status = "ai_service_down"
    else:
        overall_status = "initializing"
    
    status_info = {
        'timestamp': datetime.now().isoformat(),
        'models': {
            'face_detector': ai_service_ready,
            'face_recognizer': ai_service_ready,
            'embedding_db': ai_service_ready
        },
        'database': {
            'suspects_count': 4,  # 현재 등록된 용의자 수 (criminal, normal01, normal02, normal03)
            'embeddings_loaded': ai_service_ready
        },
        'system': {
            'status': overall_status,
            'opencv_version': cv2.__version__,
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024*1024),
            'ai_service_url': ai_service_info.get("service_url", "Unknown"),
            'ai_service_status': ai_service_info.get("service_status", "Unknown"),
            'fallback_mode_disabled': ai_service_info.get("fallback_mode_disabled", True),
            'requires_real_ai': ai_service_info.get("requires_real_ai", True)
        }
    }
    
    return jsonify(status_info)

@app.route('/api/ai-service/health')
def ai_service_health():
    """
    AI 서비스 전용 상태 확인 - 폴백 모드 비활성화 검증
    ---
    tags:
      - system
    summary: AI 서비스 상태 및 실제 모델 동작 확인
    description: BentoML AI 서비스가 실제 AI 모델로 동작 중인지 확인하고 폴백 모드가 비활성화되었는지 검증
    responses:
      200:
        description: AI 서비스 상태 정보
        schema:
          type: object
          properties:
            ai_service_ready:
              type: boolean
              description: AI 서비스 준비 상태
            real_ai_models_active:
              type: boolean
              description: 실제 AI 모델 활성 상태
            fallback_mode_disabled:
              type: boolean
              description: 폴백 모드 비활성화 여부
            service_details:
              type: object
              description: 상세 서비스 정보
      503:
        description: AI 서비스 사용 불가
        schema:
          type: object
          properties:
            error:
              type: string
              description: 오류 메시지
            action_required:
              type: string
              description: 필요한 조치
    """
    global bento_client
    
    if not bento_client:
        return jsonify({
            "ai_service_ready": False,
            "real_ai_models_active": False,
            "fallback_mode_disabled": True,
            "error": "BentoML 클라이언트가 초기화되지 않았습니다",
            "action_required": "Flask 서버 재시작 필요"
        }), 503
    
    # AI 서비스 준비 상태 확인
    is_ready = bento_client.ensure_ai_service_ready()
    ai_info = bento_client.get_ai_service_info()
    
    if is_ready:
        return jsonify({
            "ai_service_ready": True,
            "real_ai_models_active": True,
            "fallback_mode_disabled": True,
            "service_details": ai_info,
            "status": "✅ 실제 AI 모델 동작 중 - 폴백 모드 없음",
            "timestamp": datetime.now().isoformat()
        })
    else:
        return jsonify({
            "ai_service_ready": False,
            "real_ai_models_active": False,
            "fallback_mode_disabled": True,
            "service_details": ai_info,
            "error": "🔴 AI 서비스 연결 실패 - 실제 AI 모델 사용 불가",
            "action_required": "bentoml serve 명령으로 AI 서비스를 재시작하세요",
            "timestamp": datetime.now().isoformat()
        }), 503

@app.route('/api/detect_frame', methods=['POST'])
def detect_frame():
    """
    단일 프레임에서 얼굴 감지 및 인식
    ---
    tags:
      - detection
    summary: 실시간 프레임 얼굴 감지 및 인식
    description: 카메라나 이미지에서 얼굴을 감지하고 용의자와 매칭합니다.
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - name: frame_data
        in: body
        required: true
        schema:
          type: object
          properties:
            image:
              type: string
              format: base64
              description: Base64로 인코딩된 이미지 데이터
            timestamp:
              type: string
              format: date-time
              description: 프레임 타임스탬프
            camera_id:
              type: string
              description: 카메라 ID
              default: "main_camera"
    responses:
      200:
        description: 얼굴 감지 결과
        schema:
          type: object
          properties:
            detected_faces:
              type: array
              items:
                type: object
                properties:
                  bbox:
                    type: array
                    items:
                      type: number
                    description: 경계 박스 [x1, y1, x2, y2]
                  confidence:
                    type: number
                    description: 감지 신뢰도
                  suspect_match:
                    type: object
                    nullable: true
                    properties:
                      id:
                        type: string
                        description: 용의자 ID
                      name:
                        type: string
                        description: 용의자 이름
                      similarity:
                        type: number
                        description: 유사도 (0-1)
                      is_criminal:
                        type: boolean
                        description: 범죄자 여부
                      risk_level:
                        type: string
                        enum: ["low", "medium", "high"]
                        description: 위험 등급
            processing_time:
              type: number
              description: 처리 시간 (밀리초)
            timestamp:
              type: string
              format: date-time
              description: 처리 완료 시간
      400:
        description: 잘못된 요청 (이미지 데이터 없음)
      500:
        description: 서버 오류 (모델 초기화 실패 등)
    """
    global bento_client
    
    try:
        # BentoML 클라이언트 확인
        if not bento_client:
            return jsonify({
                'success': False,
                'error': 'AI 서비스가 연결되지 않았습니다.',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # Base64 이미지 데이터 받기
        data = request.get_json()
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': '이미지 데이터가 제공되지 않았습니다.'
            }), 400
            
        image_data = data['image']
        
        # 선택된 용의자 ID와 임계값
        target_suspect_id = data.get('target_suspect_id', '1')
        detection_threshold = data.get('detection_threshold', 0.8)
        matching_threshold = data.get('matching_threshold', 0.7)
        
        # BentoML 서비스로 용의자 인식 요청
        result = bento_client.recognize_suspects(
            image_data=image_data,
            detection_threshold=detection_threshold,
            matching_threshold=matching_threshold
        )
        
        if not result.get('success'):
            return jsonify({
                'success': False,
                'error': result.get('error', '용의자 인식에 실패했습니다.'),
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # 결과 포맷팅
        detections = []
        recognition_results = result.get('recognition_results', [])
        
        for recognition in recognition_results:
            face_bbox = recognition.get('face_bbox', [])
            suspect_match = recognition.get('suspect_match', {})
            
            detection = {
                'bbox': face_bbox,
                'confidence': recognition.get('detection_confidence', 0.0),
                'suspect_match': suspect_match if suspect_match.get('similarity', 0) >= matching_threshold else None,
                'timestamp': datetime.now().isoformat()
            }
            detections.append(detection)
        
        return jsonify({
            'success': True,
            'detected_faces': detections,
            'processing_time_ms': result.get('processing_time_ms', 0),
            'timestamp': datetime.now().isoformat(),
            'frame_info': {
                'total_faces': len(detections),
                'matched_faces': len([d for d in detections if d['suspect_match']]),
                'target_suspect_id': target_suspect_id
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'서버 오류: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/suspects')
def get_suspects():
    """등록된 용의자 목록 반환"""
    global embedding_db
    
    try:
        suspects = embedding_db.get_all_suspects()
        return jsonify({
            'success': True,
            'suspects': suspects
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/add_suspect', methods=['POST'])  
def add_suspect():
    """새로운 용의자 추가"""
    global face_detector, face_recognizer, embedding_db
    
    try:
        # 폼 데이터 받기
        name = request.form.get('name')
        criminal_record = request.form.get('criminal_record', '')
        risk_level = request.form.get('risk_level', 'medium')
        
        # 이미지 파일 받기
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
            
        # 이미지 처리
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 얼굴 검출 및 임베딩 추출
        faces = face_detector.detect_faces(image)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in image'}), 400
            
        # 첫 번째 얼굴 사용
        bbox, landmarks, confidence = faces[0]
        x1, y1, x2, y2 = bbox.astype(int)
        face_roi = image[y1:y2, x1:x2]
        
        # 임베딩 추출
        embedding = face_recognizer.extract_embedding(face_roi)
        
        # 데이터베이스에 저장
        suspect_id = embedding_db.add_suspect(
            name=name,
            embedding=embedding,
            criminal_record=criminal_record.split(',') if criminal_record else [],
            risk_level=risk_level
        )
        
        return jsonify({
            'success': True,
            'suspect_id': suspect_id,
            'message': f'용의자 {name}이 성공적으로 등록되었습니다.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Blueprint 등록
app.register_blueprint(upload_bp, url_prefix='/api')
app.register_blueprint(detect_bp, url_prefix='/api') 
app.register_blueprint(suspects_bp, url_prefix='/api')

@app.errorhandler(413)
def too_large(e):
    """파일 크기 초과 에러 핸들러"""
    return jsonify({
        'error': 'File too large. Maximum size is 100MB.'
    }), 413

if __name__ == '__main__':
    # 디렉터리 생성
    base_dir = os.path.dirname(os.path.dirname(__file__))
    os.makedirs(os.path.join(base_dir, 'data', 'videos'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'data', 'suspects'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'data', 'embeddings'), exist_ok=True)
    
    # BentoML 클라이언트 초기화
    if initialize_bento_client():
        print("🚀 서버 시작 중...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    else:
        print("⚠️ BentoML 클라이언트 초기화 실패. 폴백 모드로 서버를 시작합니다.")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )