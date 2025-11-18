"""
실시간 감지 API 엔드포인트 - BentoML 클라이언트 사용
"""

from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
import base64
import numpy as np
import cv2
from datetime import datetime

detect_bp = Blueprint('detect', __name__)

@detect_bp.route('/detect', methods=['POST'])
def detect_suspects():
    """
    실시간 용의자 감지
    ---
    tags:
      - detection
    summary: 실시간 용의자 감지 및 매칭
    description: 웹캠 또는 업로드된 이미지에서 용의자를 실시간으로 감지하고 매칭합니다.
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - name: detection_data
        in: body
        required: true
        schema:
          type: object
          properties:
            image:
              type: string
              format: base64
              description: Base64로 인코딩된 이미지 데이터
            detection_threshold:
              type: number
              minimum: 0
              maximum: 1
              default: 0.6
              description: 감지 신뢰도 임계값
            matching_threshold:
              type: number
              minimum: 0
              maximum: 1
              default: 0.7
              description: 매칭 유사도 임계값
    responses:
      200:
        description: 감지 결과
        schema:
          type: object
          properties:
            success:
              type: boolean
            results:
              type: array
              items:
                type: object
                properties:
                  face_id:
                    type: string
                    description: 감지된 얼굴 ID
                  bbox:
                    type: array
                    items:
                      type: number
                    description: 바운딩 박스 좌표
                  confidence:
                    type: number
                    description: 감지 신뢰도
                  matched_suspect:
                    type: object
                    nullable: true
                    description: 매칭된 용의자 정보
            timestamp:
              type: string
              format: date-time
    """
    try:
        # BentoML 클라이언트 가져오기
        from app import bento_client
        
        if not bento_client:
            return jsonify({
                'success': False,
                'error': 'AI 서비스가 초기화되지 않았습니다.'
            }), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': '이미지 데이터가 필요합니다.'
            }), 400
        
        image_data = data['image']
        detection_threshold = data.get('detection_threshold', 0.6)
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
                'error': result.get('error', '용의자 인식에 실패했습니다.')
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
                'matched_suspect': suspect_match if suspect_match.get('similarity', 0) >= matching_threshold else None
            }
            detections.append(detection)
        
        response_data = {
            'success': True,
            'detections': detections,
            'processing_time_ms': result.get('processing_time_ms', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'서버 오류: {str(e)}'
        }), 500

@detect_bp.route('/video_analysis', methods=['POST'])  
def analyze_video():
    """업로드된 비디오 전체 분석"""
    try:
        # BentoML 클라이언트 가져오기
        from app import bento_client
        
        if not bento_client:
            return jsonify({
                'success': False,
                'error': 'AI 서비스가 초기화되지 않았습니다.'
            }), 500
        
        data = request.get_json()
        video_path = data.get('video_path')
        target_suspect_id = data.get('target_suspect_id', '1')
        detection_threshold = data.get('detection_threshold', 0.6)
        matching_threshold = data.get('matching_threshold', 0.7)
        
        if not video_path:
            return jsonify({'error': '비디오 경로가 제공되지 않았습니다.'}), 400
        
        # 비디오 분석 로직 (간단한 구현)
        detections = []
        total_frames = 0
        faces_detected = 0
        suspect_matches = 0
        
        try:
            # OpenCV로 비디오 읽기
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                total_frames += 1
                
                # 매 30프레임마다 분석 (성능 최적화)
                if frame_count % 30 == 0:
                    # 프레임을 Base64로 인코딩
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # BentoML 서비스로 분석 요청
                    result = bento_client.recognize_suspects(
                        image_data=frame_base64,
                        detection_threshold=detection_threshold,
                        matching_threshold=matching_threshold
                    )
                    
                    if result.get('success'):
                        recognition_results = result.get('recognition_results', [])
                        faces_detected += len(recognition_results)
                        
                        for recognition in recognition_results:
                            suspect_match = recognition.get('suspect_match', {})
                            if suspect_match.get('similarity', 0) >= matching_threshold:
                                suspect_matches += 1
                                
                                detections.append({
                                    'frame_number': frame_count,
                                    'timestamp_ms': cap.get(cv2.CAP_PROP_POS_MSEC),
                                    'bbox': recognition.get('face_bbox'),
                                    'suspect_info': suspect_match
                                })
            
            cap.release()
            
        except Exception as video_error:
            return jsonify({
                'success': False,
                'error': f'비디오 처리 오류: {str(video_error)}'
            }), 500
        
        return jsonify({
            'success': True,
            'message': '비디오 분석이 완료되었습니다.',
            'detections': detections,
            'summary': {
                'total_frames': total_frames,
                'faces_detected': faces_detected,
                'suspect_matches': suspect_matches,
                'analysis_frames': total_frames // 30  # 실제 분석한 프레임 수
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'서버 오류: {str(e)}'
        }), 500