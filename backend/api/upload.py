"""
파일 업로드 API 엔드포인트
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['POST'])
def upload_video():
    """비디오 파일 업로드"""
    # ===============================================================================
    # **중요: 실제 얼굴 인식 처리 로직 구현 필요**
    # ===============================================================================
    # TODO: 업로드된 비디오를 즉시 AI 모델로 처리하는 기능 추가 필요
    # TODO: 용의자 매칭 결과를 실시간으로 반환하는 기능 필요
    # ===============================================================================
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # 파일명 보안 처리
        filename = secure_filename(file.filename)
        
        # 업로드 디렉터리 확인
        upload_dir = 'data/videos'
        os.makedirs(upload_dir, exist_ok=True)
        
        # 파일 저장
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        # 비디오 정보 확인
        cap = cv2.VideoCapture(filepath)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return jsonify({
            'success': True,
            'filepath': filepath,
            'filename': filename,
            'video_info': {
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500