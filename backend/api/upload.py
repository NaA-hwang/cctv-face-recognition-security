"""
파일 업로드 API 엔드포인트 - BentoML 연동 준비
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import base64

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['POST'])
def upload_video():
    """비디오 파일 업로드"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': '비디오 파일이 제공되지 않았습니다.'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': '비디오 파일이 선택되지 않았습니다.'}), 400
        
        # 파일 확장자 검증
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({
                'error': '지원되지 않는 비디오 형식입니다. 지원 형식: ' + ', '.join(allowed_extensions)
            }), 400
        
        # 파일명 보안 처리
        filename = secure_filename(file.filename)
        
        # 업로드 디렉터리 확인
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        upload_dir = os.path.join(base_dir, 'data', 'videos')
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
        
        # 파일 크기 확인
        file_size = os.path.getsize(filepath)
        
        return jsonify({
            'success': True,
            'message': '비디오가 성공적으로 업로드되었습니다.',
            'filepath': filepath,
            'filename': filename,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'video_info': {
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'duration_seconds': round(duration, 2),
                'estimated_analysis_frames': frame_count // 30  # 매 30프레임마다 분석 예정
            },
            'next_step': '비디오 분석을 시작하려면 /api/detect/video_analysis 엔드포인트를 사용하세요.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'파일 업로드 오류: {str(e)}'
        }), 500

@upload_bp.route('/upload/image', methods=['POST'])
def upload_suspect_image():
    """용의자 이미지 업로드"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '이미지 파일이 제공되지 않았습니다.'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '이미지 파일이 선택되지 않았습니다.'}), 400
        
        # 이미지 확장자 검증
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({
                'error': '지원되지 않는 이미지 형식입니다. 지원 형식: ' + ', '.join(allowed_extensions)
            }), 400
        
        # 요청 데이터에서 용의자 정보 가져오기
        suspect_id = request.form.get('suspect_id')
        category = request.form.get('category', 'criminal')
        
        if not suspect_id:
            return jsonify({'error': '용의자 ID가 필요합니다.'}), 400
        
        # 파일명 보안 처리
        original_filename = secure_filename(file.filename)
        extension = original_filename.rsplit('.', 1)[1].lower()
        filename = f"{suspect_id}.{extension}"
        
        # 업로드 디렉터리 확인 (카테고리별)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        upload_dir = os.path.join(base_dir, 'data', 'suspects', category)
        os.makedirs(upload_dir, exist_ok=True)
        
        # 파일 저장
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        # 이미지를 Base64로 인코딩 (BentoML 전송용)
        with open(filepath, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'message': '용의자 이미지가 성공적으로 업로드되었습니다.',
            'filepath': filepath,
            'filename': filename,
            'suspect_id': suspect_id,
            'category': category,
            'image_base64': image_base64,  # BentoML 서비스에서 사용 가능
            'next_step': '용의자를 등록하려면 /api/suspects 엔드포인트를 사용하세요.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'이미지 업로드 오류: {str(e)}'
        }), 500