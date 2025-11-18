"""
용의자 관리 API 엔드포인트 - BentoML 클라이언트 사용
"""

from flask import Blueprint, request, jsonify
from flasgger import swag_from
import base64
import os
from datetime import datetime

suspects_bp = Blueprint('suspects', __name__)

@suspects_bp.route('/suspects', methods=['GET'])
def list_suspects():
    """용의자 목록 조회"""
    try:
        # 실제로는 데이터베이스에서 조회해야 함
        # 현재는 폴더 구조를 기반으로 간단히 구현
        suspects_data = []
        
        # criminal 폴더의 용의자들 조회
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        criminal_folder = os.path.join(base_dir, 'data', 'suspects', 'criminal')
        if os.path.exists(criminal_folder):
            for filename in os.listdir(criminal_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    suspect_id = os.path.splitext(filename)[0]
                    suspects_data.append({
                        'id': suspect_id,
                        'name': suspect_id.replace('_', ' ').title(),
                        'category': 'criminal',
                        'risk_level': 'high',
                        'image_path': f'{criminal_folder}/{filename}'
                    })
        
        return jsonify({
            'success': True,
            'suspects': suspects_data,
            'total_count': len(suspects_data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@suspects_bp.route('/suspects', methods=['POST'])
def create_suspect():
    """새 용의자 등록"""
    try:
        # BentoML 클라이언트 가져오기
        from app import bento_client
        
        if not bento_client:
            return jsonify({
                'success': False,
                'error': 'AI 서비스가 초기화되지 않았습니다.'
            }), 500
        
        data = request.get_json()
        
        # 필수 필드 검증
        required_fields = ['suspect_id', 'name', 'image']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'error': f'{field} 필드가 필요합니다.'
                }), 400
        
        suspect_id = data['suspect_id']
        name = data['name']
        image_data = data['image']
        category = data.get('category', 'criminal')
        metadata = data.get('metadata', {})
        
        # BentoML 서비스로 용의자 추가 요청
        result = bento_client.add_suspect(
            suspect_id=suspect_id,
            name=name,
            image_data=image_data,
            metadata={
                'category': category,
                'created_at': datetime.now().isoformat(),
                **metadata
            }
        )
        
        if not result.get('success'):
            return jsonify({
                'success': False,
                'error': result.get('error', '용의자 추가에 실패했습니다.')
            }), 500
        
        return jsonify({
            'success': True,
            'message': f'{name} 용의자가 성공적으로 등록되었습니다.',
            'suspect_id': suspect_id,
            'processing_time_ms': result.get('processing_time_ms', 0)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'서버 오류: {str(e)}'
        }), 500

@suspects_bp.route('/suspects/<string:suspect_id>', methods=['GET'])
def get_suspect(suspect_id):
    """특정 용의자 정보 조회"""
    try:
        # 실제로는 데이터베이스에서 조회해야 함
        # 현재는 파일 시스템을 기반으로 간단히 구현
        
        suspect_folders = ['criminal', 'normal01', 'normal02', 'normal03']
        suspect_info = None
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        for folder in suspect_folders:
            folder_path = os.path.join(base_dir, 'data', 'suspects', folder)
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.startswith(suspect_id) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        suspect_info = {
                            'id': suspect_id,
                            'name': suspect_id.replace('_', ' ').title(),
                            'category': folder,
                            'risk_level': 'high' if folder == 'criminal' else 'low',
                            'image_path': f'{folder_path}/{filename}',
                            'created_at': datetime.fromtimestamp(
                                os.path.getctime(os.path.join(folder_path, filename))
                            ).isoformat()
                        }
                        break
                        
        if not suspect_info:
            return jsonify({
                'success': False,
                'error': '용의자를 찾을 수 없습니다.'
            }), 404
        
        return jsonify({
            'success': True,
            'suspect': suspect_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@suspects_bp.route('/suspects/<string:suspect_id>/logs', methods=['GET'])
def get_suspect_logs(suspect_id):
    """특정 용의자의 감지 로그 조회"""
    try:
        # 실제로는 데이터베이스에서 감지 로그를 조회해야 함
        # 현재는 예시 데이터 반환
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        logs = [
            {
                'id': 1,
                'timestamp': datetime.now().isoformat(),
                'location': 'Camera_01',
                'confidence': 0.95,
                'image_path': os.path.join(base_dir, 'data', 'detections', f'{suspect_id}_detection_001.jpg')
            }
        ]
        
        return jsonify({
            'success': True,
            'logs': logs,
            'total_count': len(logs)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500