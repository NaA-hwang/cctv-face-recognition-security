#!/usr/bin/env python3
"""
얼굴 임베딩 생성 스크립트
실제 InsightFace 모델을 사용하여 팀원들의 얼굴 임베딩을 생성합니다.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import insightface
    from insightface.app import FaceAnalysis
    print("✅ InsightFace 모듈 로드 성공")
except ImportError as e:
    print(f"❌ InsightFace 모듈 로드 실패: {e}")
    print("다음 명령어로 설치하세요: pip install insightface")
    sys.exit(1)

class FaceEmbeddingGenerator:
    def __init__(self):
        """InsightFace 모델 초기화"""
        print("🚀 FaceAnalysis 모델 초기화 중...")
        
        # InsightFace FaceAnalysis 초기화
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace 모델 준비 완료")
        
        # 결과 저장 경로
        self.embeddings_dir = project_root / "data" / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # 팀원 정보
        self.team_members = {
            'normal01': {
                'name': '윤정아',
                'age': 28,
                'occupation': '디자이너',
                'features': '짧은 머리, 밝은 표정'
            },
            'normal02': {
                'name': '신종우', 
                'age': 32,
                'occupation': '소프트웨어 엔지니어',
                'features': '안경, 수염'
            },
            'normal03': {
                'name': '이지선',
                'age': 35,
                'occupation': '데이터 분석가', 
                'features': '긴 검은 머리, 안경 착용'
            }
        }
    
    def extract_face_embedding(self, image_path):
        """단일 이미지에서 얼굴 임베딩 추출"""
        try:
            # 이미지 로드
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"❌ 이미지 로드 실패: {image_path}")
                return None
            
            # RGB로 변환 (InsightFace는 RGB 사용)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 얼굴 검출 및 임베딩 추출
            faces = self.app.get(img_rgb)
            
            if len(faces) == 0:
                print(f"⚠️ 얼굴을 찾을 수 없음: {image_path}")
                return None
            elif len(faces) > 1:
                print(f"⚠️ 여러 얼굴 검출됨 (첫 번째 사용): {image_path}")
            
            # 첫 번째 얼굴의 임베딩 사용
            face = faces[0]
            embedding = face.embedding
            
            print(f"✅ 임베딩 추출 성공: {image_path.name} (크기: {embedding.shape})")
            return embedding.tolist()  # numpy array를 list로 변환
            
        except Exception as e:
            print(f"❌ 임베딩 추출 오류: {image_path} - {e}")
            return None
    
    def generate_embeddings_for_person(self, person_folder):
        """한 사람의 모든 이미지에 대해 임베딩 생성"""
        person_id = person_folder.name
        person_info = self.team_members.get(person_id, {})
        person_name = person_info.get('name', person_id)
        
        print(f"\n👤 {person_name} ({person_id}) 임베딩 생성 중...")
        
        # 이미지 파일 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(person_folder.glob(f"*{ext}"))
            image_files.extend(person_folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ {person_folder}에서 이미지 파일을 찾을 수 없습니다")
            return None
        
        print(f"📷 찾은 이미지: {len(image_files)}개")
        
        # 각 이미지에서 임베딩 추출
        embeddings = {}
        successful_embeddings = 0
        
        for img_file in image_files:
            embedding = self.extract_face_embedding(img_file)
            if embedding is not None:
                embeddings[img_file.name] = embedding
                successful_embeddings += 1
        
        if successful_embeddings == 0:
            print(f"❌ {person_name}의 임베딩 생성 실패")
            return None
        
        # 평균 임베딩 계산
        embedding_arrays = [np.array(emb) for emb in embeddings.values()]
        mean_embedding = np.mean(embedding_arrays, axis=0)
        
        # 결과 저장
        person_data = {
            'person_id': person_id,
            'name': person_name,
            'info': person_info,
            'images_processed': successful_embeddings,
            'total_images': len(image_files),
            'embeddings': embeddings,
            'mean_embedding': mean_embedding.tolist(),
            'generated_at': datetime.now().isoformat(),
            'embedding_model': 'InsightFace-buffalo_l'
        }
        
        print(f"✅ {person_name} 임베딩 완료: {successful_embeddings}/{len(image_files)} 성공")
        return person_data
    
    def save_embeddings(self, all_embeddings):
        """모든 임베딩을 파일로 저장"""
        
        # 개별 파일로 저장
        for person_data in all_embeddings:
            person_id = person_data['person_id']
            file_path = self.embeddings_dir / f"{person_id}_embeddings.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(person_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 저장됨: {file_path}")
        
        # 통합 파일로 저장
        combined_data = {
            'generated_at': datetime.now().isoformat(),
            'model_info': 'InsightFace-buffalo_l',
            'total_persons': len(all_embeddings),
            'persons': all_embeddings
        }
        
        combined_file = self.embeddings_dir / "all_embeddings.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 통합 파일 저장됨: {combined_file}")
        
        # 요약 정보 생성
        self.generate_summary(all_embeddings)
    
    def generate_summary(self, all_embeddings):
        """임베딩 생성 요약 정보"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_persons': len(all_embeddings),
            'persons_summary': []
        }
        
        total_images = 0
        total_successful = 0
        
        for person_data in all_embeddings:
            person_summary = {
                'person_id': person_data['person_id'],
                'name': person_data['name'],
                'images_processed': person_data['images_processed'],
                'total_images': person_data['total_images'],
                'success_rate': f"{person_data['images_processed']/person_data['total_images']*100:.1f}%"
            }
            summary['persons_summary'].append(person_summary)
            
            total_images += person_data['total_images']
            total_successful += person_data['images_processed']
        
        summary['overall_stats'] = {
            'total_images': total_images,
            'successful_embeddings': total_successful,
            'overall_success_rate': f"{total_successful/total_images*100:.1f}%"
        }
        
        # 요약 파일 저장
        summary_file = self.embeddings_dir / "embedding_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 임베딩 생성 요약:")
        print(f"   총 인원: {summary['total_persons']}명")
        print(f"   총 이미지: {total_images}개")
        print(f"   성공적 임베딩: {total_successful}개")
        print(f"   전체 성공률: {summary['overall_stats']['overall_success_rate']}")
        print(f"💾 요약 저장됨: {summary_file}")

def main():
    """메인 실행 함수"""
    print("🎯 CCTV 얼굴 인식 시스템 - 임베딩 생성기")
    print("=" * 50)
    
    generator = FaceEmbeddingGenerator()
    
    # 이미지 폴더 경로
    images_base = project_root / "data" / "suspects" / "images"
    
    if not images_base.exists():
        print(f"❌ 이미지 폴더를 찾을 수 없습니다: {images_base}")
        return
    
    # 각 사람별 폴더 처리
    all_embeddings = []
    person_folders = [f for f in images_base.iterdir() if f.is_dir() and f.name.startswith('normal')]
    
    if not person_folders:
        print(f"❌ 'normal'로 시작하는 폴더를 찾을 수 없습니다: {images_base}")
        return
    
    print(f"📁 찾은 인원 폴더: {len(person_folders)}개")
    
    for person_folder in sorted(person_folders):
        person_data = generator.generate_embeddings_for_person(person_folder)
        if person_data:
            all_embeddings.append(person_data)
    
    if not all_embeddings:
        print("❌ 임베딩 생성에 실패했습니다")
        return
    
    # 결과 저장
    print(f"\n💾 임베딩 저장 중...")
    generator.save_embeddings(all_embeddings)
    
    print(f"\n🎉 임베딩 생성 완료!")
    print(f"📂 저장 위치: {generator.embeddings_dir}")

if __name__ == "__main__":
    main()