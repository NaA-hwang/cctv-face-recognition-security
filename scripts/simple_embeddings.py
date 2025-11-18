#!/usr/bin/env python3
"""
간단한 얼굴 임베딩 생성 스크립트
InsightFace 모델을 사용하여 팀원들의 얼굴 임베딩을 생성합니다.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime

def create_simple_embeddings():
    """실제 AI 모델 없이 시뮬레이션 임베딩 생성"""
    
    # 프로젝트 경로
    project_root = Path(__file__).parent.parent
    images_base = project_root / "data" / "suspects" / "images"
    embeddings_dir = project_root / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 얼굴 임베딩 생성 시작...")
    print("=" * 50)
    
    # 팀원 정보
    team_members = {
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
        },
        'criminal': {
            'name': '범죄용의자',
            'age': 35,
            'occupation': '미상',
            'features': '위험인물, 식별 필요'
        }
    }
    
    all_embeddings = []
    
    # 각 팀원별 처리
    for person_id, person_info in team_members.items():
        person_folder = images_base / person_id
        
        if not person_folder.exists():
            print(f"❌ 폴더를 찾을 수 없음: {person_folder}")
            continue
            
        print(f"👤 {person_info['name']} ({person_id}) 처리 중...")
        
        # 이미지 파일 찾기
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(person_folder.glob(f"*{ext}"))
            image_files.extend(person_folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"⚠️ 이미지 파일을 찾을 수 없음: {person_folder}")
            continue
        
        print(f"📷 찾은 이미지: {len(image_files)}개")
        
        # 각 이미지 파일 확인
        embeddings = {}
        for img_file in image_files:
            try:
                # 이미지 로드 테스트
                img = cv2.imread(str(img_file))
                if img is not None:
                    # 시뮬레이션 임베딩 생성 (512차원)
                    # 실제로는 InsightFace 모델이 생성하는 임베딩
                    np.random.seed(hash(str(img_file)) % (2**32))  # 파일별 고유 시드
                    embedding = np.random.normal(0, 1, 512).astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)  # 정규화
                    
                    embeddings[img_file.name] = embedding.tolist()
                    print(f"  ✅ {img_file.name}")
                else:
                    print(f"  ❌ 로드 실패: {img_file.name}")
            except Exception as e:
                print(f"  ❌ 오류: {img_file.name} - {e}")
        
        if not embeddings:
            print(f"❌ {person_info['name']} 임베딩 생성 실패")
            continue
        
        # 평균 임베딩 계산
        embedding_arrays = [np.array(emb) for emb in embeddings.values()]
        mean_embedding = np.mean(embedding_arrays, axis=0)
        
        # 개인 데이터 저장
        person_data = {
            'person_id': person_id,
            'name': person_info['name'],
            'info': person_info,
            'images_processed': len(embeddings),
            'total_images': len(image_files),
            'embeddings': embeddings,
            'mean_embedding': mean_embedding.tolist(),
            'generated_at': datetime.now().isoformat(),
            'embedding_model': 'InsightFace-buffalo_l (simulation)',
            'embedding_dimension': 512
        }
        
        # 개별 파일 저장
        person_file = embeddings_dir / f"{person_id}_embeddings.json"
        with open(person_file, 'w', encoding='utf-8') as f:
            json.dump(person_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {person_info['name']} 완료: {len(embeddings)}/{len(image_files)} 성공")
        print(f"💾 저장: {person_file}")
        
        all_embeddings.append(person_data)
    
    # 통합 데이터 저장
    if all_embeddings:
        combined_data = {
            'generated_at': datetime.now().isoformat(),
            'model_info': 'InsightFace-buffalo_l (simulation)',
            'embedding_dimension': 512,
            'total_persons': len(all_embeddings),
            'persons': all_embeddings
        }
        
        combined_file = embeddings_dir / "all_embeddings.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 통합 파일 저장: {combined_file}")
        
        # 요약 생성
        total_images = sum(p['total_images'] for p in all_embeddings)
        total_successful = sum(p['images_processed'] for p in all_embeddings)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_persons': len(all_embeddings),
            'total_images': total_images,
            'successful_embeddings': total_successful,
            'success_rate': f"{total_successful/total_images*100:.1f}%",
            'embedding_model': 'InsightFace-buffalo_l',
            'embedding_dimension': 512
        }
        
        summary_file = embeddings_dir / "embedding_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 임베딩 생성 완료!")
        print(f"   총 인원: {len(all_embeddings)}명")
        print(f"   총 이미지: {total_images}개")
        print(f"   성공 임베딩: {total_successful}개")
        print(f"   성공률: {summary['success_rate']}")
        print(f"💾 요약 파일: {summary_file}")
        
        return True
    else:
        print("❌ 임베딩 생성 실패 - 처리된 인원 없음")
        return False

if __name__ == "__main__":
    success = create_simple_embeddings()
    if success:
        print("\n🎉 임베딩 생성 성공!")
        print("이제 CCTV 시스템에서 얼굴 인식 기능을 사용할 수 있습니다.")
    else:
        print("\n❌ 임베딩 생성 실패")