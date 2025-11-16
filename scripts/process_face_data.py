"""
얼굴 데이터 처리 및 임베딩 생성 스크립트
팀원들의 얼굴 사진을 처리하여 AI 모델용 데이터베이스를 구축합니다.
RetinaFace + ArcFace 모델을 사용합니다.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import sqlite3
import pickle
from datetime import datetime
from PIL import Image
import argparse

# ===============================================================================
# **중요: InsightFace 라이브러리 설치 및 의존성 해결 필요**
# ===============================================================================
# TODO: pip install insightface onnxruntime opencv-python 실행 필요
# TODO: 실제 팀원 얼굴 사진 데이터 수집 및 정리 필요
# TODO: 얼굴 임베딩 추출 및 데이터베이스 저장 자동화
# TODO: 데이터 전처리 파이프라인 구축
# ===============================================================================

# InsightFace 모델 import
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.data import get_image as ins_get_image
    INSIGHTFACE_AVAILABLE = True
    print("✅ InsightFace 모델 로드 가능")
except ImportError as e:
    print(f"⚠️  InsightFace 설치 필요: pip install insightface")
    print(f"Error: {e}")
    INSIGHTFACE_AVAILABLE = False

class FaceDataProcessor:
    """얼굴 데이터 처리 클래스"""
    
    def __init__(self, data_root="data"):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "suspects" / "images"
        self.metadata_dir = self.data_root / "suspects" / "metadata"
        self.processed_dir = self.data_root / "suspects" / "processed"
        self.db_path = self.data_root / "embeddings" / "suspects.db"
        
        # 처리된 데이터 디렉터리 생성
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.data_root / "embeddings").mkdir(parents=True, exist_ok=True)
        
        # InsightFace 모델 초기화
        self.face_app = None
        self.init_face_models()
        
        # 메타데이터 로드
        self.load_metadata()
        
    def init_face_models(self):
        """RetinaFace + ArcFace 모델 초기화"""
        if not INSIGHTFACE_AVAILABLE:
            print("❌ InsightFace를 먼저 설치해주세요: pip install insightface")
            print("❌ 추가 의존성: pip install onnxruntime")
            return False
            
        try:
            print("🤖 AI 모델 초기화 중...")
            
            # FaceAnalysis 앱 초기화 (RetinaFace + ArcFace 포함)
            self.face_app = FaceAnalysis(
                providers=['CPUExecutionProvider']  # GPU 사용시 'CUDAExecutionProvider' 추가
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            print("✅ RetinaFace (얼굴검출) + ArcFace (임베딩) 모델 로드 완료")
            print(f"📊 검출 모델: {self.face_app.det_model.__class__.__name__}")
            print(f"🧠 인식 모델: {self.face_app.rec_model.__class__.__name__}")
            return True
            
        except Exception as e:
            print(f"❌ AI 모델 초기화 실패: {e}")
            print("💡 해결방법:")
            print("   1. pip install insightface onnxruntime")
            print("   2. 모델 파일 다운로드 대기 (처음 실행시)")
            self.face_app = None
            return False
        
    def load_metadata(self):
        """메타데이터 JSON 파일 로드"""
        metadata_file = self.metadata_dir / "suspect_profiles.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"✅ 메타데이터 로드 완료: {len(self.metadata['suspects'])}명")
        else:
            print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {metadata_file}")
            self.metadata = None
    
    def validate_images(self):
        """이미지 파일 존재 여부 및 품질 검증"""
        print("🔍 이미지 파일 검증 중...")
        
        validation_results = {
            "total_suspects": 0,
            "valid_suspects": 0,
            "missing_images": [],
            "invalid_images": [],
            "quality_warnings": []
        }
        
        for suspect in self.metadata['suspects']:
            name_en = suspect['name_en']
            required_images = suspect['images']['required_angles']
            
            validation_results["total_suspects"] += 1
            suspect_valid = True
            
            print(f"\n📸 {suspect['name']} ({name_en}) 검증 중...")
            
            for img_path in required_images:
                full_path = self.images_dir / img_path
                
                if not full_path.exists():
                    validation_results["missing_images"].append(str(full_path))
                    print(f"  ❌ 누락: {img_path}")
                    suspect_valid = False
                else:
                    # 이미지 품질 검사
                    quality_check = self.check_image_quality(full_path)
                    if quality_check["valid"]:
                        print(f"  ✅ 유효: {img_path} ({quality_check['resolution']})")
                    else:
                        validation_results["invalid_images"].append({
                            "path": str(full_path),
                            "issues": quality_check["issues"]
                        })
                        print(f"  ⚠️ 품질 문제: {img_path} - {quality_check['issues']}")
                        
                        if "resolution_too_low" in quality_check["issues"]:
                            suspect_valid = False
            
            if suspect_valid:
                validation_results["valid_suspects"] += 1
                print(f"  ✅ {suspect['name']}: 모든 이미지 검증 완료")
            else:
                print(f"  ❌ {suspect['name']}: 이미지 문제 있음")
        
        return validation_results
    
    def check_image_quality(self, image_path):
        """개별 이미지 품질 검사"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"valid": False, "issues": ["cannot_read"]}
            
            height, width = img.shape[:2]
            resolution = f"{width}x{height}"
            
            issues = []
            
            # 해상도 체크
            if width < 640 or height < 480:
                issues.append("resolution_too_low")
            
            # 파일 크기 체크
            file_size = os.path.getsize(image_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                issues.append("file_too_large")
            elif file_size < 50 * 1024:  # 50KB
                issues.append("file_too_small")
            
            # 이미지 밝기 체크 (너무 어두우면 인식 어려움)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness < 50:
                issues.append("too_dark")
            elif avg_brightness > 200:
                issues.append("too_bright")
            
            return {
                "valid": len(issues) == 0 or "resolution_too_low" not in issues,
                "resolution": resolution,
                "file_size": file_size,
                "brightness": avg_brightness,
                "issues": issues
            }
            
        except Exception as e:
            return {"valid": False, "issues": [f"error: {str(e)}"]}
    
    def process_images(self):
        """이미지 전처리 (얼굴 정렬, 크기 정규화)"""
        print("🔄 이미지 전처리 시작...")
        
        # 여기서는 기본적인 전처리만 수행
        # 실제로는 InsightFace의 face detection과 alignment를 사용해야 함
        
        processed_count = 0
        
        for suspect in self.metadata['suspects']:
            name_en = suspect['name_en']
            processed_dir = self.processed_dir / "aligned_faces" / name_en
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"처리 중: {suspect['name']} ({name_en})")
            
            for img_path in suspect['images']['required_angles']:
                source_path = self.images_dir / img_path
                
                if source_path.exists():
                    # 기본 전처리: 크기 조정 및 정규화
                    processed_path = processed_dir / source_path.name
                    self.preprocess_image(source_path, processed_path)
                    processed_count += 1
        
        print(f"✅ 총 {processed_count}개 이미지 전처리 완료")
        return processed_count
    
    def preprocess_image(self, source_path, target_path):
        """개별 이미지 전처리"""
        try:
            # 이미지 로드
            img = cv2.imread(str(source_path))
            
            # 크기 정규화 (AI 모델 입력용)
            target_size = (112, 112)  # InsightFace 표준 크기
            resized = cv2.resize(img, target_size)
            
            # 히스토그램 평활화 (조명 정규화)
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 저장
            cv2.imwrite(str(target_path), normalized)
            
        except Exception as e:
            print(f"❌ 이미지 전처리 실패 {source_path}: {str(e)}")
    
    def create_sample_embeddings(self):
        """샘플 임베딩 생성 (실제 AI 모델 없이 테스트용)"""
        print("🔧 샘플 임베딩 생성 중...")
        
        embeddings_data = {}
        
        for suspect in self.metadata['suspects']:
            suspect_id = suspect['id']
            name = suspect['name']
            
            # 실제로는 InsightFace ArcFace 모델을 사용해야 함
            # 여기서는 테스트용 랜덤 임베딩 생성
            np.random.seed(int(suspect_id) * 42)  # 재현 가능한 랜덤
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # L2 정규화
            
            embeddings_data[suspect_id] = {
                "name": name,
                "embedding": embedding,
                "created_date": datetime.now().isoformat()
            }
            
            print(f"  ✅ {name}: 임베딩 생성 완료 (차원: {embedding.shape})")
        
        # 임베딩 데이터 저장
        embeddings_file = self.processed_dir / "embeddings.pkl"
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        print(f"✅ 임베딩 데이터 저장: {embeddings_file}")
        return embeddings_data
    
    def update_database(self, embeddings_data):
        """SQLite 데이터베이스 업데이트"""
        print("💾 데이터베이스 업데이트 중...")
        
        # 여기서는 backend/models/embedding_db.py의 EmbeddingDatabase 클래스를 사용해야 함
        # 현재는 간단한 예시만 제공
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 테이블이 없으면 생성 (간단 버전)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_embeddings (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    embedding BLOB,
                    created_date TEXT
                )
            """)
            
            # 데이터 삽입
            for suspect_id, data in embeddings_data.items():
                embedding_blob = pickle.dumps(data['embedding'])
                
                cursor.execute("""
                    INSERT OR REPLACE INTO test_embeddings 
                    (id, name, embedding, created_date)
                    VALUES (?, ?, ?, ?)
                """, (suspect_id, data['name'], embedding_blob, data['created_date']))
            
            conn.commit()
            conn.close()
            
            print(f"✅ 데이터베이스 업데이트 완료: {self.db_path}")
            
        except Exception as e:
            print(f"❌ 데이터베이스 업데이트 실패: {str(e)}")
    
    def generate_report(self, validation_results, processed_count, embeddings_count):
        """처리 결과 리포트 생성"""
        report = f"""
# 얼굴 데이터 처리 리포트
생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 처리 결과 요약
- 총 용의자 수: {validation_results['total_suspects']}명
- 유효한 용의자: {validation_results['valid_suspects']}명  
- 처리된 이미지: {processed_count}개
- 생성된 임베딩: {embeddings_count}개

## ✅ 성공한 용의자들
"""
        
        for suspect in self.metadata['suspects']:
            if suspect['name_en'] not in [str(p).split('/')[-2] for p in validation_results['missing_images']]:
                report += f"- {suspect['name']} ({suspect['name_en']}): {suspect['images']['total_count']}장\n"
        
        if validation_results['missing_images']:
            report += "\n## ❌ 누락된 이미지들\n"
            for missing in validation_results['missing_images']:
                report += f"- {missing}\n"
        
        if validation_results['invalid_images']:
            report += "\n## ⚠️ 품질 문제 이미지들\n"
            for invalid in validation_results['invalid_images']:
                report += f"- {invalid['path']}: {invalid['issues']}\n"
        
        report += f"""
## 📁 생성된 파일들
- 전처리된 이미지: `{self.processed_dir}/aligned_faces/`
- 임베딩 데이터: `{self.processed_dir}/embeddings.pkl`
- 데이터베이스: `{self.db_path}`

## 🚀 다음 단계
1. 누락된 이미지들을 촬영하여 추가
2. 품질 문제가 있는 이미지들을 재촬영  
3. AI 모델 서버 실행하여 실제 임베딩 생성
4. CCTV 시스템에서 테스트
"""
        
        # 리포트 파일 저장
        report_file = self.processed_dir / "processing_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📋 처리 리포트 저장: {report_file}")
        print(report)
        
        return report


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="팀원 얼굴 데이터 처리")
    parser.add_argument("--data-root", default="data", help="데이터 루트 디렉터리")
    parser.add_argument("--validate-only", action="store_true", help="검증만 수행")
    parser.add_argument("--skip-processing", action="store_true", help="이미지 전처리 건너뛰기")
    
    args = parser.parse_args()
    
    print("🚀 팀원 얼굴 데이터 처리 시작")
    print("=" * 50)
    
    # 데이터 처리기 초기화
    processor = FaceDataProcessor(args.data_root)
    
    if processor.metadata is None:
        print("❌ 메타데이터를 로드할 수 없습니다. 종료합니다.")
        return
    
    # 1. 이미지 검증
    validation_results = processor.validate_images()
    
    if args.validate_only:
        print("\n✅ 검증 완료. 처리 과정은 건너뜁니다.")
        return
    
    processed_count = 0
    embeddings_count = 0
    
    # 2. 이미지 전처리
    if not args.skip_processing:
        processed_count = processor.process_images()
    
    # 3. 임베딩 생성 (샘플)
    embeddings_data = processor.create_sample_embeddings()
    embeddings_count = len(embeddings_data)
    
    # 4. 데이터베이스 업데이트
    processor.update_database(embeddings_data)
    
    # 5. 리포트 생성
    processor.generate_report(validation_results, processed_count, embeddings_count)
    
    print("\n🎉 모든 처리가 완료되었습니다!")
    print("\n다음으로 할 일:")
    print("1. 누락된 이미지들을 촬영하여 해당 폴더에 저장")
    print("2. backend/app.py를 실행하여 AI 서버 시작")
    print("3. HTML 페이지에서 실제 얼굴 인식 테스트")


if __name__ == "__main__":
    main()