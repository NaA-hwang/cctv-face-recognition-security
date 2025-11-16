"""
Modern Embedding Database Management
RetinaFace + ArcFace 모델을 위한 현대적인 얼굴 임베딩 데이터베이스
"""

import sqlite3
import json
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import hashlib
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SuspectProfile:
    """용의자 프로필 데이터 클래스"""
    id: str
    name: str
    name_en: str
    age: int
    gender: str
    occupation: str
    role: str  # 'thief' or 'civilian'
    is_criminal: bool
    is_target: bool
    risk_level: str
    folder_name: str
    criminal_record: List[str]
    features: Dict[str, str]
    notes: str = ""

@dataclass 
class EmbeddingData:
    """임베딩 데이터 클래스"""
    suspect_id: str
    embedding_vector: np.ndarray
    image_path: str
    confidence_score: float
    detection_bbox: Optional[Tuple[int, int, int, int]] = None
    landmarks: Optional[List[Tuple[float, float]]] = None
    model_version: str = "arcface_r100"
    processing_time_ms: int = 0

@dataclass
class DetectionResult:
    """얼굴 감지 결과 데이터 클래스"""
    suspect_id: str
    name: str
    similarity_score: float
    confidence_score: float
    bbox: Tuple[int, int, int, int]
    is_criminal: bool
    risk_level: str
    timestamp: datetime
    alert_triggered: bool = False

class ModernEmbeddingDB:
    """리팩토링된 임베딩 데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "data/embeddings/face_recognition.db", 
                 config_path: str = "data/suspects/metadata/suspect_profiles.json"):
        """
        ModernEmbeddingDB 초기화
        
        Args:
            db_path: SQLite 데이터베이스 파일 경로
            config_path: 용의자 프로필 JSON 설정 파일 경로
        """
        self.db_path = Path(db_path)
        self.config_path = Path(config_path)
        self.embedding_dim = 512  # ArcFace 임베딩 차원
        self.similarity_threshold = 0.6
        
        # 디렉터리 생성
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 초기화
        self._create_modern_schema()
        
        # 설정 파일에서 용의자 정보 로드
        self._load_suspects_from_config()
        
        logger.info(f"✅ 모던 임베딩 DB 초기화 완료: {self.db_path}")
    
    def _create_modern_schema(self):
        """현대적인 데이터베이스 스키마 생성"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 용의자 프로필 테이블 (확장된 스키마)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS suspects (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        name_en TEXT,
                        age INTEGER,
                        gender TEXT CHECK(gender IN ('male', 'female', 'other')),
                        occupation TEXT,
                        role TEXT CHECK(role IN ('thief', 'civilian', 'unknown')),
                        is_criminal BOOLEAN DEFAULT FALSE,
                        is_target BOOLEAN DEFAULT FALSE,
                        risk_level TEXT CHECK(risk_level IN ('low', 'medium', 'high', 'test')) DEFAULT 'low',
                        folder_name TEXT UNIQUE,
                        criminal_record_json TEXT,  -- JSON 배열
                        features_json TEXT,         -- JSON 객체
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # 얼굴 임베딩 테이블 (최적화된 스키마)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        suspect_id TEXT NOT NULL,
                        embedding_vector BLOB NOT NULL,  -- 512차원 float32 배열
                        image_path TEXT,
                        image_hash TEXT,  -- 이미지 중복 방지
                        confidence_score REAL DEFAULT 1.0,
                        detection_bbox TEXT,  -- JSON: [x1,y1,x2,y2]
                        landmarks_json TEXT,  -- JSON: 5점 랜드마크
                        model_version TEXT DEFAULT 'arcface_r100',
                        processing_time_ms INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (suspect_id) REFERENCES suspects (id) ON DELETE CASCADE
                    )
                """)
                
                # 실시간 감지 로그 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        suspect_id TEXT,
                        matched_embedding_id INTEGER,
                        similarity_score REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        detection_bbox TEXT,  -- JSON: [x1,y1,x2,y2]
                        camera_id TEXT DEFAULT 'main_camera',
                        frame_timestamp TIMESTAMP,
                        processing_time_ms INTEGER,
                        alert_triggered BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (suspect_id) REFERENCES suspects (id),
                        FOREIGN KEY (matched_embedding_id) REFERENCES face_embeddings (id)
                    )
                """)
                
                # 시스템 메타데이터 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        data_type TEXT CHECK(data_type IN ('string', 'integer', 'float', 'boolean', 'json')),
                        description TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 인덱스 생성 (성능 최적화)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_suspects_is_criminal ON suspects(is_criminal)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_suspects_active ON suspects(is_active)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_suspect ON face_embeddings(suspect_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_logs(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_similarity ON detection_logs(similarity_score)")
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_image_hash ON face_embeddings(image_hash)")
                
                # 시스템 메타데이터 초기화
                cursor.execute("""
                    INSERT OR IGNORE INTO system_metadata (key, value, data_type, description)
                    VALUES 
                    ('db_version', '2.0', 'string', '데이터베이스 스키마 버전'),
                    ('embedding_dim', '512', 'integer', 'ArcFace 임베딩 차원'),
                    ('similarity_threshold', '0.6', 'float', '기본 유사도 임계값'),
                    ('total_suspects', '0', 'integer', '등록된 용의자 수'),
                    ('total_embeddings', '0', 'integer', '저장된 임베딩 수')
                """)
                
                conn.commit()
                logger.info("✅ 현대적 데이터베이스 스키마 생성 완료")
                
        except Exception as e:
            logger.error(f"❌ 데이터베이스 스키마 생성 실패: {e}")
            raise e
    
    def _load_suspects_from_config(self):
        """설정 파일에서 용의자 정보 로드"""
        if not self.config_path.exists():
            logger.warning(f"⚠️ 설정 파일이 없습니다: {self.config_path}")
            self._create_default_suspects()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            suspects_data = config_data.get('suspects', [])
            
            for suspect_data in suspects_data:
                profile = SuspectProfile(
                    id=suspect_data['id'],
                    name=suspect_data['name'],
                    name_en=suspect_data['name_en'],
                    age=suspect_data['age'],
                    gender=suspect_data['gender'],
                    occupation=suspect_data['occupation'],
                    role=suspect_data['role'],
                    is_criminal=suspect_data.get('is_criminal', False),
                    is_target=suspect_data.get('is_target', False),
                    risk_level=suspect_data['risk_level'],
                    folder_name=suspect_data['folder_name'],
                    criminal_record=suspect_data['criminal_record'],
                    features=suspect_data['features'],
                    notes=suspect_data.get('notes', '')
                )
                
                self._upsert_suspect_profile(profile)
            
            self._update_system_metadata('total_suspects', len(suspects_data))
            logger.info(f"✅ 설정 파일에서 {len(suspects_data)}명의 용의자 정보 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {e}")
            self._create_default_suspects()
    
    def _create_default_suspects(self):
        """기본 용의자 데이터 생성"""
        default_suspects = [
            SuspectProfile(
                id="1", name="황윤하", name_en="hwang_yunha", age=37, gender="female",
                occupation="백수", role="thief", is_criminal=True, is_target=True,
                risk_level="high", folder_name="hwang_yunha",
                criminal_record=["절도 5회"], 
                features={"hair_style": "앞머리", "gender": "여성"},
                notes="절도범, 주요 타겟"
            ),
            SuspectProfile(
                id="2", name="순대국", name_en="sundaeguk", age=54, gender="female",
                occupation="쉐프", role="civilian", is_criminal=False, is_target=False,
                risk_level="low", folder_name="sundaeguk",
                criminal_record=["새치기 23회"],
                features={"facial_features": "다듬지 않은 눈썹", "occupation": "쉐프"},
                notes="일반인"
            ),
            SuspectProfile(
                id="3", name="하니짱", name_en="hanijjang", age=28, gender="male",
                occupation="간호사", role="civilian", is_criminal=False, is_target=False,
                risk_level="low", folder_name="hanijjang",
                criminal_record=["골목길 무단횡단"],
                features={"hair_style": "짧은 머리", "occupation": "간호사"},
                notes="일반인"
            ),
            SuspectProfile(
                id="4", name="이지선", name_en="leejisun", age=39, gender="female",
                occupation="운동선수", role="civilian", is_criminal=False, is_target=False,
                risk_level="low", folder_name="leejisun",
                criminal_record=["밥도둑"],
                features={"hair_style": "흑발 긴머리", "occupation": "운동선수"},
                notes="일반인"
            )
        ]
        
        for profile in default_suspects:
            self._upsert_suspect_profile(profile)
        
        self._update_system_metadata('total_suspects', len(default_suspects))
        logger.info("✅ 기본 용의자 데이터 생성 완료")
    
    def _upsert_suspect_profile(self, profile: SuspectProfile):
        """용의자 프로필을 데이터베이스에 삽입 또는 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO suspects 
                    (id, name, name_en, age, gender, occupation, role, is_criminal, is_target,
                     risk_level, folder_name, criminal_record_json, features_json, notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.id, profile.name, profile.name_en, profile.age, profile.gender,
                    profile.occupation, profile.role, profile.is_criminal, profile.is_target,
                    profile.risk_level, profile.folder_name,
                    json.dumps(profile.criminal_record, ensure_ascii=False),
                    json.dumps(profile.features, ensure_ascii=False),
                    profile.notes, datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ 용의자 프로필 저장 실패: {e}")
            raise e
    
    def add_face_embedding(self, embedding_data: EmbeddingData) -> int:
        """얼굴 임베딩을 데이터베이스에 추가"""
        try:
            # 이미지 해시 생성 (중복 방지)
            image_hash = self._generate_image_hash(embedding_data.image_path)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 중복 확인
                cursor.execute("SELECT id FROM face_embeddings WHERE image_hash = ?", (image_hash,))
                if cursor.fetchone():
                    logger.warning(f"⚠️ 이미지가 이미 존재합니다: {embedding_data.image_path}")
                    return None
                
                # 임베딩 벡터를 바이너리로 변환
                embedding_blob = embedding_data.embedding_vector.astype(np.float32).tobytes()
                
                # 바운딩 박스와 랜드마크를 JSON으로 변환
                bbox_json = json.dumps(embedding_data.detection_bbox) if embedding_data.detection_bbox else None
                landmarks_json = json.dumps(embedding_data.landmarks) if embedding_data.landmarks else None
                
                cursor.execute("""
                    INSERT INTO face_embeddings 
                    (suspect_id, embedding_vector, image_path, image_hash, confidence_score,
                     detection_bbox, landmarks_json, model_version, processing_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    embedding_data.suspect_id, embedding_blob, embedding_data.image_path,
                    image_hash, embedding_data.confidence_score, bbox_json, landmarks_json,
                    embedding_data.model_version, embedding_data.processing_time_ms
                ))
                
                embedding_id = cursor.lastrowid
                conn.commit()
                
                # 총 임베딩 수 업데이트
                cursor.execute("SELECT COUNT(*) FROM face_embeddings")
                total_embeddings = cursor.fetchone()[0]
                self._update_system_metadata('total_embeddings', total_embeddings)
                
                logger.info(f"✅ 임베딩 추가 완료: {embedding_data.suspect_id} (ID: {embedding_id})")
                return embedding_id
                
        except Exception as e:
            logger.error(f"❌ 임베딩 추가 실패: {e}")
            return None
    
    def find_matching_face(self, query_embedding: np.ndarray, 
                          target_suspect_id: Optional[str] = None,
                          threshold: float = None) -> Optional[DetectionResult]:
        """쿼리 임베딩과 가장 유사한 얼굴 찾기"""
        if threshold is None:
            threshold = self.similarity_threshold
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 검색 쿼리 구성
                if target_suspect_id:
                    sql_query = """
                        SELECT s.id, s.name, s.is_criminal, s.risk_level, 
                               fe.embedding_vector, fe.confidence_score, fe.id
                        FROM suspects s
                        JOIN face_embeddings fe ON s.id = fe.suspect_id
                        WHERE s.id = ? AND s.is_active = TRUE
                    """
                    params = (target_suspect_id,)
                else:
                    sql_query = """
                        SELECT s.id, s.name, s.is_criminal, s.risk_level,
                               fe.embedding_vector, fe.confidence_score, fe.id
                        FROM suspects s
                        JOIN face_embeddings fe ON s.id = fe.suspect_id
                        WHERE s.is_active = TRUE
                        ORDER BY s.is_criminal DESC
                    """
                    params = ()
                
                cursor.execute(sql_query, params)
                results = cursor.fetchall()
                
                best_match = None
                best_similarity = 0.0
                
                # 각 임베딩과 유사도 계산
                for row in results:
                    suspect_id, name, is_criminal, risk_level, embedding_blob, confidence, embedding_id = row
                    
                    # 임베딩 벡터 복원
                    stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # 코사인 유사도 계산
                    similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                    
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = DetectionResult(
                            suspect_id=suspect_id,
                            name=name,
                            similarity_score=similarity,
                            confidence_score=confidence,
                            bbox=(0, 0, 0, 0),  # 실제 구현시 채움
                            is_criminal=bool(is_criminal),
                            risk_level=risk_level,
                            timestamp=datetime.now(),
                            alert_triggered=bool(is_criminal) and similarity > 0.8
                        )
                
                return best_match
                
        except Exception as e:
            logger.error(f"❌ 얼굴 매칭 실패: {e}")
            return None
    
    def log_detection(self, detection_result: DetectionResult, 
                     camera_id: str = "main_camera",
                     processing_time_ms: int = 0):
        """감지 결과를 로그에 기록"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                bbox_json = json.dumps(detection_result.bbox)
                
                cursor.execute("""
                    INSERT INTO detection_logs
                    (suspect_id, similarity_score, confidence_score, detection_bbox,
                     camera_id, frame_timestamp, processing_time_ms, alert_triggered)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    detection_result.suspect_id, detection_result.similarity_score,
                    detection_result.confidence_score, bbox_json, camera_id,
                    detection_result.timestamp.isoformat(), processing_time_ms,
                    detection_result.alert_triggered
                ))
                
                conn.commit()
                logger.info(f"✅ 감지 로그 기록: {detection_result.name} (유사도: {detection_result.similarity_score:.3f})")
                
        except Exception as e:
            logger.error(f"❌ 감지 로그 기록 실패: {e}")
    
    def get_suspects_info(self, active_only: bool = True) -> List[Dict]:
        """모든 용의자 정보 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                sql_query = """
                    SELECT id, name, name_en, age, gender, occupation, role, 
                           is_criminal, risk_level, criminal_record_json, features_json
                    FROM suspects
                """
                if active_only:
                    sql_query += " WHERE is_active = TRUE"
                
                cursor.execute(sql_query)
                results = cursor.fetchall()
                
                suspects = []
                for row in results:
                    suspect = {
                        'id': row[0],
                        'name': row[1], 
                        'name_en': row[2],
                        'age': row[3],
                        'gender': row[4],
                        'occupation': row[5],
                        'role': row[6],
                        'is_criminal': bool(row[7]),
                        'risk_level': row[8],
                        'criminal_record': json.loads(row[9]) if row[9] else [],
                        'features': json.loads(row[10]) if row[10] else {}
                    }
                    suspects.append(suspect)
                
                return suspects
                
        except Exception as e:
            logger.error(f"❌ 용의자 정보 조회 실패: {e}")
            return []
    
    def get_detection_stats(self, hours: int = 24) -> Dict:
        """감지 통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 최근 N시간 내 감지 통계
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_detections,
                        COUNT(DISTINCT suspect_id) as unique_suspects,
                        SUM(CASE WHEN alert_triggered = 1 THEN 1 ELSE 0 END) as alerts,
                        AVG(similarity_score) as avg_similarity,
                        MAX(similarity_score) as max_similarity
                    FROM detection_logs 
                    WHERE created_at >= datetime('now', '-{} hours')
                """.format(hours))
                
                result = cursor.fetchone()
                
                stats = {
                    'total_detections': result[0] or 0,
                    'unique_suspects': result[1] or 0, 
                    'alerts_triggered': result[2] or 0,
                    'average_similarity': round(result[3] or 0, 3),
                    'max_similarity': round(result[4] or 0, 3),
                    'time_period_hours': hours
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ 감지 통계 조회 실패: {e}")
            return {}
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        try:
            # L2 정규화
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # 코사인 유사도
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"❌ 유사도 계산 실패: {e}")
            return 0.0
    
    def _generate_image_hash(self, image_path: str) -> str:
        """이미지 경로와 크기를 기반으로 해시 생성"""
        try:
            if os.path.exists(image_path):
                stat = os.stat(image_path)
                content = f"{image_path}_{stat.st_size}_{stat.st_mtime}"
            else:
                content = f"{image_path}_{time.time()}"
            
            return hashlib.md5(content.encode()).hexdigest()
            
        except Exception:
            return hashlib.md5(f"{image_path}_{time.time()}".encode()).hexdigest()
    
    def _update_system_metadata(self, key: str, value: Union[str, int, float]):
        """시스템 메타데이터 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE system_metadata 
                    SET value = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE key = ?
                """, (str(value), key))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ 메타데이터 업데이트 실패: {e}")
    
    def cleanup_old_logs(self, days: int = 30):
        """오래된 로그 정리"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM detection_logs 
                    WHERE created_at < datetime('now', '-{} days')
                """.format(days))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"✅ {deleted_count}개의 오래된 로그 정리 완료")
                
        except Exception as e:
            logger.error(f"❌ 로그 정리 실패: {e}")

# 편의 함수들
def create_embedding_database(db_path: str = None) -> ModernEmbeddingDB:
    """임베딩 데이터베이스 인스턴스 생성"""
    if db_path is None:
        db_path = "data/embeddings/face_recognition.db"
    return ModernEmbeddingDB(db_path)

if __name__ == "__main__":
    # 테스트 코드
    print("🚀 Modern Embedding Database 테스트")
    
    # 데이터베이스 생성
    db = create_embedding_database("data/embeddings/test_face_db.db")
    
    # 용의자 정보 조회
    suspects = db.get_suspects_info()
    print(f"📊 등록된 용의자: {len(suspects)}명")
    
    for suspect in suspects:
        print(f"  - {suspect['name']} ({suspect['age']}세, {suspect['gender']}, {'범죄자' if suspect['is_criminal'] else '일반인'})")
    
    # 감지 통계
    stats = db.get_detection_stats()
    print(f"📈 감지 통계: {stats}")
    
    print("✅ 테스트 완료!")
