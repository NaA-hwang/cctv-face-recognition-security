# src/utils/gallery_loader.py
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Optional

def load_gallery(emb_dir: Path, use_bank: bool = True) -> Dict[str, np.ndarray]:
    """
    갤러리 로드 (bank 우선, 없으면 centroid)
    
    Args:
        emb_dir: 임베딩 파일이 있는 디렉토리
        use_bank: True면 bank 파일 우선 로드, False면 centroid만 로드
    
    Returns:
        gallery: {person_id: embedding} 또는 {person_id: bank_array}
        use_bank=True일 때: bank가 있으면 (N, 512), 없으면 (512,)
    """
    gallery = {}
    
    # 먼저 모든 person_id 목록 수집 (중복 방지)
    person_ids = set()
    
    for npy_path in emb_dir.glob("*.npy"):
        person_id = npy_path.stem
        
        # _bank, _centroid 접미사 제거
        if person_id.endswith("_bank"):
            person_id = person_id[:-5]
        elif person_id.endswith("_centroid"):
            person_id = person_id[:-9]
        
        person_ids.add(person_id)
    
    # 각 person_id에 대해 bank 또는 centroid 로드
    for person_id in person_ids:
        if use_bank:
            # Bank 파일 우선 확인
            bank_path = emb_dir / f"{person_id}_bank.npy"
            if bank_path.exists():
                bank = np.load(bank_path)  # (N, 512)
                # Bank 내 각 임베딩 정규화
                norms = np.linalg.norm(bank, axis=1, keepdims=True) + 1e-6
                bank = bank / norms
                gallery[person_id] = bank
                continue
        
        # Centroid 파일 확인
        centroid_path = emb_dir / f"{person_id}_centroid.npy"
        legacy_path = emb_dir / f"{person_id}.npy"
        
        if centroid_path.exists():
            emb = np.load(centroid_path)
        elif legacy_path.exists():
            emb = np.load(legacy_path)
        else:
            continue
        
        # 정규화
        emb = emb.astype("float32")
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        gallery[person_id] = emb
    
    return gallery

def match_with_bank(face_emb: np.ndarray, gallery: Dict[str, np.ndarray]) -> Tuple[Optional[str], float]:
    """
    Bank 또는 centroid 기반 매칭
    
    Args:
        face_emb: 얼굴 임베딩 벡터 (512,)
        gallery: load_gallery()로 로드한 갤러리 딕셔너리
    
    Returns:
        (best_person_id, best_similarity)
    """
    face_emb = face_emb.astype("float32")
    face_emb = face_emb / (np.linalg.norm(face_emb) + 1e-6)
    
    best_person = None
    best_sim = -1.0
    
    for person_id, ref_data in gallery.items():
        if ref_data.ndim == 2:  # Bank: (N, 512)
            # 행렬 곱으로 모든 임베딩과 유사도 계산
            sims = ref_data @ face_emb  # (N,)
            sim = float(np.max(sims))  # 최대 유사도
        else:  # Centroid: (512,)
            sim = float(np.dot(ref_data, face_emb))
        
        if sim > best_sim:
            best_sim = sim
            best_person = person_id
    
    return best_person, best_sim




