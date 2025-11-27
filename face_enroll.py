# src/face_enroll.py
"""
얼굴 임베딩 추출 스크립트
프로젝트 폴더의 정면 사진에서 얼굴 임베딩을 추출하여 bank/centroid 생성

주요 기능:
1. 기본 등록: enroll 폴더에서 이미지 읽어 bank/centroid 생성
2. 수동 추가: 특정 이미지 폴더나 파일들을 기존 bank에 추가

참고: 영상에서 임베딩 수집은 face_match_cctv.py에서 처리합니다.
"""
import sys  # 파이썬 인터프리터 관련 기능(경로, argv 등)을 다루는 표준 라이브러리

from pathlib import Path  # 경로를 객체 지향적으로 다루기 위한 Path 클래스 로드.

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent  # PROJECT_ROOT = ...: 현재 파일의 상위 상위 폴더를 프로젝트 루트로 계산.현재 파일 기준 상위 두 단계(프로젝트 루트) 계산함 

sys.path.insert(0, str(PROJECT_ROOT))   #  sys.path.insert(...): 해당 경로를 모듈 탐색 경로 최우선으로 추가해 내부 모듈 임포트 가능하게 함

# CUDA 경로를 먼저 설정
from src.utils.device_config import _ensure_cuda_in_path  # CUDA DLL 경로 확보 유틸
_ensure_cuda_in_path()  # CUDA 관련 환경 변수를 선행 설정

from insightface.app import FaceAnalysis  # 얼굴 검출/정렬/임베딩 등 핵심 기능을 제공하는 클래스
import cv2  # OpenCV: 이미지 로딩/처리/변환을 위한 컴퓨터 비전 라이브러리
import numpy as np  # 벡터/행렬 연산을 위한 NumPy
from src.utils.device_config import get_device_id, safe_prepare_insightface  # 디바이스 선택 및 InsightFace 초기화 보조 함수 임포트.

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # 허용할 이미지 확장자 집합


def l2_normalize(vec: np.ndarray) -> np.ndarray:  # def l2_normalize(...): 벡터 길이를 1로 만드는 함수 정의.
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)  # 벡터의 L2 노름(길이) 계산
    if norm == 0:   # 영벡터인 경우
        return vec  # 그대로 반환하여 0으로 나눔 방지
    return vec / norm  # 그 외에는 길이 1로 정규화된 벡터 반환


def get_main_face_embedding(app: FaceAnalysis, img_path: Path) -> np.ndarray | None:
    """이미지에서 가장 큰 얼굴 한 개의 임베딩을 반환 (측면 얼굴 감지 개선)"""
    img = cv2.imread(str(img_path))  # 파일 경로를 문자열로 변환해 이미지 로드
    if img is None:                  # 로딩 실패 시
        print(f"  ⚠️ 이미지 읽기 실패: {img_path}")
        return None                  # 경고 출력 후 None

    # 먼저 원본 이미지로 시도
    faces = app.get(img)  # InsightFace로 얼굴 감지 및 임베딩 산출
    
    # 얼굴을 찾지 못한 경우, 이미지 전처리 후 재시도
    if len(faces) == 0:
        # 이미지 밝기/대비 조정
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # LAB 색공간으로 변환
        l, a, b = cv2.split(lab)                    # 밝기(L)와 색상 채널 분리
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE 객체 생성
        l = clahe.apply(l)                         # 밝기 채널에 국부 대비 향상 적용
        enhanced = cv2.merge([l, a, b])            # 강화된 L과 기존 a/b를 합쳐 LAB 이미지 구성
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)  # 다시 BGR 공간으로 변환
        
        # 전처리된 이미지로 재시도
        faces = app.get(enhanced)  # 보정정 이미지에서 얼굴 재검출
        
        # 여전히 실패하면 업스케일링 후 재시도
        if len(faces) == 0:
            h, w = img.shape[:2]      # 원본 높이/너비 추출
            if h < 1280 or w < 1280:  # 둘 중 하나라도 기준보다 작으면
                scale = max(1280 / h, 1280 / w)  # 1280 이상이 되도록 스케일 계산
                new_h, new_w = int(h * scale), int(w * scale)  # 업스케일 후 크기
                upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)  # 부드럽게 확대
                faces = app.get(upscaled)  # 고해상 이미지로 다시 감지
    
    if len(faces) == 0:
        print(f"  ⚠️ 얼굴 미검출: {img_path}")
        return None

    # 가장 큰 얼굴 선택
    # faces_sorted = sorted(...): 감지된 얼굴을 면적 기준 내림차순 정렬.
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    main_face = faces_sorted[0]
    emb = main_face.embedding.astype("float32")  # InsightFace가 준 임베딩을 float32로 변환
    emb = l2_normalize(emb)  # 임베딩을 L2 정규화해 비교 안정화
    return emb  # 최종 임베딩 반환


def save_embeddings(person_id: str, emb_list: list[np.ndarray], out_dir: Path, 
                   save_bank: bool = True, save_centroid: bool = True):
    """
    임베딩 리스트를 bank_base와 centroid_base로 저장 (사람별 폴더 구조)
    
    Enrollment 시에는 기준 Bank(base)만 생성하고, dynamic은 생성하지 않습니다.
    
    Args:
        person_id: 사람 ID
        emb_list: 임베딩 리스트
        out_dir: 저장 디렉토리 (예: outputs/embeddings)
        save_bank: bank 저장 여부
        save_centroid: centroid 저장 여부
    """
    if not emb_list:  # 비어 있으면 아무 것도 저장하지 않음
        return
    
    embs = np.stack(emb_list, axis=0)  # np.stack(...): 리스트를 행렬로 변환.(N, 512) 형태로 임베딩들을 하나의 배열로 변환
    centroid = embs.mean(axis=0)       # (512,) 평균 벡터 계산
    centroid = l2_normalize(centroid)  # 계산된 평균 벡터 정규화
    
    # 사람별 폴더 생성
    person_dir = out_dir / person_id               # 사람람별 저장 경로
    person_dir.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성
    
    if save_bank:
        # 얼굴 인식 시스템에서 추출한 임베딩(특징 벡터)을 bank_base.npy로로 저장 (기준 Bank, read-only)
        bank_base_path = person_dir / "bank_base.npy"  # base bank 저장 경로
        np.save(bank_base_path, embs)                  # 고정 기준 bank 저장
        print(f"     Base Bank 저장: {bank_base_path} ({embs.shape[0]}개 임베딩)")
        
        # Backward compatibility: 기존 bank.npy가 없으면 생성 (legacy 지원)
        legacy_bank_path = person_dir / "bank.npy"  # 예전 구조에서 사용하던 파일명
        if not legacy_bank_path.exists():     # 기존 파일이 없다면
            np.save(legacy_bank_path, embs)   # 동일한 데이터를 legacy 이름으로 저장
            print(f"     Legacy Bank 저장 (backward compatibility): {legacy_bank_path}")
    
    if save_centroid:
        # centroid_base.npy 저장 (기준 Centroid, read-only)
        centroid_base_path = person_dir / "centroid_base.npy"  # 기준 centroid 경로
        np.save(centroid_base_path, centroid)                  # 고정 centroid 저장
        print(f"     Base Centroid 저장: {centroid_base_path}")
        
        # Backward compatibility: 기존 centroid.npy가 없으면 생성 (legacy 지원)
        legacy_centroid_path = person_dir / "centroid.npy"  # 예전 파일명
        if not legacy_centroid_path.exists():               # legacy 파일이 없을 때만 생성
            np.save(legacy_centroid_path, centroid)         # 호환성을 위해 저장
            print(f"     Legacy Centroid 저장 (backward compatibility): {legacy_centroid_path}")
    
    print(f"     L2 norm: {np.linalg.norm(centroid):.4f}")  # 정규화 상태를 확인차 출력


# ===== MODE 1: 기본 등록 =====
def mode_basic_enroll(app: FaceAnalysis, enroll_root: Path, out_dir: Path, 
                     save_bank: bool = True, save_centroid: bool = True):
    """
    enroll 폴더에서 모든 사람의 이미지를 읽어 bank/centroid 생성
    
    Args:
        app: FaceAnalysis 인스턴스
        enroll_root: enroll 폴더 경로 (예: images/enroll)
        out_dir: 출력 디렉토리 (예: outputs/embeddings)
        save_bank: bank 저장 여부
        save_centroid: centroid 저장 여부
    """
    print(f"{'='*70}")  # 구분선 출력
    print(f"📝 MODE 1: 기본 등록 (Basic Enrollment)")  # 현재 모드 안내
    print(f"{'='*70}")
    print(f"   입력 폴더: {enroll_root}")  # 입력 이미지 경로 출력
    print(f"   출력 폴더: {out_dir}")  # 임베딩 저장 위치 출력
    print()
    
    if not enroll_root.exists():  # enroll 폴더가 없으면 즉시 에러
        raise FileNotFoundError(f"enroll 폴더를 찾을 수 없음: {enroll_root}")
    
    person_dirs = [p for p in enroll_root.iterdir() if p.is_dir()]  # 각 사람별 하위 폴더 수집
    if not person_dirs:  # 한 명도 없을 때 경고 후 종료
        print(f"⚠️ {enroll_root} 안에 사람별 폴더가 없습니다.")
        return
    
    print("👥 등록 대상 사람 목록:")
    for d in person_dirs:  # 폴더명을 통해 등록 대상 출력
        print(f"  - {d.name}")
    print()
    
    for person_dir in person_dirs:  # 각 사람 폴더 순회
        person_id = person_dir.name  # 폴더명을 ID로 사용
        print(f"\n===== {person_id} 등록 시작 =====")
        
        emb_list = []  # 해당 사람의 임베딩 누적 리스트
        for img_path in sorted(person_dir.glob("*")):  # 폴더 내 파일들을 정렬 후 순회
            if img_path.suffix.lower() not in IMG_EXTS:  # 이미지 확장자 아니면 건너뜀
                continue
            
            print(f"  ▶ 이미지 처리: {img_path.name}")  # 현재 처리 중인 이미지 출력
            emb = get_main_face_embedding(app, img_path)  # 대표 얼굴 임베딩 추출
            if emb is None:  # 얼굴 없으면 그냥 다음 이미지
                continue
            emb_list.append(emb)  # 얻은 임베딩을 리스트에 추가
        
        if not emb_list:  # 유효한 임베딩이 하나도 없을 때
            print(f"  ❌ 유효한 얼굴 임베딩 없음 → {person_id} 스킵")
            continue
        
        print(f"  ✅ {person_id} 등록 완료 ({len(emb_list)}장 사용)")  # 몇 장이 사용됐는지 보고
        save_embeddings(person_id, emb_list, out_dir, save_bank, save_centroid)  # 저장 함수 호출
    
    print(f"\n🎉 기본 등록 완료!")  # 전체 모드 완료 안내


# ===== MODE 2: 수동 추가 =====
def mode_manual_add(app: FaceAnalysis, person_id: str, image_paths: list[Path],
                   out_dir: Path, similarity_threshold: float = 0.95):
    """
    특정 이미지들을 bank에 수동으로 추가
    
    Args:
        app: FaceAnalysis 인스턴스
        person_id: 사람 ID
        image_paths: 추가할 이미지 경로 리스트
        out_dir: 출력 디렉토리
        similarity_threshold: 중복 체크 임계값
    
    Returns:
        추가된 임베딩 개수
    """
    print(f"{'='*70}")  # 구분선
    print(f"📁 MODE 2: 수동 추가 (Manual Add)")  # 수동 모드 안내
    print(f"{'='*70}")
    print(f"   대상 인물: {person_id}")  # 추가 대상 ID 출력
    print(f"   이미지 개수: {len(image_paths)}개")  # 총 처리 이미지 수
    print(f"   중복 체크 임계값: {similarity_threshold}")  # 유사도 기준 노출
    print()
    
    # 사람별 폴더 우선, 없으면 루트에서 찾기
    person_dir = out_dir / person_id  # 기본적으로 사람별 폴더 우선
    bank_path = person_dir / "bank.npy"  # 최신 규격 위치
    if not bank_path.exists():  # 없을 경우 레거시 위치 시도
        bank_path = out_dir / f"{person_id}_bank.npy"
    
    # 기존 bank 로드
    if bank_path.exists():  # 기존 bank 존재 시
        bank = np.load(bank_path)  # numpy 배열로 로드
        print(f"📚 기존 bank: {bank.shape[0]}개 임베딩 ({bank_path})")
    else:
        bank = np.empty((0, 512), dtype=np.float32)  # 빈 bank 초기화
        print(f"📚 새 bank 생성")
    
    new_embeddings = []   # 새롭게 추가할 임베딩 누적 리스트
    skipped_count = 0     # 스킵된 이미지 수 집계용
    
    for img_path in image_paths:  # 입력 이미지 목록 순회
        if img_path.suffix.lower() not in IMG_EXTS:  # 이미지가 아니면 스킵
            continue
        
        print(f"  ▶ 처리 중: {img_path.name}")  # 현재 처리 파일 로그
        emb = get_main_face_embedding(app, img_path)  # ★★ 임베딩 추출
        
        if emb is None:         # 얼굴 미검출 시
            skipped_count += 1  # 스킵 카운트 증가
            continue
        
        # 중복 체크
        if bank.shape[0] > 0:                    # 기존 임베딩이 있으면,
            max_sim = float(np.max(bank @ emb))  # ★★ 코사인 유사도로 중복 검사, 최대값 (정규화 기반 내적)
            if max_sim >= similarity_threshold:  # ★★ 임계 이상이면 중복으로 판단
                print(f"     ⏭ 스킵 (기존 임베딩과 유사도 {max_sim:.3f} >= {similarity_threshold})")
                skipped_count += 1
                continue
        
        new_embeddings.append(emb)  # 새 임베딩 추가
        max_sim = float(np.max(bank @ emb)) if bank.shape[0] > 0 else 0.0  # 참고용 최대 유사도
        print(f"     ✅ 추가 (기존 bank와 최대 유사도: {max_sim:.3f})")
    
    if not new_embeddings:  # 단 하나도 추가되지 않았으면
        print(f"\n⚠️ 추가할 새로운 임베딩이 없습니다. (스킵: {skipped_count}개)")
        return 0
    
    # Bank에 추가
    new_embs_array = np.stack(new_embeddings, axis=0)  # 새 임베딩들을 배열로 변환
    updated_bank = np.vstack([bank, new_embs_array])   # 기존 bank와 새로운 배열을 위아래로 결합
    
    # Centroid 재계산
    updated_centroid = updated_bank.mean(axis=0)       # ★★ 같은 차원 값(열)끼리 평균 임베딩 계산
    updated_centroid = l2_normalize(updated_centroid)  # ★★ 벡터의 길이를 1로 만드는 정규화로 비교 안정성 확보(코사인 유사도)
    
    # 저장 (사람별 폴더에 저장)
    person_dir = out_dir / person_id  # 최종 저장 경로
    person_dir.mkdir(parents=True, exist_ok=True)      # 폴더가 없으면 생성
    
    bank_path_new = person_dir / "bank.npy"  # bank 저장 경로
    np.save(bank_path_new, updated_bank)     # 업데이트된 bank 저장
    
    centroid_path_new = person_dir / "centroid.npy"  # centroid 저장 경로
    np.save(centroid_path_new, updated_centroid)  # 업데이트된 centroid 저장
    
    print(f"\n✅ Bank 업데이트 완료!")  # 성공 메시지
    print(f"   추가된 임베딩: {len(new_embeddings)}개")  # 이번에 추가된 개수
    print(f"   총 임베딩 수: {updated_bank.shape[0]}개 (기존 {bank.shape[0]}개 + 신규 {len(new_embeddings)}개)")
    print(f"   저장 위치: {person_dir}")  # 저장된 디렉토리
    
    return len(new_embeddings)  # 호출자에게 추가된 개수 반환


def main():
    # ===== 설정 =====
    MODE = 1  # 1: 기본 등록, 2: 수동 추가
    
    enroll_root = Path("images") / "enroll"
    out_dir = Path("outputs") / "embeddings"
    
    # MODE 2 설정
    person_id = "hani"  # 대상 인물 ID
    image_folder = Path("images") / "extracted_frames" / person_id  # 수동 추가할 폴더
    
    print(f"{'='*70}")
    print(f"🎯 얼굴 임베딩 추출 시스템")
    print(f"{'='*70}")
    print(f"   모드: {MODE} (1: 기본 등록, 2: 수동 추가)")
    print(f"   출력 폴더: {out_dir}")
    print()
    
    # InsightFace 초기화
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    if actual_device_id != device_id:
        print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
    print()
    
    # 모드별 실행
    if MODE == 1:
        # 기본 등록: enroll 폴더에서 모든 사람 등록
        mode_basic_enroll(app, enroll_root, out_dir, save_bank=True, save_centroid=True)
    
    elif MODE == 2:
        # 수동 추가: 특정 폴더의 이미지들을 bank에 추가
        if not image_folder.exists():
            print(f"⚠️ 이미지 폴더를 찾을 수 없음: {image_folder}")
            return
        
        image_paths = [p for p in sorted(image_folder.glob("*")) 
                      if p.suffix.lower() in IMG_EXTS]
        
        if not image_paths:
            print(f"⚠️ {image_folder} 안에 이미지 파일이 없습니다.")
            return
        
        added_count = mode_manual_add(
            app=app,
            person_id=person_id,
            image_paths=image_paths,
            out_dir=out_dir,
            similarity_threshold=0.95
        )
        
        if added_count > 0:
            print(f"\n💡 다음 단계:")
            print(f"   python src/face_match_cctv.py 실행하여 인식 성능 확인")
    
    else:
        print(f"❌ 잘못된 모드: {MODE} (1 또는 2 중 선택)")
    
    print(f"\n{'='*70}")
    print(f"✅ 작업 완료!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

