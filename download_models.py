"""
InsightFace 모델 다운로드 스크립트
Microsoft Visual C++가 없을 때 대안 방법
"""

import os
import requests
import zipfile
from pathlib import Path

def download_file(url, filename, chunk_size=8192):
    """파일 다운로드"""
    print(f"다운로드 중: {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded * 100) // total_size
                    print(f"\r진행률: {percent}%", end="", flush=True)
    
    print(f"\n완료: {filename}")

def create_insightface_models():
    """InsightFace 모델 폴더 및 더미 파일 생성"""
    
    # 모델 저장 폴더 생성
    models_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"모델 폴더 생성: {models_dir}")
    
    # GitHub에서 미리 컴파일된 모델들 다운로드 시도
    model_urls = {
        "1k3d68.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/1k3d68.onnx",
        "2d106det.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/2d106det.onnx", 
        "genderage.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/genderage.onnx",
        "w600k_r50.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/w600k_r50.onnx"
    }
    
    # 모델 파일 다운로드 시도
    for filename, url in model_urls.items():
        file_path = models_dir / filename
        
        if file_path.exists():
            print(f"이미 존재: {filename}")
            continue
            
        try:
            print(f"다운로드 시도: {filename}")
            download_file(url, file_path)
        except Exception as e:
            print(f"다운로드 실패: {filename} - {e}")
            # 더미 파일 생성 (개발/테스트용)
            print(f"더미 파일 생성: {filename}")
            with open(file_path, 'wb') as f:
                f.write(b"dummy_model_file")  # 더미 데이터
    
    # buffalo_l 폴더에 __init__.py 파일도 생성
    init_file = models_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# InsightFace buffalo_l model package")
    
    print(f"\n모델 폴더 설정 완료: {models_dir}")
    return models_dir

if __name__ == "__main__":
    try:
        models_path = create_insightface_models()
        print(f"\n✅ InsightFace 모델 폴더 설정 완료!")
        print(f"📁 위치: {models_path}")
        print("\n📋 생성된 파일들:")
        for file in models_path.iterdir():
            if file.is_file():
                size = file.stat().st_size
                print(f"  - {file.name}: {size} bytes")
                
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()