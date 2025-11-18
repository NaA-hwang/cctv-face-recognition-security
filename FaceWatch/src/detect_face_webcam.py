import cv2
from ultralytics import YOLO

MODEL_PATH = r"models\yolov12n-face.pt"


def main():
    # 1. 얼굴 전용 YOLO 모델 로드
    model = YOLO(MODEL_PATH)

    # 2. 웹캠 열기 (0번: 기본 카메라)
    cap = cv2.VideoCapture(0)  # 노트북 내장 카메라면 보통 0

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다. 다른 프로그램이 카메라를 쓰고 있는지 확인해줘.")
        return

    print("✅ 웹캠 시작! 창에서 'q' 키를 누르면 종료됩니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다. 종료합니다.")
            break

        # 3. YOLO로 얼굴 탐지 (실시간이니까 verbose=False로 조용히)
        results = model(frame, conf=0.35, save=False, verbose=False)

        # 4. 탐지 결과를 프레임에 그리기
        annotated_frame = results[0].plot()

        # 5. 화면에 출력
        cv2.imshow("FaceWatch - Webcam Face Detection", annotated_frame)

        # 6. 'q' 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 7. 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("👋 종료되었습니다.")


if __name__ == "__main__":
    main()
