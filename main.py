import cv2
from pathlib import Path


class VideoUtils:
    def __init__(self, path_captures: Path) -> None:
        self.path_captures = path_captures
        self.path_captures.mkdir(exist_ok=True)
        self.capture_count = 0

    def stream_and_capture(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("No se pudo acceder a la cámara")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                self.capture_count += 1
                path_img = self.path_captures / f"{self.capture_count:03d}.jpg"
                cv2.imwrite(str(path_img), frame)
                print(f"[✓] Imagen guardada: {path_img}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    path_data = Path("data")
    path_data.mkdir(exist_ok=True)

    path_captures = path_data / "captures"
    path_captures.mkdir(exist_ok=True)

    vu = VideoUtils(path_captures)
    vu.stream_and_capture()
