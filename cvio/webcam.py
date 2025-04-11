from typing import Optional, List, Tuple
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from cv2.typing import MatLike

#"yolov8n.pt"
#"yolov8s.pt"
#"yolov8m.pt"
#"yolov8l.pt"
#"yolov8x.pt"
PATH_MODEL = Path("data/yolov8n.pt")
WEBCAM_TITLE = "Webcam"
LEN_LIST = 5  # Máximo de puntos de referencia
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)


def label_from_cls_id(*, cls_id: int) -> Optional[str]:
    if cls_id == 0:
        return "person"
    elif cls_id == 15:
        return "cat"
    return None


def obtain_keypoint(*, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    cx = (x1 + x2) // 2
    cy = y2 - (y2 - y1) // 10
    return cx, cy


class WebcamYOLO:
    def __init__(self, path_captures: Path) -> None:
        self.path_captures = path_captures
        self.path_captures.mkdir(exist_ok=True)
        self.capture_count = 0
        self.model = YOLO(PATH_MODEL)

        self.ref_points: List[Tuple[int, int]] = []
        self.ref_points_arr: Optional[np.ndarray] = None
        self.drawing_enabled = True

    def detect_objects(self, *, frame: MatLike) -> None:
        results: Results = self.model(frame, verbose=False)[0]
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            label = label_from_cls_id(cls_id=int(cls))
            if label is not None:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = obtain_keypoint(x1=x1, y1=y1, x2=x2, y2=y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
                cv2.circle(frame, (cx, cy), 6, BLUE, -1)

    def click_event(self, event, x: int, y: int, flags, param) -> None:
        if self.drawing_enabled and event == cv2.EVENT_LBUTTONDOWN:
            self.ref_points.append((x, y))
            print(f"Referencia añadida: ({x}, {y})")
            if len(self.ref_points) == LEN_LIST:
                self.drawing_enabled = False
                self.ref_points_arr = np.array(self.ref_points, dtype=np.float32)
                print("Se alcanzó el límite de puntos de referencia.")

    def run_webcam(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("No se pudo acceder a la cámara")

        cv2.namedWindow(WEBCAM_TITLE)
        cv2.setMouseCallback(WEBCAM_TITLE, self.click_event)

        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            self.detect_objects(frame=frame)
            if self.ref_points_arr is not None:
                pass

            for (x, y) in self.ref_points:
                cv2.circle(frame, (x, y), 5, YELLOW, -1)

            # Se grafica la imagen.
            cv2.imshow(WEBCAM_TITLE, frame)

            # Se verifican las presiones de teclas.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False
            elif key == ord("c"):
                self.capture_count += 1
                path_img = self.path_captures / f"{self.capture_count:03d}.jpg"
                cv2.imwrite(str(path_img), frame)
                print(f"Save img: {path_img}")

        cap.release()
        cv2.destroyAllWindows()
