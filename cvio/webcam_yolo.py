from typing import Optional, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

import cv2
from cv2.typing import MatLike
import numpy as np

from cvio.yolo import YOLOWrapper, BoxYOLO
from cvio.typings import Color
from cvio.plot_cv2 import PlotCV2

#"yolov8n.pt"
#"yolov8s.pt"
#"yolov8m.pt"
#"yolov8l.pt"
#"yolov8x.pt"
PATH_MODEL = Path("data/yolov8l.pt")
WEBCAM_TITLE = "Webcam"
LEN_LIST = 5  # Máximo de puntos de referencia

# Colors.
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)

def get_key_pressed() -> Optional[str]:
    key = cv2.waitKey(1) & 0xFF
    return chr(key) if key != 255 else None


class BaseWebcamYOLO(ABC):
    def __init__(self, *, path_captures: Path) -> None:
        self.running = False
        self.path_captures = path_captures
        self.path_captures.mkdir(exist_ok=True)
        self.capture_count = 0

        self.yolo_wrapper = YOLOWrapper(path_model=PATH_MODEL)
        self.ref_points: List[Tuple[int, int]] = []
        self.ref_points_arr: Optional[np.ndarray] = None
        self.drawing_enabled = True

    def click_event(self, event, x: int, y: int, flags, param) -> None:
        if self.drawing_enabled and event == cv2.EVENT_LBUTTONDOWN:
            self.ref_points.append((x, y))
            print(f"Referencia añadida: ({x}, {y})")
            if len(self.ref_points) == LEN_LIST:
                self.drawing_enabled = False
                self.ref_points_arr = np.array(self.ref_points, dtype=np.float32)
                print("Se alcanzó el límite de puntos de referencia.")

    def plot_box_yolo(self, *, frame: MatLike, box_yolo: BoxYOLO, color: Color) -> None:
        PlotCV2.box(
            frame=frame,
            label=box_yolo.label,
            x1=box_yolo.x1,
            y1=box_yolo.y1,
            x2=box_yolo.x2,
            y2=box_yolo.y2,
            color=color
        )

    def stop_running(self, *, frame: MatLike) -> None:
        self.running = False

    def take_capture(self, *, frame: MatLike) -> None:
        self.capture_count += 1
        path_img = self.path_captures / f"{self.capture_count:03d}.jpg"
        cv2.imwrite(str(path_img), frame)
        print(f"Save img: {path_img}")

    def run_webcam(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("No se pudo acceder a la cámara")

        cv2.namedWindow(WEBCAM_TITLE)
        cv2.setMouseCallback(WEBCAM_TITLE, self.click_event)

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self.process_frame(frame=frame)

            # Se grafica la imagen.
            cv2.imshow(WEBCAM_TITLE, frame)

            actions_key_pressed = {
                "q": self.stop_running,
                "c": self.take_capture
            }
            # Se verifican las presiones de teclas.
            key = get_key_pressed()
            if key in actions_key_pressed:
                actions_key_pressed[key](frame=frame)

        cap.release()
        cv2.destroyAllWindows()

    @abstractmethod
    def process_frame(self, *, frame: MatLike) -> None:
        ...



class WebcamYOLO(BaseWebcamYOLO):
    def __init__(self, *, path_captures):
        super().__init__(path_captures=path_captures)

    def process_frame(self, *, frame: MatLike) -> None:
        boxes_yolo = self.yolo_wrapper.detect_objects(frame=frame)
        for box_yolo in boxes_yolo:
            self.plot_box_yolo(frame=frame, box_yolo=box_yolo, color=GREEN)
            PlotCV2.point(frame=frame, x=box_yolo.cx, y=box_yolo.cy, color=BLUE, radius=6)

        if self.ref_points_arr is not None:
            pass

        for (x, y) in self.ref_points:
            cv2.circle(frame, (x, y), 5, YELLOW, -1)