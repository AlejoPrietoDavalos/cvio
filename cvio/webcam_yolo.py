from typing import Optional, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json

import cv2
from cv2.typing import MatLike
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np

from cvio.yolo import YOLOWrapper, BoxYOLO
from cvio.typings import Color
from cvio.plot_cv2 import PlotCV2

#"yolov8n.pt"
#"yolov8s.pt"
#"yolov8m.pt"
#"yolov8l.pt"
#"yolov8x.pt"
#{
#    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
#    7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
#    12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
#    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
#    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
#    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
#    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
#    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
#    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
#    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
#    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
#    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
#    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
#    78: 'hair drier', 79: 'toothbrush'
#}
PATH_MODEL = Path("data/yolov8l.pt")
WEBCAM_TITLE = "Webcam"
NUM_POINTS_REF = 10                              # FIXME: Deshardcodear.

# Colors.
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)


def plot_box_yolo(*, frame: MatLike, box_yolo: BoxYOLO, color: Color) -> None:
    PlotCV2.box(
        frame=frame,
        label=f"id={box_yolo.id} - {box_yolo.label}",
        x1=box_yolo.x1,
        y1=box_yolo.y1,
        x2=box_yolo.x2,
        y2=box_yolo.y2,
        color=color
    )

def plot_boxes_yolo(
        *,
        frame: MatLike,
        boxes_yolo: List[BoxYOLO],
        color_box: Color,
        color_keypoint: Color = None,
        radius: int = 6
) -> None:
    for box_yolo in boxes_yolo:
        # Dibuja las cajas de los objetos detectados.
        plot_box_yolo(frame=frame, box_yolo=box_yolo, color=color_box)

        # Dibuja los key_points de las cajas.
        if color_keypoint is not None:
            PlotCV2.circle(frame=frame, x=box_yolo.cx, y=box_yolo.cy, color=color_keypoint, radius=radius)




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
        self.actions_key_pressed = {"q": self.stop_running, "c": self.take_capture, "r": self.toggle_recording}

        self.recording = False
        self.frames_to_save: List[MatLike] = []
        self.boxes_yolo_list: List[List[BoxYOLO]] = []

        self.points_ref_list: List[Tuple[int, int]] = []

    def add_reference_point(self, event, x: int, y: int, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points_ref_list) < NUM_POINTS_REF:
            self.points_ref_list.append((x, y))

    def stop_running(self, *, frame: MatLike) -> None:
        self.running = False

    def take_capture(self, *, frame: MatLike) -> None:
        self.capture_count += 1
        path_img = self.path_captures / f"{self.capture_count:03d}.jpg"
        cv2.imwrite(str(path_img), frame)
        print(f"Save img: {path_img}")

    def toggle_recording(self, *, frame: MatLike) -> None:
        if len(self.points_ref_list) < NUM_POINTS_REF:
            print(f"[INFO] Necesitás {NUM_POINTS_REF}, tenés {len(self.points_ref_list)}.")
            return None
        elif not self.recording:
            print("[INFO] Iniciando grabación...")
            self.recording = True
        else:
            print("[INFO] Finalizando grabación...")
            self.recording = False

            now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

            # Guardar video
            h, w = self.frames_to_save[0].shape[:2]
            fourcc = VideoWriter_fourcc(*'mp4v')
            path_video = self.path_captures / f"{now_str}.mp4"
            out = VideoWriter(str(path_video), fourcc, 20.0, (w, h))
            for f in self.frames_to_save:
                out.write(f)
            out.release()
            print(f"[INFO] Video guardado: {path_video}")

            # Guardar boxes_yolo en JSON
            path_points_ref = self.path_captures / f"{now_str}.json"
            with open(path_points_ref, "w") as f:
                json.dump(self.points_ref_list, f)

            # Limpio ejecución anterior.
            self.frames_to_save.clear()
            print(f"[INFO] Detecciones guardadas: {path_points_ref}")

    def run_webcam(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("No se pudo acceder a la cámara")

        cv2.namedWindow(WEBCAM_TITLE)
        cv2.setMouseCallback(WEBCAM_TITLE, self.add_reference_point)        # TODO: Abstraer.

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Se grafica la imagen.
            if self.recording:
                self.frames_to_save.append(frame.copy())
            
            self.process_frame(frame=frame)
            
            cv2.imshow(WEBCAM_TITLE, frame)

            # Se verifican las presiones de teclas.
            key = get_key_pressed()
            if key in self.actions_key_pressed:
                self.actions_key_pressed[key](frame=frame)
        cap.release()
        cv2.destroyAllWindows()

    @abstractmethod
    def process_frame(self, *, frame: MatLike) -> None:
        ...





class WebcamYOLO(BaseWebcamYOLO):
    def __init__(self, *, path_captures):
        super().__init__(path_captures=path_captures)

    def process_frame(self, *, frame: MatLike) -> None:
        # Detecta los objetos.
        boxes_yolo = self.yolo_wrapper.detect_objects(frame=frame)

        plot_boxes_yolo(frame=frame, boxes_yolo=boxes_yolo, color_box=GREEN, color_keypoint=BLUE)

        # Dibuja los puntos de referencia..
        for (x, y) in self.points_ref_list:
            PlotCV2.circle(frame=frame, x=x, y=y, color=YELLOW, radius=6)
