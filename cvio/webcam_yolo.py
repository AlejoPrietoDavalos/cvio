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
PATH_MODEL = Path("data/yolov8l.pt")
WEBCAM_TITLE = "Webcam"
NUM_POINTS_REF = 10                              # FIXME: Deshardcodear.

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
        self.actions_key_pressed = {"q": self.stop_running, "c": self.take_capture, "r": self.toggle_recording}

        self.recording = False
        self.frames_to_save: List[MatLike] = []
        self.boxes_yolo_list: List[List[BoxYOLO]] = []

        self.points_ref_list: List[Tuple[int, int]] = []
        #points_real = np.array([
        #    [275, 375],             # P1
        #    [62.5, 273],            # P2
        #    [126.5, 273],           # P3
        #    [126.5, 213]            # P4
        #], dtype=np.float32)
        #points_ref = np.array(self.points_ref_list, dtype=np.float32)
        #self.homography = Homography(points_real=points_real, points_ref=points_ref)

    def add_reference_point(self, event, x: int, y: int, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.boxes_yolo_list) < NUM_POINTS_REF:
            self.points_ref_list.append((x, y))

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
            boxes_json_list = [[box.model_dump() for box in boxes_yolo] for boxes_yolo in self.boxes_yolo_list]
            path_boxes = self.path_captures / f"{now_str}.json"
            with open(path_boxes, "w") as f:
                json.dump({"points_ref_list": self.points_ref_list, "boxes_json_list": boxes_json_list}, f)

            # Limpio ejecución anterior.
            self.frames_to_save.clear()
            self.boxes_yolo_list.clear()
            print(f"[INFO] Detecciones guardadas: {path_boxes}")

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



def _process_frame_plot(self, *, frame: MatLike, boxes_yolo: List[BoxYOLO]) -> None:
    h1 = frame.shape[0]
    h2, w2 = self.plot_img.shape[:2]
    
    # Si 'plot_resized' es None, lo inicializamos
    if self.plot_resized is None:
        self.plot_resized = np.ones_like(self.plot_img) * 255  # Crear una copia blanca de la imagen base
    
    # Redimensionar 'plot_img' al tamaño de la ventana de la cámara
    self.plot_resized = cv2.resize(self.plot_img.copy(), (int(w2 * h1 / h2), h1))

    if self.homography.H is None or not boxes_yolo:
        return

    points = np.array([[box_yolo.cx, box_yolo.cy] for box_yolo in boxes_yolo], dtype=np.float32)
    projected = self.homography.project_points(points=points)

    scale = 3
    padding = 100
    max_y = 458 * scale

    # Asegúrate de que plot_resized es un np.ndarray adecuado para OpenCV
    if self.plot_resized is None:
        print("ERROR: plot_resized es None")
        return

    for idx, pt in enumerate(projected):
        x, y = pt * scale
        x += padding
        y = max_y - y + padding
        print(f"[INFO] Punto proyectado ajustado (plot): x={x:.2f}, y={y:.2f}")
        
        # Asegúrate de dibujar sobre la imagen numpy (en formato de imagen OpenCV)
        PlotCV2.circle(frame=self.plot_resized, x=int(x), y=int(y), color=YELLOW, radius=6)

        if idx == 0:
            PlotCV2.circle(frame=self.plot_resized, x=int(x), y=int(y), color=BLUE, radius=10)



class WebcamYOLO(BaseWebcamYOLO):
    def __init__(self, *, path_captures):
        super().__init__(path_captures=path_captures)

    def process_frame(self, *, frame: MatLike) -> None:
        # Detecta los objetos.
        boxes_yolo = self.yolo_wrapper.detect_objects(frame=frame)

        for box_yolo in boxes_yolo:
            # Dibuja las cajas de los objetos detectados.
            self.plot_box_yolo(frame=frame, box_yolo=box_yolo, color=GREEN)

            # Dibuja los key_points de las cajas.
            PlotCV2.circle(frame=frame, x=box_yolo.cx, y=box_yolo.cy, color=BLUE, radius=6)

        # Dibuja los puntos de referencia..
        for (x, y) in self.points_ref_list:
            PlotCV2.circle(frame=frame, x=x, y=y, color=YELLOW, radius=6)
        
        if self.recording:
            self.frames_to_save.append(frame.copy())
            self.boxes_yolo_list.append(boxes_yolo)
        return frame
