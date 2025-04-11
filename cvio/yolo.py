from typing import Dict, List
from pathlib import Path

from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.engine.results import Results
from cv2.typing import MatLike


class BoxYOLO(BaseModel):
    label: str      # TODO: Poner como Litearal.
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return self.y2 - (self.y2 - self.y1) // 10


class YOLOWrapper:
    def __init__(self, *, path_model: Path, categorie2label: Dict[int, str] = None):
        self.path_model = path_model
        self.model = YOLO(self.path_model)
        if categorie2label is None:
            categorie2label = {0: "person", 15: "cat"}      # FIXME: Debería retornar todas las categorías y labels.
        self.categorie2label = categorie2label

    def detect_objects(self, *, frame: MatLike) -> List[BoxYOLO]:
        boxes_yolo = []
        results: Results = self.model(frame, verbose=False)[0]
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            label = self.categorie2label.get(int(cls), None)
            if label is not None:
                x1, y1, x2, y2 = map(int, box)
                boxes_yolo.append(BoxYOLO(label=label, x1=x1, y1=y1, x2=x2, y2=y2))
        return boxes_yolo
