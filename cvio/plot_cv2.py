import cv2
from cv2.typing import MatLike

from cvio.typings import Color


class PlotCV2:
    @staticmethod
    def point(*, frame: MatLike, x: int, y: int, color: Color, radius: int = 6) -> None:
        cv2.circle(frame, (x, y), radius, color, -1)    # TODO: shift -1, para que sirve?

    @staticmethod
    def box(*, frame: MatLike, label: str, x1: int, y1: int, x2: int, y2: int, color: Color) -> None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
