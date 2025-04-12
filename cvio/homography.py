import cv2
import numpy as np

class Homography:
    """
    - points_real: Debe contener las coordenadas reales x,y de los puntos de referencia.
    """
    def __init__(self, *, points_real: np.ndarray, points_ref: np.ndarray):
        self.points_real: np.ndarray = points_real
        self.points_ref: np.ndarray = points_ref
        self.H, _ = cv2.findHomography(self.points_ref, self.points_real)

    def project_points(self, *, points: np.ndarray) -> np.ndarray:
        """ Proyecta puntos desde la imagen al plano real usando la homografía. """
        if points.shape[0] == 0:
            raise ValueError("Se requiere al menos 1 punto para proyectar.")
        projected: np.ndarray = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), self.H)
        return projected.reshape(-1, 2)
