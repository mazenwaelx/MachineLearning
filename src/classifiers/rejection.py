import numpy as np
import cv2

class UnknownClassHandler:

    UNKNOWN_CLASS_ID = 6
    UNKNOWN_CLASS_NAME = 'Unknown'

    def __init__(self, confidence_threshold: float = 0.5, blur_threshold: float = 100.0):
        self.confidence_threshold = confidence_threshold
        self.blur_threshold = blur_threshold

    def check_blur(self, image: np.ndarray) -> bool:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return variance < self.blur_threshold

    def apply_rejection(
        self,
        prediction: int,
        confidence: float,
        is_blurry: bool = False
    ) -> tuple[int, float]:
        if is_blurry:
            return (self.UNKNOWN_CLASS_ID, confidence)

        if confidence < self.confidence_threshold:
            return (self.UNKNOWN_CLASS_ID, confidence)

        return (prediction, confidence)

    def get_blur_score(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = threshold

    def set_blur_threshold(self, threshold: float) -> None:
        self.blur_threshold = threshold

    def get_params(self) -> dict:
        return {
            'confidence_threshold': self.confidence_threshold,
            'blur_threshold': self.blur_threshold
        }
