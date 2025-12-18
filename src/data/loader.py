import os
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_MAPPING = {
    'glass': 0,
    'paper': 1,
    'cardboard': 2,
    'plastic': 3,
    'metal': 4,
    'trash': 5
}

CLASS_NAMES = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash', 'Unknown']

class ImageLoadError(Exception):
    pass

class DataLoader:

    def __init__(self, dataset_path: str, image_size: Tuple[int, int] = (224, 224)):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self._class_mapping = CLASS_MAPPING.copy()

    def get_class_mapping(self) -> dict:
        return self._class_mapping.copy()

    def _safe_load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            img = cv2.imread(path)
            if img is None:
                raise ImageLoadError(f"Failed to load: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            logger.warning(f"Skipping corrupted image: {path}, Error: {e}")
            return None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.image_size)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        images = []
        labels = []

        for class_name, class_label in self._class_mapping.items():
            class_folder = os.path.join(self.dataset_path, class_name)

            if not os.path.isdir(class_folder):
                logger.warning(f"Class folder not found: {class_folder}")
                continue

            image_files = [
                f for f in os.listdir(class_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            ]

            logger.info(f"Loading {len(image_files)} images from class '{class_name}'")

            for filename in image_files:
                filepath = os.path.join(class_folder, filename)

                img = self._safe_load_image(filepath)
                if img is None:
                    continue

                processed_img = self._preprocess_image(img)

                images.append(processed_img)
                labels.append(class_label)

        if len(images) == 0:
            logger.warning("No images loaded from dataset")
            return np.array([]), np.array([])

        images_array = np.array(images, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)

        logger.info(f"Loaded {len(images_array)} images total")
        return images_array, labels_array
