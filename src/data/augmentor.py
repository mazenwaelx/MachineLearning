import logging
from typing import Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class Augmentor:

    def __init__(self, target_samples_per_class: int = 500):
        self.target_samples_per_class = target_samples_per_class

        self.rotation_range = (-30, 30)
        self.scale_range = (0.8, 1.2)
        self.brightness_range = (-0.2, 0.2)
        self.contrast_range = (0.8, 1.2)

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated

    def _flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    def _flip_vertical(self, image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 0)

    def _scale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        scaled = cv2.resize(image, (new_w, new_h))

        if scale_factor > 1.0:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            result = scaled[start_y:start_y + h, start_x:start_x + w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            result = cv2.copyMakeBorder(
                scaled, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                borderType=cv2.BORDER_REFLECT
            )

        return result

    def _adjust_brightness(self, image: np.ndarray, delta: float) -> np.ndarray:
        adjusted = image + delta
        return np.clip(adjusted, 0.0, 1.0)

    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0.0, 1.0)

    def _apply_transforms(self, image: np.ndarray) -> List[np.ndarray]:
        augmented = []

        angle = np.random.uniform(*self.rotation_range)
        rotated = self._rotate(image, angle)
        augmented.append(rotated)

        if np.random.random() > 0.5:
            flipped_h = self._flip_horizontal(image)
            augmented.append(flipped_h)

        if np.random.random() > 0.5:
            flipped_v = self._flip_vertical(image)
            augmented.append(flipped_v)

        scale_factor = np.random.uniform(*self.scale_range)
        scaled = self._scale(image, scale_factor)
        augmented.append(scaled)

        brightness_delta = np.random.uniform(*self.brightness_range)
        contrast_factor = np.random.uniform(*self.contrast_range)
        jittered = self._adjust_brightness(image, brightness_delta)
        jittered = self._adjust_contrast(jittered, contrast_factor)
        augmented.append(jittered)

        combined = self._rotate(image, np.random.uniform(*self.rotation_range))
        combined = self._adjust_brightness(combined, np.random.uniform(*self.brightness_range))
        combined = self._adjust_contrast(combined, np.random.uniform(*self.contrast_range))
        augmented.append(combined)

        return augmented

    def augment_dataset(
        self, images: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(images) == 0:
            logger.warning("Empty dataset provided for augmentation")
            return images, labels

        unique_classes = np.unique(labels)
        class_counts = {cls: np.sum(labels == cls) for cls in unique_classes}

        logger.info(f"Original class distribution: {class_counts}")

        augmented_images = list(images)
        augmented_labels = list(labels)

        for cls in unique_classes:
            current_count = class_counts[cls]
            samples_needed = max(0, self.target_samples_per_class - current_count)

            if samples_needed == 0:
                continue

            class_indices = np.where(labels == cls)[0]

            logger.info(f"Class {cls}: augmenting from {current_count} to ~{self.target_samples_per_class} samples")

            samples_added = 0
            while samples_added < samples_needed:
                idx = np.random.choice(class_indices)
                original_image = images[idx]

                new_images = self._apply_transforms(original_image)

                for aug_img in new_images:
                    if samples_added >= samples_needed:
                        break
                    augmented_images.append(aug_img)
                    augmented_labels.append(cls)
                    samples_added += 1

        result_images = np.array(augmented_images, dtype=np.float32)
        result_labels = np.array(augmented_labels, dtype=np.int32)

        final_counts = {cls: np.sum(result_labels == cls) for cls in unique_classes}
        logger.info(f"Augmented class distribution: {final_counts}")
        logger.info(f"Dataset size: {len(images)} -> {len(result_images)} "
                   f"(+{len(result_images) - len(images)} samples, "
                   f"{((len(result_images) - len(images)) / len(images) * 100):.1f}% increase)")

        return result_images, result_labels
