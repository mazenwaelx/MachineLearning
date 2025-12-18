import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from typing import Optional

class FeatureExtractor:

    def __init__(self, color_bins: int = 32, hog_orientations: int = 9,
                 hog_pixels_per_cell: tuple = (16, 16),
                 hog_cells_per_block: tuple = (2, 2)):
        self.color_bins = color_bins
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.scaler: Optional[StandardScaler] = None

    def extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        histograms = []
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, 
                               [self.color_bins], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            histograms.append(hist)

        return np.concatenate(histograms)

    def extract_hog(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        features = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True
        )

        return features

    def extract(self, image: np.ndarray) -> np.ndarray:
        color_features = self.extract_color_histogram(image)
        hog_features = self.extract_hog(image)

        return np.concatenate([color_features, hog_features])

    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        features = []
        for image in images:
            features.append(self.extract(image))

        return np.array(features)

    def fit_scaler(self, features: np.ndarray) -> None:
        self.scaler = StandardScaler()
        self.scaler.fit(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")

        return self.scaler.transform(features)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit_scaler(features)
        return self.transform(features)

    def get_feature_dim(self, image_shape: tuple = (224, 224)) -> int:
        color_dim = self.color_bins * 3

        h, w = image_shape
        n_cells_y = h // self.hog_pixels_per_cell[0]
        n_cells_x = w // self.hog_pixels_per_cell[1]
        n_blocks_y = n_cells_y - self.hog_cells_per_block[0] + 1
        n_blocks_x = n_cells_x - self.hog_cells_per_block[1] + 1
        hog_dim = (n_blocks_y * n_blocks_x * 
                   self.hog_cells_per_block[0] * self.hog_cells_per_block[1] * 
                   self.hog_orientations)

        return color_dim + hog_dim
