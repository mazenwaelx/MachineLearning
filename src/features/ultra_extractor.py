import numpy as np
import cv2
from typing import Optional
from sklearn.preprocessing import StandardScaler

class UltraFeatureExtractor:

    def __init__(self, feature_version: str = 'auto'):
        self.scaler: Optional[StandardScaler] = None
        self.feature_version = feature_version
        self.feature_dim = 186

    def _determine_feature_version(self) -> int:
        if self.feature_version == 'auto':
            if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
                expected = self.scaler.n_features_in_
                if expected == 161:
                    return 161
                elif expected == 170:
                    return 170
                elif expected == 178:
                    return 178
                elif expected == 186:
                    return 186
            return 186
        elif self.feature_version == '161':
            return 161
        elif self.feature_version == '170' or self.feature_version == '171':
            return 170
        elif self.feature_version == '178':
            return 178
        elif self.feature_version == '186':
            return 186

        return 186

    def extract(self, img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        feature_vector = []

        for i in range(3):
            channel = img[:,:,i]
            feature_vector.extend([
                np.mean(channel), np.std(channel), np.median(channel),
                np.percentile(channel, 25), np.percentile(channel, 75),
                np.min(channel), np.max(channel)
            ])

        for i in range(3):
            channel = hsv[:,:,i]
            feature_vector.extend([
                np.mean(channel), np.std(channel), np.median(channel),
                np.percentile(channel, 25), np.percentile(channel, 75)
            ])

        for i in range(3):
            channel = lab[:,:,i]
            feature_vector.extend([
                np.mean(channel), np.std(channel), np.median(channel)
            ])

        feature_vector.extend([
            np.mean(gray), np.std(gray), np.var(gray),
            np.min(gray), np.max(gray), np.ptp(gray),
            np.percentile(gray, 25), np.percentile(gray, 75)
        ])

        edges_fine = cv2.Canny(gray, 50, 150)
        edges_medium = cv2.Canny(gray, 30, 100)
        edges_coarse = cv2.Canny(gray, 20, 80)

        edge_density_fine = np.sum(edges_fine > 0) / (edges_fine.shape[0] * edges_fine.shape[1])
        edge_density_medium = np.sum(edges_medium > 0) / (edges_medium.shape[0] * edges_medium.shape[1])
        edge_density_coarse = np.sum(edges_coarse > 0) / (edges_coarse.shape[0] * edges_coarse.shape[1])

        feature_vector.extend([edge_density_fine, edge_density_medium, edge_density_coarse])

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        feature_vector.extend([
            np.var(laplacian), np.mean(np.abs(laplacian)), np.std(laplacian)
        ])

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        feature_vector.extend([
            np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag),
            np.percentile(grad_mag, 90)
        ])

        brightness_variance = np.var(gray)
        local_contrast = np.std(gray) / (np.mean(gray) + 1e-8)

        bright_threshold = np.percentile(gray, 95)
        bright_spots = np.sum(gray > bright_threshold) / (gray.shape[0] * gray.shape[1])

        edge_sharpness = np.mean(grad_mag[grad_mag > np.percentile(grad_mag, 90)])

        h, w = gray.shape
        region_size = 8
        local_variances = []
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = gray[i:i+region_size, j:j+region_size]
                local_variances.append(np.var(region))
        transparency_indicator = np.std(local_variances) if local_variances else 0

        very_bright_threshold = np.percentile(gray, 98)
        specular_highlights = np.sum(gray > very_bright_threshold) / (gray.shape[0] * gray.shape[1])

        feature_vector.extend([
            brightness_variance, local_contrast, bright_spots,
            edge_sharpness, transparency_indicator, specular_highlights
        ])

        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [14], [0, 256])
            hist = hist.flatten() / (img.shape[0] * img.shape[1])
            feature_vector.extend(hist)

        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [10], [0, 256])
            hist = hist.flatten() / (hsv.shape[0] * hsv.shape[1])
            feature_vector.extend(hist)

        for color_img in [img, hsv]:
            for i in range(3):
                channel = color_img[:,:,i].flatten()
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                if std_val > 0:
                    skewness = np.mean(((channel - mean_val) / std_val) ** 3)
                    kurtosis = np.mean(((channel - mean_val) / std_val) ** 4)
                else:
                    skewness = 0
                    kurtosis = 0
                feature_vector.extend([skewness, kurtosis])

        h, w = gray.shape
        quadrants = [
            gray[:h//2, :w//2],
            gray[:h//2, w//2:],
            gray[h//2:, :w//2],
            gray[h//2:, w//2:]
        ]

        for quad in quadrants:
            if quad.size > 0:
                feature_vector.extend([np.mean(quad), np.std(quad)])
            else:
                feature_vector.extend([0, 0])

        lbp_features = []
        step = max(2, min(h, w) // 20)
        for i in range(1, h-1, step):
            for j in range(1, w-1, step):
                center = gray[i, j]
                pattern = 0
                pattern |= (gray[i-1, j-1] >= center) << 7
                pattern |= (gray[i-1, j] >= center) << 6
                pattern |= (gray[i-1, j+1] >= center) << 5
                pattern |= (gray[i, j+1] >= center) << 4
                pattern |= (gray[i+1, j+1] >= center) << 3
                pattern |= (gray[i+1, j] >= center) << 2
                pattern |= (gray[i+1, j-1] >= center) << 1
                pattern |= (gray[i, j-1] >= center) << 0
                lbp_features.append(pattern)

        if lbp_features:
            lbp_array = np.array(lbp_features)
            feature_vector.extend([
                np.mean(lbp_array), np.std(lbp_array),
                np.percentile(lbp_array, 25), np.percentile(lbp_array, 75)
            ])
        else:
            feature_vector.extend([0, 0, 0, 0])

        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        low_freq_energy = np.mean(fft_magnitude[:h//4, :w//4])
        high_freq_energy = np.mean(fft_magnitude[h//2:, w//2:])
        feature_vector.extend([low_freq_energy, high_freq_energy, low_freq_energy / (high_freq_energy + 1e-8)])

        density_score = np.sum(gray > np.percentile(gray, 50)) / gray.size
        feature_vector.append(density_score)

        color_uniformity = 1.0 / (1.0 + np.std([np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])]))
        feature_vector.append(color_uniformity)

        grad_percentiles = [
            np.percentile(grad_mag, 10),
            np.percentile(grad_mag, 25),
            np.percentile(grad_mag, 50),
            np.percentile(grad_mag, 75),
            np.percentile(grad_mag, 90)
        ]
        smoothness_score = np.mean(grad_percentiles[1:4])
        feature_vector.append(smoothness_score)

        moderate_bright = np.percentile(gray, 70)
        bright_pixels = np.sum((gray > moderate_bright) & (gray <= bright_threshold)) / gray.size
        sheen_ratio = bright_pixels / (bright_spots + 1e-8)
        feature_vector.append(sheen_ratio)

        saturation = hsv[:,:,1]
        mean_saturation = np.mean(saturation)
        std_saturation = np.std(saturation)
        feature_vector.extend([mean_saturation, std_saturation])

        edge_sharpness_distribution = [
            np.percentile(grad_mag, 25),
            np.percentile(grad_mag, 50),
            np.percentile(grad_mag, 75)
        ]
        feature_vector.extend(edge_sharpness_distribution)

        texture_uniformity = 1.0 / (1.0 + np.std(local_variances) if local_variances else 1.0)
        feature_vector.append(texture_uniformity)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        feature_vector.append(edge_density)

        h, w = gray.shape
        edge_regions = [
            np.sum(edges[:h//3, :] > 0),
            np.sum(edges[h//3:2*h//3, :] > 0),
            np.sum(edges[2*h//3:, :] > 0),
            np.sum(edges[:, :w//3] > 0),
            np.sum(edges[:, w//3:2*w//3] > 0),
            np.sum(edges[:, 2*w//3:] > 0)
        ]
        edge_distribution_irregularity = np.std(edge_regions) if edge_regions else 0
        feature_vector.append(edge_distribution_irregularity)

        color_variance = np.var([np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])])
        feature_vector.append(color_variance)

        brightness_variance = np.var(gray)
        feature_vector.append(brightness_variance)

        texture_complexity = np.std(local_variances) if local_variances else 0
        feature_vector.append(texture_complexity)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_direction = np.arctan2(grad_y, grad_x)
        pattern_irregularity = np.std(grad_direction)
        feature_vector.append(pattern_irregularity)

        hist, bins = np.histogram(gray.flatten(), bins=32)
        peak_threshold = np.max(hist) * 0.3
        distinct_regions = np.sum(hist > peak_threshold)
        feature_vector.append(distinct_regions)

        edge_coords = np.column_stack(np.where(edges > 0))
        if len(edge_coords) > 10:
            if len(edge_coords) > 0:
                x_coords = edge_coords[:, 1]
                y_coords = edge_coords[:, 0]
                spatial_chaos = np.std(x_coords) + np.std(y_coords)
            else:
                spatial_chaos = 0
        else:
            spatial_chaos = 0
        feature_vector.append(spatial_chaos)

        version = self._determine_feature_version()

        if version == 161:
            feature_vector = feature_vector[:-25]
            self.feature_dim = 161
        elif version == 170:
            feature_vector = feature_vector[:-16]
            self.feature_dim = 170
        elif version == 178:
            feature_vector = feature_vector[:-8]
            self.feature_dim = 178
        else:
            self.feature_dim = 186

        return np.array(feature_vector)

    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        features = []
        for image in images:
            features.append(self.extract(image))
        return np.array(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Scaler not set. Load scaler before transforming features.")

        if hasattr(self.scaler, 'n_features_in_'):
            expected = self.scaler.n_features_in_
            actual = features.shape[1] if len(features.shape) > 1 else len(features)
            if expected != actual:
                raise ValueError(
                    f"Feature dimension mismatch! Scaler expects {expected} features, "
                    f"but got {actual}. Please use the matching model/scaler pair."
                )

        return self.scaler.transform(features)

class FinalFeatureExtractor:

    def __init__(self):
        self.scaler = None
        self.feature_dim = 187
        self.image_size = (96, 96)

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)

        features = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        h, w = gray.shape
        total_pixels = h * w

        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            features.extend([
                np.mean(channel), np.std(channel), np.median(channel),
                np.percentile(channel, 25), np.percentile(channel, 75),
                np.min(channel), np.max(channel)
            ])

        for i in range(3):
            channel = hsv[:, :, i].astype(np.float32)
            features.extend([
                np.mean(channel), np.std(channel), np.median(channel),
                np.percentile(channel, 25), np.percentile(channel, 75)
            ])

        for i in range(3):
            channel = lab[:, :, i].astype(np.float32)
            features.extend([np.mean(channel), np.std(channel), np.median(channel)])

        gray_f = gray.astype(np.float32)
        features.extend([
            np.mean(gray_f), np.std(gray_f), np.var(gray_f),
            np.min(gray_f), np.max(gray_f), np.ptp(gray_f),
            np.percentile(gray_f, 25), np.percentile(gray_f, 75)
        ])

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([np.var(laplacian), np.mean(np.abs(laplacian)), np.std(laplacian)])

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)

        features.extend([
            np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag),
            np.percentile(grad_mag, 90),
            np.mean(grad_dir), np.std(grad_dir),
            np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y))
        ])

        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        features.extend([
            edge_pixels / total_pixels,
            np.mean(edges),
            np.std(edges),
            np.sum(edges) / 255.0 / total_pixels
        ])

        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [8], [0, 256])
            hist = hist.flatten() / total_pixels
            features.extend(hist)

        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [8], [0, 256])
            hist = hist.flatten() / total_pixels
            features.extend(hist)

        brightness_var = np.var(gray_f)
        local_contrast = np.std(gray_f) / (np.mean(gray_f) + 1e-8)
        bright_threshold = np.percentile(gray_f, 95)
        bright_spots = np.sum(gray_f > bright_threshold) / total_pixels
        dark_spots = np.sum(gray_f < np.percentile(gray_f, 5)) / total_pixels
        features.extend([brightness_var, local_contrast, bright_spots, dark_spots, np.ptp(gray_f)])

        cell_h, cell_w = h // 2, w // 2

        for cy in range(2):
            for cx in range(2):
                cell_grad_mag = grad_mag[cy*cell_h:(cy+1)*cell_h, cx*cell_w:(cx+1)*cell_w]
                cell_grad_dir = grad_dir[cy*cell_h:(cy+1)*cell_h, cx*cell_w:(cx+1)*cell_w]

                cell_grad_dir = np.abs(cell_grad_dir) * 180 / np.pi

                hist = np.zeros(9)
                for i in range(9):
                    bin_low = i * 20
                    bin_high = (i + 1) * 20
                    mask = (cell_grad_dir >= bin_low) & (cell_grad_dir < bin_high)
                    hist[i] = np.sum(cell_grad_mag[mask])

                hist_sum = np.sum(hist) + 1e-8
                hist = hist / hist_sum
                features.extend(hist)

        features.extend([
            np.percentile(grad_mag, 10),
            np.percentile(grad_mag, 30),
            np.percentile(grad_mag, 50),
            np.percentile(grad_mag, 70),
            np.mean(laplacian),
            np.percentile(laplacian, 25),
            np.percentile(laplacian, 75),
            np.std(laplacian) / (np.mean(np.abs(laplacian)) + 1e-8)
        ])

        lbp_hist = np.zeros(16)
        for row in range(1, h-1, 2):
            for col in range(1, w-1, 2):
                center = gray[row, col]
                pattern = 0
                if gray[row-1, col] >= center: pattern |= 1
                if gray[row, col+1] >= center: pattern |= 2
                if gray[row+1, col] >= center: pattern |= 4
                if gray[row, col-1] >= center: pattern |= 8
                lbp_hist[pattern] += 1
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)
        features.extend(lbp_hist)

        for color_space in [image, hsv]:
            for ch in range(3):
                channel = color_space[:, :, ch]
                mean_val = np.mean(channel)
                coherence = np.sum(np.abs(channel - mean_val) < 30) / total_pixels
                features.append(coherence)

        return np.array(features, dtype=np.float32)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Scaler not set. Load scaler before transforming features.")

        if hasattr(self.scaler, 'n_features_in_'):
            expected = self.scaler.n_features_in_
            actual = features.shape[1] if len(features.shape) > 1 else len(features)
            if expected != actual:
                raise ValueError(
                    f"Feature dimension mismatch! Scaler expects {expected} features, "
                    f"but got {actual}. Please use the matching model/scaler pair."
                )

        return self.scaler.transform(features)
