import numpy as np
import pytest

from src.features.extractor import FeatureExtractor

class TestFeatureExtractor:

    def test_color_histogram_dimensions(self):
        extractor = FeatureExtractor(color_bins=32)

        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        hist_features = extractor.extract_color_histogram(image)

        assert hist_features.shape == (96,)

    def test_hog_features_extraction(self):
        extractor = FeatureExtractor()

        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        hog_features = extractor.extract_hog(image)

        assert len(hog_features.shape) == 1
        assert len(hog_features) > 0

    def test_combined_feature_vector_dimensions(self):
        extractor = FeatureExtractor(color_bins=32)

        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        features = extractor.extract(image)
        expected_dim = extractor.get_feature_dim((224, 224))

        assert features.shape == (expected_dim,)

    def test_scaler_fitting(self):
        extractor = FeatureExtractor()

        images = np.random.randint(0, 256, (5, 224, 224, 3), dtype=np.uint8)
        features = extractor.extract_batch(images)

        extractor.fit_scaler(features)

        assert extractor.scaler is not None

    def test_scaler_transformation(self):
        extractor = FeatureExtractor()

        images = np.random.randint(0, 256, (10, 224, 224, 3), dtype=np.uint8)
        features = extractor.extract_batch(images)

        normalized = extractor.fit_transform(features)

        assert np.abs(normalized.mean()) < 0.1
        assert np.abs(normalized.std() - 1.0) < 0.1

    def test_transform_without_fit_raises_error(self):
        extractor = FeatureExtractor()

        features = np.random.rand(5, 100)

        with pytest.raises(ValueError):
            extractor.transform(features)

    def test_batch_extraction(self):
        extractor = FeatureExtractor()

        images = np.random.randint(0, 256, (3, 224, 224, 3), dtype=np.uint8)

        features = extractor.extract_batch(images)

        assert features.shape[0] == 3
        assert features.shape[1] == extractor.get_feature_dim((224, 224))
