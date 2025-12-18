import numpy as np
import pytest

from src.classifiers.svm import SVMClassifier
from src.classifiers.knn import KNNClassifier
from src.classifiers.rejection import UnknownClassHandler

class TestSVMClassifier:

    def test_svm_training_and_prediction(self):
        classifier = SVMClassifier(kernel='rbf', C=1.0)

        X_train = np.random.rand(50, 10)
        y_train = np.array([0] * 25 + [1] * 25)

        classifier.train(X_train, y_train)

        predictions = classifier.predict(X_train)

        assert predictions.shape == (50,)
        assert set(predictions).issubset({0, 1})

    def test_svm_predict_proba(self):
        classifier = SVMClassifier()

        X_train = np.random.rand(30, 10)
        y_train = np.array([0] * 10 + [1] * 10 + [2] * 10)

        classifier.train(X_train, y_train)

        proba = classifier.predict_proba(X_train[:5])

        assert proba.shape == (5, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_svm_predict_before_train_raises_error(self):
        classifier = SVMClassifier()

        X = np.random.rand(5, 10)

        with pytest.raises(RuntimeError):
            classifier.predict(X)

class TestKNNClassifier:

    def test_knn_training_and_prediction(self):
        classifier = KNNClassifier(n_neighbors=3)

        X_train = np.random.rand(30, 10)
        y_train = np.array([0] * 10 + [1] * 10 + [2] * 10)

        classifier.train(X_train, y_train)

        predictions = classifier.predict(X_train)

        assert predictions.shape == (30,)
        assert set(predictions).issubset({0, 1, 2})

    def test_knn_predict_proba(self):
        classifier = KNNClassifier(n_neighbors=5)

        X_train = np.random.rand(30, 10)
        y_train = np.array([0] * 10 + [1] * 10 + [2] * 10)

        classifier.train(X_train, y_train)

        proba = classifier.predict_proba(X_train[:5])

        assert proba.shape == (5, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_knn_predict_before_train_raises_error(self):
        classifier = KNNClassifier()

        X = np.random.rand(5, 10)

        with pytest.raises(RuntimeError):
            classifier.predict(X)

class TestUnknownClassHandler:

    def test_rejection_below_threshold(self):
        handler = UnknownClassHandler(confidence_threshold=0.5)

        final_class, final_conf = handler.apply_rejection(
            prediction=1, confidence=0.3, is_blurry=False
        )

        assert final_class == 6
        assert final_conf == 0.3

    def test_acceptance_above_threshold(self):
        handler = UnknownClassHandler(confidence_threshold=0.5)

        final_class, final_conf = handler.apply_rejection(
            prediction=2, confidence=0.8, is_blurry=False
        )

        assert final_class == 2
        assert final_conf == 0.8

    def test_rejection_for_blurry_image(self):
        handler = UnknownClassHandler(confidence_threshold=0.5)

        final_class, _ = handler.apply_rejection(
            prediction=3, confidence=0.9, is_blurry=True
        )

        assert final_class == 6

    def test_blur_detection(self):
        handler = UnknownClassHandler(blur_threshold=100.0)

        sharp_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        blurry_img = np.full((100, 100), 128, dtype=np.uint8)

        assert handler.check_blur(sharp_img) == False
        assert handler.check_blur(blurry_img) == True
