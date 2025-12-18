import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

from src.data.loader import DataLoader, CLASS_NAMES
from src.data.augmentor import Augmentor
from src.features.extractor import FeatureExtractor
from src.classifiers.svm import SVMClassifier
from src.classifiers.knn import KNNClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        self.image_size = self.config.get('image_size', (224, 224))
        self.target_samples_per_class = self.config.get('target_samples_per_class', 500)
        self.train_split = self.config.get('train_split', 0.8)

        self.color_bins = self.config.get('color_bins', 32)
        self.hog_orientations = self.config.get('hog_orientations', 9)

        self.svm_c_range = self.config.get('svm_c_range', [0.1, 1, 10, 100])
        self.svm_gamma_range = self.config.get('svm_gamma_range', ['scale', 'auto', 0.01, 0.1])

        self.knn_k_range = self.config.get('knn_k_range', range(3, 21, 2))

        self.data_loader: Optional[DataLoader] = None
        self.augmentor: Optional[Augmentor] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.svm_classifier: Optional[SVMClassifier] = None
        self.knn_classifier: Optional[KNNClassifier] = None

        self.metrics: Dict[str, Any] = {}
        self.training_time: Dict[str, float] = {}

    def _load_data(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Loading dataset...")
        self.data_loader = DataLoader(dataset_path, image_size=self.image_size)
        images, labels = self.data_loader.load_dataset()
        logger.info(f"Loaded {len(images)} images")
        return images, labels

    def _augment_data(
        self, images: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Augmenting dataset...")
        self.augmentor = Augmentor(target_samples_per_class=self.target_samples_per_class)
        aug_images, aug_labels = self.augmentor.augment_dataset(images, labels)
        return aug_images, aug_labels

    def _split_data(
        self, images: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info(f"Splitting data with {self.train_split:.0%} train / {1-self.train_split:.0%} val")
        return train_test_split(
            images, labels,
            train_size=self.train_split,
            stratify=labels,
            random_state=42
        )

    def _extract_features(
        self, train_images: np.ndarray, val_images: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Extracting features...")
        self.feature_extractor = FeatureExtractor(
            color_bins=self.color_bins,
            hog_orientations=self.hog_orientations
        )

        train_images_uint8 = (train_images * 255).astype(np.uint8)
        val_images_uint8 = (val_images * 255).astype(np.uint8)

        logger.info("Extracting training features...")
        train_features = []
        for img in tqdm(train_images_uint8, desc="Training features"):
            train_features.append(self.feature_extractor.extract(img))
        train_features = np.array(train_features)

        logger.info("Extracting validation features...")
        val_features = []
        for img in tqdm(val_images_uint8, desc="Validation features"):
            val_features.append(self.feature_extractor.extract(img))
        val_features = np.array(val_features)

        logger.info("Normalizing features...")
        self.feature_extractor.fit_scaler(train_features)
        train_features = self.feature_extractor.transform(train_features)
        val_features = self.feature_extractor.transform(val_features)

        logger.info(f"Feature vector dimension: {train_features.shape[1]}")
        return train_features, val_features

    def _train_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> SVMClassifier:
        logger.info("Training SVM classifier with hyperparameter tuning...")
        start_time = time.time()

        self.svm_classifier = SVMClassifier()
        tune_results = self.svm_classifier.tune_hyperparameters(
            X_train, y_train,
            c_range=self.svm_c_range,
            gamma_range=self.svm_gamma_range
        )

        self.training_time['svm'] = time.time() - start_time
        logger.info(f"SVM best params: {tune_results['best_params']}")
        logger.info(f"SVM CV score: {tune_results['best_score']:.4f}")
        logger.info(f"SVM training time: {self.training_time['svm']:.2f}s")

        return self.svm_classifier

    def _train_knn(self, X_train: np.ndarray, y_train: np.ndarray) -> KNNClassifier:
        logger.info("Training k-NN classifier with k tuning...")
        start_time = time.time()

        self.knn_classifier = KNNClassifier()
        best_k = self.knn_classifier.tune_k(
            X_train, y_train,
            k_range=self.knn_k_range
        )

        self.training_time['knn'] = time.time() - start_time
        logger.info(f"k-NN best k: {best_k}")
        logger.info(f"k-NN training time: {self.training_time['knn']:.2f}s")

        return self.knn_classifier

    def _compute_metrics(
        self,
        classifier_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [CLASS_NAMES[i] for i in classes]

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred,
                target_names=class_names,
                zero_division=0
            )
        }

        logger.info(f"\n{classifier_name.upper()} Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")

        return metrics

    def _evaluate_classifiers(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        logger.info("\nEvaluating classifiers on validation set...")

        results = {}

        if self.svm_classifier is not None:
            start_time = time.time()
            svm_pred = self.svm_classifier.predict(X_val)
            svm_inference_time = time.time() - start_time

            results['svm'] = self._compute_metrics('svm', y_val, svm_pred)
            results['svm']['inference_time'] = svm_inference_time
            results['svm']['training_time'] = self.training_time.get('svm', 0)
            logger.info(f"  SVM inference time: {svm_inference_time:.4f}s for {len(y_val)} samples")

        if self.knn_classifier is not None:
            start_time = time.time()
            knn_pred = self.knn_classifier.predict(X_val)
            knn_inference_time = time.time() - start_time

            results['knn'] = self._compute_metrics('knn', y_val, knn_pred)
            results['knn']['inference_time'] = knn_inference_time
            results['knn']['training_time'] = self.training_time.get('knn', 0)
            logger.info(f"  k-NN inference time: {knn_inference_time:.4f}s for {len(y_val)} samples")

        return results

    def run(self, dataset_path: str) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 60)

        total_start_time = time.time()

        images, labels = self._load_data(dataset_path)
        if len(images) == 0:
            raise ValueError("No images loaded from dataset")

        aug_images, aug_labels = self._augment_data(images, labels)

        X_train_img, X_val_img, y_train, y_val = self._split_data(aug_images, aug_labels)
        logger.info(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")

        X_train, X_val = self._extract_features(X_train_img, X_val_img)

        self._train_svm(X_train, y_train)
        self._train_knn(X_train, y_train)

        self.metrics = self._evaluate_classifiers(X_val, y_val)

        total_time = time.time() - total_start_time
        logger.info("=" * 60)
        logger.info(f"Training Pipeline Complete - Total time: {total_time:.2f}s")
        logger.info("=" * 60)

        self._print_comparison_summary()

        return self.metrics

    def _print_comparison_summary(self) -> None:
        if not self.metrics:
            return

        logger.info("\n" + "=" * 60)
        logger.info("CLASSIFIER COMPARISON SUMMARY")
        logger.info("=" * 60)

        headers = ["Metric", "SVM", "k-NN", "Best"]
        rows = []

        svm_metrics = self.metrics.get('svm', {})
        knn_metrics = self.metrics.get('knn', {})

        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            svm_val = svm_metrics.get(metric, 0)
            knn_val = knn_metrics.get(metric, 0)
            best = "SVM" if svm_val >= knn_val else "k-NN"
            rows.append([metric.capitalize(), f"{svm_val:.4f}", f"{knn_val:.4f}", best])

        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(4)]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = "-+-".join("-" * w for w in col_widths)

        logger.info(header_line)
        logger.info(separator)
        for row in rows:
            logger.info(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

        svm_acc = svm_metrics.get('accuracy', 0)
        knn_acc = knn_metrics.get('accuracy', 0)
        best_model = "SVM" if svm_acc >= knn_acc else "k-NN"
        logger.info(f"\nBest performing model: {best_model} (accuracy: {max(svm_acc, knn_acc):.4f})")

    def get_best_model(self) -> Tuple[str, Any]:
        if not self.metrics:
            raise RuntimeError("Pipeline must be run before getting best model")

        svm_acc = self.metrics.get('svm', {}).get('accuracy', 0)
        knn_acc = self.metrics.get('knn', {}).get('accuracy', 0)

        if svm_acc >= knn_acc:
            return 'svm', self.svm_classifier
        return 'knn', self.knn_classifier

    def save_models(self, output_dir: str = 'models') -> Dict[str, str]:
        if self.svm_classifier is None or self.knn_classifier is None:
            raise RuntimeError("Pipeline must be run before saving models")

        if self.feature_extractor is None or self.feature_extractor.scaler is None:
            raise RuntimeError("Feature extractor with fitted scaler required")

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        saved_paths = {}

        svm_path = os.path.join(output_dir, f'svm_classifier_{timestamp}.joblib')
        joblib.dump(self.svm_classifier, svm_path)
        saved_paths['svm'] = svm_path
        logger.info(f"Saved SVM classifier to: {svm_path}")

        knn_path = os.path.join(output_dir, f'knn_classifier_{timestamp}.joblib')
        joblib.dump(self.knn_classifier, knn_path)
        saved_paths['knn'] = knn_path
        logger.info(f"Saved k-NN classifier to: {knn_path}")

        scaler_path = os.path.join(output_dir, f'feature_scaler_{timestamp}.joblib')
        joblib.dump(self.feature_extractor.scaler, scaler_path)
        saved_paths['scaler'] = scaler_path
        logger.info(f"Saved feature scaler to: {scaler_path}")

        config = {
            'color_bins': self.feature_extractor.color_bins,
            'hog_orientations': self.feature_extractor.hog_orientations,
            'hog_pixels_per_cell': self.feature_extractor.hog_pixels_per_cell,
            'hog_cells_per_block': self.feature_extractor.hog_cells_per_block,
            'image_size': self.image_size,
            'timestamp': timestamp,
            'metrics': self.metrics
        }
        config_path = os.path.join(output_dir, f'config_{timestamp}.joblib')
        joblib.dump(config, config_path)
        saved_paths['config'] = config_path
        logger.info(f"Saved configuration to: {config_path}")

        manifest = {
            'timestamp': timestamp,
            'svm_path': os.path.basename(svm_path),
            'knn_path': os.path.basename(knn_path),
            'scaler_path': os.path.basename(scaler_path),
            'config_path': os.path.basename(config_path),
            'best_model': self.get_best_model()[0] if self.metrics else 'svm'
        }
        manifest_path = os.path.join(output_dir, 'latest_manifest.joblib')
        joblib.dump(manifest, manifest_path)
        saved_paths['manifest'] = manifest_path
        logger.info(f"Saved manifest to: {manifest_path}")

        logger.info(f"All models saved successfully with timestamp: {timestamp}")
        return saved_paths
