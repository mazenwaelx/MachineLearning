import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from skimage.feature import hog

CLASS_MAPPING = {'glass': 0, 'paper': 1, 'cardboard': 2, 'plastic': 3, 'metal': 4, 'trash': 5}
CLASS_NAMES = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate Kaggle-trained models directly',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to dataset folder containing class subfolders'
    )
    parser.add_argument(
        '--kaggle-models', '-k',
        type=str,
        required=True,
        help='Directory containing Kaggle-trained models'
    )
    parser.add_argument(
        '--split',
        type=float,
        default=0.2,
        help='Validation set proportion (0.0 to 1.0)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=300,
        help='Target samples per class (should match Kaggle training)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def find_kaggle_models(kaggle_dir: str) -> dict:
    files = {}

    for filename in os.listdir(kaggle_dir):
        filepath = os.path.join(kaggle_dir, filename)
        if not os.path.isfile(filepath):
            continue

        if filename.startswith('svm_') and filename.endswith('.joblib'):
            files['svm'] = filepath
        elif filename.startswith('knn_') and filename.endswith('.joblib'):
            files['knn'] = filepath
        elif filename.startswith('scaler_') and filename.endswith('.joblib'):
            files['scaler'] = filepath
        elif filename.startswith('results_') and filename.endswith('.joblib'):
            files['results'] = filepath

    return files

def load_dataset(dataset_path: str, image_size=(224, 224)):
    images, labels = [], []

    class_mapping = {}
    if os.path.exists(dataset_path):
        subdirs = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))]

        for subdir in subdirs:
            subdir_lower = subdir.lower()
            for expected in ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']:
                if expected in subdir_lower and expected in CLASS_MAPPING:
                    class_mapping[subdir] = CLASS_MAPPING[expected]
                    print(f"Detected: '{subdir}' -> label {CLASS_MAPPING[expected]}")
                    break

    for class_name, class_label in class_mapping.items():
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        image_files = [f for f in os.listdir(class_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        print(f"Loading {len(image_files)} images from '{class_name}'")

        for filename in image_files:
            try:
                filepath = os.path.join(class_folder, filename)
                img = cv2.imread(filepath)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(img, image_size)
                normalized = resized.astype(np.float32) / 255.0
                images.append(normalized)
                labels.append(class_label)
            except:
                continue

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

def augment_image(image: np.ndarray) -> np.ndarray:
    aug = image.copy()

    if np.random.random() > 0.5:
        aug = cv2.flip(aug, 1)

    delta = np.random.uniform(-0.2, 0.2)
    aug = np.clip(aug + delta, 0.0, 1.0)

    return aug

def augment_dataset(images: np.ndarray, labels: np.ndarray, target_samples_per_class: int = 300):
    if len(images) == 0:
        return images, labels

    unique_classes = np.unique(labels)
    class_counts = {cls: np.sum(labels == cls) for cls in unique_classes}
    print(f"Original distribution: {class_counts}")

    augmented_images = list(images)
    augmented_labels = list(labels)

    for cls in unique_classes:
        current_count = class_counts[cls]
        samples_needed = max(0, target_samples_per_class - current_count)

        if samples_needed == 0:
            continue

        class_indices = np.where(labels == cls)[0]
        print(f"Class {cls}: augmenting from {current_count} to ~{target_samples_per_class}")

        for _ in range(samples_needed):
            idx = np.random.choice(class_indices)
            aug_img = augment_image(images[idx])
            augmented_images.append(aug_img)
            augmented_labels.append(cls)

    return np.array(augmented_images, dtype=np.float32), np.array(augmented_labels, dtype=np.int32)

def extract_features(image: np.ndarray, color_bins: int = 32, hog_orientations: int = 9):
    histograms = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [color_bins], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        histograms.append(hist)
    color_features = np.concatenate(histograms)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hog_features = hog(gray, orientations=hog_orientations,
                      pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys', feature_vector=True)

    return np.concatenate([color_features, hog_features])

def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray, model_name: str):
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_val)
    elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
        y_pred = model.model.predict(X_val)
    else:
        raise ValueError(f"Cannot find predict method for {model_name}")

    classes = np.unique(np.concatenate([y_val, y_pred]))
    class_names = [CLASS_NAMES[i] for i in classes]

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_val, y_pred),
        'classification_report': classification_report(
            y_val, y_pred, target_names=class_names, zero_division=0
        )
    }

    return metrics

def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("KAGGLE MODEL EVALUATION")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Kaggle models: {args.kaggle_models}")
    print(f"Validation split: {args.split:.0%}")
    print("=" * 70)

    try:
        model_files = find_kaggle_models(args.kaggle_models)

        required_files = ['svm', 'knn', 'scaler']
        missing_files = [f for f in required_files if f not in model_files]
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            logger.info("Expected files: svm_*.joblib, knn_*.joblib, scaler_*.joblib")
            return 1

        print("\nLoading models...")
        svm_model = joblib.load(model_files['svm'])
        knn_model = joblib.load(model_files['knn'])
        scaler = joblib.load(model_files['scaler'])

        results = None
        if 'results' in model_files:
            results = joblib.load(model_files['results'])
            print("Loaded training results")

        print(f"  SVM: {os.path.basename(model_files['svm'])}")
        print(f"  k-NN: {os.path.basename(model_files['knn'])}")
        print(f"  Scaler: {os.path.basename(model_files['scaler'])}")

        print("\nLoading dataset...")
        images, labels = load_dataset(args.dataset)

        if len(images) == 0:
            logger.error("No images loaded from dataset")
            return 1

        print(f"Loaded {len(images)} images")

        print("\nAugmenting dataset...")
        aug_images, aug_labels = augment_dataset(images, labels, args.samples)
        print(f"Augmented to {len(aug_images)} images")

        print("\nSplitting data...")
        _, X_val_img, _, y_val = train_test_split(
            aug_images, aug_labels,
            test_size=args.split,
            stratify=aug_labels,
            random_state=42
        )
        print(f"Validation samples: {len(y_val)}")

        print("\nExtracting features...")
        X_val_uint8 = (X_val_img * 255).astype(np.uint8)

        val_features = []
        for img in tqdm(X_val_uint8, desc="Extracting features"):
            features = extract_features(img)
            val_features.append(features)

        X_val = np.array(val_features)
        X_val = scaler.transform(X_val)
        print(f"Feature dimension: {X_val.shape[1]}")

        print("\n" + "=" * 70)
        print("SVM EVALUATION")
        print("=" * 70)
        svm_metrics = evaluate_model(svm_model, X_val, y_val, 'SVM')

        print(f"Accuracy:  {svm_metrics['accuracy']:.4f}")
        print(f"Precision: {svm_metrics['precision']:.4f}")
        print(f"Recall:    {svm_metrics['recall']:.4f}")
        print(f"F1 Score:  {svm_metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(svm_metrics['classification_report'])

        print("\n" + "=" * 70)
        print("k-NN EVALUATION")
        print("=" * 70)
        knn_metrics = evaluate_model(knn_model, X_val, y_val, 'k-NN')

        print(f"Accuracy:  {knn_metrics['accuracy']:.4f}")
        print(f"Precision: {knn_metrics['precision']:.4f}")
        print(f"Recall:    {knn_metrics['recall']:.4f}")
        print(f"F1 Score:  {knn_metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(knn_metrics['classification_report'])

        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            svm_val = svm_metrics[metric]
            knn_val = knn_metrics[metric]
            best = "SVM" if svm_val >= knn_val else "k-NN"
            print(f"{metric:12} | SVM: {svm_val:.4f} | k-NN: {knn_val:.4f} | Best: {best}")

        if results:
            print("\nOriginal Kaggle training results:")
            if 'svm_metrics' in results:
                orig_svm = results['svm_metrics']['accuracy']
                print(f"  SVM training accuracy: {orig_svm:.4f}")
            if 'knn_metrics' in results:
                orig_knn = results['knn_metrics']['accuracy']
                print(f"  k-NN training accuracy: {orig_knn:.4f}")

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1

if __name__ == '__main__':
    sys.exit(main())
