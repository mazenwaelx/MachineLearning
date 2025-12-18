import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate trained waste material classification models (SVM and k-NN)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to dataset folder containing class subfolders'
    )
    parser.add_argument(
        '--models', '-m',
        type=str,
        required=True,
        help='Directory containing trained models'
    )

    parser.add_argument(
        '--split',
        type=float,
        default=0.2,
        help='Validation set proportion (0.0 to 1.0)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size (width and height) for resizing'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=500,
        help='Target samples per class after augmentation'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )

    return parser.parse_args()

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def find_model_files(models_dir: str) -> dict:
    files = {}

    manifest_path = os.path.join(models_dir, 'latest_manifest.joblib')
    if os.path.exists(manifest_path):
        manifest = joblib.load(manifest_path)
        files['svm'] = os.path.join(models_dir, manifest.get('svm_path', ''))
        files['knn'] = os.path.join(models_dir, manifest.get('knn_path', ''))
        files['scaler'] = os.path.join(models_dir, manifest.get('scaler_path', ''))
        files['config'] = os.path.join(models_dir, manifest.get('config_path', ''))
        return files

    for filename in os.listdir(models_dir):
        filepath = os.path.join(models_dir, filename)
        if 'svm_classifier' in filename.lower():
            files['svm'] = filepath
        elif 'knn_classifier' in filename.lower():
            files['knn'] = filepath
        elif 'scaler' in filename.lower():
            files['scaler'] = filepath
        elif 'config' in filename.lower() and filename.endswith('.joblib'):
            files['config'] = filepath

    return files

def print_confusion_matrix(cm: np.ndarray, class_names: list, title: str) -> None:
    print(f"\n{title}")
    print("=" * 60)

    col_width = max(len(name) for name in class_names) + 2
    col_width = max(col_width, 8)

    header = "Actual\\Pred".ljust(col_width)
    for name in class_names:
        header += name[:col_width-1].center(col_width)
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = class_names[i][:col_width-1].ljust(col_width)
        for val in row:
            row_str += str(val).center(col_width)
        print(row_str)

def print_comparison_table(svm_metrics: dict, knn_metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("CLASSIFIER COMPARISON SUMMARY")
    print("=" * 60)

    headers = ["Metric", "SVM", "k-NN", "Best"]
    rows = []

    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        svm_val = svm_metrics.get(metric, 0)
        knn_val = knn_metrics.get(metric, 0)
        best = "SVM" if svm_val >= knn_val else "k-NN"
        rows.append([metric.replace('_', ' ').title(), f"{svm_val:.4f}", f"{knn_val:.4f}", best])

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(4)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

def evaluate_classifier(classifier, X_val: np.ndarray, y_val: np.ndarray, name: str) -> dict:
    y_pred = classifier.predict(X_val)

    classes = np.unique(np.concatenate([y_val, y_pred]))
    class_names = [CLASS_NAMES[i] for i in classes]

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_val, y_pred),
        'classification_report': classification_report(
            y_val, y_pred,
            target_names=class_names,
            zero_division=0
        ),
        'class_names': class_names
    }

    return metrics

def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.dataset):
        logger.error(f"Dataset path does not exist: {args.dataset}")
        return 1

    if not os.path.isdir(args.models):
        logger.error(f"Models directory does not exist: {args.models}")
        return 1

    print("=" * 60)
    print("Material Stream Identification - Evaluation Script")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Models directory: {args.models}")
    print(f"Validation split: {args.split:.0%}")
    print("=" * 60)

    try:
        model_files = find_model_files(args.models)

        if 'svm' not in model_files or not os.path.exists(model_files.get('svm', '')):
            logger.error("SVM model file not found")
            return 1

        if 'knn' not in model_files or not os.path.exists(model_files.get('knn', '')):
            logger.error("k-NN model file not found")
            return 1

        if 'scaler' not in model_files or not os.path.exists(model_files.get('scaler', '')):
            logger.error("Feature scaler file not found")
            return 1

        print("\nLoading trained models...")
        svm_classifier = joblib.load(model_files['svm'])
        knn_classifier = joblib.load(model_files['knn'])
        scaler = joblib.load(model_files['scaler'])

        config = {}
        if 'config' in model_files and os.path.exists(model_files.get('config', '')):
            config = joblib.load(model_files['config'])

        print(f"  Loaded SVM: {model_files['svm']}")
        print(f"  Loaded k-NN: {model_files['knn']}")
        print(f"  Loaded scaler: {model_files['scaler']}")

        print("\nLoading dataset...")
        image_size = (args.image_size, args.image_size)
        data_loader = DataLoader(args.dataset, image_size=image_size)
        images, labels = data_loader.load_dataset()

        if len(images) == 0:
            logger.error("No images loaded from dataset")
            return 1

        print(f"  Loaded {len(images)} images")

        print("\nAugmenting dataset...")
        augmentor = Augmentor(target_samples_per_class=args.samples)
        aug_images, aug_labels = augmentor.augment_dataset(images, labels)
        print(f"  Augmented to {len(aug_images)} images")

        print("\nSplitting data...")
        _, X_val_img, _, y_val = train_test_split(
            aug_images, aug_labels,
            test_size=args.split,
            stratify=aug_labels,
            random_state=42
        )
        print(f"  Validation samples: {len(y_val)}")

        print("\nExtracting features...")
        feature_extractor = FeatureExtractor(
            color_bins=config.get('color_bins', 32),
            hog_orientations=config.get('hog_orientations', 9)
        )
        feature_extractor.scaler = scaler

        X_val_uint8 = (X_val_img * 255).astype(np.uint8)

        val_features = []
        for img in tqdm(X_val_uint8, desc="Extracting features"):
            val_features.append(feature_extractor.extract(img))
        X_val = np.array(val_features)

        X_val = feature_extractor.transform(X_val)
        print(f"  Feature vector dimension: {X_val.shape[1]}")

        print("\n" + "=" * 60)
        print("SVM CLASSIFIER EVALUATION")
        print("=" * 60)
        svm_metrics = evaluate_classifier(svm_classifier, X_val, y_val, 'SVM')

        print(f"\nAccuracy:  {svm_metrics['accuracy']:.4f}")
        print(f"Precision: {svm_metrics['precision']:.4f}")
        print(f"Recall:    {svm_metrics['recall']:.4f}")
        print(f"F1 Score:  {svm_metrics['f1_score']:.4f}")

        print("\nClassification Report:")
        print(svm_metrics['classification_report'])

        print_confusion_matrix(
            svm_metrics['confusion_matrix'],
            svm_metrics['class_names'],
            "SVM Confusion Matrix"
        )

        print("\n" + "=" * 60)
        print("k-NN CLASSIFIER EVALUATION")
        print("=" * 60)
        knn_metrics = evaluate_classifier(knn_classifier, X_val, y_val, 'k-NN')

        print(f"\nAccuracy:  {knn_metrics['accuracy']:.4f}")
        print(f"Precision: {knn_metrics['precision']:.4f}")
        print(f"Recall:    {knn_metrics['recall']:.4f}")
        print(f"F1 Score:  {knn_metrics['f1_score']:.4f}")

        print("\nClassification Report:")
        print(knn_metrics['classification_report'])

        print_confusion_matrix(
            knn_metrics['confusion_matrix'],
            knn_metrics['class_names'],
            "k-NN Confusion Matrix"
        )

        print_comparison_table(svm_metrics, knn_metrics)

        svm_acc = svm_metrics['accuracy']
        knn_acc = knn_metrics['accuracy']
        best_model = "SVM" if svm_acc >= knn_acc else "k-NN"
        print(f"\nBest performing model: {best_model} (accuracy: {max(svm_acc, knn_acc):.4f})")

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1

if __name__ == '__main__':
    sys.exit(main())
