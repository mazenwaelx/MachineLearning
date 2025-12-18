import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from typing import Dict, Any, Optional

from src.classifiers.svm import SVMClassifier
from src.classifiers.knn import KNNClassifier
from src.features.extractor import FeatureExtractor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Load Kaggle-trained models and integrate with project structure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--kaggle-models', '-k',
        type=str,
        required=True,
        help='Directory containing Kaggle-trained models (svm_*.joblib, knn_*.joblib, scaler_*.joblib)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for integrated models'
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

def find_kaggle_models(kaggle_dir: str) -> Dict[str, str]:
    files = {}

    if not os.path.exists(kaggle_dir):
        raise FileNotFoundError(f"Kaggle models directory not found: {kaggle_dir}")

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

class KaggleModelAdapter:

    def __init__(self, kaggle_svm, kaggle_knn, scaler, results: Optional[Dict] = None):
        self.kaggle_svm = kaggle_svm
        self.kaggle_knn = kaggle_knn
        self.scaler = scaler
        self.results = results or {}

    def create_project_svm(self) -> SVMClassifier:
        svm_classifier = SVMClassifier()

        if hasattr(self.kaggle_svm, 'model'):
            svm_classifier.model = self.kaggle_svm.model
            svm_classifier.C = getattr(self.kaggle_svm, 'C', 1.0)
            svm_classifier.gamma = getattr(self.kaggle_svm, 'gamma', 'scale')
        else:
            svm_classifier.model = self.kaggle_svm
            if hasattr(self.kaggle_svm, 'C'):
                svm_classifier.C = self.kaggle_svm.C
            if hasattr(self.kaggle_svm, 'gamma'):
                svm_classifier.gamma = self.kaggle_svm.gamma

        return svm_classifier

    def create_project_knn(self) -> KNNClassifier:
        knn_classifier = KNNClassifier()

        if hasattr(self.kaggle_knn, 'model'):
            knn_classifier.model = self.kaggle_knn.model
            knn_classifier.n_neighbors = getattr(self.kaggle_knn, 'n_neighbors', 5)
        else:
            knn_classifier.model = self.kaggle_knn
            if hasattr(self.kaggle_knn, 'n_neighbors'):
                knn_classifier.n_neighbors = self.kaggle_knn.n_neighbors

        return knn_classifier

    def create_project_feature_extractor(self) -> FeatureExtractor:
        config = self.results.get('config', {})

        feature_extractor = FeatureExtractor(
            color_bins=config.get('color_bins', 32),
            hog_orientations=config.get('hog_orientations', 9)
        )
        feature_extractor.scaler = self.scaler

        return feature_extractor

def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("KAGGLE MODEL INTEGRATION")
    print("=" * 70)
    print(f"Kaggle models: {args.kaggle_models}")
    print(f"Output directory: {args.output}")
    print("=" * 70)

    try:
        logger.info("Finding Kaggle model files...")
        kaggle_files = find_kaggle_models(args.kaggle_models)

        required_files = ['svm', 'knn', 'scaler']
        missing_files = [f for f in required_files if f not in kaggle_files]
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return 1

        logger.info("Found files:")
        for name, path in kaggle_files.items():
            logger.info(f"  {name}: {os.path.basename(path)}")

        logger.info("\nLoading Kaggle models...")
        kaggle_svm = joblib.load(kaggle_files['svm'])
        kaggle_knn = joblib.load(kaggle_files['knn'])
        scaler = joblib.load(kaggle_files['scaler'])

        results = None
        if 'results' in kaggle_files:
            results = joblib.load(kaggle_files['results'])
            logger.info("Loaded training results and config")

        logger.info("Converting models to project structure...")
        adapter = KaggleModelAdapter(kaggle_svm, kaggle_knn, scaler, results)

        project_svm = adapter.create_project_svm()
        project_knn = adapter.create_project_knn()
        project_feature_extractor = adapter.create_project_feature_extractor()

        os.makedirs(args.output, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        svm_path = os.path.join(args.output, f'svm_classifier_{timestamp}.joblib')
        knn_path = os.path.join(args.output, f'knn_classifier_{timestamp}.joblib')
        scaler_path = os.path.join(args.output, f'scaler_{timestamp}.joblib')
        config_path = os.path.join(args.output, f'config_{timestamp}.joblib')
        manifest_path = os.path.join(args.output, 'latest_manifest.joblib')

        logger.info("Saving converted models...")
        joblib.dump(project_svm, svm_path)
        joblib.dump(project_knn, knn_path)
        joblib.dump(project_feature_extractor.scaler, scaler_path)

        config_data = {
            'image_size': (224, 224),
            'color_bins': project_feature_extractor.color_bins,
            'hog_orientations': project_feature_extractor.hog_orientations,
            'timestamp': timestamp,
            'source': 'kaggle_integration',
            'original_results': results
        }
        joblib.dump(config_data, config_path)

        manifest = {
            'svm_path': os.path.basename(svm_path),
            'knn_path': os.path.basename(knn_path),
            'scaler_path': os.path.basename(scaler_path),
            'config_path': os.path.basename(config_path),
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat()
        }
        joblib.dump(manifest, manifest_path)

        print("\n" + "=" * 70)
        print("INTEGRATION COMPLETE")
        print("=" * 70)

        print("\nSaved models:")
        print(f"  SVM classifier: {svm_path}")
        print(f"  k-NN classifier: {knn_path}")
        print(f"  Feature scaler: {scaler_path}")
        print(f"  Configuration: {config_path}")
        print(f"  Manifest: {manifest_path}")

        if results:
            print("\nOriginal Kaggle training results:")
            if 'svm_metrics' in results:
                svm_metrics = results['svm_metrics']
                print(f"  SVM accuracy: {svm_metrics.get('accuracy', 'N/A'):.4f}")
            if 'knn_metrics' in results:
                knn_metrics = results['knn_metrics']
                print(f"  k-NN accuracy: {knn_metrics.get('accuracy', 'N/A'):.4f}")
            if 'training_time' in results:
                print(f"  Training time: {results['training_time']:.1f}s")

        print("\nYou can now use these models with:")
        print(f"  python scripts/evaluate.py --dataset ./dataset --models {args.output}")
        print(f"  python scripts/run_camera.py --models {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Integration failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1

if __name__ == '__main__':
    sys.exit(main())
