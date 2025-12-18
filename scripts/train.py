import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.training import TrainingPipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train waste material classification models (SVM and k-NN)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to dataset folder containing class subfolders'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Directory to save trained models'
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
        '--train-split',
        type=float,
        default=0.8,
        help='Training set proportion (0.0 to 1.0)'
    )

    parser.add_argument(
        '--color-bins',
        type=int,
        default=32,
        help='Number of bins per channel for color histogram'
    )
    parser.add_argument(
        '--hog-orientations',
        type=int,
        default=9,
        help='Number of orientation bins for HOG features'
    )

    parser.add_argument(
        '--svm-c',
        type=float,
        nargs='+',
        default=[0.1, 1, 10, 100],
        help='C values for SVM hyperparameter tuning'
    )
    parser.add_argument(
        '--svm-gamma',
        nargs='+',
        default=['scale', 'auto', 0.01, 0.1],
        help='Gamma values for SVM hyperparameter tuning'
    )

    parser.add_argument(
        '--knn-k-min',
        type=int,
        default=3,
        help='Minimum k value for k-NN tuning'
    )
    parser.add_argument(
        '--knn-k-max',
        type=int,
        default=21,
        help='Maximum k value for k-NN tuning'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (logs to console if not specified)'
    )

    return parser.parse_args()

def setup_logging(verbose: bool, log_file: str = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def build_config(args: argparse.Namespace) -> dict:
    gamma_values = []
    for g in args.svm_gamma:
        if isinstance(g, str) and g in ('scale', 'auto'):
            gamma_values.append(g)
        else:
            try:
                gamma_values.append(float(g))
            except (ValueError, TypeError):
                gamma_values.append(g)

    return {
        'image_size': (args.image_size, args.image_size),
        'target_samples_per_class': args.samples,
        'train_split': args.train_split,
        'color_bins': args.color_bins,
        'hog_orientations': args.hog_orientations,
        'svm_c_range': args.svm_c,
        'svm_gamma_range': gamma_values,
        'knn_k_range': range(args.knn_k_min, args.knn_k_max, 2),
    }

def main() -> int:
    args = parse_args()
    setup_logging(args.verbose, args.log_file)

    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.dataset):
        logger.error(f"Dataset path does not exist: {args.dataset}")
        return 1

    logger.info("=" * 60)
    logger.info("Material Stream Identification - Training Script")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Image size: {args.image_size}x{args.image_size}")
    logger.info(f"Target samples per class: {args.samples}")
    logger.info(f"Train/Val split: {args.train_split:.0%}/{1-args.train_split:.0%}")
    logger.info("=" * 60)

    try:
        config = build_config(args)

        pipeline = TrainingPipeline(config)
        metrics = pipeline.run(args.dataset)

        logger.info("\nSaving trained models...")
        saved_paths = pipeline.save_models(args.output)

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

        for model_name in ['svm', 'knn']:
            if model_name in metrics:
                m = metrics[model_name]
                logger.info(f"\n{model_name.upper()} Results:")
                logger.info(f"  Accuracy:  {m['accuracy']:.4f}")
                logger.info(f"  Precision: {m['precision']:.4f}")
                logger.info(f"  Recall:    {m['recall']:.4f}")
                logger.info(f"  F1 Score:  {m['f1_score']:.4f}")

        logger.info("\nSaved models:")
        for name, path in saved_paths.items():
            logger.info(f"  {name}: {path}")

        best_name, _ = pipeline.get_best_model()
        best_acc = metrics[best_name]['accuracy']
        logger.info(f"\nRecommended model: {best_name.upper()} (accuracy: {best_acc:.4f})")

        return 0

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1

if __name__ == '__main__':
    sys.exit(main())
