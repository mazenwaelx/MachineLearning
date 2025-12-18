#!/usr/bin/env python
"""Real-time camera classification script for Material Stream Identification System.

This script initializes the CameraClassifier and starts real-time classification
of waste materials using a connected camera device.

Usage:
    python scripts/run_camera.py --model models/svm_model.joblib
    python scripts/run_camera.py --model models/knn_model.joblib --model-type knn --camera 1
"""

import argparse
import logging
import sys
import os
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.realtime.camera import CameraClassifier, CameraNotFoundError
from src.pipeline.inference import ModelLoadError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Real-time waste material classification using camera feed.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default SVM model and camera 0
    python scripts/run_camera.py --model models/svm_model.joblib

    # Run with k-NN model and camera 1
    python scripts/run_camera.py --model models/knn_model.joblib --model-type knn --camera 1

    # Run with custom thresholds
    python scripts/run_camera.py --model models/svm_model.joblib --confidence 0.6 --blur 150
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to the trained classifier model file (e.g., models/svm_model.joblib)'
    )
    
    parser.add_argument(
        '--model-type', '-t',
        type=str,
        choices=['svm', 'knn'],
        default=None,
        help='Type of classifier model (default: auto-detect from filename)'
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--scaler', '-s',
        type=str,
        default=None,
        help='Path to the feature scaler file (optional, auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--confidence', '-conf',
        type=float,
        default=0.3,
        help='Confidence threshold for classification (default: 0.3, lower is more lenient)'
    )
    
    parser.add_argument(
        '--blur', '-b',
        type=float,
        default=20.0,
        help='Blur threshold for image quality check (default: 20.0, lower is more lenient)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Target frames per second (minimum: 10, default: 10)'
    )
    
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available camera devices and exit'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save captured images (default: camera_captures). Press "s" during capture to save frames.'
    )
    
    return parser.parse_args()


def list_available_cameras() -> None:
    """Detect and list available camera devices."""
    import cv2
    
    print("\nDetecting available cameras...")
    print("-" * 40)
    
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  Camera {i}: {width}x{height} @ {fps:.1f} FPS")
            cap.release()
    
    if not available:
        print("  No cameras detected")
    else:
        print("-" * 40)
        print(f"Found {len(available)} camera(s): {available}")
    print()


def auto_detect_scaler(model_path: str) -> Optional[str]:
    """Auto-detect scaler path based on model path.
    
    Args:
        model_path: Path to the model file.
        
    Returns:
        Path to the scaler file if found, None otherwise.
    """
    model_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)
    
    # Check if this is an ultra or final model
    is_ultra = 'ultra' in model_filename.lower()
    is_final = 'final' in model_filename.lower()
    
    # Look for matching scaler
    if os.path.isdir(model_dir):
        scaler_files = [f for f in os.listdir(model_dir) if 'scaler' in f.lower()]
        
        if is_final:
            # Prefer final scaler for final models
            final_scalers = [f for f in scaler_files if 'final' in f.lower()]
            if final_scalers:
                # Try to match timestamp if possible
                if '_final_' in model_filename:
                    timestamp = model_filename.split('_final_')[1].split('.')[0]
                    matching = [f for f in final_scalers if timestamp in f]
                    if matching:
                        return os.path.join(model_dir, matching[0])
                return os.path.join(model_dir, final_scalers[0])
        elif is_ultra:
            # Prefer ultra scaler for ultra models
            ultra_scalers = [f for f in scaler_files if 'ultra' in f.lower()]
            if ultra_scalers:
                # Try to match timestamp if possible
                if '_ultra_' in model_filename:
                    timestamp = model_filename.split('_ultra_')[1].split('.')[0]
                    matching = [f for f in ultra_scalers if timestamp in f]
                    if matching:
                        return os.path.join(model_dir, matching[0])
                return os.path.join(model_dir, ultra_scalers[0])
        
        # Fallback to any scaler
        if scaler_files:
            return os.path.join(model_dir, scaler_files[0])
    
    return None


def main() -> int:
    """Main entry point for real-time camera classification.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_args()
    
    # Handle list cameras option
    if args.list_cameras:
        list_available_cameras()
        return 0
    
    # Validate model path
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        print(f"\nError: Model file not found: {args.model}")
        print("Please provide a valid path to a trained model file.")
        return 1
    
    # Auto-detect model type if not provided
    model_type = args.model_type
    if model_type is None:
        model_filename = os.path.basename(args.model).lower()
        if 'svm' in model_filename or 'knn' not in model_filename:
            model_type = 'svm'
        else:
            model_type = 'knn'
        logger.info(f"Auto-detected model type: {model_type}")
    
    # Auto-detect scaler if not provided
    scaler_path = args.scaler
    if scaler_path is None:
        scaler_path = auto_detect_scaler(args.model)
        if scaler_path:
            logger.info(f"Auto-detected scaler: {scaler_path}")
        else:
            logger.warning("No scaler found - features may not be normalized correctly")
    
    # Initialize classifier
    classifier = None
    try:
        print("\n" + "=" * 50)
        print("Material Stream Identification System")
        print("=" * 50)
        print(f"Model: {args.model}")
        print(f"Model Type: {model_type.upper()}")
        print(f"Camera ID: {args.camera}")
        print(f"Confidence Threshold: {args.confidence}")
        print(f"Blur Threshold: {args.blur}")
        print(f"Target FPS: {args.fps}")
        if 'ultra' in os.path.basename(args.model).lower():
            print("Ultra Model: Using 96x96 image size and 158-feature extractor")
        print("=" * 50)
        
        classifier = CameraClassifier(
            model_path=args.model,
            camera_id=args.camera,
            model_type=model_type,
            scaler_path=scaler_path,
            confidence_threshold=args.confidence,
            blur_threshold=args.blur,
            target_fps=args.fps,
            save_directory=args.save_dir
        )
        
        # Start real-time classification loop
        classifier.start()
        
        return 0
        
    except CameraNotFoundError as e:
        logger.error(f"Camera error: {e}")
        print(f"\nCamera Error: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check if a camera is connected")
        print("  2. Try a different camera ID with --camera <id>")
        print("  3. Run with --list-cameras to see available devices")
        return 2
        
    except ModelLoadError as e:
        logger.error(f"Model loading error: {e}")
        print(f"\nModel Error: {e}")
        print("\nMake sure you have trained a model first:")
        print("  python scripts/train.py --dataset dataset/")
        return 3
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\nShutting down...")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")
        return 4
        
    finally:
        # Ensure cleanup happens
        if classifier is not None:
            classifier.stop()
            print("Cleanup complete. Goodbye!")


if __name__ == '__main__':
    sys.exit(main())
