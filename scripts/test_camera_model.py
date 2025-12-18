import argparse
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.inference import InferencePipeline, ModelLoadError
from src.data.loader import CLASS_NAMES as DEFAULT_CLASS_NAMES

def test_image(image_path: str, model_path: str, model_type: str = 'svm', scaler_path: str = None):

    print("=" * 60)
    print("Model Diagnostic Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Model Type: {model_type}")
    print(f"Image: {image_path}")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return

    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: [{image.min()}, {image.max()}]")
    print()

    try:
        pipeline = InferencePipeline(
            model_path=model_path,
            model_type=model_type,
            scaler_path=scaler_path,
            confidence_threshold=0.1,
            blur_threshold=20.0
        )

        print(f"Pipeline initialized:")
        print(f"  - Ultra model: {pipeline.is_ultra_model}")
        print(f"  - Image size: {pipeline.image_size}")
        print(f"  - Scaler loaded: {pipeline.feature_extractor.scaler is not None}")
        print()

        processed = pipeline._preprocess_image(image)
        print(f"Processed image shape: {processed.shape}")
        print(f"Processed image dtype: {processed.dtype}")
        print(f"Processed image range: [{processed.min()}, {processed.max()}]")
        print()

        features = pipeline.feature_extractor.extract(processed)
        print(f"Features extracted:")
        print(f"  - Feature dimension: {len(features)}")
        print(f"  - Feature dtype: {features.dtype}")
        print(f"  - Feature range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"  - Feature mean: {features.mean():.2f}")
        print(f"  - Feature std: {features.std():.2f}")
        print()

        if pipeline.feature_extractor.scaler is not None:
            features_scaled = pipeline.feature_extractor.transform(features.reshape(1, -1))
            print(f"Features after scaling:")
            print(f"  - Scaled range: [{features_scaled.min():.2f}, {features_scaled.max():.2f}]")
            print(f"  - Scaled mean: {features_scaled.mean():.2f}")
            print(f"  - Scaled std: {features_scaled.std():.2f}")
            print()

        class_id, class_name, confidence = pipeline.predict(image)
        debug_info = pipeline.get_last_debug_info()

        class_names = getattr(pipeline, 'class_names', DEFAULT_CLASS_NAMES)

        print("=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Final Prediction: {class_name} (ID: {class_id})")
        print(f"Final Confidence: {confidence:.1%}")
        print()

        if debug_info:
            print("Debug Information:")
            raw_pred_id = debug_info['raw_prediction']
            raw_class_name = class_names[raw_pred_id] if raw_pred_id < len(class_names) else f"Class{raw_pred_id}"
            print(f"  - Raw Prediction: {raw_class_name} (ID: {raw_pred_id})")
            print(f"  - Raw Confidence: {debug_info['raw_confidence']:.1%}")
            print(f"  - Max Confidence: {debug_info['max_confidence']:.1%}")
            print(f"  - Blur Score: {debug_info['blur_score']:.1f}")
            print(f"  - Is Blurry: {debug_info['is_blurry']}")
            print()

            if debug_info.get('all_probabilities'):
                probs = np.array(debug_info['all_probabilities'])
                print("All Class Probabilities:")
                print("-" * 60)
                sorted_indices = np.argsort(probs)[::-1]
                for idx in sorted_indices:
                    prob = probs[idx]
                    class_name_prob = class_names[idx] if idx < len(class_names) else f"Class{idx}"
                    marker = " <-- PREDICTED" if idx == debug_info['raw_prediction'] else ""
                    print(f"  {class_name_prob:15s}: {prob:6.1%}{marker}")
                print()

        print("=" * 60)

    except ModelLoadError as e:
        print(f"ERROR: Failed to load model: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Test camera model with an image')
    parser.add_argument('--image', '-i', required=True, help='Path to test image')
    parser.add_argument('--model', '-m', required=True, help='Path to model file')
    parser.add_argument('--model-type', '-t', default='svm', choices=['svm', 'knn'], help='Model type')
    parser.add_argument('--scaler', '-s', default=None, help='Path to scaler file (auto-detected if not provided)')

    args = parser.parse_args()
    test_image(args.image, args.model, args.model_type, args.scaler)

if __name__ == '__main__':
    main()
