import sys
import os
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_label_mapping(model_path: str):

    model_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)

    print("=" * 60)
    print("Label Mapping Checker")
    print("=" * 60)
    print(f"Model: {model_path}")
    print()

    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        return

    results_files = [f for f in os.listdir(model_dir) 
                    if 'results' in f.lower() and 'ultra' in f.lower()]

    if not results_files:
        print("ERROR: No results file found!")
        print("Looking for files with 'results' and 'ultra' in the name.")
        return

    if '_ultra_' in model_filename:
        timestamp = model_filename.split('_ultra_')[1].split('.')[0]
        matching = [f for f in results_files if timestamp in f]
        if matching:
            results_files = matching
            print(f"Found matching results file: {results_files[0]}")
        else:
            print(f"Using most recent results file: {results_files[0]}")
    else:
        print(f"Using most recent results file: {results_files[0]}")

    print()

    results_path = os.path.join(model_dir, results_files[0])
    try:
        results_data = joblib.load(results_path)

        print("=" * 60)
        print("RESULTS FILE CONTENTS")
        print("=" * 60)

        label_map = results_data.get('label_map', None)
        if label_map:
            print("\nLabel Mapping (from training):")
            print("-" * 60)
            sorted_map = sorted(label_map.items(), key=lambda x: x[1])
            for class_name, label_id in sorted_map:
                print(f"  {class_name:15s} -> ID {label_id}")

            print("\nClass Order (as model expects):")
            print("-" * 60)
            for idx, (class_name, _) in enumerate(sorted_map):
                print(f"  ID {idx}: {class_name.capitalize()}")
        else:
            print("WARNING: No label_map found in results file!")

        print("\nOther information:")
        print("-" * 60)
        print(f"  Feature dimension: {results_data.get('feature_dim', 'N/A')}")
        print(f"  Training samples: {results_data.get('training_samples', 'N/A')}")
        print(f"  Validation samples: {results_data.get('validation_samples', 'N/A')}")

        if 'accuracies' in results_data:
            print("\nModel Accuracies:")
            print("-" * 60)
            for model_name, acc in results_data['accuracies'].items():
                print(f"  {model_name.upper()}: {acc:.4f}")

        print("\n" + "=" * 60)
        print("IMPORTANT:")
        print("=" * 60)
        if label_map:
            print("The model was trained with the above label mapping.")
            print("Make sure the inference pipeline uses this same mapping!")
        else:
            print("WARNING: Could not find label mapping!")
            print("The model may use alphabetical sorting:")
            print("  cardboard=0, glass=1, metal=2, paper=3, plastic=4, trash=5")

    except Exception as e:
        print(f"ERROR: Failed to load results file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_label_mapping.py <model_path>")
        print("Example: python scripts/check_label_mapping.py models/svm_ultra_20251215_193443.joblib")
        sys.exit(1)

    check_label_mapping(sys.argv[1])
