import argparse
import cv2
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import CLASS_MAPPING

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def capture_images_for_class(class_name: str, output_dir: str, num_images: int = 50, camera_id: int = 0):
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        return

    print(f"\n{'='*60}")
    print(f"Capturing {num_images} images for class: {class_name.upper()}")
    print(f"{'='*60}")
    print("Instructions:")
    print("  - Press SPACE to capture an image")
    print("  - Press 'q' to quit and move to next class")
    print("  - Show different examples of the material")
    print("  - Vary lighting, angles, and backgrounds")
    print(f"{'='*60}\n")

    captured = 0
    window_name = f"Capture {class_name.upper()} - Press SPACE to capture, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Class: {class_name.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Captured: {captured}/{num_images}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "SPACE = Capture | Q = Quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{class_name}_{timestamp}.jpg"
            filepath = os.path.join(class_dir, filename)

            cv2.imwrite(filepath, frame)
            captured += 1
            print(f"  ✓ Captured image {captured}/{num_images}: {filename}")

            cv2.putText(display_frame, "CAPTURED!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow(window_name, display_frame)
            cv2.waitKey(500)

        elif key == ord('q'):
            print(f"\nStopped capturing. Captured {captured} images for {class_name}.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Saved {captured} images to {class_dir}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Capture training images from camera for each material class'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='camera_training_data',
        help='Output directory for captured images (default: camera_training_data)'
    )
    parser.add_argument(
        '--num-images', '-n',
        type=int,
        default=50,
        help='Number of images to capture per class (default: 50)'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--material-class', '--class', '-cls',
        type=str,
        default=None,
        dest='material_class',
        choices=CLASS_NAMES,
        help='Capture images for specific class only (default: all classes)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Camera Training Image Capture")
    print("="*60)
    print(f"Output directory: {args.output}")
    print(f"Images per class: {args.num_images}")
    print(f"Camera ID: {args.camera}")
    print("="*60)

    classes_to_capture = [args.material_class] if args.material_class else CLASS_NAMES

    for class_name in classes_to_capture:
        capture_images_for_class(class_name, args.output, args.num_images, args.camera)
        if args.material_class:
            break

    print("\n" + "="*60)
    print("Capture Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review captured images in: {args.output}")
    print(f"2. Copy good images to your dataset folder")
    print(f"3. Retrain the model with: python KAGGLE_ULTRA_FAST_20K.py")
    print()

if __name__ == '__main__':
    main()
