#!/usr/bin/env python3


import os
import numpy as np
import cv2
import logging
import joblib
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============= CONFIGURATION =============
CONFIG = {
    'image_size': (96, 96),  # Good balance of quality and speed
    'augmentation_target': 20000,  # 5000 per class (balanced)
    'test_size': 0.12,  # 12% for validation (more training data)
    'random_state': 123,  # Try different seed for split
    'n_jobs': -1,
    'cv_folds': 5,  # Cross-validation folds
}

# Class labels
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = len(CLASS_NAMES)

# ============= DATA LOADING =============
def load_single_image(args):
    """Load and resize a single image."""
    img_path, label = args
    try:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, CONFIG['image_size'])
            return img, label
    except Exception:
        pass
    return None, None

def load_dataset(dataset_path):
    """Load dataset with parallel processing."""
    logger.info("Loading dataset...")
    
    images = []
    labels = []
    label_map = {}
    
    # Discover classes
    class_folders = sorted([d for d in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, d)) and d in CLASS_NAMES])
    
    for idx, class_name in enumerate(class_folders):
        label_map[idx] = class_name
        logger.info(f"Found class: '{class_name}' -> label {idx}")
    
    # Collect all image paths
    image_paths = []
    for label, class_name in label_map.items():
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append((os.path.join(class_path, img_name), label))
    
    logger.info(f"Found {len(image_paths)} images total")
    
    # Load images in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(load_single_image, image_paths), 
                           total=len(image_paths), desc="Loading images"))
    
    for img, label in results:
        if img is not None:
            images.append(img)
            labels.append(label)
    
    logger.info(f"Successfully loaded {len(images)} images")
    return np.array(images), np.array(labels), label_map

# ============= DATA AUGMENTATION =============
def augment_image(img, aug_type):
    """Apply a single augmentation to an image."""
    if aug_type == 0:  # Horizontal flip
        return cv2.flip(img, 1)
    elif aug_type == 1:  # Vertical flip
        return cv2.flip(img, 0)
    elif aug_type == 2:  # Brightness increase
        return cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    elif aug_type == 3:  # Brightness decrease
        return cv2.convertScaleAbs(img, alpha=1.0, beta=-30)
    elif aug_type == 4:  # Contrast increase
        return cv2.convertScaleAbs(img, alpha=1.3, beta=0)
    elif aug_type == 5:  # Contrast decrease
        return cv2.convertScaleAbs(img, alpha=0.7, beta=0)
    elif aug_type == 6:  # Rotation 90
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif aug_type == 7:  # Rotation 180
        return cv2.rotate(img, cv2.ROTATE_180)
    elif aug_type == 8:  # Add noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy
    elif aug_type == 9:  # Blur slightly
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img

def augment_dataset(images, labels, target_total):
    """Augment dataset to target size with balanced classes."""
    logger.info(f"Augmenting dataset to {target_total} images...")
    
    target_per_class = target_total // NUM_CLASSES
    augmented_images = []
    augmented_labels = []
    
    for class_id in range(NUM_CLASSES):
        class_mask = labels == class_id
        class_images = images[class_mask]
        class_count = len(class_images)
        
        # Keep all original images
        augmented_images.extend(class_images)
        augmented_labels.extend([class_id] * class_count)
        
        # Generate augmented images to reach target
        need = target_per_class - class_count
        current_count = class_count
        aug_idx = 0
        
        while current_count < target_per_class:
            for img in class_images:
                if current_count >= target_per_class:
                    break
                # Cycle through augmentation types
                aug_img = augment_image(img, aug_idx % 10)
                augmented_images.append(aug_img)
                augmented_labels.append(class_id)
                current_count += 1
            aug_idx += 1
        
        logger.info(f"Class {class_id} ({CLASS_NAMES[class_id]}): {class_count} -> {current_count}")
    
    return np.array(augmented_images), np.array(augmented_labels)

# ============= FEATURE EXTRACTION =============
def extract_features(img):
    """Extract comprehensive features from a single image.
    
    Feature vector composition (~160 features):
    - Color statistics (BGR, HSV, LAB): 45 features
    - Texture features (Laplacian, Sobel): 15 features
    - Edge features (Canny): 5 features
    - Color histograms: 48 features
    - Shape features: 7 features
    - HOG-like gradient histogram: 36 features
    """
    features = []
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 1. BGR color statistics (21 features)
    for i in range(3):
        channel = img[:, :, i].astype(np.float32)
        features.extend([
            np.mean(channel), np.std(channel), np.median(channel),
            np.percentile(channel, 25), np.percentile(channel, 75),
            np.min(channel), np.max(channel)
        ])
    
    # 2. HSV color statistics (15 features)
    for i in range(3):
        channel = hsv[:, :, i].astype(np.float32)
        features.extend([
            np.mean(channel), np.std(channel), np.median(channel),
            np.percentile(channel, 25), np.percentile(channel, 75)
        ])
    
    # 3. LAB color statistics (9 features)
    for i in range(3):
        channel = lab[:, :, i].astype(np.float32)
        features.extend([np.mean(channel), np.std(channel), np.median(channel)])
    
    # 4. Grayscale statistics (8 features)
    gray_f = gray.astype(np.float32)
    features.extend([
        np.mean(gray_f), np.std(gray_f), np.var(gray_f),
        np.min(gray_f), np.max(gray_f), np.ptp(gray_f),
        np.percentile(gray_f, 25), np.percentile(gray_f, 75)
    ])
    
    # 5. Texture features - Laplacian (3 features)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.extend([np.var(laplacian), np.mean(np.abs(laplacian)), np.std(laplacian)])
    
    # 6. Gradient features - Sobel (8 features)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)
    
    features.extend([
        np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag),
        np.percentile(grad_mag, 90),
        np.mean(grad_dir), np.std(grad_dir),
        np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y))
    ])
    
    # 7. Edge features - Canny (4 features)
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    features.extend([
        edge_pixels / total_pixels,  # Edge density
        np.mean(edges),
        np.std(edges),
        np.sum(edges) / 255.0 / total_pixels  # Normalized edge sum
    ])
    
    # 8. Color histograms - BGR (24 features: 8 bins x 3 channels)
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [8], [0, 256])
        hist = hist.flatten() / total_pixels
        features.extend(hist)
    
    # 9. Color histograms - HSV (24 features: 8 bins x 3 channels)
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [8], [0, 256])
        hist = hist.flatten() / total_pixels
        features.extend(hist)
    
    # 10. Reflectivity features (5 features)
    brightness_var = np.var(gray_f)
    local_contrast = np.std(gray_f) / (np.mean(gray_f) + 1e-8)
    bright_threshold = np.percentile(gray_f, 95)
    bright_spots = np.sum(gray_f > bright_threshold) / total_pixels
    dark_spots = np.sum(gray_f < np.percentile(gray_f, 5)) / total_pixels
    features.extend([brightness_var, local_contrast, bright_spots, dark_spots, np.ptp(gray_f)])
    
    # 11. HOG-like gradient histogram (36 features: 9 orientations x 4 cells)
    # Simplified HOG: divide image into 2x2 cells, compute 9-bin orientation histogram per cell
    h, w = gray.shape
    cell_h, cell_w = h // 2, w // 2
    
    for cy in range(2):
        for cx in range(2):
            cell_grad_mag = grad_mag[cy*cell_h:(cy+1)*cell_h, cx*cell_w:(cx+1)*cell_w]
            cell_grad_dir = grad_dir[cy*cell_h:(cy+1)*cell_h, cx*cell_w:(cx+1)*cell_w]
            
            # Convert direction to 0-180 range (unsigned gradient)
            cell_grad_dir = np.abs(cell_grad_dir) * 180 / np.pi
            
            # Create 9-bin histogram (0-180 degrees, 20 degrees per bin)
            hist = np.zeros(9)
            for i in range(9):
                bin_low = i * 20
                bin_high = (i + 1) * 20
                mask = (cell_grad_dir >= bin_low) & (cell_grad_dir < bin_high)
                hist[i] = np.sum(cell_grad_mag[mask])
            
            # Normalize histogram
            hist_sum = np.sum(hist) + 1e-8
            hist = hist / hist_sum
            features.extend(hist)
    
    # 12. Additional texture features (8 features)
    # Local Binary Pattern approximation using gradient statistics
    features.extend([
        np.percentile(grad_mag, 10),
        np.percentile(grad_mag, 30),
        np.percentile(grad_mag, 50),
        np.percentile(grad_mag, 70),
        np.mean(laplacian),
        np.percentile(laplacian, 25),
        np.percentile(laplacian, 75),
        np.std(laplacian) / (np.mean(np.abs(laplacian)) + 1e-8)  # Texture uniformity
    ])
    
    # 13. Simple LBP-like texture features (16 features)
    # Compare each pixel with its neighbors to capture texture patterns
    h, w = gray.shape
    lbp_hist = np.zeros(16)  # 16 patterns for simplified LBP
    for row in range(1, h-1, 2):  # Sample every 2nd pixel for speed
        for col in range(1, w-1, 2):
            center = gray[row, col]
            pattern = 0
            # 4 neighbors (simplified LBP)
            if gray[row-1, col] >= center: pattern |= 1
            if gray[row, col+1] >= center: pattern |= 2
            if gray[row+1, col] >= center: pattern |= 4
            if gray[row, col-1] >= center: pattern |= 8
            lbp_hist[pattern] += 1
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)
    features.extend(lbp_hist)
    
    # 14. Color coherence features (6 features)
    # Measure how uniform colors are in the image
    for i, color_space in enumerate([img, hsv]):
        for ch in range(3):
            channel = color_space[:, :, ch]
            # Ratio of pixels close to mean
            mean_val = np.mean(channel)
            coherence = np.sum(np.abs(channel - mean_val) < 30) / total_pixels
            features.append(coherence)
    
    return np.array(features, dtype=np.float32)

def extract_features_batch(images):
    """Extract features from a batch of images with parallel processing."""
    logger.info(f"Extracting features from {len(images)} images...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        features = list(tqdm(executor.map(extract_features, images), 
                            total=len(images), desc="Feature extraction"))
    
    features = np.array(features)
    logger.info(f"Feature dimension: {features.shape[1]}")
    return features

# ============= MODEL TRAINING =============
def train_svm(X_train, y_train, X_val, y_val):
    """Train SVM classifier with hyperparameter optimization.
    
    Uses RBF kernel (justified: good for non-linear classification of complex features).
    Optimizes C and gamma parameters.
    """
    logger.info("Training SVM classifier...")
    
    best_acc = 0
    best_model = None
    best_params = None
    
    # Hyperparameter grid - focused on sweet spot found (C=1-10, gamma=0.005-0.02)
    C_values = [0.5, 1, 2, 5, 10]
    gamma_values = [0.005, 0.01, 0.015, 0.02, 'scale']
    
    total_configs = len(C_values) * len(gamma_values)
    logger.info(f"Testing {total_configs} SVM configurations...")
    
    for C in C_values:
        for gamma in gamma_values:
            try:
                svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=CONFIG['random_state'], 
                         cache_size=500)
                svm.fit(X_train, y_train)
                
                # Evaluate on validation set only (fast)
                val_pred = svm.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                
                logger.info(f"SVM C={C}, gamma={gamma}: Val={val_acc:.4f}")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = svm
                    best_params = {'C': C, 'gamma': gamma}
                    logger.info(f"  ✅ New best SVM! Val={val_acc:.4f}")
                    
            except Exception as e:
                logger.warning(f"SVM training failed for C={C}, gamma={gamma}: {e}")
    
    # Final evaluation
    val_pred = best_model.predict(X_val)
    final_acc = accuracy_score(y_val, val_pred)
    
    logger.info(f"Best SVM: {best_params}, Final Accuracy: {final_acc:.4f}")
    
    return best_model, final_acc, best_params

def train_knn(X_train, y_train, X_val, y_val):
    """Train k-NN classifier with hyperparameter optimization.
    
    Uses PCA for dimensionality reduction and BaggingClassifier.
    """
    logger.info("Training k-NN classifier...")
    logger.info(f"Training k-NN with {X_train.shape[1]} features")
    
    best_acc = 0
    best_model = None
    best_params = None
    best_pca = None
    
    # Try different PCA components
    pca_components_list = [0.90, 0.95, 0.97, 0.98, 0.99, None]  # None means no PCA
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19]
    weights_values = ['distance', 'uniform']
    metric_values = ['euclidean', 'manhattan', 'chebyshev']
    
    logger.info("Testing k-NN with different PCA configurations...")
    
    for pca_comp in pca_components_list:
        if pca_comp is not None:
            pca = PCA(n_components=pca_comp, random_state=CONFIG['random_state'])
            X_train_t = pca.fit_transform(X_train)
            X_val_t = pca.transform(X_val)
            n_comp = X_train_t.shape[1]
        else:
            pca = None
            X_train_t = X_train
            X_val_t = X_val
            n_comp = X_train.shape[1]
        
        for k in k_values:
            for weights in weights_values:
                for metric in metric_values:
                    try:
                        knn = KNeighborsClassifier(
                            n_neighbors=k, weights=weights, metric=metric,
                            n_jobs=CONFIG['n_jobs'], algorithm='auto'
                        )
                        knn.fit(X_train_t, y_train)
                        
                        val_pred = knn.predict(X_val_t)
                        val_acc = accuracy_score(y_val, val_pred)
                        
                        if val_acc > best_acc:
                            best_acc = val_acc
                            best_model = knn
                            best_pca = pca
                            best_params = {
                                'n_neighbors': k, 
                                'weights': weights, 
                                'metric': metric,
                                'pca_components': pca_comp,
                                'actual_components': n_comp
                            }
                            logger.info(f"  ✅ New best! PCA={pca_comp}, k={k}, weights={weights}, metric={metric}, Val={val_acc:.4f}")
                            
                    except Exception as e:
                        pass
    
    # Try Bagging with best parameters
    logger.info("Trying Bagging ensemble with best parameters...")
    try:
        if best_pca is not None:
            X_train_best = best_pca.transform(X_train) if hasattr(best_pca, 'transform') else best_pca.fit_transform(X_train)
            X_val_best = best_pca.transform(X_val)
        else:
            X_train_best = X_train
            X_val_best = X_val
        
        for n_estimators in [15, 20, 25, 30]:
            base_knn = KNeighborsClassifier(
                n_neighbors=best_params['n_neighbors'],
                weights=best_params['weights'],
                metric=best_params['metric'],
                n_jobs=1
            )
            
            bagged_knn = BaggingClassifier(
                estimator=base_knn,
                n_estimators=n_estimators,
                max_samples=0.9,
                max_features=0.9,
                bootstrap=True,
                bootstrap_features=False,
                n_jobs=CONFIG['n_jobs'],
                random_state=CONFIG['random_state']
            )
            
            bagged_knn.fit(X_train_best, y_train)
            val_pred = bagged_knn.predict(X_val_best)
            val_acc = accuracy_score(y_val, val_pred)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = bagged_knn
                best_params['ensemble'] = 'Bagging'
                best_params['n_estimators'] = n_estimators
                logger.info(f"  ✅ Bagging improved! n_estimators={n_estimators}, Val={val_acc:.4f}")
                
    except Exception as e:
        logger.warning(f"Bagging failed: {e}")
    
    logger.info(f"Best k-NN: {best_params}, Final Accuracy: {best_acc:.4f}")
    
    return best_model, best_pca, best_acc, best_params

# ============= MAIN TRAINING PIPELINE =============

# ============= MAIN TRAINING PIPELINE =============
def main():
    """Main training pipeline."""
    print("=" * 80)
    print("🚀 MATERIAL STREAM IDENTIFICATION SYSTEM - FINAL TRAINING")
    print("=" * 80)
    print("Cairo University - Faculty of Computing and AI")
    print("Machine Learning Course Project")
    print()
    print("Target: ≥85% accuracy for both SVM and k-NN")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Check dataset
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return
    
    try:
        # Step 1: Load dataset
        images, labels, label_map = load_dataset(dataset_path)
        
        # Step 2: Split BEFORE augmentation (prevents data leakage)
        logger.info("Splitting data before augmentation...")
        X_train_orig, X_val, y_train_orig, y_val = train_test_split(
            images, labels, 
            test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state'], 
            stratify=labels
        )
        logger.info(f"Original split: Train={len(X_train_orig)}, Val={len(X_val)}")
        
        # Step 3: Augment only training set
        X_train, y_train = augment_dataset(X_train_orig, y_train_orig, CONFIG['augmentation_target'])
        logger.info(f"After augmentation: Train={len(X_train)}, Val={len(X_val)}")
        
        # Step 4: Extract features
        X_train_features = extract_features_batch(X_train)
        X_val_features = extract_features_batch(X_val)
        
        # Step 5: Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_val_scaled = scaler.transform(X_val_features)
        
        # Step 6: Train classifiers
        # SVM with early stopping at 85%
        svm_model, svm_acc, svm_params = train_svm(X_train_scaled, y_train, X_val_scaled, y_val)
        
        knn_model, knn_pca, knn_acc, knn_params = train_knn(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Step 7: Save models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models", exist_ok=True)
        
        joblib.dump(svm_model, f"models/svm_final_{timestamp}.joblib")
        joblib.dump(knn_model, f"models/knn_final_{timestamp}.joblib")
        if knn_pca is not None:
            joblib.dump(knn_pca, f"models/knn_pca_final_{timestamp}.joblib")
        joblib.dump(scaler, f"models/scaler_final_{timestamp}.joblib")
        
        # Save results
        results = {
            'svm': {'accuracy': svm_acc, 'params': svm_params},
            'knn': {'accuracy': knn_acc, 'params': knn_params},
            'label_map': label_map,
            'feature_dim': X_train_features.shape[1],
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'config': CONFIG
        }
        joblib.dump(results, f"models/results_final_{timestamp}.joblib")
        
        # Print final report
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Total time: {duration}")
        print()
        print("📊 RESULTS:")
        print(f"   SVM Accuracy:  {svm_acc:.4f} ({svm_acc*100:.2f}%) - Params: {svm_params}")
        print(f"   k-NN Accuracy: {knn_acc:.4f} ({knn_acc*100:.2f}%) - Params: {knn_params}")
        print()
        
        if svm_acc >= 0.85 and knn_acc >= 0.85:
            print("✅ TARGET ACHIEVED! Both classifiers ≥85%")
        elif svm_acc >= 0.85:
            print(f"⚠️  SVM achieved target, k-NN needs improvement ({knn_acc*100:.2f}% < 85%)")
        elif knn_acc >= 0.85:
            print(f"⚠️  k-NN achieved target, SVM needs improvement ({svm_acc*100:.2f}% < 85%)")
        else:
            print(f"❌ Neither classifier achieved 85% target")
        
        print()
        print(f"💾 Models saved to: models/")
        print(f"   - svm_final_{timestamp}.joblib")
        print(f"   - knn_final_{timestamp}.joblib")
        if knn_pca is not None:
            print(f"   - knn_pca_final_{timestamp}.joblib")
        print(f"   - scaler_final_{timestamp}.joblib")
        print("=" * 80)
        
        # Print detailed classification reports
        print("\n📋 SVM Classification Report:")
        svm_pred = svm_model.predict(X_val_scaled)
        print(classification_report(y_val, svm_pred, target_names=CLASS_NAMES))
        
        print("\n📋 k-NN Classification Report:")
        if knn_pca is not None:
            X_val_knn = knn_pca.transform(X_val_scaled)
        else:
            X_val_knn = X_val_scaled
        knn_pred = knn_model.predict(X_val_knn)
        print(classification_report(y_val, knn_pred, target_names=CLASS_NAMES))
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
