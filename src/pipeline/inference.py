"""Inference pipeline for Material Stream Identification System."""

import os
import logging
from typing import Any, Optional, Tuple

import cv2
import joblib
import numpy as np

from src.data.loader import CLASS_NAMES
from src.features.extractor import FeatureExtractor
from src.features.ultra_extractor import UltraFeatureExtractor, FinalFeatureExtractor
from src.classifiers.svm import SVMClassifier
from src.classifiers.knn import KNNClassifier
from src.classifiers.rejection import UnknownClassHandler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when model file cannot be loaded."""
    pass


def load_model(path: str) -> Any:
    """Load model with version checking.
    
    Args:
        path: Path to the model file.
        
    Returns:
        Loaded model object.
        
    Raises:
        ModelLoadError: If model file cannot be found or loaded.
    """
    if not os.path.exists(path):
        raise ModelLoadError(f"Model file not found: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {path}: {e}")


class InferencePipeline:
    """Inference pipeline for waste material classification.
    
    Loads trained models and provides single-image prediction with
    unknown class rejection handling.
    """
    
    _probability_warning_shown = False  # Class-level flag to show warning only once
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'svm',
        scaler_path: Optional[str] = None,
        config_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        blur_threshold: float = 100.0
    ):
        """Load trained model and feature extractor.
        
        Args:
            model_path: Path to the saved classifier model file.
            model_type: Type of classifier ('svm' or 'knn'). Default 'svm'.
            scaler_path: Path to the saved feature scaler. If None, attempts
                to find it in the same directory as the model.
            config_path: Path to the saved configuration. If None, uses defaults.
            confidence_threshold: Minimum confidence for accepting predictions.
            blur_threshold: Minimum Laplacian variance for sharp images.
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        
        # Detect if this is an "ultra" model (from KAGGLE_ULTRA_FAST_20K.py)
        self.is_ultra_model = self._detect_ultra_model(model_path)
        
        # Detect if this is a "final" model (from FINAL_TRAINING.py)
        self.is_final_model = self._detect_final_model(model_path)
        
        # Load label mapping from results file for ultra models
        self.label_map = None
        self.class_names = CLASS_NAMES.copy()  # Default class names
        if self.is_ultra_model:
            self._load_label_mapping(model_path)
        elif self.is_final_model:
            self._load_final_label_mapping(model_path)
        
        # Load classifier
        self.classifier = self._load_classifier(model_path)
        
        # Load or create feature extractor
        self.feature_extractor = self._setup_feature_extractor(
            scaler_path, config_path, model_path
        )
        
        # Initialize rejection handler (tailored to model type)
        # Different models may need different confidence thresholds
        # SVM: Uses probability estimates (more calibrated)
        # k-NN: Uses neighbor voting (may need different threshold)
        model_specific_threshold = confidence_threshold
        if self.model_type == 'knn':
            # k-NN confidence is based on neighbor voting, may be more conservative
            # Slightly lower threshold for k-NN to account for different confidence scale
            model_specific_threshold = confidence_threshold * 0.9  # 10% more lenient for k-NN
        
        self.rejection_handler = UnknownClassHandler(
            confidence_threshold=model_specific_threshold,
            blur_threshold=blur_threshold
        )
        logger.info(f"Rejection mechanism configured for {self.model_type.upper()}: "
                   f"confidence_threshold={model_specific_threshold:.2f}, blur_threshold={blur_threshold}")
        
        # Store image size from config or use default
        # Ultra models use 96x96, others use 224x224
        if self.is_ultra_model or self.is_final_model:
            self.image_size = (96, 96)
        else:
            self.image_size = (224, 224)
        
        if config_path and os.path.exists(config_path):
            config = load_model(config_path)
            self.image_size = config.get('image_size', self.image_size)
        
        # Enable adaptive normalization by default
        self._enable_adaptive_normalization = True
        
        # Normalization parameters (can be adjusted for different environments)
        self._normalization_strength = 1.0  # 0.0 = disabled, 1.0 = full strength
        self._target_brightness = 120.0
        self._target_contrast = 50.0
        
        logger.info(f"Inference pipeline initialized with {model_type} classifier")
        if self.is_final_model:
            logger.info("Detected final model - using 96x96 image size and 187-feature extractor")
            if self.label_map:
                logger.info(f"Loaded label mapping: {self.label_map}")
                logger.info(f"Class names order: {self.class_names[:-1]}")
        elif self.is_ultra_model:
            logger.info("Detected ultra model - using 96x96 image size and 158-feature extractor")
            if self.label_map:
                logger.info(f"Loaded label mapping: {self.label_map}")
                logger.info(f"Class names order: {self.class_names[:-1]}")
            else:
                logger.warning("WARNING: Label mapping not loaded - using default class names!")
                logger.warning("This may cause incorrect classifications!")

    def _load_label_mapping(self, model_path: str) -> None:
        """Load label mapping from results file for ultra models.
        
        The training script uses alphabetical sorting which may differ from
        the fixed CLASS_NAMES. This loads the actual mapping used during training.
        
        Args:
            model_path: Path to the model file.
        """
        model_dir = os.path.dirname(model_path)
        if not os.path.isdir(model_dir):
            return
        
        # Look for results file
        results_files = [f for f in os.listdir(model_dir) 
                        if 'results' in f.lower() and 'ultra' in f.lower()]
        
        if not results_files:
            logger.warning("No results file found - using alphabetical class mapping (training default)")
            # Fallback: Use alphabetical order as training script does
            # This matches: sorted(['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'])
            self.label_map = {
                'cardboard': 0,
                'glass': 1,
                'metal': 2,
                'paper': 3,
                'plastic': 4,
                'trash': 5
            }
            self.class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash', 'Unknown']
            logger.info(f"Using fallback label mapping: {self.label_map}")
            logger.info(f"Class order: {self.class_names[:-1]}")
            return
        
        # Try to find matching results file (same timestamp)
        model_filename = os.path.basename(model_path)
        if '_ultra_' in model_filename:
            timestamp = model_filename.split('_ultra_')[1].split('.')[0]
            matching = [f for f in results_files if timestamp in f]
            if matching:
                results_files = matching
        
        # Load the first matching results file
        results_path = os.path.join(model_dir, results_files[0])
        try:
            results_data = load_model(results_path)
            label_map = results_data.get('label_map', None)
            
            if label_map:
                self.label_map = label_map
                # Create class names array based on label map
                # Sort by label value to get correct order
                sorted_classes = sorted(label_map.items(), key=lambda x: x[1])
                self.class_names = [class_name.capitalize() for class_name, _ in sorted_classes]
                self.class_names.append('Unknown')  # Add Unknown at the end
                logger.info(f"Loaded label mapping from {results_files[0]}")
                logger.info(f"Class order: {self.class_names[:-1]}")
        except Exception as e:
            logger.warning(f"Failed to load label mapping from results file: {e}")
            logger.warning("Using alphabetical class mapping as fallback")
            # Fallback: Use alphabetical order as training script does
            self.label_map = {
                'cardboard': 0,
                'glass': 1,
                'metal': 2,
                'paper': 3,
                'plastic': 4,
                'trash': 5
            }
            self.class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash', 'Unknown']
            logger.info(f"Using fallback label mapping: {self.label_map}")
            logger.info(f"Class order: {self.class_names[:-1]}")
    
    def _detect_ultra_model(self, model_path: str) -> bool:
        """Detect if this is an ultra model from KAGGLE_ULTRA_FAST_20K.py.
        
        Args:
            model_path: Path to the model file.
            
        Returns:
            True if this is an ultra model, False otherwise.
        """
        # Check filename for "ultra"
        filename = os.path.basename(model_path).lower()
        if 'ultra' in filename:
            return True
        
        # Check for results file in same directory
        model_dir = os.path.dirname(model_path)
        if os.path.isdir(model_dir):
            results_files = [f for f in os.listdir(model_dir) if 'results' in f.lower() and 'ultra' in f.lower()]
            if results_files:
                return True
        
        return False
    
    def _detect_final_model(self, model_path: str) -> bool:
        """Detect if this is a final model from FINAL_TRAINING.py (187 features).
        
        Args:
            model_path: Path to the model file.
            
        Returns:
            True if this is a final model, False otherwise.
        """
        # Check filename for "final"
        filename = os.path.basename(model_path).lower()
        if 'final' in filename:
            return True
        
        # Check for results file in same directory
        model_dir = os.path.dirname(model_path)
        if os.path.isdir(model_dir):
            results_files = [f for f in os.listdir(model_dir) if 'results' in f.lower() and 'final' in f.lower()]
            if results_files:
                return True
        
        return False
    
    def _load_final_label_mapping(self, model_path: str) -> None:
        """Load label mapping from results file for final models.
        
        Args:
            model_path: Path to the model file.
        """
        model_dir = os.path.dirname(model_path)
        if not os.path.isdir(model_dir):
            return
        
        # Look for results file
        results_files = [f for f in os.listdir(model_dir) 
                        if 'results' in f.lower() and 'final' in f.lower()]
        
        if not results_files:
            logger.info("No results file found for final model - using alphabetical class mapping")
            # Fallback: Use alphabetical order as training script does
            self.label_map = {
                'cardboard': 0,
                'glass': 1,
                'metal': 2,
                'paper': 3,
                'plastic': 4,
                'trash': 5
            }
            self.class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash', 'Unknown']
            logger.info(f"Using label mapping: {self.label_map}")
            return
        
        # Try to find matching results file (same timestamp)
        model_filename = os.path.basename(model_path)
        if '_final_' in model_filename:
            timestamp = model_filename.split('_final_')[1].split('.')[0]
            matching = [f for f in results_files if timestamp in f]
            if matching:
                results_files = matching
        
        # Load the first matching results file
        results_path = os.path.join(model_dir, results_files[0])
        try:
            results_data = load_model(results_path)
            label_map = results_data.get('label_map', None)
            
            if label_map:
                self.label_map = label_map
                # Create class names array based on label map
                sorted_classes = sorted(label_map.items(), key=lambda x: x[1])
                self.class_names = [class_name.capitalize() for class_name, _ in sorted_classes]
                self.class_names.append('Unknown')
                logger.info(f"Loaded label mapping from {results_files[0]}")
                logger.info(f"Class order: {self.class_names[:-1]}")
            else:
                # Use fallback
                self.label_map = {
                    'cardboard': 0,
                    'glass': 1,
                    'metal': 2,
                    'paper': 3,
                    'plastic': 4,
                    'trash': 5
                }
                self.class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash', 'Unknown']
        except Exception as e:
            logger.warning(f"Failed to load label mapping: {e}")
            self.label_map = {
                'cardboard': 0,
                'glass': 1,
                'metal': 2,
                'paper': 3,
                'plastic': 4,
                'trash': 5
            }
            self.class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash', 'Unknown']

    def _load_classifier(self, model_path: str) -> Any:
        """Load the classifier from file.
        
        Args:
            model_path: Path to the classifier file.
            
        Returns:
            Loaded classifier instance.
            
        Raises:
            ModelLoadError: If classifier cannot be loaded.
            ValueError: If model type is invalid.
        """
        if self.model_type not in ('svm', 'knn'):
            raise ValueError(f"Invalid model type: {self.model_type}. Must be 'svm' or 'knn'")
        
        classifier = load_model(model_path)
        
        # Ultra/Final models use sklearn models directly, not custom classes
        if self.is_ultra_model or self.is_final_model:
            if self.model_type == 'svm' and not isinstance(classifier, SVC):
                logger.warning(f"Expected sklearn SVC but got {type(classifier).__name__}")
            elif self.model_type == 'knn' and not isinstance(classifier, KNeighborsClassifier):
                logger.warning(f"Expected sklearn KNeighborsClassifier but got {type(classifier).__name__}")
        else:
            # Regular models use custom classifier classes
            if self.model_type == 'svm' and not isinstance(classifier, SVMClassifier):
                logger.warning(f"Expected SVMClassifier but got {type(classifier).__name__}")
            elif self.model_type == 'knn' and not isinstance(classifier, KNNClassifier):
                logger.warning(f"Expected KNNClassifier but got {type(classifier).__name__}")
        
        return classifier
    
    def _setup_feature_extractor(
        self,
        scaler_path: Optional[str],
        config_path: Optional[str],
        model_path: str
    ):
        """Set up feature extractor with scaler.
        
        Args:
            scaler_path: Path to saved scaler, or None to auto-detect.
            config_path: Path to saved config, or None to use defaults.
            model_path: Path to model file (used for auto-detection).
            
        Returns:
            Configured FeatureExtractor, UltraFeatureExtractor, or FinalFeatureExtractor instance.
        """
        # Use final extractor for final models (187 features)
        if self.is_final_model:
            extractor = FinalFeatureExtractor()
            logger.info("Using FinalFeatureExtractor (187 features)")
        # Use ultra extractor for ultra models
        elif self.is_ultra_model:
            extractor = UltraFeatureExtractor(feature_version='auto')
            logger.info("Using UltraFeatureExtractor")
        else:
            # Load configuration if available
            config = {}
            if config_path and os.path.exists(config_path):
                config = load_model(config_path)
            
            # Create regular feature extractor with config
            extractor = FeatureExtractor(
                color_bins=config.get('color_bins', 32),
                hog_orientations=config.get('hog_orientations', 9),
                hog_pixels_per_cell=config.get('hog_pixels_per_cell', (16, 16)),
                hog_cells_per_block=config.get('hog_cells_per_block', (2, 2))
            )
        
        # Load scaler
        if scaler_path and os.path.exists(scaler_path):
            extractor.scaler = load_model(scaler_path)
            logger.info(f"Loaded scaler from: {scaler_path}")
        else:
            # Try to find scaler in same directory as model
            model_dir = os.path.dirname(model_path)
            scaler_files = [
                f for f in os.listdir(model_dir) if 'scaler' in f.lower()
            ] if os.path.isdir(model_dir) else []
            
            if scaler_files:
                # Prefer final scaler for final models
                if self.is_final_model:
                    final_scalers = [f for f in scaler_files if 'final' in f.lower()]
                    if final_scalers:
                        scaler_files = final_scalers
                        
                        # Try to match timestamp with model
                        model_filename = os.path.basename(model_path)
                        if '_final_' in model_filename:
                            timestamp = model_filename.split('_final_')[1].split('.')[0]
                            matching_scalers = [f for f in scaler_files if timestamp in f]
                            if matching_scalers:
                                scaler_files = matching_scalers
                                logger.info(f"Found matching final scaler with timestamp {timestamp}")
                # Prefer ultra scaler for ultra models
                elif self.is_ultra_model:
                    ultra_scalers = [f for f in scaler_files if 'ultra' in f.lower()]
                    if ultra_scalers:
                        scaler_files = ultra_scalers
                        
                        # Try to match timestamp with model
                        model_filename = os.path.basename(model_path)
                        if '_ultra_' in model_filename:
                            timestamp = model_filename.split('_ultra_')[1].split('.')[0]
                            matching_scalers = [f for f in scaler_files if timestamp in f]
                            if matching_scalers:
                                scaler_files = matching_scalers
                                logger.info(f"Found matching scaler with timestamp {timestamp}")
                            else:
                                logger.warning(f"No scaler found with matching timestamp {timestamp}")
                                logger.warning(f"Using most recent ultra scaler: {scaler_files[0]}")
                                logger.warning("This may cause feature dimension mismatch!")
                
                auto_scaler_path = os.path.join(model_dir, scaler_files[0])
                extractor.scaler = load_model(auto_scaler_path)
                logger.info(f"Auto-loaded scaler from: {auto_scaler_path}")
                
                # Verify feature dimension matches (for ultra models)
                # Update extractor's feature version based on scaler
                if self.is_ultra_model and hasattr(extractor.scaler, 'n_features_in_'):
                    expected_features = extractor.scaler.n_features_in_
                    # Update extractor to match scaler's expected features
                    if expected_features == 161:
                        extractor.feature_version = '161'
                        extractor.feature_dim = 161
                        logger.info(f"Using 161-feature extractor (old model)")
                    elif expected_features == 170:
                        extractor.feature_version = '170'
                        extractor.feature_dim = 170
                        logger.info(f"Using 170-feature extractor (medium model)")
                    elif expected_features == 178:
                        extractor.feature_version = '178'
                        extractor.feature_dim = 178
                        logger.info(f"Using 178-feature extractor (new model with plastic features)")
                    else:
                        logger.warning(f"Unknown feature dimension {expected_features}, using default")
            else:
                logger.warning("No scaler found. Features will not be normalized.")
        
        return extractor
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for feature extraction with adaptive normalization.
        
        Applies intelligent preprocessing to make camera images more similar to
        training data by normalizing lighting, contrast, and color distribution.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            
        Returns:
            Preprocessed image resized to target size, ensuring uint8 format.
        """
        # Resize to target dimensions first
        resized = cv2.resize(image, self.image_size)
        
        # Ensure image is uint8 (0-255) to match training
        if resized.dtype != np.uint8:
            if resized.max() <= 1.0:
                # Image is normalized (0-1), convert back to uint8
                resized = (resized * 255).astype(np.uint8)
            else:
                resized = resized.astype(np.uint8)
        
        # Apply adaptive preprocessing for better real-world robustness
        # Only apply if enabled (can be disabled for testing)
        if getattr(self, '_enable_adaptive_normalization', True):
            processed = self._adaptive_normalize(resized)
        else:
            processed = resized
        
        return processed
    
    def _adaptive_normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive normalization to improve robustness to lighting variations.
        
        This helps camera images match the training data distribution better by:
        - Always applying intelligent normalization (not just for extreme cases)
        - Normalizing brightness and contrast to match training distribution
        - Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Adjusting color balance to handle different lighting temperatures
        - Reducing false metal detections from reflections
        - Preserving glass characteristics (transparency, reflections)
        
        Args:
            image: Input image as uint8 numpy array (BGR format).
            
        Returns:
            Normalized image as uint8 numpy array.
        """
        # Apply normalization strength (0.0 = no normalization, 1.0 = full, >1.0 = extra strong)
        if self._normalization_strength <= 0:
            return image  # Skip normalization if disabled
        
        # Scale adjustment amounts by strength
        strength = min(self._normalization_strength, 2.0)  # Cap at 2x for safety
        
        # Target statistics (can be calibrated for different environments)
        # Blend between calibrated values and default training values based on strength
        default_mean = 120.0
        default_std = 50.0
        TARGET_MEAN = self._target_brightness * self._normalization_strength + default_mean * (1 - self._normalization_strength)
        TARGET_STD = self._target_contrast * self._normalization_strength + default_std * (1 - self._normalization_strength)
        
        # Calculate current statistics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Detect potential glass characteristics
        bright_threshold = np.percentile(gray, 95)
        bright_spots_ratio = np.sum(gray > bright_threshold) / gray.size
        brightness_variance = np.var(gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate local variances for transparency detection (glass shows background through it)
        h, w = gray.shape
        region_size = 8
        local_variances = []
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = gray[i:i+region_size, j:j+region_size]
                local_variances.append(np.var(region))
        
        # Improved glass detection - balanced to detect glass correctly but not too strict
        # Glass characteristics: high variance, transparency, sharp edges, specular highlights
        transparency_score = np.std(local_variances) if local_variances else 0
        very_bright_threshold = np.percentile(gray, 98)
        specular_ratio = np.sum(gray > very_bright_threshold) / gray.size
        
        # More lenient glass detection - glass can have varying characteristics
        is_likely_glass = (
            (brightness_variance > 500 or transparency_score > 150) and  # High variance OR transparency
            (specular_ratio > 0.015 or edge_density > 0.10) and  # Some specular highlights OR decent edge density
            mean_brightness > 70 and  # Not too dark
            edge_density > 0.08  # Glass has decent edge density (lowered from 0.12)
        )
        
        # Improved plastic detection - more distinct from glass
        # Plastic characteristics: moderate smoothness, characteristic sheen, moderate saturation, solid (not transparent)
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]
        mean_saturation = np.mean(saturation)
        moderate_bright = np.percentile(gray, 70)
        moderate_bright_ratio = np.sum((gray > moderate_bright) & (gray <= bright_threshold)) / gray.size
        
        # Plastic is solid (not transparent like glass) - key distinguishing feature
        transparency_score = np.std(local_variances) if local_variances else 0
        
        # More distinct plastic detection - emphasize solidity and lower edge density
        is_likely_plastic = (
            transparency_score < 120 and  # Low transparency (plastic is solid, not transparent) - KEY FEATURE
            brightness_variance < 600 and  # Less variance than glass (more uniform)
            edge_density < 0.15 and  # Lower edge density than glass
            (mean_saturation > 30 or moderate_bright_ratio > 0.10) and  # Some color or sheen
            specular_ratio < 0.012  # Fewer specular highlights than glass
        )
        
        # Metal detection - distinguish from paper
        # Metal characteristics: high reflectivity, specular highlights, sharp edges, low saturation, solid
        is_likely_metal = (
            specular_ratio > 0.018 and  # High specular highlights (metal reflects light strongly)
            edge_density > 0.10 and  # High edge density (metal has sharp edges)
            mean_saturation < 50 and  # Low saturation (metallic surfaces are often desaturated)
            transparency_score < 100 and  # Solid (not transparent)
            brightness_variance > 400  # Moderate to high variance (reflections create variance)
        )
        
        # Start with image copy
        normalized = image.copy()
        
        # Step 1: Always apply CLAHE for better contrast (but more conservative for glass)
        # This helps normalize different lighting conditions
        lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adjust CLAHE strength based on image characteristics
        if is_likely_glass:
            # More conservative for glass to preserve characteristics
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        elif is_likely_plastic:
            # Moderate for plastic - preserve sheen but normalize lighting
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        elif is_likely_metal:
            # Moderate for metal - preserve reflections but normalize lighting
            clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
        else:
            # More aggressive for other materials to normalize lighting
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Recalculate statistics after CLAHE
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Step 2: Normalize brightness to target mean
        # This is crucial for handling different lighting environments
        brightness_diff = TARGET_MEAN - mean_brightness
        
        # Apply brightness adjustment (more aggressive for non-glass, moderate for plastic and metal)
        if is_likely_glass:
            # Conservative adjustment for glass (preserve reflections)
            brightness_diff = int(np.clip(brightness_diff * strength * 0.5, -20 * strength, 20 * strength))
        elif is_likely_plastic:
            # Moderate adjustment for plastic (preserve sheen but normalize)
            brightness_diff = int(np.clip(brightness_diff * strength * 0.75, -30 * strength, 30 * strength))
        elif is_likely_metal:
            # Moderate adjustment for metal (preserve reflections but normalize)
            brightness_diff = int(np.clip(brightness_diff * strength * 0.7, -28 * strength, 28 * strength))
        else:
            # Full adjustment for other materials (scaled by strength)
            brightness_diff = int(np.clip(brightness_diff * strength, -40 * strength, 40 * strength))
        
        if abs(brightness_diff) > 5:  # Only adjust if significant difference
            normalized = cv2.add(normalized, brightness_diff)
            gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
        
        # Step 3: Normalize contrast to target std
        # This helps reduce false metal detections from reflections
        if std_brightness > 0:
            contrast_ratio = TARGET_STD / std_brightness
            
            # Apply contrast adjustment (more conservative for glass, moderate for plastic and metal)
            if is_likely_glass:
                # Limit contrast adjustment for glass
                contrast_ratio = np.clip(contrast_ratio, 0.85, 1.15)
            elif is_likely_plastic:
                # Moderate contrast adjustment for plastic (preserve sheen)
                contrast_ratio = 1.0 + (contrast_ratio - 1.0) * strength * 0.8
                contrast_ratio = np.clip(contrast_ratio, 0.8, 1.25)
            elif is_likely_metal:
                # Moderate contrast adjustment for metal (preserve reflections)
                contrast_ratio = 1.0 + (contrast_ratio - 1.0) * strength * 0.75
                contrast_ratio = np.clip(contrast_ratio, 0.8, 1.25)
            else:
                # More aggressive for other materials (scaled by strength)
                # Interpolate between no change (1.0) and full adjustment based on strength
                contrast_ratio = 1.0 + (contrast_ratio - 1.0) * strength
                contrast_ratio = np.clip(contrast_ratio, 0.7, 1.3)
            
            if abs(contrast_ratio - 1.0) > 0.05:  # Only adjust if significant difference
                # Apply contrast adjustment
                normalized = cv2.convertScaleAbs(
                    normalized,
                    alpha=contrast_ratio,
                    beta=(TARGET_MEAN - mean_brightness * contrast_ratio)
                )
        
        # Step 4: Color balance normalization (handle different lighting temperatures)
        # This helps when moving to different locations with different light sources
        b_mean = np.mean(normalized[:,:,0])  # Blue channel
        g_mean = np.mean(normalized[:,:,1])  # Green channel
        r_mean = np.mean(normalized[:,:,2])  # Red channel
        
        # Target: balanced color (all channels similar mean)
        target_color_mean = (b_mean + g_mean + r_mean) / 3.0
        
        # Calculate color balance adjustments
        b_adj = target_color_mean - b_mean
        g_adj = target_color_mean - g_mean
        r_adj = target_color_mean - r_mean
        
        # Apply color balance (conservative to avoid over-correction)
        if not is_likely_glass and (abs(b_adj) > 5 or abs(g_adj) > 5 or abs(r_adj) > 5):
            # Split channels
            b, g, r = cv2.split(normalized)
            
            # Apply adjustments (limited to prevent artifacts)
            b = cv2.add(b, int(np.clip(b_adj * 0.3, -15, 15)))
            g = cv2.add(g, int(np.clip(g_adj * 0.3, -15, 15)))
            r = cv2.add(r, int(np.clip(r_adj * 0.3, -15, 15)))
            
            # Merge back
            normalized = cv2.merge([b, g, r])
        
        # Step 5: Reduce excessive reflections that cause false metal detections
        # This is important when lighting creates strong reflections on plastic/glass
        if not is_likely_glass:
            gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            very_bright = np.percentile(gray, 98)
            very_bright_ratio = np.sum(gray > very_bright) / gray.size
            
            # If there are too many very bright pixels (likely reflections), tone them down
            if very_bright_ratio > 0.05 and very_bright > 240:
                # Create mask for very bright areas
                mask = gray > very_bright
                # Tone down bright areas slightly
                normalized[mask] = (normalized[mask] * 0.92).astype(np.uint8)
        
        return normalized
    
    def predict(self, image: np.ndarray, return_debug: bool = False) -> Tuple[int, str, float]:
        """Predict class for single image.
        
        Performs preprocessing, feature extraction, classification,
        and rejection handling for unknown/low-confidence predictions.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            return_debug: If True, returns additional debug info (for internal use).
            
        Returns:
            Tuple of (class_id, class_name, confidence) where:
                - class_id: Integer class label (0-5 for known, 6 for unknown)
                - class_name: Human-readable class name
                - confidence: Prediction confidence score (0-1)
            If return_debug=True, also returns (blur_score, raw_confidence, raw_prediction)
        """
        # Check for blur
        blur_score = self.rejection_handler.get_blur_score(image)
        is_blurry = blur_score < self.rejection_handler.blur_threshold
        
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Extract features
        features = self.feature_extractor.extract(processed)
        features = features.reshape(1, -1)
        
        # Normalize features if scaler is available
        if self.feature_extractor.scaler is not None:
            features = self.feature_extractor.transform(features)
        else:
            logger.warning("No scaler available - features may not be normalized correctly")
        
        # Get prediction and probabilities
        original_prediction = int(self.classifier.predict(features)[0])
        prediction = original_prediction  # Will be modified by post-processing if needed
        
        # Handle both sklearn models (ultra) and custom models
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = float(probabilities[prediction])
            max_confidence = float(np.max(probabilities))
            
            # INTELLIGENT POST-PROCESSING: Correct common misclassifications using material characteristics
            # Get class indices for common confusions
            glass_idx = None
            plastic_idx = None
            paper_idx = None
            metal_idx = None
            trash_idx = None
            
            for i, name in enumerate(self.class_names):
                if name.lower() == 'glass':
                    glass_idx = i
                elif name.lower() == 'plastic':
                    plastic_idx = i
                elif name.lower() == 'paper':
                    paper_idx = i
                elif name.lower() == 'metal':
                    metal_idx = i
                elif name.lower() == 'trash':
                    trash_idx = i
            
            # Analyze image characteristics for material detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            region_size = 8
            local_variances = []
            for i in range(0, h - region_size, region_size):
                for j in range(0, w - region_size, region_size):
                    region = gray[i:i+region_size, j:j+region_size]
                    local_variances.append(np.var(region))
            transparency_score = np.std(local_variances) if local_variances else 0
            brightness_variance = np.var(gray)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]
            mean_saturation = np.mean(saturation)
            
            # Calculate additional features for plastic vs glass distinction
            # Plastic bottles often have labels, caps, and more color variation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue_std = np.std(hsv[:,:,0])  # Plastic has more hue variation (labels, colors)
            value_channel = hsv[:,:,2]
            bright_pixels = np.sum(value_channel > 200) / value_channel.size  # High brightness spots
            
            # Plastic characteristics: more color variation, less pure transparency
            # Glass characteristics: uniform color/clear, high transparency, reflective edges
            
            # Correct plastic vs glass confusion
            if plastic_idx is not None and glass_idx is not None:
                plastic_prob = probabilities[plastic_idx] if probabilities is not None else 0.5
                glass_prob = probabilities[glass_idx] if probabilities is not None else 0.5
                
                # Apply correction more aggressively - plastic bottles often misclassified as glass
                # Check if predicted as glass OR probabilities are close
                should_check_plastic = (prediction == glass_idx) or (abs(plastic_prob - glass_prob) < 0.25)
                
                if should_check_plastic:
                    # Plastic bottle indicators:
                    # - Has some color variation (hue_std > 10) from labels/caps
                    # - Less uniform transparency (transparency_score < 140)
                    # - Moderate saturation (from colored plastic or labels)
                    # - Less bright spots than pure glass
                    
                    plastic_score = 0
                    glass_score = 0
                    
                    # Hue variation - plastic has more (labels, colored caps)
                    if hue_std > 15:
                        plastic_score += 2
                    elif hue_std > 8:
                        plastic_score += 1
                    else:
                        glass_score += 1
                    
                    # Transparency - glass is more uniformly transparent
                    if transparency_score < 100:
                        plastic_score += 2
                    elif transparency_score < 140:
                        plastic_score += 1
                    elif transparency_score > 160:
                        glass_score += 2
                    
                    # Brightness variance - plastic is more variable
                    if brightness_variance < 500:
                        plastic_score += 1
                    elif brightness_variance > 700:
                        glass_score += 1
                    
                    # Saturation - colored plastic has higher saturation
                    if mean_saturation > 30:
                        plastic_score += 1
                    elif mean_saturation < 15:
                        glass_score += 1
                    
                    # Bright spots - glass reflects more uniform bright spots
                    if bright_pixels > 0.15:
                        glass_score += 1
                    elif bright_pixels < 0.08:
                        plastic_score += 1
                    
                    # Make correction based on scores
                    if plastic_score >= glass_score + 2 and prediction == glass_idx:
                        prediction = plastic_idx
                        confidence = max(plastic_prob, confidence * 0.9)
                        logger.debug(f"Corrected glass->plastic: hue_std={hue_std:.1f}, transparency={transparency_score:.1f}, plastic_score={plastic_score}, glass_score={glass_score}")
                    elif glass_score >= plastic_score + 2 and prediction == plastic_idx:
                        prediction = glass_idx
                        confidence = max(glass_prob, confidence * 0.9)
                        logger.debug(f"Corrected plastic->glass: hue_std={hue_std:.1f}, transparency={transparency_score:.1f}, plastic_score={plastic_score}, glass_score={glass_score}")
            
            # Correct glass vs paper confusion
            if glass_idx is not None and paper_idx is not None:
                glass_prob = probabilities[glass_idx]
                paper_prob = probabilities[paper_idx]
                
                # If probabilities are close, use material characteristics
                if abs(glass_prob - paper_prob) < 0.15:
                    # Glass has high edge density and variance, paper has lower
                    if edge_density > 0.10 and brightness_variance > 500:
                        # Likely glass - boost glass probability
                        if prediction == paper_idx:
                            prediction = glass_idx
                            confidence = glass_prob
                            # Boost confidence when correction is made with strong indicators
                            if edge_density > 0.12 and brightness_variance > 600:
                                confidence = min(1.0, confidence + 0.05)  # Strong match = higher confidence
                            logger.debug(f"Corrected paper->glass: edge_density={edge_density:.3f}, variance={brightness_variance:.1f}")
                    elif edge_density < 0.08 and brightness_variance < 400:
                        # Likely paper - boost paper probability
                        if prediction == glass_idx:
                            prediction = paper_idx
                            confidence = paper_prob
                            # Boost confidence when correction is made with strong indicators
                            if edge_density < 0.06 and brightness_variance < 350:
                                confidence = min(1.0, confidence + 0.05)  # Strong match = higher confidence
                            logger.debug(f"Corrected glass->paper: edge_density={edge_density:.3f}, variance={brightness_variance:.1f}")
            
            # Correct metal vs paper confusion
            if metal_idx is not None and paper_idx is not None:
                metal_prob = probabilities[metal_idx]
                paper_prob = probabilities[paper_idx]
                
                # Metal characteristics: high specular highlights, high edge density, low saturation, solid
                very_bright_threshold = np.percentile(gray, 98)
                specular_ratio = np.sum(gray > very_bright_threshold) / gray.size
                
                # If probabilities are close, use material characteristics
                if abs(metal_prob - paper_prob) < 0.15:
                    # Metal has high specular highlights, high edge density, low saturation
                    # Paper has low specular highlights, low edge density, moderate saturation
                    if specular_ratio > 0.015 and edge_density > 0.10 and mean_saturation < 50:
                        # Likely metal - boost metal probability
                        if prediction == paper_idx:
                            prediction = metal_idx
                            confidence = metal_prob
                            # Boost confidence when correction is made with strong indicators
                            if specular_ratio > 0.020 and edge_density > 0.12 and mean_saturation < 40:
                                confidence = min(1.0, confidence + 0.05)  # Strong match = higher confidence
                            logger.debug(f"Corrected paper->metal: specular={specular_ratio:.4f}, edge_density={edge_density:.3f}, saturation={mean_saturation:.1f}")
                    elif specular_ratio < 0.010 and edge_density < 0.08 and mean_saturation > 30:
                        # Likely paper - boost paper probability
                        if prediction == metal_idx:
                            prediction = paper_idx
                            confidence = paper_prob
                            # Boost confidence when correction is made with strong indicators
                            if specular_ratio < 0.008 and edge_density < 0.06:
                                confidence = min(1.0, confidence + 0.05)  # Strong match = higher confidence
                            logger.debug(f"Corrected metal->paper: specular={specular_ratio:.4f}, edge_density={edge_density:.3f}, saturation={mean_saturation:.1f}")
                # Also correct if metal probability is reasonable but paper is predicted
                elif metal_prob > 0.20 and prediction == paper_idx:
                    # If metal has decent probability and material characteristics strongly suggest metal
                    if specular_ratio > 0.018 and edge_density > 0.12 and mean_saturation < 45:
                        prediction = metal_idx
                        confidence = metal_prob
                        # Strong indicators = higher confidence
                        if specular_ratio > 0.022 and edge_density > 0.14:
                            confidence = min(1.0, confidence + 0.06)  # Very strong match
                        else:
                            confidence = min(1.0, confidence + 0.03)  # Good match
                        logger.debug(f"Corrected paper->metal (strong indicators): specular={specular_ratio:.4f}, edge_density={edge_density:.3f}, saturation={mean_saturation:.1f}")
            
            # ENHANCED TRASH DETECTION: Smart multi-characteristic analysis
            # Trash characteristics: high texture variety, irregular patterns, high edge density, variable colors, high variance, spatial chaos
            if trash_idx is not None:
                trash_prob = probabilities[trash_idx]
                
                # Calculate comprehensive trash characteristics (matching new trash-specific features)
                # 1. Color variety (trash has high color variance due to mixed materials)
                color_variance = np.var([np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])])
                
                # 2. Edge distribution irregularity (trash has irregular edge patterns)
                # Divide into 6 regions (3 horizontal + 3 vertical) for better analysis
                edge_regions = [
                    np.sum(edges[:h//3, :] > 0),
                    np.sum(edges[h//3:2*h//3, :] > 0),
                    np.sum(edges[2*h//3:, :] > 0),
                    np.sum(edges[:, :w//3] > 0),
                    np.sum(edges[:, w//3:2*w//3] > 0),
                    np.sum(edges[:, 2*w//3:] > 0)
                ]
                edge_distribution_irregularity = np.std(edge_regions) if edge_regions else 0
                
                # 3. Texture complexity (trash has high texture variety)
                # Calculate local variance distribution
                region_size = 8
                local_variances = []
                for i in range(0, h - region_size, region_size):
                    for j in range(0, w - region_size, region_size):
                        region = gray[i:i+region_size, j:j+region_size]
                        local_variances.append(np.var(region))
                texture_complexity = np.std(local_variances) if local_variances else 0
                
                # 4. Pattern irregularity (trash has irregular/chaotic patterns)
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_direction = np.arctan2(grad_y, grad_x)
                pattern_irregularity = np.std(grad_direction)
                
                # 5. Multi-material indicator (trash often contains multiple distinct regions)
                hist, bins = np.histogram(gray.flatten(), bins=32)
                peak_threshold = np.max(hist) * 0.3 if len(hist) > 0 else 0
                distinct_regions = np.sum(hist > peak_threshold) if len(hist) > 0 else 0
                
                # 6. Spatial chaos (trash has chaotic spatial distribution)
                edge_coords = np.column_stack(np.where(edges > 0))
                if len(edge_coords) > 10:
                    x_coords = edge_coords[:, 1]
                    y_coords = edge_coords[:, 0]
                    spatial_chaos = np.std(x_coords) + np.std(y_coords)
                else:
                    spatial_chaos = 0
                
                # ENHANCED: Multi-level trash detection with weighted scoring
                # Score each characteristic and combine for robust detection
                trash_score = 0.0
                max_score = 0.0
                
                # Edge density (weight: 0.20)
                if edge_density > 0.12:
                    trash_score += 0.20 * min(1.0, (edge_density - 0.12) / 0.08)  # Normalize to 0-1
                max_score += 0.20
                
                # Brightness variance (weight: 0.15)
                if brightness_variance > 400:
                    trash_score += 0.15 * min(1.0, (brightness_variance - 400) / 400)  # Normalize
                max_score += 0.15
                
                # Color variety (weight: 0.15)
                if color_variance > 500:
                    trash_score += 0.15 * min(1.0, (color_variance - 500) / 500)  # Normalize
                max_score += 0.15
                
                # Edge distribution irregularity (weight: 0.15)
                if edge_distribution_irregularity > 1000:
                    trash_score += 0.15 * min(1.0, (edge_distribution_irregularity - 1000) / 2000)  # Normalize
                max_score += 0.15
                
                # Texture complexity (weight: 0.15)
                if texture_complexity > 200:
                    trash_score += 0.15 * min(1.0, (texture_complexity - 200) / 400)  # Normalize
                max_score += 0.15
                
                # Pattern irregularity (weight: 0.10)
                if pattern_irregularity > 1.0:
                    trash_score += 0.10 * min(1.0, (pattern_irregularity - 1.0) / 1.0)  # Normalize
                max_score += 0.10
                
                # Spatial chaos (weight: 0.10)
                if spatial_chaos > 50:
                    trash_score += 0.10 * min(1.0, (spatial_chaos - 50) / 100)  # Normalize
                max_score += 0.10
                
                # Normalize trash score to 0-1
                trash_confidence = trash_score / max_score if max_score > 0 else 0.0
                
                # Determine if likely trash (threshold: 0.60 = 60% of characteristics match)
                is_likely_trash = trash_confidence > 0.60
                is_very_likely_trash = trash_confidence > 0.75  # Strong indicators
                
                # Check against each other class with smarter logic
                for other_idx, other_name in enumerate(self.class_names):
                    if other_idx == trash_idx or other_idx >= len(probabilities):
                        continue
                    
                    other_prob = probabilities[other_idx]
                    other_name_lower = other_name.lower()
                    
                    # ENHANCED: More aggressive correction when trash characteristics are strong
                    # If probabilities are close (within 20% for trash) and characteristics strongly suggest trash
                    prob_gap = abs(trash_prob - other_prob)
                    if prob_gap < 0.20:  # Increased from 0.15 to 0.20 for more aggressive correction
                        if is_likely_trash and prediction == other_idx:
                            # Boost trash if characteristics strongly suggest trash
                            prediction = trash_idx
                            confidence = trash_prob
                            # Strong indicators = higher confidence boost
                            if is_very_likely_trash:
                                confidence = min(1.0, confidence + 0.08)  # Very strong match
                            elif trash_confidence > 0.70:
                                confidence = min(1.0, confidence + 0.05)  # Strong match
                            else:
                                confidence = min(1.0, confidence + 0.03)  # Good match
                            logger.debug(f"Corrected {other_name}->trash: score={trash_confidence:.2f}, edge_density={edge_density:.3f}, variance={brightness_variance:.1f}, color_var={color_variance:.1f}, chaos={spatial_chaos:.1f}")
                
                # ENHANCED: More aggressive boosting when trash has decent probability and strong characteristics
                if trash_prob > 0.15 and is_likely_trash and prediction != trash_idx:  # Lowered threshold from 0.20 to 0.15
                    # If trash has decent probability and characteristics strongly suggest trash
                    if trash_confidence > 0.65:  # More lenient threshold
                        prediction = trash_idx
                        confidence = trash_prob
                        if is_very_likely_trash:
                            confidence = min(1.0, confidence + 0.07)  # Very strong match
                        elif trash_confidence > 0.70:
                            confidence = min(1.0, confidence + 0.04)  # Strong match
                        else:
                            confidence = min(1.0, confidence + 0.02)  # Good match
                        logger.debug(f"Corrected to trash (strong indicators): score={trash_confidence:.2f}, edge_density={edge_density:.3f}, variance={brightness_variance:.1f}, chaos={spatial_chaos:.1f}")
            
            # ENHANCED CONFIDENCE BOOSTING: Multi-level intelligent confidence enhancement
            # This makes the model "smarter" by being more confident when it should be
            second_best_prob = float(np.partition(probabilities, -2)[-2])  # Second highest probability
            confidence_gap = confidence - second_best_prob
            
            # Calculate how dominant the prediction is
            total_prob = np.sum(probabilities)
            dominance_ratio = confidence / (total_prob + 1e-8)
            
            # Level 1: Gap-based boosting (clear predictions get more confidence)
            gap_boost = 0.0
            if confidence_gap > 0.4:  # Extremely clear prediction
                gap_boost = min(0.08, confidence_gap * 0.15)  # Up to 8% boost
            elif confidence_gap > 0.3:  # Very clear prediction
                gap_boost = min(0.06, confidence_gap * 0.12)  # Up to 6% boost
            elif confidence_gap > 0.2:  # Clear prediction
                gap_boost = min(0.04, confidence_gap * 0.10)  # Up to 4% boost
            elif confidence_gap > 0.15:  # Moderately clear
                gap_boost = min(0.03, confidence_gap * 0.08)  # Up to 3% boost
            elif confidence_gap > 0.10:  # Somewhat clear
                gap_boost = min(0.02, confidence_gap * 0.05)  # Up to 2% boost
            
            # Level 2: Dominance-based boosting (when prediction dominates all others)
            dominance_boost = 0.0
            if dominance_ratio > 0.6:  # Prediction is >60% of total probability
                dominance_boost = min(0.05, (dominance_ratio - 0.6) * 0.25)
            elif dominance_ratio > 0.5:  # Prediction is >50% of total probability
                dominance_boost = min(0.03, (dominance_ratio - 0.5) * 0.20)
            
            # Level 3: Material characteristic matching boost (if prediction matches material characteristics)
            material_boost = 0.0
            predicted_class_name = self.class_names[prediction].lower() if prediction < len(self.class_names) else ""
            
            if predicted_class_name == 'glass':
                # Glass should have high transparency, high edge density, high variance
                if transparency_score > 150 and edge_density > 0.10 and brightness_variance > 500:
                    material_boost = 0.03  # Boost confidence when characteristics match
            elif predicted_class_name == 'plastic':
                # Plastic should be solid, moderate variance, moderate edge density
                if transparency_score < 120 and brightness_variance < 600 and edge_density < 0.15:
                    material_boost = 0.03
            elif predicted_class_name == 'metal':
                # Metal should have high specular highlights, high edge density, low saturation
                if specular_ratio > 0.015 and edge_density > 0.10 and mean_saturation < 50:
                    material_boost = 0.03
            elif predicted_class_name == 'paper':
                # Paper should have low edge density, low variance, moderate saturation
                if edge_density < 0.08 and brightness_variance < 400 and mean_saturation > 20:
                    material_boost = 0.03
            elif predicted_class_name == 'trash':
                # Trash should have high edge density, high variance, high color variety, irregular patterns
                color_variance = np.var([np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])])
                edge_distribution = np.std([np.sum(edges[:h//3, :] > 0), np.sum(edges[h//3:2*h//3, :] > 0), np.sum(edges[2*h//3:, :] > 0)])
                if edge_density > 0.12 and brightness_variance > 400 and color_variance > 500 and edge_distribution > 1000:
                    material_boost = 0.03
            
            # Level 4: Correction confidence boost (if we made a correction, we're more confident)
            correction_boost = 0.0
            if prediction != original_prediction:
                # We made a correction based on material characteristics - boost confidence
                correction_boost = 0.02  # 2% boost for intelligent corrections
            
            # Apply all boosts
            total_boost = gap_boost + dominance_boost + material_boost + correction_boost
            confidence = min(1.0, confidence + total_boost)
            
            # Level 5: High confidence normalization (if confidence is already high, boost it more)
            if confidence > 0.85:
                # Very high confidence predictions get additional boost
                confidence = min(1.0, confidence + 0.02)
            elif confidence > 0.75:
                # High confidence predictions get small boost
                confidence = min(1.0, confidence + 0.01)
            
            # Level 6: Top-3 consensus boost (if top 3 predictions agree on material type)
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            top3_probs = probabilities[top3_indices]
            
            # Check if top 3 are all similar materials (e.g., all reflective, all solid, etc.)
            consensus_boost = 0.0
            if len(top3_indices) >= 3:
                top3_names = [self.class_names[i].lower() if i < len(self.class_names) else "" for i in top3_indices]
                
                # Reflective materials (metal, glass)
                reflective_classes = {'metal', 'glass'}
                # Solid materials (plastic, metal, cardboard)
                solid_classes = {'plastic', 'metal', 'cardboard'}
                
                reflective_count = sum(1 for name in top3_names if name in reflective_classes)
                solid_count = sum(1 for name in top3_names if name in solid_classes)
                
                # If top 3 are all reflective or all solid, and prediction matches, boost confidence
                predicted_name = self.class_names[prediction].lower() if prediction < len(self.class_names) else ""
                if (reflective_count >= 2 and predicted_name in reflective_classes) or \
                   (solid_count >= 2 and predicted_name in solid_classes):
                    consensus_boost = 0.02  # 2% boost for consensus
            
            confidence = min(1.0, confidence + consensus_boost)
            
            # Re-normalize probabilities after boosting (optional, for display)
            # We only boost the winning class, so probabilities won't sum to 1
            # This is fine - we're making the model more confident when it should be
        else:
            # Fallback for models without probability prediction
            # Use decision_function to get dynamic confidence for SVM
            if hasattr(self.classifier, 'decision_function'):
                decision_values = self.classifier.decision_function(features)
                if len(decision_values.shape) == 1:
                    # Binary classification
                    confidence = 1.0 / (1.0 + np.exp(-np.abs(decision_values[0])))
                else:
                    # Multi-class: use the winning class margin
                    decision_values = decision_values[0]
                    
                    # Get the winning class decision value and the runner-up
                    sorted_indices = np.argsort(decision_values)[::-1]
                    winner_value = decision_values[sorted_indices[0]]
                    runner_up_value = decision_values[sorted_indices[1]]
                    
                    # Margin between winner and runner-up indicates confidence
                    margin = winner_value - runner_up_value
                    
                    # Use sigmoid to convert margin to confidence
                    # Larger margin = higher confidence
                    # Tuned so margin of 1.0 gives ~73% confidence, 2.0 gives ~88%
                    confidence = float(1.0 / (1.0 + np.exp(-margin * 0.8)))
                    
                    # Also compute softmax probabilities for additional info
                    # Use temperature scaling to get more spread probabilities
                    temperature = 0.5  # Lower = more confident predictions
                    exp_values = np.exp((decision_values - np.max(decision_values)) / temperature)
                    probabilities = exp_values / exp_values.sum()
                    
                    # Blend margin-based and softmax confidence
                    softmax_conf = float(probabilities[prediction])
                    confidence = 0.6 * confidence + 0.4 * softmax_conf
                    max_confidence = confidence
                    
            elif hasattr(self.classifier, 'kneighbors'):
                # For k-NN: use distance-based confidence
                distances, indices = self.classifier.kneighbors(features, n_neighbors=self.classifier.n_neighbors)
                distances = distances[0]
                
                # Count how many of the k neighbors belong to predicted class
                neighbor_labels = self.classifier._y[indices[0]]
                votes_for_prediction = np.sum(neighbor_labels == prediction)
                vote_ratio = votes_for_prediction / len(neighbor_labels)
                
                # Also consider distance - closer = more confident
                # Use inverse weighted average distance
                weights = 1.0 / (distances + 1e-8)
                weighted_vote = np.sum(weights * (neighbor_labels == prediction)) / np.sum(weights)
                
                # Blend vote ratio and weighted confidence
                confidence = float(0.5 * vote_ratio + 0.5 * weighted_vote)
                max_confidence = confidence
                probabilities = None
            else:
                confidence = 0.7  # Default moderate confidence
                max_confidence = 0.7
                probabilities = None
            
            if not InferencePipeline._probability_warning_shown:
                logger.info("Using decision_function/distance-based confidence estimation")
                InferencePipeline._probability_warning_shown = True
            
            # FAST POST-PROCESSING for glass/plastic correction (only when predicted as glass)
            glass_idx = None
            plastic_idx = None
            
            for i, name in enumerate(self.class_names):
                if name.lower() == 'glass':
                    glass_idx = i
                elif name.lower() == 'plastic':
                    plastic_idx = i
            
            # Detect plastic bottles with PRINTED LABELS
            # Key insight: printed labels have LOCALIZED vivid colors with sharp edges
            # vs glass with colored liquid has DIFFUSE color throughout
            if prediction == glass_idx and plastic_idx is not None and glass_idx is not None:
                small_img = cv2.resize(image, (48, 48))
                hsv_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
                gray_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
                
                # Detect vivid colored regions (printed label colors)
                vivid_mask = hsv_small[:,:,1] > 100  # High saturation = vivid colors
                vivid_ratio = np.sum(vivid_mask) / (48 * 48)
                
                # Detect edges in vivid regions (printed labels have sharp text/graphics)
                edges = cv2.Canny(gray_small, 50, 150)
                edges_in_vivid = np.sum((edges > 0) & vivid_mask) / (np.sum(vivid_mask) + 1)
                
                # Printed labels have high edge density in colored areas
                # Colored liquid is smooth with few edges
                has_printed_label = vivid_ratio > 0.05 and edges_in_vivid > 0.15
                
                # Also check for multiple distinct colors (blue + red like Nestle label)
                hue_in_vivid = hsv_small[:,:,0][vivid_mask] if np.any(vivid_mask) else np.array([0])
                hue_range = np.ptp(hue_in_vivid) if len(hue_in_vivid) > 10 else 0
                has_multi_color = hue_range > 40  # Multiple distinct hues in label
                
                # Require printed label OR strong multi-color evidence
                # Scale threshold by confidence
                if confidence > 0.85:
                    # Very high confidence - need printed label + multi-color
                    should_correct = has_printed_label and has_multi_color
                elif confidence > 0.70:
                    # High confidence - need printed label evidence
                    should_correct = has_printed_label and (edges_in_vivid > 0.20 or has_multi_color)
                else:
                    # Lower confidence - printed label is enough
                    should_correct = has_printed_label
                
                if should_correct:
                    prediction = plastic_idx
                    logger.debug(f"Corrected glass->plastic: vivid={vivid_ratio:.1%}, edges_in_vivid={edges_in_vivid:.2f}, hue_range={hue_range:.0f}")
        
        # Store debug info for potential access
        self._last_debug_info = {
            'blur_score': blur_score,
            'is_blurry': is_blurry,
            'raw_prediction': prediction,
            'raw_confidence': confidence,
            'max_confidence': max_confidence,
            'all_probabilities': probabilities.tolist() if probabilities is not None else None
        }
        
        # Apply rejection mechanism (tailored to model type)
        # SVM: Uses predict_proba() - probability estimates are well-calibrated
        # k-NN: Uses predict_proba() - confidence based on neighbor voting proportion
        final_class, final_confidence = self.rejection_handler.apply_rejection(
            prediction, confidence, is_blurry
        )
        
        # Model-specific post-processing for rejection (tailored to each model)
        if probabilities is not None:
            if self.model_type == 'svm':
                # SVM: Probability estimates are well-calibrated
                # Additional check: if max probability is very low, reject
                max_prob = float(np.max(probabilities))
                if max_prob < 0.25:  # Very low confidence for SVM
                    final_class = 6  # Unknown
                    final_confidence = max_prob
                    logger.debug(f"SVM: Rejected due to very low max probability ({max_prob:.3f})")
            elif self.model_type == 'knn':
                # k-NN: Confidence is proportion of neighbors, may need different handling
                # Additional check: if confidence gap between top-2 is too small, prediction is uncertain
                sorted_probs = np.sort(probabilities)[::-1]
                if len(sorted_probs) >= 2:
                    prob_gap = sorted_probs[0] - sorted_probs[1]
                    # If top-2 classes are very close AND confidence is low, prediction is uncertain
                    if prob_gap < 0.15 and confidence < 0.35:  # Close probabilities + low confidence
                        final_class = 6  # Unknown
                        final_confidence = confidence
                        logger.debug(f"k-NN: Rejected due to uncertain prediction (gap={prob_gap:.3f}, conf={confidence:.3f})")
        
        # Get class name using correct mapping
        class_name = self.class_names[final_class] if final_class < len(self.class_names) else 'Unknown'
        
        if return_debug:
            return (final_class, class_name, final_confidence, blur_score, confidence, prediction)
        
        return (final_class, class_name, final_confidence)
    
    def get_last_debug_info(self) -> Optional[dict]:
        """Get debug information from the last prediction.
        
        Returns:
            Dictionary with debug info or None if no prediction has been made yet.
        """
        return getattr(self, '_last_debug_info', None)
    
    def predict_batch(self, images: list) -> list:
        """Predict classes for multiple images.
        
        Args:
            images: List of images as numpy arrays (BGR format).
            
        Returns:
            List of (class_id, class_name, confidence) tuples.
        """
        return [self.predict(img) for img in images]
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold for rejection.
        
        Args:
            threshold: New confidence threshold (0-1).
        """
        self.rejection_handler.set_confidence_threshold(threshold)
    
    def set_blur_threshold(self, threshold: float) -> None:
        """Update the blur threshold for rejection.
        
        Args:
            threshold: New blur threshold (Laplacian variance).
        """
        self.rejection_handler.set_blur_threshold(threshold)
    
    def set_adaptive_normalization(self, enable: bool) -> None:
        """Enable or disable adaptive normalization.
        
        Args:
            enable: If True, adaptive normalization is applied. If False, only basic resizing.
        """
        self._enable_adaptive_normalization = enable
    
    def set_normalization_strength(self, strength: float) -> None:
        """Set normalization strength for different environments.
        
        Args:
            strength: Normalization strength (0.0 = disabled, 1.0 = full strength, 
                     can be > 1.0 for very different environments).
        """
        self._normalization_strength = max(0.0, strength)
    
    def calibrate_normalization(self, reference_image: np.ndarray) -> None:
        """Calibrate normalization parameters based on a reference image.
        
        This helps adapt to your specific environment. Take a picture of a known
        material (e.g., white paper or neutral background) in your environment
        and use it to calibrate.
        
        Args:
            reference_image: Reference image from your environment (BGR format).
        """
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        self._target_brightness = float(np.mean(gray))
        self._target_contrast = float(np.std(gray))
        logger.info(f"Calibrated normalization: brightness={self._target_brightness:.1f}, contrast={self._target_contrast:.1f}")
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model type, path, and parameters.
        """
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'image_size': self.image_size,
            'rejection_params': self.rejection_handler.get_params(),
            'classifier_params': self.classifier.get_params() if hasattr(self.classifier, 'get_params') else {}
        }
