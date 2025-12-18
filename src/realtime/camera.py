import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.data.loader import CLASS_NAMES as DEFAULT_CLASS_NAMES
from src.pipeline.inference import InferencePipeline, ModelLoadError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraNotFoundError(Exception):
    pass

class CameraClassifier:

    def __init__(
        self,
        model_path: str,
        camera_id: int = 0,
        model_type: str = 'svm',
        scaler_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        blur_threshold: float = 100.0,
        target_fps: int = 10,
        save_directory: Optional[str] = None
    ):
        self.camera_id = camera_id
        self.target_fps = max(target_fps, 10)
        self.frame_delay = 1.0 / self.target_fps
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

        self._prediction_history = []
        self._history_size = 5
        self._min_confidence_for_switch = 0.15

        if save_directory is None:
            save_directory = "camera_captures"
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory: {self.save_directory}")

        try:
            self.pipeline = InferencePipeline(
                model_path=model_path,
                model_type=model_type,
                scaler_path=scaler_path,
                confidence_threshold=confidence_threshold,
                blur_threshold=blur_threshold
            )
            logger.info(f"Loaded {model_type} model from: {model_path}")
        except ModelLoadError as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _initialize_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            raise CameraNotFoundError(
                f"Camera {self.camera_id} not available. "
                "Please check if a camera is connected or try a different camera ID."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        logger.info(f"Camera {self.camera_id} initialized successfully")
        return cap

    def process_frame(self, frame: np.ndarray) -> Tuple[int, str, float, float, dict]:
        start_time = time.time()

        class_id, class_name, confidence = self.pipeline.predict(frame)

        inference_time = (time.time() - start_time) * 1000

        if inference_time > 200:
            logger.warning(f"Inference time ({inference_time:.1f}ms) exceeds 200ms target")

        debug_info = self.pipeline.get_last_debug_info() or {}

        smoothed_class_id, smoothed_class_name, smoothed_confidence = self._apply_temporal_smoothing(
            class_id, class_name, confidence, debug_info
        )

        return smoothed_class_id, smoothed_class_name, smoothed_confidence, inference_time, debug_info

    def _apply_temporal_smoothing(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        debug_info: dict
    ) -> Tuple[int, str, float]:
        self._prediction_history.append({
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'all_probs': debug_info.get('all_probabilities', None)
        })

        if len(self._prediction_history) > self._history_size:
            self._prediction_history.pop(0)

        if len(self._prediction_history) < 3:
            return class_id, class_name, confidence

        recent_classes = [p['class_id'] for p in self._prediction_history[-3:]]
        most_common_class = max(set(recent_classes), key=recent_classes.count)
        most_common_count = recent_classes.count(most_common_class)

        current_pred = self._prediction_history[-1]
        previous_pred = self._prediction_history[-2] if len(self._prediction_history) > 1 else None

        if current_pred['class_id'] == most_common_class and most_common_count >= 2:
            return current_pred['class_id'], current_pred['class_name'], current_pred['confidence']

        if previous_pred and current_pred['class_id'] != previous_pred['class_id']:
            all_probs = current_pred.get('all_probs')
            if all_probs:
                probs = np.array(all_probs)
                sorted_probs = np.sort(probs)[::-1]
                confidence_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

                if confidence_gap < self._min_confidence_for_switch and most_common_count < 2:
                    return previous_pred['class_id'], previous_pred['class_name'], previous_pred['confidence']

        return current_pred['class_id'], current_pred['class_name'], current_pred['confidence']

    def display_result(
        self,
        frame: np.ndarray,
        class_name: str,
        confidence: float,
        inference_time_ms: float = 0.0,
        debug_info: Optional[dict] = None,
        stats: Optional[dict] = None
    ) -> np.ndarray:
        annotated = frame.copy()

        if class_name == "Unknown":
            color = (0, 0, 255)
            bg_color = (0, 0, 100)
        elif confidence >= 0.7:
            color = (0, 255, 0)
            bg_color = (0, 100, 0)
        elif confidence >= 0.5:
            color = (0, 255, 255)
            bg_color = (0, 100, 100)
        else:
            color = (0, 165, 255)
            bg_color = (0, 50, 100)

        accuracy_indicator = ""
        if confidence >= 0.9:
            accuracy_indicator = " [EXCELLENT]"
        elif confidence >= 0.8:
            accuracy_indicator = " [HIGH]"
        elif confidence >= 0.7:
            accuracy_indicator = " [GOOD]"
        elif confidence >= 0.5:
            accuracy_indicator = " [MEDIUM]"
        else:
            accuracy_indicator = " [LOW]"

        label_text = f"{class_name}: {confidence:.1%}{accuracy_indicator}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2 if class_name == "Unknown" else 1.0
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )

        padding = 12
        rect_x1, rect_y1 = 10, 10
        rect_x2 = rect_x1 + text_width + padding * 2
        rect_y2 = rect_y1 + text_height + padding * 2 + baseline

        cv2.rectangle(
            annotated,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            bg_color,
            -1
        )

        cv2.rectangle(
            annotated,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            color,
            2
        )

        cv2.putText(
            annotated,
            label_text,
            (rect_x1 + padding, rect_y1 + text_height + padding),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

        y_offset = rect_y2 + 25
        if inference_time_ms > 0:
            time_text = f"Time: {inference_time_ms:.1f}ms"
            time_color = (255, 255, 255) if inference_time_ms <= 200 else (0, 0, 255)
            cv2.putText(
                annotated,
                time_text,
                (rect_x1, y_offset),
                font,
                0.6,
                time_color,
                1,
                cv2.LINE_AA
            )
            y_offset += 20

        if debug_info:
            blur_score = debug_info.get('blur_score', 0)
            raw_confidence = debug_info.get('raw_confidence', 0)
            is_blurry = debug_info.get('is_blurry', False)
            all_probs = debug_info.get('all_probabilities', None)
            raw_pred = debug_info.get('raw_prediction', -1)

            class_names = getattr(self.pipeline, 'class_names', DEFAULT_CLASS_NAMES)
            label_map = getattr(self.pipeline, 'label_map', None)

            if label_map:
                id_text = f"Class ID: {raw_pred}"
                cv2.putText(
                    annotated,
                    id_text,
                    (rect_x1, y_offset),
                    font,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA
                )
                y_offset += 18

            blur_text = f"Blur: {blur_score:.1f}"
            blur_color = (0, 255, 0) if not is_blurry else (0, 0, 255)
            cv2.putText(
                annotated,
                blur_text,
                (rect_x1, y_offset),
                font,
                0.5,
                blur_color,
                1,
                cv2.LINE_AA
            )
            y_offset += 18

            if abs(raw_confidence - confidence) > 0.01:
                raw_conf_text = f"Raw: {raw_confidence:.1%}"
                cv2.putText(
                    annotated,
                    raw_conf_text,
                    (rect_x1, y_offset),
                    font,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA
                )
                y_offset += 18

            if all_probs is not None and len(all_probs) >= 6:
                class_names = getattr(self.pipeline, 'class_names', DEFAULT_CLASS_NAMES)

                prob_array = np.array(all_probs)
                top3_indices = np.argsort(prob_array)[-3:][::-1]

                top_class = class_names[top3_indices[0]] if top3_indices[0] < len(class_names) else f"Class{top3_indices[0]}"
                top_prob = prob_array[top3_indices[0]]
                top_text = f"Top: {top_class}[{top3_indices[0]}] ({top_prob:.1%})"
                top_color = (0, 255, 255) if top3_indices[0] == raw_pred else (150, 150, 150)
                cv2.putText(
                    annotated,
                    top_text,
                    (rect_x1, y_offset),
                    font,
                    0.5,
                    top_color,
                    1,
                    cv2.LINE_AA
                )
                y_offset += 18

                if len(top3_indices) > 1 and prob_array[top3_indices[1]] > 0.1:
                    second_class = class_names[top3_indices[1]] if top3_indices[1] < len(class_names) else f"Class{top3_indices[1]}"
                    second_prob = prob_array[top3_indices[1]]
                    second_text = f"2nd: {second_class}[{top3_indices[1]}] ({second_prob:.1%})"
                    cv2.putText(
                        annotated,
                        second_text,
                        (rect_x1, y_offset),
                        font,
                        0.45,
                        (120, 120, 120),
                        1,
                        cv2.LINE_AA
                    )
                    y_offset += 16

                if len(top3_indices) > 2 and prob_array[top3_indices[2]] > 0.15:
                    third_class = class_names[top3_indices[2]] if top3_indices[2] < len(class_names) else f"Class{top3_indices[2]}"
                    third_prob = prob_array[top3_indices[2]]
                    third_text = f"3rd: {third_class}[{top3_indices[2]}] ({third_prob:.1%})"
                    cv2.putText(
                        annotated,
                        third_text,
                        (rect_x1, y_offset),
                        font,
                        0.4,
                        (100, 100, 100),
                        1,
                        cv2.LINE_AA
                    )
                    y_offset += 14

            threshold_text = f"Thresh: {self.pipeline.rejection_handler.confidence_threshold:.2f}"
            cv2.putText(
                annotated,
                threshold_text,
                (rect_x1, y_offset),
                font,
                0.5,
                (150, 150, 150),
                1,
                cv2.LINE_AA
            )
            y_offset += 18

        if stats and stats.get('total_frames', 0) > 0:
            y_offset += 5
            cv2.line(annotated, (rect_x1, y_offset), (rect_x1 + 300, y_offset), (100, 100, 100), 1)
            y_offset += 12

            total = stats['total_frames']
            high_conf = stats['high_confidence_frames']
            medium_conf = stats['medium_confidence_frames']
            low_conf = stats['low_confidence_frames']
            avg_conf = stats['avg_confidence']

            high_conf_rate = (high_conf / total) * 100 if total > 0 else 0
            accuracy_text = f"Accuracy: {high_conf_rate:.1f}% (High Conf)"
            accuracy_color = (0, 255, 0) if high_conf_rate > 70 else (0, 255, 255) if high_conf_rate > 50 else (0, 165, 255)
            cv2.putText(
                annotated,
                accuracy_text,
                (rect_x1, y_offset),
                font,
                0.6,
                accuracy_color,
                2,
                cv2.LINE_AA
            )
            y_offset += 22

            avg_conf_text = f"Avg Confidence: {avg_conf:.1%}"
            avg_conf_color = (0, 255, 0) if avg_conf > 0.8 else (0, 255, 255) if avg_conf > 0.6 else (0, 165, 255)
            cv2.putText(
                annotated,
                avg_conf_text,
                (rect_x1, y_offset),
                font,
                0.55,
                avg_conf_color,
                1,
                cv2.LINE_AA
            )
            y_offset += 18

            dist_text = f"Distribution: H:{high_conf} M:{medium_conf} L:{low_conf}"
            cv2.putText(
                annotated,
                dist_text,
                (rect_x1, y_offset),
                font,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )
            y_offset += 18

            frames_text = f"Frames: {total}"
            cv2.putText(
                annotated,
                frames_text,
                (rect_x1, y_offset),
                font,
                0.5,
                (150, 150, 150),
                1,
                cv2.LINE_AA
            )

        if debug_info and debug_info.get('all_probabilities'):
            all_probs = debug_info.get('all_probabilities')
            if all_probs and len(all_probs) >= 6:
                class_names = getattr(self.pipeline, 'class_names', DEFAULT_CLASS_NAMES)

                right_x = frame.shape[1] - 220
                right_y = 10
                panel_width = 210
                panel_height = min(200, len(all_probs) * 25 + 30)

                overlay = annotated.copy()
                cv2.rectangle(
                    overlay,
                    (right_x - 5, right_y - 5),
                    (right_x + panel_width, right_y + panel_height),
                    (0, 0, 0),
                    -1
                )
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

                cv2.rectangle(
                    annotated,
                    (right_x - 5, right_y - 5),
                    (right_x + panel_width, right_y + panel_height),
                    (100, 100, 100),
                    1
                )

                cv2.putText(
                    annotated,
                    "All Probabilities:",
                    (right_x, right_y + 15),
                    font,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                right_y += 25

                prob_array = np.array(all_probs)
                sorted_indices = np.argsort(prob_array)[::-1]

                current_pred_id = debug_info.get('raw_prediction', -1) if debug_info else -1

                for idx in sorted_indices:
                    if idx < len(class_names):
                        class_name_display = class_names[idx]
                        prob_value = prob_array[idx]

                        if prob_value > 0.5:
                            prob_color = (0, 255, 0)
                        elif prob_value > 0.2:
                            prob_color = (0, 255, 255)
                        else:
                            prob_color = (150, 150, 150)

                        if idx == current_pred_id:
                            prob_color = (0, 255, 255)
                            cv2.rectangle(
                                annotated,
                                (right_x - 3, right_y - 2),
                                (right_x + panel_width - 2, right_y + 14),
                                (50, 50, 50),
                                -1
                            )

                        prob_text = f"{class_name_display:12s}: {prob_value:5.1%}"
                        cv2.putText(
                            annotated,
                            prob_text,
                            (right_x, right_y + 12),
                            font,
                            0.45,
                            prob_color,
                            1,
                            cv2.LINE_AA
                        )
                        right_y += 18

        if hasattr(self, '_last_fps'):
            fps_text = f"FPS: {self._last_fps:.1f}"
            fps_color = (255, 255, 255)
            if self._last_fps < 10:
                fps_color = (0, 0, 255)

            cv2.putText(
                annotated,
                fps_text,
                (10, frame.shape[0] - 10),
                font,
                0.7,
                fps_color,
                2,
                cv2.LINE_AA
            )

        return annotated

    def start(self) -> None:
        try:
            self._cap = self._initialize_camera()
        except CameraNotFoundError as e:
            logger.error(str(e))
            print(f"\nError: {e}")
            print("Available options:")
            print("  - Connect a camera device")
            print("  - Try a different camera ID (e.g., camera_id=1)")
            print("  - Use a video file as input source")
            raise

        self._running = True
        window_name = "Material Stream Identification"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        logger.info("Starting real-time classification. Press 'q' to quit, 's' to save frame.")
        print("\n" + "=" * 50)
        print("Material Stream Identification - Real-Time Mode")
        print("=" * 50)
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'c' to calibrate normalization (point at white/neutral surface)")
        print("Press '+' to increase normalization strength")
        print("Press '-' to decrease normalization strength")
        print(f"Saved images will be stored in: {self.save_directory}")
        print("-" * 50)

        frame_count = 0
        fps_start_time = time.time()
        self._last_fps = 0.0
        self._save_confirmation = False
        self._save_confirmation_time = 0.0

        self._prediction_stats = {
            'total_frames': 0,
            'high_confidence_frames': 0,
            'medium_confidence_frames': 0,
            'low_confidence_frames': 0,
            'avg_confidence': 0.0,
            'confidence_sum': 0.0
        }

        try:
            while self._running:
                loop_start = time.time()

                ret, frame = self._cap.read()

                if not ret:
                    logger.warning("Failed to capture frame")
                    continue

                class_id, class_name, confidence, inference_time, debug_info = self.process_frame(frame)

                self._prediction_stats['total_frames'] += 1
                self._prediction_stats['confidence_sum'] += confidence
                self._prediction_stats['avg_confidence'] = self._prediction_stats['confidence_sum'] / self._prediction_stats['total_frames']

                if confidence >= 0.8:
                    self._prediction_stats['high_confidence_frames'] += 1
                elif confidence >= 0.5:
                    self._prediction_stats['medium_confidence_frames'] += 1
                else:
                    self._prediction_stats['low_confidence_frames'] += 1

                annotated_frame = self.display_result(frame, class_name, confidence, inference_time, debug_info, self._prediction_stats)

                if self._save_confirmation and (time.time() - self._save_confirmation_time) < 1.0:
                    cv2.putText(
                        annotated_frame,
                        "SAVED!",
                        (annotated_frame.shape[1] - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA
                    )
                else:
                    self._save_confirmation = False

                cv2.imshow(window_name, annotated_frame)

                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    self._last_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start_time = time.time()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    logger.info("Quit key pressed")
                    break
                elif key == ord('s') or key == ord('S'):
                    self._save_frame(annotated_frame, class_name, confidence, debug_info, self._prediction_stats)
                elif key == ord('c') or key == ord('C'):
                    self.pipeline.calibrate_normalization(frame)
                    print("  ✓ Normalization calibrated based on current frame")
                elif key == ord('+') or key == ord('='):
                    current = getattr(self.pipeline, '_normalization_strength', 1.0)
                    new_strength = min(2.0, current + 0.1)
                    self.pipeline.set_normalization_strength(new_strength)
                    print(f"  ✓ Normalization strength: {new_strength:.1f}")
                elif key == ord('-') or key == ord('_'):
                    current = getattr(self.pipeline, '_normalization_strength', 1.0)
                    new_strength = max(0.0, current - 0.1)
                    self.pipeline.set_normalization_strength(new_strength)
                    print(f"  ✓ Normalization strength: {new_strength:.1f}")

                loop_time = time.time() - loop_start
                if loop_time < self.frame_delay:
                    time.sleep(self.frame_delay - loop_time)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False

        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

        cv2.destroyAllWindows()
        logger.info("Real-time classification stopped")

    def capture_single_frame(self) -> Optional[np.ndarray]:
        cap = None
        try:
            cap = self._initialize_camera()

            for _ in range(5):
                cap.read()

            ret, frame = cap.read()
            if ret:
                return frame
            return None

        finally:
            if cap is not None:
                cap.release()

    def classify_image_file(self, image_path: str) -> Tuple[int, str, float, float, dict]:
        import os
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return self.process_frame(frame)

    def _save_frame(
        self,
        annotated_frame: np.ndarray,
        class_name: str,
        confidence: float,
        debug_info: Optional[dict] = None,
        stats: Optional[dict] = None
    ) -> None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_class = class_name.lower().replace(' ', '_')
            filename = f"{safe_class}_{confidence:.0%}_{timestamp}.jpg"
            filepath = self.save_directory / filename

            cv2.imwrite(str(filepath), annotated_frame)

            logger.info(f"Saved annotated frame: {filepath}")
            print(f"  ✓ Saved annotated frame: {filename}")
            if stats and stats.get('total_frames', 0) > 0:
                avg_conf = stats.get('avg_confidence', 0)
                high_conf_rate = (stats.get('high_confidence_frames', 0) / stats.get('total_frames', 1)) * 100
                print(f"    Accuracy: {high_conf_rate:.1f}%, Avg Confidence: {avg_conf:.1%}")

            self._save_confirmation = True
            self._save_confirmation_time = time.time()

        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            print(f"  ✗ Failed to save: {e}")

    def get_available_cameras(self, max_cameras: int = 5) -> list:
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
