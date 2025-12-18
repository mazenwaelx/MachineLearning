import numpy as np
import pytest
import os
import tempfile
import cv2

from src.data.loader import DataLoader, CLASS_MAPPING
from src.data.augmentor import Augmentor

class TestDataLoader:

    def test_class_mapping_returns_correct_labels(self):
        loader = DataLoader("dummy_path")
        mapping = loader.get_class_mapping()

        assert len(mapping) == 6
        assert mapping['glass'] == 0
        assert mapping['paper'] == 1
        assert mapping['cardboard'] == 2
        assert mapping['plastic'] == 3
        assert mapping['metal'] == 4
        assert mapping['trash'] == 5

    def test_image_resizing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class_folder = os.path.join(tmpdir, 'glass')
            os.makedirs(class_folder)

            test_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(class_folder, 'test.jpg'), test_img)

            loader = DataLoader(tmpdir, image_size=(224, 224))
            images, labels = loader.load_dataset()

            assert images.shape[1:3] == (224, 224)
            assert images.shape[3] == 3

    def test_image_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class_folder = os.path.join(tmpdir, 'paper')
            os.makedirs(class_folder)

            test_img = np.full((50, 50, 3), 128, dtype=np.uint8)
            cv2.imwrite(os.path.join(class_folder, 'test.jpg'), test_img)

            loader = DataLoader(tmpdir)
            images, _ = loader.load_dataset()

            assert images.min() >= 0.0
            assert images.max() <= 1.0
            assert images.dtype == np.float32

    def test_empty_dataset_returns_empty_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)
            images, labels = loader.load_dataset()

            assert len(images) == 0
            assert len(labels) == 0

class TestAugmentor:

    def test_augmentation_produces_valid_images(self):
        augmentor = Augmentor(target_samples_per_class=10)

        images = np.random.rand(5, 64, 64, 3).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2], dtype=np.int32)

        aug_images, aug_labels = augmentor.augment_dataset(images, labels)

        assert aug_images.min() >= 0.0
        assert aug_images.max() <= 1.0
        assert aug_images.dtype == np.float32

    def test_augmentation_preserves_image_shape(self):
        augmentor = Augmentor(target_samples_per_class=10)

        images = np.random.rand(3, 64, 64, 3).astype(np.float32)
        labels = np.array([0, 1, 2], dtype=np.int32)

        aug_images, _ = augmentor.augment_dataset(images, labels)

        assert aug_images.shape[1:] == (64, 64, 3)

    def test_augmentation_increases_dataset_size(self):
        augmentor = Augmentor(target_samples_per_class=20)

        images = np.random.rand(10, 32, 32, 3).astype(np.float32)
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)

        aug_images, aug_labels = augmentor.augment_dataset(images, labels)

        increase_ratio = (len(aug_images) - len(images)) / len(images)
        assert increase_ratio >= 0.3

    def test_rotation_transform(self):
        augmentor = Augmentor()

        image = np.random.rand(64, 64, 3).astype(np.float32)
        rotated = augmentor._rotate(image, 15)

        assert rotated.shape == image.shape

    def test_flip_transforms(self):
        augmentor = Augmentor()

        image = np.random.rand(64, 64, 3).astype(np.float32)

        h_flipped = augmentor._flip_horizontal(image)
        v_flipped = augmentor._flip_vertical(image)

        assert h_flipped.shape == image.shape
        assert v_flipped.shape == image.shape
