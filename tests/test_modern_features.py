"""
Unit tests for ModernFeatureExtractor (Level 1: Static Embeddings).
"""

import os
import sys
import shutil

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from modules.config import ModernFeatureConfig
from modules.modern_features import ModernFeatureExtractor
from modules.feature_io import load_features


class DummyKeyedVectors:
    """Mock KeyedVectors object mimicking gensim.models.KeyedVectors."""
    def __init__(self, vocab_dim=5):
        self.vector_size = vocab_dim
        self.key_to_index = {
            "good": 0,
            "great": 1,
            "sad": 2,
            "happy": 3,
            "love": 4,
        }
        np.random.seed(42)
        self.vectors = np.random.randn(len(self.key_to_index), vocab_dim).astype(np.float32)

    def __getitem__(self, key):
        return self.vectors[self.key_to_index[key]]

    def __contains__(self, key):
        return key in self.key_to_index


def test_level1_word2vec_mean_pooling():
    dummy_model = DummyKeyedVectors(vocab_dim=4)
    cfg = ModernFeatureConfig(
        embedding_type="word2vec",
        features_dir="features_test",
        save_format="npy",
        normalize_embeddings=False,
    )
    extractor = ModernFeatureExtractor(config=cfg, model=dummy_model)

    texts = [
        "Good, happy love!",      # all 3 words in vocab
        "Unknown words xyz abc",  # OOV words -> zero vector
        "",                       # empty string -> zero vector
    ]

    features = extractor.extract(texts)
    assert isinstance(features, np.ndarray)
    assert features.shape == (3, 4)
    assert features.dtype == np.float32

    # Vector 1 should be the average of good, happy, love
    expected_vec_0 = np.mean([
        dummy_model["good"],
        dummy_model["happy"],
        dummy_model["love"]
    ], axis=0)
    np.testing.assert_allclose(features[0], expected_vec_0, rtol=1e-5)

    # Vector 1 and 2 should be zeros
    np.testing.assert_allclose(features[1], np.zeros(4), rtol=1e-5)
    np.testing.assert_allclose(features[2], np.zeros(4), rtol=1e-5)


def test_level1_save_and_load_npy_and_h5():
    test_dir = "features_test"
    os.makedirs(test_dir, exist_ok=True)
    try:
        dummy_model = DummyKeyedVectors(vocab_dim=6)
        texts = ["Love and happy", "Sad"]

        # 1. Test NPY
        cfg_npy = ModernFeatureConfig(
            embedding_type="word2vec",
            features_dir=test_dir,
            save_format="npy",
            normalize_embeddings=True,
        )
        extractor_npy = ModernFeatureExtractor(config=cfg_npy, model=dummy_model)
        saved_npy = extractor_npy.extract_and_save(texts, "test_emb")

        loaded_npy = load_features("test_emb", features_dir=test_dir, save_format="npy")
        assert loaded_npy.shape == (2, 6)
        np.testing.assert_allclose(saved_npy, loaded_npy, rtol=1e-5)

        # Check normalization
        norms = np.linalg.norm(saved_npy, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)

        # 2. Test H5
        cfg_h5 = ModernFeatureConfig(
            embedding_type="glove",
            features_dir=test_dir,
            save_format="h5",
        )
        extractor_h5 = ModernFeatureExtractor(config=cfg_h5, model=dummy_model)
        saved_h5 = extractor_h5.extract_and_save(texts, "test_glove")

        loaded_h5 = load_features("test_glove", features_dir=test_dir, save_format="h5")
        assert loaded_h5.shape == (2, 6)
        np.testing.assert_allclose(saved_h5, loaded_h5, rtol=1e-5)

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_level1_word2vec_mean_pooling()
    test_level1_save_and_load_npy_and_h5()
    print("All Level 1 Modern Feature Extraction tests passed successfully!")
