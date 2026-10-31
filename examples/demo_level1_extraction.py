"""
Demo: Modern Feature Extraction - Level 1 (Static Embeddings: Word2Vec & GloVe)

This script demonstrates how to:
1. Configure ModernFeatureConfig for Word2Vec or GloVe.
2. Initialize ModernFeatureExtractor.
3. Extract dense sentence embeddings using word-level mean pooling.
4. Save features to .npy and .h5 formats in the `features/` directory.
5. Load them back via `feature_io` and feed them to downstream classifiers.
"""

import os
import sys

# Ensure root package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from modules.config import ModernFeatureConfig, ClassifierConfig, FEATURES_DIR
from modules.modern_features import ModernFeatureExtractor
from modules.feature_io import load_features


def run_demo():
    print("=" * 60)
    print("  MODERN FEATURE EXTRACTION - LEVEL 1 DEMO (STATIC EMBEDDINGS)  ")
    print("=" * 60)

    # Sample sentences mimicking emotion text data (e.g., from GoEmotions)
    sample_texts = [
        "I am so proud of your achievement, this is wonderful news!",
        "I am really sorry for what happened, please forgive me.",
        "That was totally unexpected and unbelievable!",
        "I feel completely neutral about this topic.",
    ]

    print("\n[Step 1] Sample input texts:")
    for i, t in enumerate(sample_texts, 1):
        print(f"  {i}. {t}")

    # Configuration for Level 1: GloVe or Word2Vec
    # Note: 'glove-wiki-gigaword-100' (~128MB) is fast to download and evaluate.
    # For full Word2Vec, set embedding_type='word2vec' (defaults to 'word2vec-google-news-300', ~1.6GB).
    config_glove = ModernFeatureConfig(
        embedding_type="glove",
        model_name="glove-wiki-gigaword-100",
        save_format="npy",
        features_dir=FEATURES_DIR,
        normalize_embeddings=True,
    )

    print(f"\n[Step 2] Initializing ModernFeatureExtractor with config:")
    print(f"  - Embedding type: {config_glove.embedding_type}")
    print(f"  - Model identifier: {config_glove.model_name}")
    print(f"  - Save format: {config_glove.save_format}")
    print(f"  - Output dir: {config_glove.features_dir}")

    try:
        extractor = ModernFeatureExtractor(config=config_glove)
    except ImportError as e:
        print(f"\n[Notice] {e}")
        print("To run with pre-trained Gensim models on Colab / Linux:")
        print("  pip install gensim")
        return

    # Extract and save
    print("\n[Step 3] Extracting features and saving to disk...")
    features_train = extractor.extract_and_save(sample_texts, filename="demo_glove_train")

    print(f"  Extracted feature array shape: {features_train.shape} (N={features_train.shape[0]}, D={features_train.shape[1]})")
    print(f"  Dtype: {features_train.dtype}")

    # Load back using feature_io
    print("\n[Step 4] Verifying feature loading via load_features()...")
    loaded_features = load_features("demo_glove_train", features_dir=FEATURES_DIR, save_format="npy")
    print(f"  Loaded array shape: {loaded_features.shape}")
    assert np.allclose(features_train, loaded_features)
    print("  Verification successful: Saved and loaded features match perfectly!")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
