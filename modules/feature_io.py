"""
Module: Feature IO
Shared save/load utilities for feature arrays (.npy and .h5).

Both `TraditionalFeatureExtractor` and `ModernFeatureExtractor` use these
functions so that their outputs are interchangeable — any saved feature
file can be loaded and fed directly into the shared `Classifier`.
"""

import os
import numpy as np
import h5py

from modules.config import FEATURES_DIR


def save_features(embeddings, filename, features_dir=FEATURES_DIR, save_format="npy"):
    """
    Saves a feature array to disk.

    Implementation Guide:
    1. `os.makedirs(features_dir, exist_ok=True)`.
    2. Build the full path: `path = os.path.join(features_dir, filename)`.
    3. If `save_format == 'npy'`:
       - If the array is a scipy sparse matrix, convert first: `embeddings = embeddings.toarray()`.
       - `np.save(path + '.npy', embeddings)`.
    4. If `save_format == 'h5'`:
       - Convert to dense if sparse.
       - ```
         with h5py.File(path + '.h5', 'w') as f:
             f.create_dataset('embeddings', data=embeddings)
         ```
    5. Print the saved path and array shape for confirmation.

    Args:
        embeddings (np.ndarray | scipy.sparse matrix): Feature array of shape (N, D).
        filename (str): Base filename (without extension).
        features_dir (str): Directory to save into.
        save_format (str): 'npy' or 'h5'.
    """
    pass


def load_features(filename, features_dir=FEATURES_DIR, save_format="npy"):
    """
    Loads a feature array from disk.

    Implementation Guide:
    1. Build the full path with the correct extension.
    2. If `save_format == 'npy'`: `return np.load(path, allow_pickle=False)`.
    3. If `save_format == 'h5'`:
       ```
       with h5py.File(path, 'r') as f:
           return f['embeddings'][:]
       ```

    Args:
        filename (str): Base filename (without extension).
        features_dir (str): Directory to load from.
        save_format (str): 'npy' or 'h5'.
    Returns:
        np.ndarray: Loaded feature array of shape (N, D).
    """
    pass
