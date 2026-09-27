"""
Module: Traditional Feature Extraction
BoW, TF-IDF, and character n-gram feature extraction.

Produces dense numpy arrays and saves them via `feature_io` so they are
interchangeable with the modern feature extractor outputs.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from modules.config import TraditionalFeatureConfig
from modules.feature_io import save_features, load_features


# ──────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────

def _build_vectorizer(config: TraditionalFeatureConfig):
    """
    Returns a scikit-learn vectorizer based on the config.

    Implementation Guide:
    1. If `config.feature_type == 'bow'`:
       return `CountVectorizer(max_features=config.max_features, ngram_range=config.ngram_range)`.
    2. If `'tfidf'`:
       return `TfidfVectorizer(max_features=config.max_features, ngram_range=config.ngram_range)`.
    3. If `'ngram'`:
       return `TfidfVectorizer(max_features=config.max_features, ngram_range=config.ngram_range, analyzer='char_wb')`.
       (Character n-grams capture sub-word patterns.)
    4. Else raise `ValueError`.
    """
    pass


# ──────────────────────────────────────────────
#  Extractor
# ──────────────────────────────────────────────

class TraditionalFeatureExtractor:
    def __init__(self, config: TraditionalFeatureConfig = None):
        """
        Initializes the vectorizer from a config.

        Implementation Guide:
        1. Default to `TraditionalFeatureConfig()` if `config` is None.
        2. `self.vectorizer = _build_vectorizer(config)`.
        """
        self.config = config or TraditionalFeatureConfig()
        self.vectorizer = None

    def extract(self, texts, is_train=True):
        """
        Fits (if train) and transforms texts into a dense numpy array.

        Implementation Guide:
        1. If `is_train`: `sparse_matrix = self.vectorizer.fit_transform(texts)`.
        2. Else: `sparse_matrix = self.vectorizer.transform(texts)`.
        3. Convert to dense: `return sparse_matrix.toarray()`.
           - Note: If `max_features` is large (>50k) and dataset is big, this may
             use a lot of RAM.  Consider keeping it at 10k–20k.

        Args:
            texts (list[str] | pd.Series): Input texts.
            is_train (bool): Whether to fit the vectorizer.
        Returns:
            np.ndarray: Dense feature array of shape (N, max_features).
        """
        pass

    def extract_and_save(self, texts, filename, is_train=True):
        """
        Extracts features and saves them to disk in one step.

        Implementation Guide:
        1. `embeddings = self.extract(texts, is_train)`.
        2. `save_features(embeddings, filename, self.config.features_dir, self.config.save_format)`.
        3. Return `embeddings`.

        Args:
            texts: Input texts.
            filename (str): Base filename (e.g. 'tfidf_train').
            is_train (bool): Whether to fit the vectorizer.
        Returns:
            np.ndarray: Dense feature array.
        """
        pass
