"""
Module: Configuration
Centralized configuration for the entire pipeline using Python dataclasses.
All configurable options (preprocessing, feature extraction, model selection,
training hyperparameters) are defined here and passed into the pipeline modules.
"""

from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
#  GoEmotions Label Constants
# ──────────────────────────────────────────────

LABEL_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

NUM_LABELS = len(LABEL_NAMES)

# Default directory for saving/loading feature arrays
FEATURES_DIR = "features"


# ──────────────────────────────────────────────
#  Preprocessing Configuration
# ──────────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    """Controls which preprocessing steps are applied and in what order.

    Implementation Guide:
    - Each boolean flag toggles a specific preprocessing step.
    - The `preprocess_dataset` function reads this config and applies only
      the steps that are set to True, in the order listed below.
    - This lets you easily compare different preprocessing strategies by
      creating multiple configs.

    Example usage in notebook:
        cfg = PreprocessingConfig(lowercase=True, remove_stopwords=True, stemming=True)
        df_train = preprocess_dataset(df_train, cfg)
    """

    lowercase: bool = True
    remove_punctuation: bool = False # to keep punctuation (e.g. "happy! vs happy!!!")
    remove_stopwords: bool = False   # to keep common stopwords (e.g. "the", "and"). may be change later

    normalize_unicode: bool = True   # to normalize unicode characters (e.g. ế can be represented 2 ways in NFD. normalize that)
    decode_html: bool = True         # to decode HTML (e.g. &amp; -> &)
    replace_urls: bool = True       # to replace URLs with a placeholder token (e.g. "http://example.com" -> "URLTOKEN")
    replace_users: bool = True      # to replace user mentions with a placeholder token (e.g. "@user" -> "USERTOKEN")
    preserve_negation: bool = True   # if remove_stopwords, this will help keep negation (e.g. "not", "no")
    tokenizer: str = "regex"         # regex | whitespace
    empty_token: str = "EMPTYTOKEN"  # to replace empty tokens with a placeholder

    remove_numbers: bool = False
    stemming: bool = False          # Use Porter Stemmer
    lemmatization: bool = False     # Use WordNet Lemmatizer (mutually exclusive with stemming)
    min_word_length: int = 1        # Drop words shorter than this


# ──────────────────────────────────────────────
#  Traditional Feature Extraction Configuration
# ──────────────────────────────────────────────

@dataclass
class TraditionalFeatureConfig:
    """Controls traditional feature extraction (BoW, TF-IDF, n-grams).

    The extractor produces a dense numpy array and saves it to
    `features_dir` so that the shared `Classifier` can consume it
    identically to modern embeddings.

    Implementation Guide:
    - `feature_type`: One of 'bow', 'tfidf', 'ngram'.
    - `ngram_range`: Tuple for n-gram range, e.g. (1, 1) for unigrams, (1, 2) for uni+bigrams.
    - `max_features`: Maximum number of features for the vectorizer.
    - `save_format`: One of 'npy' or 'h5'.

    Example usage in notebook:
        cfg = TraditionalFeatureConfig(feature_type='tfidf', ngram_range=(1, 2))
        extractor = TraditionalFeatureExtractor(cfg)
        X_train = extractor.extract_and_save(train_texts, 'tfidf_train')
    """

    feature_type: str = "tfidf"          # 'bow' | 'tfidf' | 'ngram'
    ngram_range: tuple = (1, 1)          # e.g. (1,2) for bigrams
    max_features: Optional[int] = 10_000
    save_format: str = "npy"             # 'npy' | 'h5'
    features_dir: str = FEATURES_DIR


# ──────────────────────────────────────────────
#  Modern Feature Extraction Configuration
# ──────────────────────────────────────────────

@dataclass
class ModernFeatureConfig:
    """Controls neural embedding extraction (Word2Vec, GloVe, frozen BERT).

    The extractor produces a dense numpy array and saves it to
    `features_dir` so that the shared `Classifier` can consume it
    identically to traditional features.

    Implementation Guide:
    - `embedding_type`: One of 'word2vec', 'glove', 'bert'.
    - `model_name`: HuggingFace model name (only used when embedding_type is 'bert').
    - `pooling_strategy`: How to aggregate token embeddings into a single vector.
        - 'mean': Average all token embeddings.
        - 'cls': Use the [CLS] token output (BERT only).
    - `save_format`: One of 'npy' or 'h5'.

    Example usage in notebook:
        cfg = ModernFeatureConfig(embedding_type='bert', pooling_strategy='cls')
        extractor = ModernFeatureExtractor(cfg)
        X_train = extractor.extract_and_save(train_texts, 'bert_cls_train')
    """

    embedding_type: str = "word2vec"          # 'word2vec' | 'glove' | 'bert'
    model_name: str = "bert-base-uncased"     # HuggingFace model identifier
    pooling_strategy: str = "mean"            # 'mean' | 'cls'
    save_format: str = "npy"                  # 'npy' | 'h5'
    features_dir: str = FEATURES_DIR


# ──────────────────────────────────────────────
#  Shared Classifier Configuration
# ──────────────────────────────────────────────

@dataclass
class ClassifierConfig:
    """Controls the classifier that trains on ANY feature array (traditional or modern).

    Both `TraditionalFeatureExtractor` and `ModernFeatureExtractor` produce
    numpy arrays of shape (N, D).  This single classifier consumes them.

    Implementation Guide:
    - `classifier_type`: One of 'logistic_regression', 'svm', 'random_forest',
      'naive_bayes', 'mlp'.
    - `tuning_method`: One of 'grid', 'random', or None to skip tuning.

    Example usage in notebook:
        # Same classifier, different features
        clf_cfg = ClassifierConfig(classifier_type='svm', tuning_method='grid')

        clf_trad = Classifier(clf_cfg)
        clf_trad.train(X_tfidf_train, y_train)

        clf_modern = Classifier(clf_cfg)
        clf_modern.train(X_bert_train, y_train)
    """

    classifier_type: str = "logistic_regression"  # 'logistic_regression' | 'svm' | 'random_forest' | 'naive_bayes' | 'mlp'
    tuning_method: Optional[str] = None           # 'grid' | 'random' | None


# ──────────────────────────────────────────────
#  Deep Learning Pipeline Configuration
# ──────────────────────────────────────────────

@dataclass
class DLPipelineConfig:
    """Controls end-to-end deep learning fine-tuning.

    Implementation Guide:
    - `model_name`: Any HuggingFace model that supports SequenceClassification.
    - `max_length`: Maximum token length for the tokenizer.
    - `batch_size`, `epochs`, `learning_rate`: Standard training hyperparameters.
    - `threshold`: Sigmoid threshold for converting logits to binary predictions.

    Example usage in notebook:
        cfg = DLPipelineConfig(
            model_name='distilbert-base-uncased',
            epochs=5,
            learning_rate=3e-5,
            batch_size=32,
        )
        pipeline = EndToEndDLPipeline(cfg)
    """

    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    threshold: float = 0.5
    num_labels: int = NUM_LABELS


# ──────────────────────────────────────────────
#  Evaluation Configuration
# ──────────────────────────────────────────────

@dataclass
class EvaluationConfig:
    """Controls evaluation metrics and visualisation options.

    Implementation Guide:
    - `average_types`: Which F1 averages to compute (e.g. 'micro', 'macro', 'weighted').
    - `top_n_classes`: Number of classes to show in per-class plots (use None for all 28).
    - `show_classification_report`: Whether to print sklearn's full classification report.

    Example usage in notebook:
        cfg = EvaluationConfig(top_n_classes=10, show_classification_report=True)
        metrics = compute_metrics(y_true, y_pred, cfg)
    """

    average_types: list = field(default_factory=lambda: ["micro", "macro", "weighted"])
    top_n_classes: Optional[int] = None     # None = show all 28
    show_classification_report: bool = True
