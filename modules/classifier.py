"""
Module: Classifier
Shared multi-label classifier that trains on ANY feature array — whether
produced by `TraditionalFeatureExtractor` or `ModernFeatureExtractor`.

Both extractors save dense numpy arrays of shape (N, D), so this classifier
consumes them identically.
"""

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from modules.config import ClassifierConfig


# ──────────────────────────────────────────────
#  Factory helpers
# ──────────────────────────────────────────────

def _build_classifier(config: ClassifierConfig):
    """
    Returns a multi-label classifier based on the config.

    Implementation Guide:
    1. Map `config.classifier_type` to a base estimator:
       - 'logistic_regression' -> LogisticRegression(max_iter=1000)
       - 'svm'                -> LinearSVC(max_iter=2000)
       - 'random_forest'      -> RandomForestClassifier(n_estimators=100)
       - 'naive_bayes'        -> MultinomialNB()
       - 'mlp'                -> MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20)
    2. Wrap in `OneVsRestClassifier(base_estimator)`.
    3. Return the wrapped classifier.

    Note:
    - `naive_bayes` requires non-negative input, so it works with BoW/TF-IDF
      but NOT with Word2Vec/BERT embeddings (which can be negative).
      The implementer should add a check or document this limitation.
    """
    pass


def _get_default_param_grid(config: ClassifierConfig):
    """
    Returns a sensible default hyperparameter grid for the chosen classifier.

    Implementation Guide:
    1. If 'logistic_regression': `{'estimator__C': [0.01, 0.1, 1, 10]}`.
    2. If 'svm': `{'estimator__C': [0.01, 0.1, 1, 10]}`.
    3. If 'random_forest': `{'estimator__n_estimators': [50, 100, 200]}`.
    4. If 'naive_bayes': `{'estimator__alpha': [0.1, 0.5, 1.0]}`.
    5. If 'mlp': `{'estimator__hidden_layer_sizes': [(128,), (256, 128)], 'estimator__alpha': [1e-4, 1e-3]}`.
    6. Return the dict.
    """
    pass


# ──────────────────────────────────────────────
#  Classifier class
# ──────────────────────────────────────────────

class Classifier:
    def __init__(self, config: ClassifierConfig = None):
        """
        Initializes the classifier from a config.

        Implementation Guide:
        1. Default to `ClassifierConfig()` if `config` is None.
        2. `self.model = _build_classifier(config)`.

        Example usage in notebook:
            # Exact same classifier on two different feature sets
            clf_cfg = ClassifierConfig(classifier_type='logistic_regression')

            clf = Classifier(clf_cfg)
            clf.train(X_tfidf_train, y_train)
            y_pred_trad = clf.predict(X_tfidf_test)

            clf2 = Classifier(clf_cfg)
            clf2.train(X_bert_train, y_train)
            y_pred_modern = clf2.predict(X_bert_test)
        """
        self.config = config or ClassifierConfig()
        self.model = None

    def train(self, X_train, y_train):
        """
        Trains the multi-label classifier.

        Implementation Guide:
        1. `self.model.fit(X_train, y_train)`.

        Args:
            X_train (np.ndarray): Feature array of shape (N, D).
            y_train (np.ndarray): Binary label matrix of shape (N, 28).
        """
        pass

    def predict(self, X_test):
        """
        Predicts labels for the test set.

        Implementation Guide:
        1. `return self.model.predict(X_test)`.

        Args:
            X_test (np.ndarray): Feature array of shape (N, D).
        Returns:
            np.ndarray: Binary prediction matrix of shape (N, 28).
        """
        pass

    def predict_proba(self, X_test):
        """
        Returns prediction probabilities or decision scores.

        Implementation Guide:
        1. If `hasattr(self.model, 'predict_proba')`:
           return `self.model.predict_proba(X_test)`.
        2. Else: return `self.model.decision_function(X_test)`.

        Args:
            X_test (np.ndarray): Feature array of shape (N, D).
        Returns:
            np.ndarray: Score matrix of shape (N, 28).
        """
        pass

    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        """
        Performs hyperparameter search based on `self.config.tuning_method`.

        Implementation Guide:
        1. Default `param_grid` to `_get_default_param_grid(self.config)` if None.
        2. If `self.config.tuning_method == 'grid'`:
           `search = GridSearchCV(self.model, param_grid, scoring='f1_micro', cv=3)`.
        3. If `'random'`:
           `search = RandomizedSearchCV(self.model, param_grid, scoring='f1_micro', cv=3, n_iter=10)`.
        4. `search.fit(X_train, y_train)`.
        5. `self.model = search.best_estimator_`.
        6. Return `search.best_params_` and `search.best_score_`.

        Args:
            X_train (np.ndarray): Feature array.
            y_train (np.ndarray): Label matrix.
            param_grid (dict | None): Hyperparameter grid.
        Returns:
            tuple[dict, float]: (best_params, best_score).
        """
        pass

    def run(self, X_train, y_train, X_test):
        """
        Convenience method: optionally tune, train, and predict.

        Implementation Guide:
        1. If `self.config.tuning_method` is not None:
           `self.hyperparameter_tuning(X_train, y_train)`.
        2. `self.train(X_train, y_train)`.
        3. `return self.predict(X_test)`.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Test features.
        Returns:
            np.ndarray: Predictions.
        """
        pass
