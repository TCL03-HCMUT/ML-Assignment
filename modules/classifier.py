"""Module: Classifier
Multi-label classifier training on traditional or modern feature matrices.
Supports linear models, Naive Bayes, tree ensembles, nearest neighbors, and MLPs.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
    Perceptron,
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score

from modules.config import ClassifierConfig


@dataclass
class ClassifierSpec:
    name: str
    family: str
    speed: str
    sparse_input: bool
    dense_input: bool
    non_negative_only: bool
    native_multilabel: bool
    description: str
    builder: Callable[[int], object]


def _registry() -> Dict[str, ClassifierSpec]:
    entries = [
        ClassifierSpec("logistic_regression", "linear", "fast", True, True, False, False,
                       "L2-regularized Logistic Regression",
                       lambda rs: LogisticRegression(max_iter=1000, random_state=rs)),
        ClassifierSpec("svm", "linear", "fast", True, True, False, False,
                       "Linear Support Vector Classifier",
                       lambda rs: LinearSVC(max_iter=2000, random_state=rs)),
        ClassifierSpec("svm_calibrated", "linear", "medium", True, True, False, False,
                       "Probability-calibrated Linear SVC",
                       lambda rs: CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=rs))),
        ClassifierSpec("ridge", "linear", "fast", True, True, False, True,
                       "Ridge Classifier with closed-form L2 solver",
                       lambda rs: RidgeClassifier(random_state=rs)),
        ClassifierSpec("sgd", "linear", "fast", True, True, False, False,
                       "SGD with log-loss (logistic loss)",
                       lambda rs: SGDClassifier(loss="log_loss", max_iter=1000, random_state=rs)),
        ClassifierSpec("sgd_log", "linear", "fast", True, True, False, False,
                       "SGD with log-loss (logistic loss)",
                       lambda rs: SGDClassifier(loss="log_loss", max_iter=1000, random_state=rs)),
        ClassifierSpec("sgd_hinge", "linear", "fast", True, True, False, False,
                       "SGD with linear hinge loss (Linear SVM)",
                       lambda rs: SGDClassifier(loss="hinge", max_iter=1000, random_state=rs)),
        ClassifierSpec("passive_aggressive", "linear", "fast", True, True, False, False,
                       "Passive-Aggressive online classifier",
                       lambda rs: PassiveAggressiveClassifier(max_iter=1000, random_state=rs)),
        ClassifierSpec("perceptron", "linear", "fast", True, True, False, False,
                       "Classical Perceptron linear classifier",
                       lambda rs: Perceptron(max_iter=1000, random_state=rs)),
        ClassifierSpec("naive_bayes", "naive_bayes", "fast", True, True, True, False,
                       "Multinomial Naive Bayes",
                       lambda rs: MultinomialNB()),
        ClassifierSpec("multinomial_nb", "naive_bayes", "fast", True, True, True, False,
                       "Multinomial Naive Bayes",
                       lambda rs: MultinomialNB()),
        ClassifierSpec("complement_nb", "naive_bayes", "fast", True, True, True, False,
                       "Complement Naive Bayes for imbalanced classes",
                       lambda rs: ComplementNB()),
        ClassifierSpec("bernoulli_nb", "naive_bayes", "fast", True, True, False, False,
                       "Bernoulli Naive Bayes for binary features",
                       lambda rs: BernoulliNB()),
        ClassifierSpec("knn", "neighbors", "slow", True, True, False, True,
                       "K-Nearest Neighbors (k=5)",
                       lambda rs: KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ClassifierSpec("nearest_centroid", "neighbors", "fast", True, True, False, False,
                       "Nearest Centroid classification",
                       lambda rs: NearestCentroid()),
        ClassifierSpec("decision_tree", "tree_ensemble", "medium", True, True, False, True,
                       "Decision Tree (max_depth=20)",
                       lambda rs: DecisionTreeClassifier(max_depth=20, random_state=rs)),
        ClassifierSpec("random_forest", "tree_ensemble", "medium", True, True, False, True,
                       "Random Forest (100 trees)",
                       lambda rs: RandomForestClassifier(n_estimators=100, max_depth=25, random_state=rs, n_jobs=-1)),
        ClassifierSpec("extra_trees", "tree_ensemble", "medium", True, True, False, True,
                       "Extremely Randomized Trees (100 trees)",
                       lambda rs: ExtraTreesClassifier(n_estimators=100, max_depth=25, random_state=rs, n_jobs=-1)),
        ClassifierSpec("hist_gradient_boosting", "tree_ensemble", "medium", False, True, False, False,
                       "Histogram-based Gradient Boosting",
                       lambda rs: HistGradientBoostingClassifier(max_iter=50, random_state=rs)),
        ClassifierSpec("adaboost", "tree_ensemble", "medium", True, True, False, False,
                       "AdaBoost with decision stumps",
                       lambda rs: AdaBoostClassifier(n_estimators=50, random_state=rs)),
        ClassifierSpec("mlp", "neural", "medium", True, True, False, True,
                       "Multi-Layer Perceptron (256, 128)",
                       lambda rs: MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=25, random_state=rs)),
        ClassifierSpec("dummy", "baseline", "fast", True, True, False, True,
                       "Prior-class Dummy Baseline",
                       lambda rs: DummyClassifier(strategy="prior")),
    ]
    return {entry.name: entry for entry in entries}


CLASSIFIER_REGISTRY = _registry()


def list_classifiers() -> pd.DataFrame:
    rows = []
    for spec in CLASSIFIER_REGISTRY.values():
        rows.append({
            "classifier": spec.name,
            "family": spec.family,
            "speed": spec.speed,
            "sparse_input": spec.sparse_input,
            "dense_input": spec.dense_input,
            "non_negative_only": spec.non_negative_only,
            "native_multilabel": spec.native_multilabel,
            "description": spec.description,
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["classifier"]).reset_index(drop=True)


def check_compatibility(classifier_type: str, X) -> Tuple[bool, str]:
    spec = CLASSIFIER_REGISTRY.get(classifier_type)
    if spec is None:
        return False, f"Unknown classifier type: {classifier_type}"
    is_sp = sparse.issparse(X)
    if is_sp and not spec.sparse_input:
        return False, f"{classifier_type} requires dense arrays"
    if spec.non_negative_only:
        min_val = X.data.min() if is_sp else np.min(X)
        if min_val < 0:
            return False, f"{classifier_type} requires non-negative features (min={min_val:.3f})"
    return True, "Compatible"


def _build_classifier(config: ClassifierConfig):
    model_name = config.classifier_type
    spec = CLASSIFIER_REGISTRY.get(model_name)
    if spec is None:
        raise ValueError(f"Unsupported classifier_type '{model_name}'. Choose from: {list(CLASSIFIER_REGISTRY.keys())}")

    base = spec.builder(config.random_state)
    strat = getattr(config, "multilabel_strategy", "ovr")

    if strat == "chain":
        return ClassifierChain(base, random_state=config.random_state)
    return OneVsRestClassifier(base, n_jobs=getattr(config, "n_jobs", -1))


def _get_default_param_grid(config: ClassifierConfig) -> dict:
    t = config.classifier_type
    prefix = "base_estimator__" if getattr(config, "multilabel_strategy", "ovr") == "chain" else "estimator__"
    if t in ("logistic_regression", "svm", "svm_calibrated"):
        return {f"{prefix}C": [0.1, 0.5, 1.0, 2.0]}
    if t in ("sgd", "sgd_log", "sgd_hinge"):
        return {f"{prefix}alpha": [1e-5, 1e-4, 1e-3]}
    if t in ("naive_bayes", "multinomial_nb", "complement_nb"):
        return {f"{prefix}alpha": [0.1, 0.5, 1.0]}
    if t in ("random_forest", "extra_trees"):
        return {f"{prefix}n_estimators": [50, 100]}
    if t == "mlp":
        return {f"{prefix}alpha": [1e-4, 1e-3]}
    return {}


class Classifier:
    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self.spec = CLASSIFIER_REGISTRY.get(self.config.classifier_type)
        self.model = _build_classifier(self.config)
        self.thresholds_ = None

    def _prepare_X(self, X):
        if self.spec and not self.spec.sparse_input and sparse.issparse(X):
            return X.toarray()
        return X

    def train(self, X_train, y_train):
        X = self._prepare_X(X_train)
        self.model.fit(X, y_train)
        return self

    def predict_proba(self, X_test) -> np.ndarray:
        X = self._prepare_X(X_test)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if isinstance(probs, list):
                probs = np.column_stack([p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p for p in probs])
            return np.nan_to_num(probs, nan=0.0)

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-np.clip(scores, -25, 25)))

        preds = self.model.predict(X)
        return preds.astype(float)

    def tune_thresholds(self, X_val, y_val, strategy: str = "per_class") -> np.ndarray:
        probs = self.predict_proba(X_val)
        n_classes = y_val.shape[1]
        thresholds = np.full(n_classes, 0.5)

        candidates = np.linspace(0.1, 0.9, 17)
        for c in range(n_classes):
            best_f1, best_th = -1.0, 0.5
            y_c = y_val[:, c]
            p_c = probs[:, c]
            for th in candidates:
                pred_c = (p_c >= th).astype(int)
                score = f1_score(y_c, pred_c, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_th = th
            thresholds[c] = best_th

        self.thresholds_ = thresholds
        return self.thresholds_

    def predict(self, X_test) -> np.ndarray:
        X = self._prepare_X(X_test)
        if self.thresholds_ is not None:
            probs = self.predict_proba(X)
            return (probs >= self.thresholds_).astype(int)
        return self.model.predict(X)

    def hyperparameter_tuning(self, X_train, y_train, param_grid: Optional[dict] = None) -> Tuple[dict, float]:
        X = self._prepare_X(X_train)
        grid = param_grid or _get_default_param_grid(self.config)
        cv = getattr(self.config, "cv_folds", 3)
        method = getattr(self.config, "tuning_method", "grid")

        if method == "random":
            search = RandomizedSearchCV(self.model, grid, n_iter=10, scoring="f1_micro", cv=cv, random_state=self.config.random_state)
        else:
            search = GridSearchCV(self.model, grid, scoring="f1_micro", cv=cv)

        search.fit(X, y_train)
        self.model = search.best_estimator_
        return search.best_params_, float(search.best_score_)

    def run(self, X_train, y_train, X_test) -> np.ndarray:
        if getattr(self.config, "tuning_method", None):
            self.hyperparameter_tuning(X_train, y_train)
        self.train(X_train, y_train)
        return self.predict(X_test)
