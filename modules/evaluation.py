"""
Module: Evaluation
Metrics computation, per-class analysis, and pipeline comparison plots.

Options are driven by an `EvaluationConfig` object.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

from modules.config import EvaluationConfig, LABEL_NAMES


# ──────────────────────────────────────────────
#  Core Metrics
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred, config: EvaluationConfig = None):
    """
    Computes a suite of evaluation metrics for multi-label classification.

    Implementation Guide:
    1. Default to `EvaluationConfig()` if `config` is None.
    2. Compute subset accuracy: `accuracy_score(y_true, y_pred)`.
    3. For each average type in `config.average_types`, compute:
       - `f1_score(y_true, y_pred, average=avg, zero_division=0)`.
       - `precision_score(y_true, y_pred, average=avg, zero_division=0)`.
       - `recall_score(y_true, y_pred, average=avg, zero_division=0)`.
    4. If `config.show_classification_report`:
       - `report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0)`.
       - `print(report)`.
    5. Return a dict: `{'subset_accuracy': ..., 'f1_micro': ..., 'f1_macro': ..., ...}`.

    Args:
        y_true (np.ndarray): Ground truth binary matrix (N, 28).
        y_pred (np.ndarray): Predicted binary matrix (N, 28).
        config (EvaluationConfig | None): Configuration.
    Returns:
        dict: Computed metrics.
    """
    pass


def compute_per_class_f1(y_true, y_pred):
    """
    Computes the F1-score for each of the 28 emotion classes individually.

    Implementation Guide:
    1. `per_class = f1_score(y_true, y_pred, average=None, zero_division=0)`.
    2. Return a dict mapping `LABEL_NAMES[i] -> per_class[i]`.

    Args:
        y_true (np.ndarray): Ground truth.
        y_pred (np.ndarray): Predictions.
    Returns:
        dict[str, float]: Per-class F1 scores.
    """
    pass


# ──────────────────────────────────────────────
#  Visualisation
# ──────────────────────────────────────────────

def plot_per_class_f1(y_true, y_pred, config: EvaluationConfig = None):
    """
    Plots a horizontal bar chart of per-class F1 scores.

    Implementation Guide:
    1. Call `compute_per_class_f1(y_true, y_pred)`.
    2. Sort the classes by F1 (ascending for horizontal bars).
    3. If `config.top_n_classes` is set, show only the top/bottom N.
    4. Plot with `plt.barh`.
    5. Add title and axis labels.

    Args:
        y_true (np.ndarray): Ground truth.
        y_pred (np.ndarray): Predictions.
        config (EvaluationConfig | None): Configuration.
    """
    pass


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plots per-label confusion matrices for the top-N most frequent classes.

    Implementation Guide:
    1. `mcm = multilabel_confusion_matrix(y_true, y_pred)`.
    2. Default `class_names` to `LABEL_NAMES`.
    3. Pick the top N classes by support (sum of y_true columns).
    4. For each selected class, plot a 2x2 heatmap subplot using `sns.heatmap`.
    5. Title each subplot with the class name.

    Args:
        y_true (np.ndarray): Ground truth.
        y_pred (np.ndarray): Predictions.
        class_names (list | None): Label names.
    """
    pass


def compare_pipelines(results_dict):
    """
    Plots a grouped bar chart comparing metrics across pipelines.

    Implementation Guide:
    1. `results_dict` format:
       ```
       {
           'TF-IDF + LogReg': {'f1_micro': 0.45, 'f1_macro': 0.30},
           'BERT Frozen + MLP': {'f1_micro': 0.50, 'f1_macro': 0.38},
           'DistilBERT E2E': {'f1_micro': 0.60, 'f1_macro': 0.48},
       }
       ```
    2. For each metric, create a grouped bar using `np.arange` offsets.
    3. Add legend, title, axis labels.
    4. Use `plt.tight_layout()`.

    Args:
        results_dict (dict[str, dict[str, float]]): Pipeline name → metric dict.
    """
    pass
