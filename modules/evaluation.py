"""Module: Evaluation
Metrics computation, per-class analysis, threshold tuning, and visualization.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

from modules.config import EvaluationConfig, LABEL_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, config: Optional[EvaluationConfig] = None) -> Dict[str, float]:
    cfg = config or EvaluationConfig()
    metrics = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    for avg in cfg.average_types:
        metrics[f"f1_{avg}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        metrics[f"precision_{avg}"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        metrics[f"recall_{avg}"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))

    if cfg.show_classification_report:
        n_classes = y_true.shape[1]
        target_names = LABEL_NAMES[:n_classes] if n_classes <= len(LABEL_NAMES) else None
        print("\n=== Classification Report ===")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    return metrics


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, float]:
    names = class_names or LABEL_NAMES[:y_true.shape[1]]
    scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {name: float(scores[i]) for i, name in enumerate(names)}


def plot_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, config: Optional[EvaluationConfig] = None, class_names: Optional[List[str]] = None):
    scores = compute_per_class_f1(y_true, y_pred, class_names=class_names)
    series = pd.Series(scores).sort_values()

    top_n = config.top_n_classes if config else None
    if top_n is not None and top_n < len(series):
        series = series.iloc[-top_n:]

    fig, ax = plt.subplots(figsize=(8, max(4, len(series) * 0.28)))
    bars = ax.barh(series.index, series.values, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1-Score")
    ax.set_title("Per-Class F1 Score")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None, top_n: int = 6):
    names = class_names or LABEL_NAMES[:y_true.shape[1]]
    support = y_true.sum(axis=0)
    top_indices = np.argsort(support)[::-1][:top_n]

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    cols = 3
    rows = int(np.ceil(len(top_indices) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.0))
    axes = np.array(axes).reshape(-1)

    for i, idx in enumerate(top_indices):
        cm = mcm[idx]
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        ax.set_title(f"{names[idx]} (N={support[idx]})", fontsize=10)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    for j in range(len(top_indices), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Top {len(top_indices)} Classes Confusion Matrices", y=1.02, fontsize=12)
    plt.tight_layout()
    return fig


def compare_pipelines(results_dict: Dict[str, Dict[str, float]], metrics: Optional[List[str]] = None):
    metric_keys = metrics or ["f1_micro", "f1_macro", "subset_accuracy"]
    df = pd.DataFrame(results_dict).T
    available = [m for m in metric_keys if m in df.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 4.5))
    df[available].plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Pipeline Architecture Comparison")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def find_best_thresholds(y_true: np.ndarray, y_probs: np.ndarray, candidates: Optional[np.ndarray] = None) -> np.ndarray:
    cand = candidates if candidates is not None else np.linspace(0.1, 0.9, 17)
    n_classes = y_true.shape[1]
    best_thresholds = np.full(n_classes, 0.5)

    for c in range(n_classes):
        y_c = y_true[:, c]
        p_c = y_probs[:, c]
        best_score, best_th = -1.0, 0.5
        for th in cand:
            pred_c = (p_c >= th).astype(int)
            score = f1_score(y_c, pred_c, zero_division=0)
            if score > best_score:
                best_score = score
                best_th = th
        best_thresholds[c] = best_th

    return best_thresholds


def apply_thresholds(y_probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (y_probs >= thresholds).astype(int)


def plot_results_heatmap(results_df: pd.DataFrame, index: str = "model", columns: str = "features",
                         value: str = "f1_micro", title: Optional[str] = None):
    pivot = results_df.pivot_table(index=index, columns=columns, values=value)
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.8), max(5, len(pivot) * 0.35)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, ax=ax, linewidths=0.5)
    ax.set_title(title or f"{value.upper()} Comparison")
    plt.tight_layout()
    return fig


def plot_top_configurations(results_df: pd.DataFrame, label_cols: List[str] = ["features", "model"],
                            metric: str = "f1_micro", top_n: int = 10, title: Optional[str] = None):
    df = results_df.copy()
    df["label"] = df[label_cols].astype(str).agg(" | ".join, axis=1)
    df = df.sort_values(metric, ascending=False).head(top_n).sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, max(4, top_n * 0.35)))
    bars = ax.barh(df["label"], df[metric], color="teal", edgecolor="black", alpha=0.85)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(metric.upper())
    ax.set_title(title or f"Top {top_n} Configurations by {metric.upper()}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    return fig
