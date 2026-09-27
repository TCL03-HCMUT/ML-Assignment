"""
Module: Data Loader
Responsible for downloading, loading, and exploring the GoEmotions dataset.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

from modules.config import LABEL_NAMES, NUM_LABELS


# ──────────────────────────────────────────────
#  Loading
# ──────────────────────────────────────────────

def load_goemotions_data():
    """
    Downloads the GoEmotions dataset using the HuggingFace datasets library
    and returns it as Pandas DataFrames.

    Implementation Guide:
    1. Use `load_dataset("google-research-datasets/goemotions", "simplified")` to
       fetch the dataset.  The "simplified" config gives a single `labels` column
       that is a list of integer label IDs.
    2. Convert each split to a DataFrame (`dataset["train"].to_pandas()`, etc.).
    3. Call `binarize_labels()` on each DataFrame to expand the `labels` list
       column into 28 binary columns (one per emotion).
    4. Return the three splits: (df_train, df_val, df_test).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    pass


def binarize_labels(df, labels_col="labels"):
    """
    Converts a column of label-ID lists into 28 binary indicator columns.

    Implementation Guide:
    1. Use `sklearn.preprocessing.MultiLabelBinarizer` with `classes=range(NUM_LABELS)`.
    2. `fit_transform(df[labels_col])` produces a (N, 28) binary array.
    3. Assign the 28 columns to the DataFrame using `LABEL_NAMES` as column names.
    4. (Optional) Drop the original `labels_col` to keep the DataFrame tidy.
    5. Return the updated DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the raw labels column.
        labels_col (str): Name of the column with label-ID lists.
    Returns:
        pd.DataFrame: DataFrame with 28 new binary columns.
    """
    pass


# ──────────────────────────────────────────────
#  Exploratory Data Analysis
# ──────────────────────────────────────────────

def get_eda_statistics(df, text_col="text"):
    """
    Computes descriptive statistics for text lengths and label distributions.

    Implementation Guide:
    1. Add a `word_count` column: `df[text_col].str.split().str.len()`.
    2. Compute mean, std, median, min, max of `word_count`.
    3. Compute label frequencies: `df[LABEL_NAMES].sum()`.
    4. Compute multi-label stats: how many samples have >1 label.
    5. Return a dict with keys like "word_count_stats", "label_frequencies",
       "multi_label_ratio".

    Args:
        df (pd.DataFrame): A split with text and binarized label columns.
        text_col (str): Name of the text column.
    Returns:
        dict: Computed statistics.
    """
    pass


def plot_class_distribution(df, label_cols=None):
    """
    Plots a horizontal bar chart of emotion class frequencies.

    Implementation Guide:
    1. Default `label_cols` to `LABEL_NAMES` if None.
    2. Sum each label column, sort descending.
    3. Use `sns.barplot` (horizontal) or `plt.barh`.
    4. Add a title, axis labels, and call `plt.tight_layout()`.

    Args:
        df (pd.DataFrame): A split with binarized label columns.
        label_cols (list | None): Columns to plot.
    """
    pass


def plot_text_length_distribution(df, text_col="text"):
    """
    Plots a histogram of text lengths (in words).

    Implementation Guide:
    1. Compute word counts: `df[text_col].str.split().str.len()`.
    2. Plot with `plt.hist` or `sns.histplot` (bins=50).
    3. Add vertical lines for mean and median.

    Args:
        df (pd.DataFrame): A split.
        text_col (str): Name of the text column.
    """
    pass


def plot_label_cooccurrence(df, label_cols=None):
    """
    Plots a heatmap of label co-occurrence to show which emotions tend
    to appear together.

    Implementation Guide:
    1. Default `label_cols` to `LABEL_NAMES` if None.
    2. Compute co-occurrence matrix: `df[label_cols].T.dot(df[label_cols])`.
    3. Plot with `sns.heatmap`, use a mask for the upper triangle to reduce clutter.

    Args:
        df (pd.DataFrame): A split with binarized label columns.
        label_cols (list | None): Columns to analyse.
    """
    pass

"""TODO: Add more EDA functions if needed:
- stop words analysis
- vocabulary richness
- top words per each emotion class

Refer to
https://ltsach.github.io/AILearningHub/01_Data_Analysis/01_EDA/bbcnews_text_classification/eda_report_tutorial.html
for more information.""" 