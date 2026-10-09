"""
Module: Data Loader
Responsible for downloading, loading, and exploring the GoEmotions dataset.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


from modules.config import LABEL_NAMES, NUM_LABELS


DATASET_ID = 'google-research-datasets/go_emotions'
DATASET_CONFIG = 'simplified'
DATA_REVISION = 'add492243ff905527e67aeb8b80c082af02207c3' # Commit in the Hugging Face dataset repo
EXPECTED_SIZES = {'train': 43410, 'validation': 5426, 'test': 5427}

# ──────────────────────────────────────────────
#  Loading
# ──────────────────────────────────────────────

def load_goemotions_data():
    """
    Downloads the GoEmotions dataset using the HuggingFace datasets library
    and returns it as Pandas DataFrames.
    cache holds the Datasets cache and manifest. The Hub library manages

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    cache = Path('data/raw')
    cache.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATA_REVISION,
                           cache_dir=str(cache / 'huggingface'))
    if set(dataset) != set(EXPECTED_SIZES):
        raise ValueError('Unexpected dataset splits; expected train/validation/test')
    manifest = {
        'provider': 'huggingface', 'dataset_id': DATASET_ID,
        'config': DATASET_CONFIG, 'revision': DATA_REVISION,
        'source_url': f'https://huggingface.co/datasets/{DATASET_ID}/tree/{DATA_REVISION}',
        'label_names': LABEL_NAMES,
        'splits': {},
    }
    frames = []
    for split, expected_size in EXPECTED_SIZES.items():
        source = dataset[split]
        if source.features['labels'].feature.names != LABEL_NAMES:
            raise ValueError(f'Official emotion order disagrees with modules.config in {split}')
        df = source.to_pandas()
        if not {'id', 'text', 'labels'}.issubset(df.columns):
            raise ValueError(f'Missing required columns in {split}')
        if len(df) != expected_size:
            raise ValueError(f'Unexpected {split} row count: {len(df)}')
        if df['id'].isna().any() or df['id'].eq('').any() or df['id'].duplicated().any():
            raise ValueError(f'Missing/duplicate comment IDs in {split}')
        if df['text'].isna().any() or not df['text'].map(lambda t: isinstance(t, str)).all():
            raise ValueError(f'Missing/non-string comments in {split}')
        if df['text'].str.strip().eq('').any():
            raise ValueError(f'Blank comments in {split}; inspect source before cleaning')
        df = binarize_labels(df)
        df['labels'] = df['labels'].map(lambda values: [int(i) for i in values])
        manifest['splits'][split] = {'rows': len(df)}
        frames.append(df)
    (cache / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return tuple(frames)


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
    from sklearn.preprocessing import MultiLabelBinarizer
    for row, labels in enumerate(df[labels_col]):
        if not isinstance(labels, (list, tuple, np.ndarray)) or len(labels) == 0:
            raise ValueError(f'Empty or malformed labels at row {row}')
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or not 0 <= i < NUM_LABELS for i in labels):
            raise ValueError(f'Invalid label ID at row {row}: {labels}')
        if len(set(labels)) != len(labels):
            raise ValueError(f'Repeated label ID at row {row}')
    out = df.copy()
    # Use `sklearn.preprocessing.MultiLabelBinarizer` with `classes=range(NUM_LABELS)`.
    # `fit_transform(df[labels_col])` produces a (N, 28) binary array.
    y = MultiLabelBinarizer(classes=range(NUM_LABELS)).fit_transform(out[labels_col])
    out[LABEL_NAMES] = y.astype(np.uint8)
    return out


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
    counts = df[text_col].str.split().str.len()
    chars = df[text_col].str.len()
    nlabels = df[LABEL_NAMES].sum(axis=1)
    return {'n_samples': len(df),
            'word_count_stats': counts.describe(percentiles=[.25,.5,.75,.95,.99]).to_dict(),
            'char_count_stats': chars.describe(percentiles=[.25,.5,.75,.95,.99]).to_dict(),
            'label_frequencies': df[LABEL_NAMES].sum().to_dict(),
            'multi_label_ratio': float((nlabels > 1).mean()),
            'label_cardinality': float(nlabels.mean()),
            'label_density': float(nlabels.mean() / NUM_LABELS)}


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
    import matplotlib.pyplot as plt
    counts = df[label_cols or LABEL_NAMES].sum().sort_values()
    fig, ax = plt.subplots(figsize=(9,8))
    counts.plot.barh(ax=ax, color='#287c9a')
    ax.set(title='Training label support (labels may overlap)', xlabel='Comments containing label')
    fig.tight_layout()
    return fig


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
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    for ax, values, title in zip(axes, [df[text_col].str.split().str.len(), df[text_col].str.len()], ['Words (whitespace tokens)', 'Characters (Unicode code points)']):
        ax.hist(values, bins=45, color='#287c9a', alpha=.85)
        ax.axvline(values.mean(), color='#c6682c', label=f'Mean {values.mean():.1f}')
        ax.axvline(values.median(), color='#333333', linestyle='--', label=f'Median {values.median():.1f}')
        ax.set(xlabel=title, ylabel='Comments', title=f'Raw training text: {title}')
        ax.legend()
    fig.tight_layout()
    return fig


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
    import matplotlib.pyplot as plt
    import seaborn as sns
    labels = label_cols or LABEL_NAMES
    # Cast BEFORE dot product: uint8 would overflow at 255.
    y = df[labels].to_numpy(dtype=np.int64)
    co = y.T @ y
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(co, xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax,
                mask=np.triu(np.ones_like(co, dtype=bool), k=1))
    ax.set_title('Training label co-occurrence counts (diagonal = support)')
    fig.tight_layout()
    return fig

"""TODO: Add more EDA functions if needed:
- stop words analysis
- vocabulary richness
- top words per each emotion class

Refer to
https://ltsach.github.io/AILearningHub/01_Data_Analysis/01_EDA/bbcnews_text_classification/eda_report_tutorial.html
for more information.""" 