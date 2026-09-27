"""
Module: Preprocessing
Configurable text cleaning, stop-word removal, stemming, and lemmatization.

All steps are toggled via a `PreprocessingConfig` dataclass so that different
preprocessing strategies can be compared without editing this file.
"""

import re

from modules.config import PreprocessingConfig


# ──────────────────────────────────────────────
#  Atomic Preprocessing Steps
# ──────────────────────────────────────────────

def lowercase(text):
    """
    Converts text to lowercase.

    Implementation Guide:
    1. Return `text.lower()`.
    """
    pass


def remove_punctuation(text):
    """
    Removes all punctuation and special characters.

    Implementation Guide:
    1. Use `re.sub(r'[^\\w\\s]', '', text)` to strip non-alphanumeric characters.
    2. Return the result.
    """
    pass


def remove_numbers(text):
    """
    Removes all digit characters.

    Implementation Guide:
    1. Use `re.sub(r'\\d+', '', text)`.
    2. Return the result.
    """
    pass


def remove_extra_whitespace(text):
    """
    Collapses multiple spaces into a single space and strips leading/trailing spaces.

    Implementation Guide:
    1. Use `re.sub(r'\\s+', ' ', text).strip()`.
    2. Return the result.
    """
    pass


def remove_stopwords(text, stop_words):
    """
    Removes common English stop words.

    Implementation Guide:
    1. Split the text into tokens (words).
    2. Filter out any word that exists in the `stop_words` set.
    3. Join the remaining words back into a single string.

    Args:
        text (str): The cleaned text.
        stop_words (set): A set of stop words.
    Returns:
        str: Text without stop words.
    """
    pass


def apply_stemming(text):
    """
    Applies Porter Stemming to each word.

    Implementation Guide:
    1. Import `PorterStemmer` from `nltk.stem`.
    2. Instantiate the stemmer.
    3. Split text, stem each word, rejoin.

    Args:
        text (str): Text to stem.
    Returns:
        str: Stemmed text.
    """
    pass


def apply_lemmatization(text):
    """
    Applies WordNet Lemmatization to each word.

    Implementation Guide:
    1. Import `WordNetLemmatizer` from `nltk.stem`.
    2. Instantiate the lemmatizer.
    3. Split text, lemmatize each word (default POS='n'), rejoin.

    Args:
        text (str): Text to lemmatize.
    Returns:
        str: Lemmatized text.
    """
    pass


def filter_short_words(text, min_length):
    """
    Removes words shorter than `min_length`.

    Implementation Guide:
    1. Split text into words.
    2. Keep only words where `len(word) >= min_length`.
    3. Rejoin and return.

    Args:
        text (str): Input text.
        min_length (int): Minimum word length to keep.
    Returns:
        str: Filtered text.
    """
    pass


# ──────────────────────────────────────────────
#  Orchestrator
# ──────────────────────────────────────────────

def preprocess_text(text, config: PreprocessingConfig, stop_words=None):
    """
    Applies all preprocessing steps enabled in `config` to a single text string.

    Implementation Guide:
    1. If `config.lowercase` is True, call `lowercase(text)`.
    2. If `config.remove_punctuation`, call `remove_punctuation(text)`.
    3. If `config.remove_numbers`, call `remove_numbers(text)`.
    4. Always call `remove_extra_whitespace(text)`.
    5. If `config.remove_stopwords` and `stop_words` is provided, call `remove_stopwords(text, stop_words)`.
    6. If `config.stemming`, call `apply_stemming(text)`.
    7. Elif `config.lemmatization`, call `apply_lemmatization(text)`.
    8. If `config.min_word_length > 1`, call `filter_short_words(text, config.min_word_length)`.
    9. Return the fully processed text.

    Args:
        text (str): Raw text.
        config (PreprocessingConfig): Configuration object.
        stop_words (set | None): Optional stop words set.
    Returns:
        str: Processed text.
    """
    pass


def preprocess_dataset(df, config: PreprocessingConfig, text_col="text"):
    """
    Applies the full preprocessing pipeline to every row in the DataFrame.

    Implementation Guide:
    1. If `config.remove_stopwords`, load a stop word list:
       - `import nltk; nltk.download('stopwords')`
       - `stop_words = set(nltk.corpus.stopwords.words('english'))`
    2. If `config.lemmatization`, also download `nltk.download('wordnet')`.
    3. Apply `preprocess_text` to each row: `df[text_col].apply(lambda t: preprocess_text(t, config, stop_words))`.
    4. Store the result in a new column (e.g., `processed_text`) or overwrite `text_col`.
    5. Return the updated DataFrame.

    Args:
        df (pd.DataFrame): The dataset split.
        config (PreprocessingConfig): Configuration object.
        text_col (str): Name of the column containing text.
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    pass
