"""
Module: Preprocessing
Configurable text cleaning, stop-word removal, stemming, and lemmatization.

All steps are toggled via a `PreprocessingConfig` dataclass so that different
preprocessing strategies can be compared without editing this file.
"""

import re
import html
import unicodedata
from functools import lru_cache
from modules.config import PreprocessingConfig


PROTECTED_WORDS = frozenset({'no','not','nor','never','neither','cannot','very','too','so','more','most','against'}) # words that do not be removed when remove_stopwords is on
TOKEN_PATTERN = re.compile(r"\[[A-Za-z_]+\]|[^\W_]+(?:['’][^\W_]+)*|[^\w\s]", re.UNICODE) # includes word, special tokens, punctuation
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE) # e.g. "word", "word's"


# ──────────────────────────────────────────────
#  Atomic Preprocessing Steps
# ──────────────────────────────────────────────

def lowercase(text):
    """
    Converts text to lowercase.

    Implementation Guide:
    1. Return `text.lower()`.
    """
    return text.lower() # really complicated function to implement @.@


def remove_punctuation(text):
    """
    Replaces punctuation and special characters by space. keep apostrophes, emoji and symbol cues
    """
    return ''.join(' ' if unicodedata.category(c).startswith('P') and c not in "'’" else c for c in text)



def remove_numbers(text):
    return re.sub(r'\d+', ' ', text)


def remove_extra_whitespace(text):
   return re.sub(r'\s+', ' ', text).strip()


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
    return ' '.join(t for t in tokenize_text(text) if t.lower() not in stop_words)

def tokenize_text(text, method='regex'):
    if method == 'whitespace':
        return text.split()
    if method == 'regex':
        return TOKEN_PATTERN.findall(text)
    raise ValueError("tokenizer must be 'regex' or 'whitespace'")

def word_tokens(text):
    """Lexical tokens for EDA; emoji and punctuation are analyzed separately."""
    return WORD_PATTERN.findall(text.lower())

@lru_cache(maxsize=1)
def default_stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return frozenset(ENGLISH_STOP_WORDS) - PROTECTED_WORDS

def prepare_nltk_resources(config):
    """Explicitly prepare optional WordNet; defaults need no NLTK downloads."""
    if config.lemmatization:
        import nltk
        import os
        from pathlib import Path
        resource_dir = Path(os.environ.get('NLTK_DATA', '.cache/nltk_data')).resolve()
        resource_dir.mkdir(parents=True, exist_ok=True)
        if str(resource_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(resource_dir))
        try:
            nltk.corpus.wordnet.ensure_loaded()
        except LookupError:
            if not nltk.download('wordnet', download_dir=str(resource_dir), quiet=True):
                raise RuntimeError('Cannot download WordNet')


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
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(t) if t.isalpha() else t for t in tokenize_text(text))



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
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # Noun-only WordNet normalization; no claim of POS-aware lemmatization.
    return ' '.join(lemmatizer.lemmatize(t) if t.isalpha() else t for t in tokenize_text(text))



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
    return ' '.join(t for t in tokenize_text(text) if not t.isalpha() or len(t) >= min_length)



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
    if not isinstance(text, str):
        raise TypeError('Text must be a string')
    if config.stemming and config.lemmatization:
        raise ValueError('Choose stemming or lemmatization, not both')
    if config.tokenizer not in {'regex', 'whitespace'}:
        raise ValueError('Unsupported tokenizer')
    if not isinstance(config.empty_token, str) or not config.empty_token.strip():
        raise ValueError('empty_token must be a nonblank string')
    if config.min_word_length < 1:
        raise ValueError('min_word_length must be positive')
    text = unicodedata.normalize('NFC', text) if config.normalize_unicode else text
    text = html.unescape(text) if config.decode_html else text
    if config.replace_urls:
        text = re.sub(r'https?://\S+|www\.\S+', ' URLTOKEN ', text, flags=re.I)
    if config.replace_users:
        text = re.sub(r'(?<!\w)(?:/?u/|@)[\w-]+', ' USERTOKEN ', text)
    if config.lowercase:
        text = lowercase(text)
    if config.remove_punctuation:
        text = remove_punctuation(text)
    if config.remove_numbers:
        text = remove_numbers(text)
    text = remove_extra_whitespace(text)
    if config.remove_stopwords:
        stops = set(default_stopwords() if stop_words is None else stop_words)
        if stop_words is None and not config.preserve_negation:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            stops = set(ENGLISH_STOP_WORDS)
        if config.preserve_negation:
            stops -= PROTECTED_WORDS
        text = remove_stopwords(text, stops)
    if config.stemming:
        text = apply_stemming(text)
    elif config.lemmatization:
        text = apply_lemmatization(text)
    if config.min_word_length > 1:
        text = filter_short_words(text, config.min_word_length)
    # Keep empty transformations as an explicit feature
    return remove_extra_whitespace(text) or config.empty_token



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
    prepare_nltk_resources(config)
    out = df.copy()
    out['processed_text'] = out[text_col].map(lambda t: preprocess_text(t, config))
    out['tokens'] = out['processed_text'].map(lambda t: tokenize_text(t, config.tokenizer))
    return out
