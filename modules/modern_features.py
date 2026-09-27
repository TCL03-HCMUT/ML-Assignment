"""
Module: Modern Feature Extraction
Word2Vec, GloVe, and frozen BERT embedding extraction.

Produces dense numpy arrays and saves them via `feature_io` so they are
interchangeable with the traditional feature extractor outputs.
"""

import numpy as np

from modules.config import ModernFeatureConfig
from modules.feature_io import save_features, load_features


class ModernFeatureExtractor:
    def __init__(self, config: ModernFeatureConfig = None):
        """
        Initializes the embedding model based on the config.

        Implementation Guide:
        1. Default to `ModernFeatureConfig()` if `config` is None.
        2. If `config.embedding_type == 'word2vec'`:
           - `import gensim.downloader as api`
           - `self.model = api.load('word2vec-google-news-300')`
        3. If `'glove'`:
           - `self.model = api.load('glove-wiki-gigaword-100')` (or 200/300).
        4. If `'bert'`:
           - `from transformers import AutoModel, AutoTokenizer`
           - `self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)`
           - `self.model = AutoModel.from_pretrained(config.model_name)`
           - `self.model.eval()` — freeze the model.
        """
        self.config = config or ModernFeatureConfig()
        self.model = None
        self.tokenizer = None  # Only used for BERT

    def _embed_word2vec(self, texts):
        """
        Generates sentence embeddings via word-level mean pooling.

        Implementation Guide:
        1. For each text, split into words.
        2. Collect vectors for words that exist in `self.model.key_to_index`.
        3. If no words have vectors, return a zero vector of the correct dim.
        4. Otherwise return `np.mean(vectors, axis=0)`.
        5. Stack all sentence embeddings into a 2D array.

        Args:
            texts (list[str]): List of text strings.
        Returns:
            np.ndarray: Shape (N, embedding_dim).
        """
        pass

    def _embed_bert(self, texts, batch_size=64):
        """
        Generates sentence embeddings from a frozen BERT-family model.

        Implementation Guide:
        1. `import torch`.
        2. Process in batches of `batch_size` to avoid OOM.
        3. For each batch:
           - `encoded = self.tokenizer(batch, padding=True, truncation=True,
              max_length=128, return_tensors='pt')`.
           - `with torch.no_grad(): outputs = self.model(**encoded)`.
           - `hidden = outputs.last_hidden_state`.
           - If `self.config.pooling_strategy == 'cls'`:
             `emb = hidden[:, 0, :]`.
           - If `'mean'`:
             `mask = encoded['attention_mask'].unsqueeze(-1)`
             `emb = (hidden * mask).sum(1) / mask.sum(1)`.
           - Append `emb.numpy()`.
        4. Concatenate all batches and return.

        Args:
            texts (list[str]): List of text strings.
            batch_size (int): Texts to process at once.
        Returns:
            np.ndarray: Shape (N, hidden_dim).
        """
        pass

    def extract(self, texts):
        """
        Dispatches to the correct embedding method based on config.

        Implementation Guide:
        1. If `self.config.embedding_type` in ('word2vec', 'glove'):
           return `self._embed_word2vec(texts)`.
        2. If `'bert'`:
           return `self._embed_bert(texts)`.
        3. Else raise `ValueError`.

        Args:
            texts (list[str]): Input texts.
        Returns:
            np.ndarray: Dense feature array of shape (N, D).
        """
        pass

    def extract_and_save(self, texts, filename):
        """
        Extracts features and saves them to disk in one step.

        Implementation Guide:
        1. `embeddings = self.extract(texts)`.
        2. `save_features(embeddings, filename, self.config.features_dir, self.config.save_format)`.
        3. Return `embeddings`.

        Args:
            texts (list[str]): Input texts.
            filename (str): Base filename (e.g. 'bert_cls_train').
        Returns:
            np.ndarray: Dense feature array.
        """
        pass
