"""
Module: Modern Feature Extraction
Word2Vec, GloVe, and frozen BERT embedding extraction.

Produces dense numpy arrays and saves them via `feature_io` so they are
interchangeable with the traditional feature extractor outputs.
"""

import os
from typing import List, Union, Optional
import numpy as np

from modules.config import ModernFeatureConfig
from modules.feature_io import save_features, load_features


class ModernFeatureExtractor:
    """
    Feature extractor utilizing pre-trained static embeddings (Word2Vec, GloVe)
    and contextual Transformer embeddings (BERT, DistilBERT).
    """

    def __init__(self, config: ModernFeatureConfig = None, model = None):
        """
        Initializes the embedding model based on the config.

        Args:
            config (ModernFeatureConfig, optional): Configuration settings.
            model (optional): Pre-loaded model or dictionary (useful for custom/mock embeddings).
        """
        self.config = config or ModernFeatureConfig()
        self.model = model
        self.tokenizer = None

        if self.model is None:
            self._load_model()

    def _load_model(self):
        """Loads the pre-trained embedding model based on `self.config.embedding_type`."""
        emb_type = self.config.embedding_type.lower()

        if emb_type in ("word2vec", "glove"):
            try:
                import gensim.downloader as api
                from gensim.models import KeyedVectors
            except ImportError as e:
                raise ImportError(
                    "gensim is required for Word2Vec/GloVe embeddings. "
                    "Please install it via `pip install gensim` or supply a pre-loaded model."
                ) from e

            # Determine model identifier or path
            if emb_type == "word2vec":
                default_name = "word2vec-google-news-300"
            else:
                default_name = "glove-wiki-gigaword-100"

            model_identifier = self.config.model_name or default_name

            # Check if model_identifier is a local file
            if os.path.exists(model_identifier):
                print(f"Loading local embeddings from {model_identifier}...")
                is_bin = model_identifier.endswith(".bin") or model_identifier.endswith(".bin.gz")
                self.model = KeyedVectors.load_word2vec_format(model_identifier, binary=is_bin)
            else:
                print(f"Loading pre-trained {emb_type.upper()} model ({model_identifier}) via gensim downloader...")
                self.model = api.load(model_identifier)

            print(f"Loaded {emb_type.upper()} model successfully. Vector size: {getattr(self.model, 'vector_size', 'unknown')}")

        elif emb_type in ("bert", "distilbert"):
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "transformers is required for BERT embeddings. "
                    "Please install it via `pip install transformers`."
                ) from e

            model_name = self.config.model_name or "bert-base-uncased"
            print(f"Loading pre-trained Transformer ({model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()  # Freeze weights for feature extraction
            print(f"Loaded {model_name} successfully in evaluation mode.")

        else:
            raise ValueError(
                f"Unsupported embedding_type: {self.config.embedding_type!r}. "
                "Expected 'word2vec', 'glove', or 'bert'."
            )

    def _embed_word2vec(self, texts: List[str]) -> np.ndarray:
        """
        Generates sentence embeddings via word-level mean pooling.

        Args:
            texts (list[str]): List of text strings.

        Returns:
            np.ndarray: Dense feature array of shape (N, embedding_dim).
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # Determine embedding dimension
        if hasattr(self.model, "vector_size"):
            embedding_dim = self.model.vector_size
        elif hasattr(self.model, "vectors"):
            embedding_dim = self.model.vectors.shape[1]
        elif hasattr(self.model, "values"):
            first_val = next(iter(self.model.values()))
            embedding_dim = len(first_val)
        else:
            embedding_dim = 300

        has_key_to_index = hasattr(self.model, "key_to_index")
        embeddings = []

        for text in texts:
            if text is None or (isinstance(text, float) and np.isnan(text)):
                text = ""
            else:
                text = str(text)

            # Tokenize by whitespace and strip boundary punctuation
            raw_tokens = text.split()
            vectors = []

            for raw_token in raw_tokens:
                token = raw_token.strip(".,!?;:\"'()[]{}*~`-_/")
                if not token:
                    continue

                # Match word (exact case first, then lowercase)
                matched_vec = None
                if has_key_to_index:
                    if token in self.model.key_to_index:
                        matched_vec = self.model[token]
                    elif token.lower() in self.model.key_to_index:
                        matched_vec = self.model[token.lower()]
                elif token in self.model:
                    matched_vec = self.model[token]
                elif token.lower() in self.model:
                    matched_vec = self.model[token.lower()]

                if matched_vec is not None:
                    vectors.append(matched_vec)

            # Mean pooling across words, or zero-vector fallback
            if len(vectors) > 0:
                sentence_vec = np.mean(vectors, axis=0)
            else:
                sentence_vec = np.zeros(embedding_dim, dtype=np.float32)

            # Optional L2 normalization
            if getattr(self.config, "normalize_embeddings", False):
                norm = np.linalg.norm(sentence_vec)
                if norm > 0:
                    sentence_vec = sentence_vec / norm

            embeddings.append(sentence_vec)

        return np.vstack(embeddings).astype(np.float32)

    def _embed_bert(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Generates sentence embeddings from a frozen BERT-family model.

        Args:
            texts (list[str]): List of text strings.
            batch_size (int): Texts to process in each forward pass.

        Returns:
            np.ndarray: Dense feature array of shape (N, hidden_dim).
        """
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) if t is not None else "" for t in texts[i : i + batch_size]]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            hidden = outputs.last_hidden_state  # Shape: (B, Seq_Len, Hidden_Dim)

            if self.config.pooling_strategy == "cls":
                emb = hidden[:, 0, :]
            else:
                # Attention mask weighted mean pooling
                mask = encoded["attention_mask"].unsqueeze(-1)
                emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            if getattr(self.config, "normalize_embeddings", False):
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb.cpu().numpy())

        if len(all_embeddings) == 0:
            hidden_dim = getattr(self.model.config, "hidden_size", 768)
            return np.empty((0, hidden_dim), dtype=np.float32)

        return np.vstack(all_embeddings).astype(np.float32)

    def extract(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Dispatches to the correct embedding method based on config.

        Args:
            texts: List of text strings or pandas Series / numpy array.

        Returns:
            np.ndarray: Dense feature array of shape (N, D).
        """
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = list(texts)

        if self.config.embedding_type in ("word2vec", "glove"):
            return self._embed_word2vec(texts)
        elif self.config.embedding_type in ("bert", "distilbert"):
            return self._embed_bert(texts)
        else:
            raise ValueError(
                f"Unsupported embedding_type {self.config.embedding_type!r}; "
                "expected 'word2vec', 'glove', or 'bert'."
            )

    def extract_and_save(self, texts: Union[List[str], np.ndarray], filename: str) -> np.ndarray:
        """
        Extracts features and saves them to disk in one step.

        Args:
            texts: Input texts.
            filename (str): Base filename (e.g. 'word2vec_train' or 'bert_cls_train').

        Returns:
            np.ndarray: Dense feature array.
        """
        embeddings = self.extract(texts)
        save_features(
            embeddings,
            filename,
            features_dir=self.config.features_dir,
            save_format=self.config.save_format,
        )
        return embeddings
