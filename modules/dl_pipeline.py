"""
Module: Deep Learning Pipeline
End-to-End Transformer fine-tuning for multi-label classification (Bonus).

All options are driven by a `DLPipelineConfig` object.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from modules.config import DLPipelineConfig, LABEL_NAMES


# ──────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────

class EndToEndDLPipeline:
    def __init__(self, config: DLPipelineConfig = None):
        """
        Initializes the Transformer model and tokenizer from config.

        Implementation Guide:
        1. Default to `DLPipelineConfig()` if `config` is None.
        2. `from transformers import AutoTokenizer, AutoModelForSequenceClassification`.
        3. `self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)`.
        4. `self.model = AutoModelForSequenceClassification.from_pretrained(
               config.model_name,
               num_labels=config.num_labels,
               problem_type="multi_label_classification",
           )`.
        5. `self.model.to(self.device)`.
        """
        self.config = config or DLPipelineConfig()
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_history = []  # Stores per-epoch loss/metrics for plotting

    def _tokenize(self, texts):
        """
        Tokenizes a list of texts using the HuggingFace tokenizer.

        Implementation Guide:
        1. `return self.tokenizer(
               texts,
               padding='max_length',
               truncation=True,
               max_length=self.config.max_length,
               return_tensors='pt',
           )`.

        Args:
            texts (list[str]): Input texts.
        Returns:
            dict: Tokenizer output with 'input_ids' and 'attention_mask' tensors.
        """
        pass

    def prepare_dataloaders(self, train_texts, train_labels, val_texts, val_labels):
        """
        Tokenizes inputs and creates PyTorch DataLoaders.

        Implementation Guide:
        1. Tokenize train and val texts with `self._tokenize()`.
        2. Convert labels to `torch.FloatTensor` (required for BCEWithLogitsLoss).
        3. Create `TensorDataset(input_ids, attention_mask, labels)` for each split.
        4. Wrap in `DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True/False)`.
        5. Return `(train_loader, val_loader)`.

        Args:
            train_texts (list[str]): Training texts.
            train_labels (np.ndarray): Binary label matrix (N, 28).
            val_texts (list[str]): Validation texts.
            val_labels (np.ndarray): Binary label matrix (N, 28).
        Returns:
            tuple[DataLoader, DataLoader]
        """
        pass

    def train(self, train_loader, val_loader):
        """
        Fine-tunes the entire model end-to-end.

        Implementation Guide:
        1. `optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)`.
        2. For each epoch in `range(self.config.epochs)`:
           a. Training loop:
              - `self.model.train()`.
              - For each batch in `train_loader`:
                - Move `input_ids`, `attention_mask`, `labels` to `self.device`.
                - `outputs = self.model(input_ids=..., attention_mask=..., labels=...)`.
                - `loss = outputs.loss`.
                - `loss.backward()`.
                - `optimizer.step()`, `optimizer.zero_grad()`.
              - Track average training loss.
           b. Validation loop:
              - `self.model.eval()`.
              - `with torch.no_grad():` loop over `val_loader`, compute avg val loss.
           c. Append `{'epoch': ..., 'train_loss': ..., 'val_loss': ...}` to `self.train_history`.
           d. Print progress.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
        """
        pass

    def predict(self, test_loader):
        """
        Generates predictions from the fine-tuned model.

        Implementation Guide:
        1. `self.model.eval()`.
        2. `all_preds = []`.
        3. `with torch.no_grad():` loop over `test_loader`:
           - `outputs = self.model(input_ids=..., attention_mask=...)`.
           - `logits = outputs.logits`.
           - `probs = torch.sigmoid(logits)`.
           - `preds = (probs >= self.config.threshold).int()`.
           - `all_preds.append(preds.cpu().numpy())`.
        4. `return np.concatenate(all_preds, axis=0)`.

        Args:
            test_loader (DataLoader): Test data.
        Returns:
            np.ndarray: Binary prediction matrix (N, 28).
        """
        pass

    def plot_training_history(self):
        """
        Plots training and validation loss over epochs.

        Implementation Guide:
        1. Extract epoch numbers, train losses, val losses from `self.train_history`.
        2. `plt.plot(epochs, train_losses, label='Train Loss')`.
        3. `plt.plot(epochs, val_losses, label='Val Loss')`.
        4. Add labels, legend, title.
        """
        pass

    def run(self, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
        """
        Convenience method: prepare data, train, predict, return predictions.

        Implementation Guide:
        1. `train_loader, val_loader = self.prepare_dataloaders(...)`.
        2. Build a test DataLoader similarly (without labels or with dummy labels).
        3. `self.train(train_loader, val_loader)`.
        4. `y_pred = self.predict(test_loader)`.
        5. Return `y_pred`.
        """
        pass
