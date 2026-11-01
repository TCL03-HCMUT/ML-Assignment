"""
Module: Deep Learning Pipeline
End-to-End Transformer fine-tuning for multi-label classification (Bonus).

All options are driven by a `DLPipelineConfig` object.
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

from modules.config import DLPipelineConfig, LABEL_NAMES
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# ──────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────

class EndToEndDLPipeline:
    def __init__(self, config: DLPipelineConfig = None):
        """
        Initializes the Transformer model and tokenizer from config.
        """
        self.config = config
        if self.config is None:
            self.config = DLPipelineConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            problem_type="multi_label_classification"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_history = []  # Stores per-epoch loss/metrics for plotting

    def _tokenize(self, texts):
        """
        Tokenizes a list of texts using the HuggingFace tokenizer.

        Args:
            texts (list[str]): Input texts.
        Returns:
            dict: Tokenizer output with 'input_ids' and 'attention_mask' tensors.
        """
        return self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )

    def prepare_dataloaders(self, train_texts: list[str], train_labels: np.ndarray, val_texts: list[str], val_labels: np.ndarray):
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
        train_tokens = self._tokenize(train_texts)
        train_label_tensors = torch.FloatTensor(train_labels)
        
        val_tokens = self._tokenize(val_texts)
        val_label_tensors = torch.FloatTensor(val_labels)
        
        train_dataset = TensorDataset(
            train_tokens['input_ids'], 
            train_tokens['attention_mask'], 
            train_label_tensors,
        )
        
        val_dataset = TensorDataset(
            val_tokens['input_ids'],
            val_tokens['attention_mask'],
            val_label_tensors
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return train_loader, val_loader

    def train(self, train_loader, val_loader):
        """
        Fine-tunes the entire model end-to-end with progress tracking.
        """
        # 1: Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        # 2: Epoch loop
        for epoch in range(self.config.epochs):
            
            # --- a. Training loop ---
            self.model.train()
            total_train_loss = 0.0
            
            # Wrap train_loader with tqdm for a progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs} [Train]")
            
            for batch in train_pbar:
                # Move inputs and labels to the configured device
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                optimizer.zero_grad()
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                
                # Update progress bar with the current batch loss
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            avg_train_loss = total_train_loss / len(train_loader)

            # --- b. Validation loop ---
            self.model.eval()
            total_val_loss = 0.0
            
            # Wrap val_loader with tqdm
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs} [Val]")
            
            with torch.no_grad():
                for batch in val_pbar:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
            avg_val_loss = total_val_loss / len(val_loader)

            # --- c. Track history ---
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })

            # --- d. Print progress ---
            # Print a clean summary at the end of the epoch
            print(f"Epoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
                

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
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                preds = (probs >= self.config.threshold).int()
                all_preds.append(preds.cpu().numpy())

        if not all_preds:
            return np.empty((0, self.config.num_labels), dtype=int)

        return np.concatenate(all_preds, axis=0)

    def plot_training_history(self):
        """
        Plots training and validation loss over epochs.

        Implementation Guide:
        1. Extract epoch numbers, train losses, val losses from `self.train_history`.
        2. `plt.plot(epochs, train_losses, label='Train Loss')`.
        3. `plt.plot(epochs, val_losses, label='Val Loss')`.
        4. Add labels, legend, title.
        """
        if not self.train_history:
            print("No training history available yet.")
            return

        epochs = [entry['epoch'] for entry in self.train_history]
        train_losses = [entry['train_loss'] for entry in self.train_history]
        val_losses = [entry['val_loss'] for entry in self.train_history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss', marker='o')
        plt.plot(epochs, val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

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
        train_loader, val_loader = self.prepare_dataloaders(train_texts, train_labels, val_texts, val_labels)

        test_tokens = self._tokenize(test_texts)
        test_labels_tensor = torch.FloatTensor(test_labels)
        test_dataset = TensorDataset(
            test_tokens['input_ids'],
            test_tokens['attention_mask'],
            test_labels_tensor,
        )
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        self.train(train_loader, val_loader)
        y_pred = self.predict(test_loader)
        return y_pred
