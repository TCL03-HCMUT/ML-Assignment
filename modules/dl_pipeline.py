"""Module: Deep Learning Pipeline
End-to-End Transformer fine-tuning for multi-label classification.
Supports pre-trained HuggingFace models (DistilBERT, BERT, RoBERTa).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from modules.config import DLPipelineConfig, NUM_LABELS


PRETRAINED_MODELS: Dict[str, str] = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}


def resolve_model_name(name: str) -> str:
    return PRETRAINED_MODELS.get(name.lower(), name)


class EndToEndDLPipeline:
    def __init__(self, config: Optional[DLPipelineConfig] = None):
        self.config = config or DLPipelineConfig()
        model_identifier = resolve_model_name(self.config.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_identifier,
            num_labels=self.config.num_labels,
            problem_type="multi_label_classification",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_history: List[dict] = []
        self.thresholds_: Optional[np.ndarray] = None

    def _tokenize(self, texts: List[str]) -> dict:
        clean_texts = [str(t) if t is not None else "" for t in texts]
        return self.tokenizer(
            clean_texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def create_dataloader(self, texts: List[str], labels: Optional[np.ndarray] = None, shuffle: bool = False) -> DataLoader:
        tokens = self._tokenize(texts)
        if labels is not None:
            label_tensor = torch.FloatTensor(labels)
            dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], label_tensor)
        else:
            dummy_labels = torch.zeros((len(texts), self.config.num_labels), dtype=torch.float32)
            dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], dummy_labels)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def prepare_dataloaders(self, train_texts: List[str], train_labels: np.ndarray,
                            val_texts: List[str], val_labels: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        train_loader = self.create_dataloader(train_texts, train_labels, shuffle=True)
        val_loader = self.create_dataloader(val_texts, val_labels, shuffle=False)
        return train_loader, val_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader, y_train_for_loss: Optional[np.ndarray] = None):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        criterion = None
        loss_type = getattr(self.config, "loss", "bce")
        if loss_type == "bce_weighted" and y_train_for_loss is not None:
            pos_counts = np.maximum(y_train_for_loss.sum(axis=0), 1)
            neg_counts = len(y_train_for_loss) - pos_counts
            weights = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(self.device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

        for epoch in range(self.config.epochs):
            self.model.train()
            total_train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs} [Train]")

            for batch in pbar:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels) if criterion else torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / max(len(train_loader), 1)

            self.model.eval()
            total_val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs} [Val]")

            with torch.no_grad():
                for batch in val_pbar:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels) if criterion else torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
                    total_val_loss += loss.item()
                    val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_val_loss = total_val_loss / max(len(val_loader), 1)
            self.train_history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    def predict_proba(self, loader_or_texts) -> np.ndarray:
        loader = loader_or_texts if isinstance(loader_or_texts, DataLoader) else self.create_dataloader(loader_or_texts, shuffle=False)
        self.model.eval()
        probs_list = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits)
                probs_list.append(probs.cpu().numpy())

        if not probs_list:
            return np.empty((0, self.config.num_labels), dtype=float)
        return np.concatenate(probs_list, axis=0)

    def tune_thresholds(self, loader_or_texts, y_val: np.ndarray, strategy: str = "per_class") -> np.ndarray:
        from modules.evaluation import find_best_thresholds
        probs = self.predict_proba(loader_or_texts)
        self.thresholds_ = find_best_thresholds(y_val, probs)
        return self.thresholds_

    def predict(self, loader_or_texts) -> np.ndarray:
        probs = self.predict_proba(loader_or_texts)
        if self.thresholds_ is not None:
            return (probs >= self.thresholds_).astype(int)
        return (probs >= self.config.threshold).astype(int)

    def plot_training_history(self):
        if not self.train_history:
            print("No training history available.")
            return

        epochs = [e["epoch"] for e in self.train_history]
        tr = [e["train_loss"] for e in self.train_history]
        va = [e["val_loss"] for e in self.train_history]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, tr, label="Train Loss", marker="o")
        ax.plot(epochs, va, label="Val Loss", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Fine-Tuning Loss ({self.config.model_name})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        return fig

    def run(self, train_texts: List[str], train_labels: np.ndarray,
            val_texts: List[str], val_labels: np.ndarray,
            test_texts: List[str], test_labels: Optional[np.ndarray] = None) -> np.ndarray:
        train_loader, val_loader = self.prepare_dataloaders(train_texts, train_labels, val_texts, val_labels)
        test_loader = self.create_dataloader(test_texts, test_labels, shuffle=False)
        self.train(train_loader, val_loader, y_train_for_loss=train_labels)
        return self.predict(test_loader)
