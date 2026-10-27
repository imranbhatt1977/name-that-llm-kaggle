# src/dataset.py
import torch
from torch.utils.data import Dataset

class LLMDataset(Dataset):
    """PyTorch Dataset for handling text tokenization, padding, and optional stylometric features for authorship attribution."""
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, stylo_features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stylo_features = stylo_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        # Squeeze batch dimension added by the tokenizer
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.stylo_features is not None:
            item['stylo_features'] = torch.tensor(self.stylo_features[idx], dtype=torch.float)
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
