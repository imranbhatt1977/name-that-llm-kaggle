# src/model.py
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, get_cosine_schedule_with_warmup
from tqdm import tqdm


class HybridLLMClassifier(nn.Module):
    """
    Hybrid Transformer + Stylometric Fusion Classifier.
    Combines attention-masked mean-pooled DeBERTa representations with a
    projection over the 236 engineered stylometric & typographic features.
    """
    def __init__(self, model_name: str, num_labels: int, stylo_dim: int = 0, dropout: float = 0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.stylo_dim = stylo_dim

        if stylo_dim > 0:
            stylo_hidden = 128
            self.stylo_head = nn.Sequential(
                nn.LayerNorm(stylo_dim),
                nn.Linear(stylo_dim, stylo_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            combined_dim = hidden_size + stylo_hidden
        else:
            self.stylo_head = None
            combined_dim = hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, stylo_features=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        # Attention-mask weighted mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask

        if self.stylo_head is not None and stylo_features is not None:
            stylo_emb = self.stylo_head(stylo_features)
            features = torch.cat([pooled, stylo_emb], dim=1)
        else:
            features = pooled

        logits = self.classifier(features)
        return logits


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    train_loss = 0.0
    
    for batch in tqdm(dataloader, desc="[Train]", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        stylo = batch['stylo_features'].to(device) if 'stylo_features' in batch else None
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask, stylo_features=stylo)
        loss = criterion(logits, labels)
        train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
    return train_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    """Evaluates the model on validation data and returns accuracy, predictions, and probabilities."""
    model.eval()
    val_preds, actual_labels, val_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Val]", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            stylo = batch['stylo_features'].to(device) if 'stylo_features' in batch else None
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask, stylo_features=stylo)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            val_probs.append(probs.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            
    accuracy = np.mean(np.array(val_preds) == np.array(actual_labels))
    return accuracy, val_preds, np.vstack(val_probs)


def train_fold(
    fold: int,
    model_name: str,
    num_classes: int,
    train_dataset,
    val_dataset,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_weights_tensor,
    device,
    stylo_dim: int = 0
):
    """Executes training and validation for a single cross-validation fold."""
    print(f"\n--- Training Hybrid DeBERTa Fold {fold + 1} ---")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = HybridLLMClassifier(
        model_name=model_name, 
        num_labels=num_classes,
        stylo_dim=stylo_dim
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    best_val_accuracy = 0.0
    model_save_path = f"deberta_v3_fold_{fold}.pth"
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_accuracy, _, _ = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model checkpoint for fold {fold + 1}")
            
    print(f"Fold {fold+1} Best Validation Accuracy: {best_val_accuracy:.4f}")
    return model_save_path
