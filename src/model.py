# src/model.py
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    train_loss = 0.0
    
    for batch in tqdm(dataloader, desc="[Train]", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
    return train_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    """Evaluates the model on validation data and returns predictions and labels."""
    model.eval()
    val_preds, actual_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Val]", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            
    accuracy = np.mean(np.array(val_preds) == np.array(actual_labels))
    return accuracy, val_preds

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
    device
):
    """Executes training and validation for a single cross-validation fold."""
    print(f"\n--- Training Fold {fold + 1} ---")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
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
        val_accuracy, _ = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"✨ Saved new best model checkpoint for fold {fold + 1}")
            
    print(f"Fold {fold+1} Best Validation Accuracy: {best_val_accuracy:.4f}")
    return model_save_path
