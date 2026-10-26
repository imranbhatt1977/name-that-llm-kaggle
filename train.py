# train.py
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from src.utils import seed_everything
from src.dataset import LLMDataset
from src.model import train_fold

# --- CONFIGURATION ---
class Config:
    seed = 42
    model_name = "microsoft/deberta-v3-base"  # Upgrade to deberta-v3-large if GPU memory permits
    max_length = 512
    batch_size = 8
    epochs = 3
    lr = 2e-5
    weight_decay = 0.01
    n_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    seed_everything(Config.seed)
    
    # --- 1. DATA LOADING ---
    # Update paths based on your Kaggle or local directory
    train_path = 'train.csv'
    test_path = 'test.csv'
    
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # --- 2. LABEL ENCODING & CLASS WEIGHTS ---
    encoder = LabelEncoder()
    train_df['label'] = encoder.fit_transform(train_df['model'])
    num_classes = len(encoder.classes_)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(Config.device)
    
    # --- 3. TOKENIZER SETUP ---
    print(f"Loading tokenizer: {Config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    
    # --- 4. STRATIFIED GROUP K-FOLD CROSS VALIDATION ---
    sgkf = StratifiedGroupKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, train_df['label'], groups=train_df['prompt_id'])):
        train_texts = train_df['text'].values[train_idx]
        train_labels = train_df['label'].values[train_idx]
        val_texts = train_df['text'].values[val_idx]
        val_labels = train_df['label'].values[val_idx]
        
        train_dataset = LLMDataset(train_texts, train_labels, tokenizer, Config.max_length)
        val_dataset = LLMDataset(val_texts, val_labels, tokenizer, Config.max_length)
        
        # Train fold and save best weights
        train_fold(
            fold=fold,
            model_name=Config.model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=Config.batch_size,
            epochs=Config.epochs,
            lr=Config.lr,
            weight_decay=Config.weight_decay,
            class_weights_tensor=class_weights_tensor,
            device=Config.device
        )
        
        # For local testing, you can break after fold 0 or remove 'break' to run all 5 folds
        break 

    # --- 5. TEST INFERENCE & SUBMISSION ---
    print("\n--- Running Test Inference ---")
    test_dataset = LLMDataset(test_df['text'].values, tokenizer=tokenizer, max_length=Config.max_length)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    model = AutoModelForSequenceClassification.from_pretrained(Config.model_name, num_labels=num_classes)
    model.load_state_dict(torch.load("deberta_v3_fold_0.pth", map_location=Config.device))
    model.to(Config.device)
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(Config.device)
            attention_mask = batch['attention_mask'].to(Config.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    predicted_models = encoder.inverse_transform(all_preds)
    
    # --- 6. GENERATE SUBMISSION.PARQUET ---
    # Required format: id, model saved as parquet
    submission = pd.DataFrame({
        'id': test_df['id'],
        'model': predicted_models
    })
    submission.to_parquet('submission.parquet', index=False)
    print("🎉 Submission successfully saved to submission.parquet!")

if __name__ == "__main__":
    main()
