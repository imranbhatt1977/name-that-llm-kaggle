# train.py
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from src.utils import seed_everything
from src.features import process_dataframe_features
from src.tabular_model import train_and_predict_multiview_ensemble
from test_submission import validate_submission


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


def run_deberta_pipeline(train_df, test_df, y, num_classes):
    """Optional Hybrid DeBERTa-v3 + Stylometric 5-fold training and test probability prediction."""
    import torch
    from transformers import AutoTokenizer
    from src.dataset import LLMDataset
    from src.model import HybridLLMClassifier, train_fold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    print(f"\nLoading tokenizer: {Config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    X_stylo_train = process_dataframe_features(train_df).fillna(0).values
    X_stylo_test = process_dataframe_features(test_df).fillna(0).values
    stylo_dim = X_stylo_train.shape[1]

    sgkf = StratifiedGroupKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    test_probs = np.zeros((len(test_df), num_classes))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups=train_df["prompt_id"])):
        scaler = RobustScaler()
        Xs_tr = np.clip(scaler.fit_transform(X_stylo_train[train_idx]), -8.0, 8.0)
        Xs_va = np.clip(scaler.transform(X_stylo_train[val_idx]), -8.0, 8.0)
        Xs_te = np.clip(scaler.transform(X_stylo_test), -8.0, 8.0)

        train_texts = train_df["text"].values[train_idx]
        train_labels = y[train_idx]
        val_texts = train_df["text"].values[val_idx]
        val_labels = y[val_idx]

        train_dataset = LLMDataset(train_texts, train_labels, tokenizer, Config.max_length, stylo_features=Xs_tr)
        val_dataset = LLMDataset(val_texts, val_labels, tokenizer, Config.max_length, stylo_features=Xs_va)

        ckpt_path = train_fold(
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
            device=device,
            stylo_dim=stylo_dim
        )

        # Predict on test set with this fold's scaler and checkpoint
        test_dataset = LLMDataset(test_df["text"].values, tokenizer=tokenizer, max_length=Config.max_length, stylo_features=Xs_te)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

        model = HybridLLMClassifier(Config.model_name, num_labels=num_classes, stylo_dim=stylo_dim)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                stylo = batch["stylo_features"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, stylo_features=stylo)
                probs = torch.softmax(logits, dim=1)
                fold_probs.append(probs.cpu().numpy())
        test_probs += np.vstack(fold_probs) / Config.n_folds

    return test_probs


def main():
    parser = argparse.ArgumentParser(description="Train Name-That-LLM Attribution Pipeline")
    parser.add_argument("--use-deberta", action="store_true", help="Also train and ensemble 5-fold Hybrid DeBERTa-v3")
    args = parser.parse_args()

    seed_everything(Config.seed)

    # --- 1. DATA LOADING ---
    train_path = "train.csv"
    test_path = "test.csv"

    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # --- 2. LABEL ENCODING ---
    encoder = LabelEncoder()
    train_df["label"] = encoder.fit_transform(train_df["model"])
    y = train_df["label"].values
    class_names = list(encoder.classes_)
    num_classes = len(class_names)

    # --- 3. 5-FOLD HIERARCHICAL MULTI-VIEW & SIBLING SPECIALIST ENSEMBLE ---
    oof_probs, test_probs = train_and_predict_multiview_ensemble(
        train_df=train_df,
        test_df=test_df,
        y=y,
        class_names=class_names,
        n_folds=Config.n_folds,
        seed=Config.seed
    )

    # --- 4. OPTIONAL HYBRID DEBERTA-V3 BLEND ---
    if args.use_deberta:
        deberta_test_probs = run_deberta_pipeline(train_df, test_df, y, num_classes)
        test_probs = 0.55 * deberta_test_probs + 0.45 * test_probs

    # --- 5. GENERATE & VALIDATE SUBMISSION.PARQUET ---
    all_preds = np.argmax(test_probs, axis=1)
    predicted_models = encoder.inverse_transform(all_preds)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "model": predicted_models
    })
    submission.to_parquet("submission.parquet", index=False)
    print("\nSubmission successfully saved to submission.parquet!")

    validate_submission("submission.parquet", test_path, train_path)


if __name__ == "__main__":
    main()
