# src/tabular_model.py
import os
import warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.features import process_dataframe_features, to_syntactic_skeleton


def _softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    scaled = logits * temp
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_v = np.exp(scaled)
    return exp_v / np.sum(exp_v, axis=1, keepdims=True)


def compute_style_feature_weights(X_tr, y_tr_model, y_tr_prompt, power_model=0.55, power_prompt=0.35):
    """
    Computes per-feature weights strictly inside each training fold:
    Upweights features that discriminate between models (F_model)
    and downweights features driven by prompt topic (F_prompt).
    """
    f_model, _ = f_classif(X_tr, y_tr_model)
    f_prompt, _ = f_classif(X_tr, y_tr_prompt)

    f_model = np.nan_to_num(f_model, nan=0.0, posinf=0.0, neginf=0.0)
    f_prompt = np.nan_to_num(f_prompt, nan=1.0, posinf=100.0, neginf=1.0)

    signal = np.maximum(f_model - 0.85, 0.05) ** power_model
    topic_noise = (1.0 + np.maximum(f_prompt - 1.0, 0.0)) ** power_prompt
    weights = signal / topic_noise
    return weights / (np.mean(weights) + 1e-8)


def train_and_predict_multiview_ensemble(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
):
    """
    Trains a 5-fold StratifiedGroupKFold (grouped by prompt_id) Multi-View Ensemble:
      - View 1: 236 Engineered Stylometric, Lexical, Syntactic & Typography features
      - View 2: Word + Punctuation N-Grams (1-3) with topic-suppressed weighting
      - View 3: Character N-Grams (2-5, char & char_wb)
      - View 4: Topic-Masked Syntactic Skeleton N-Grams (1-4)
    Combines SGD (log-loss), LinearSVC, RidgeClassifier, Stylometric LogReg, and HistGBDT.
    All preprocessors, scalers, and vectorizers are fit strictly inside each training fold.
    """
    print("\n--- Extracting 236 Stylometric & Syntactic Skeleton Features ---")
    X_stylo_train = process_dataframe_features(train_df).fillna(0).values
    X_stylo_test = process_dataframe_features(test_df).fillna(0).values

    train_texts = train_df["text"].astype(str).values
    test_texts = test_df["text"].astype(str).values

    train_skels = np.array([to_syntactic_skeleton(t) for t in train_texts])
    test_skels = np.array([to_syntactic_skeleton(t) for t in test_texts])

    prompt_codes = pd.factorize(train_df["prompt_id"])[0]
    num_classes = len(np.unique(y))

    oof_stylo_lr = np.zeros((len(train_df), num_classes))
    oof_stylo_hgb = np.zeros((len(train_df), num_classes))
    oof_sgd = np.zeros((len(train_df), num_classes))
    oof_svc = np.zeros((len(train_df), num_classes))
    oof_ridge = np.zeros((len(train_df), num_classes))

    test_stylo_lr = np.zeros((len(test_df), num_classes))
    test_stylo_hgb = np.zeros((len(test_df), num_classes))
    test_sgd = np.zeros((len(test_df), num_classes))
    test_svc = np.zeros((len(test_df), num_classes))
    test_ridge = np.zeros((len(test_df), num_classes))

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(train_df, y, groups=train_df["prompt_id"])):
        y_tr, y_va = y[tr_idx], y[va_idx]
        p_tr = prompt_codes[tr_idx]

        # 1. Stylometric View (strictly fit on training fold)
        rob_scaler = RobustScaler()
        Xs_tr_rob = np.clip(rob_scaler.fit_transform(X_stylo_train[tr_idx]), -8.0, 8.0)
        Xs_va_rob = np.clip(rob_scaler.transform(X_stylo_train[va_idx]), -8.0, 8.0)
        Xs_te_rob = np.clip(rob_scaler.transform(X_stylo_test), -8.0, 8.0)

        w_stylo = compute_style_feature_weights(Xs_tr_rob, y_tr, p_tr, power_model=0.5, power_prompt=0.4)
        Xs_tr_w = Xs_tr_rob * w_stylo
        Xs_va_w = Xs_va_rob * w_stylo
        Xs_te_w = Xs_te_rob * w_stylo

        clf_stylo_lr = LogisticRegression(C=0.5, max_iter=800, random_state=seed + fold)
        clf_stylo_lr.fit(Xs_tr_w, y_tr)
        oof_stylo_lr[va_idx] = clf_stylo_lr.predict_proba(Xs_va_w)
        test_stylo_lr += clf_stylo_lr.predict_proba(Xs_te_w) / n_folds

        clf_stylo_hgb = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.04,
            max_leaf_nodes=25,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=seed + fold
        )
        clf_stylo_hgb.fit(Xs_tr_w, y_tr)
        oof_stylo_hgb[va_idx] = clf_stylo_hgb.predict_proba(Xs_va_w)
        test_stylo_hgb += clf_stylo_hgb.predict_proba(Xs_te_w) / n_folds

        # 2. Multi-Granularity Text & Skeleton Views (strictly fit on training fold)
        std_scaler = StandardScaler()
        Xs_tr_std = np.clip(std_scaler.fit_transform(X_stylo_train[tr_idx]), -6.0, 6.0)
        Xs_va_std = np.clip(std_scaler.transform(X_stylo_train[va_idx]), -6.0, 6.0)
        Xs_te_std = np.clip(std_scaler.transform(X_stylo_test), -6.0, 6.0)

        w_vec = TfidfVectorizer(
            ngram_range=(1, 3),
            token_pattern=r"(?u)\b\w+\b|[—–;:,!?'’“”\"()-]",
            min_df=3,
            max_df=0.95,
            max_features=35000,
            sublinear_tf=True,
            lowercase=False
        )
        Xw_tr = w_vec.fit_transform(train_texts[tr_idx])
        Xw_va = w_vec.transform(train_texts[va_idx])
        Xw_te = w_vec.transform(test_texts)

        c_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=5,
            max_df=0.95,
            max_features=50000,
            sublinear_tf=True,
            lowercase=False
        )
        Xc_tr = c_vec.fit_transform(train_texts[tr_idx])
        Xc_va = c_vec.transform(train_texts[va_idx])
        Xc_te = c_vec.transform(test_texts)

        cw_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=5,
            max_df=0.95,
            max_features=40000,
            sublinear_tf=True,
            lowercase=False
        )
        Xcw_tr = cw_vec.fit_transform(train_texts[tr_idx])
        Xcw_va = cw_vec.transform(train_texts[va_idx])
        Xcw_te = cw_vec.transform(test_texts)

        sk_vec = TfidfVectorizer(
            ngram_range=(1, 4),
            token_pattern=r"(?u)\b\w+\b|<NUM>|[—–;:,!?'’“”\"()-]",
            min_df=4,
            max_features=30000,
            sublinear_tf=True,
            lowercase=False
        )
        Xsk_tr = sk_vec.fit_transform(train_skels[tr_idx])
        Xsk_va = sk_vec.transform(train_skels[va_idx])
        Xsk_te = sk_vec.transform(test_skels)

        X_tr_joint = hstack([
            csr_matrix(Xs_tr_std * 0.10),
            Xw_tr,
            Xc_tr,
            Xcw_tr,
            Xsk_tr * 0.8
        ]).tocsr()
        X_va_joint = hstack([
            csr_matrix(Xs_va_std * 0.10),
            Xw_va,
            Xc_va,
            Xcw_va,
            Xsk_va * 0.8
        ]).tocsr()
        X_te_joint = hstack([
            csr_matrix(Xs_te_std * 0.10),
            Xw_te,
            Xc_te,
            Xcw_te,
            Xsk_te * 0.8
        ]).tocsr()

        # Model A: Sparse SGD Log-Loss
        sgd = SGDClassifier(loss="log_loss", alpha=5e-5, max_iter=45, random_state=seed + fold)
        sgd.fit(X_tr_joint, y_tr)
        oof_sgd[va_idx] = sgd.predict_proba(X_va_joint)
        test_sgd += sgd.predict_proba(X_te_joint) / n_folds

        # Model B: LinearSVC (margin-based)
        svc = LinearSVC(C=0.25, random_state=seed + fold)
        svc.fit(X_tr_joint, y_tr)
        oof_svc[va_idx] = _softmax(svc.decision_function(X_va_joint), temp=2.5)
        test_svc += _softmax(svc.decision_function(X_te_joint), temp=2.5) / n_folds

        # Model C: Fast L2 Ridge Classifier (lsqr solver)
        ridge = RidgeClassifier(alpha=3.0, solver="lsqr", tol=1e-3, random_state=seed + fold)
        ridge.fit(X_tr_joint, y_tr)
        oof_ridge[va_idx] = _softmax(ridge.decision_function(X_va_joint), temp=3.0)
        test_ridge += _softmax(ridge.decision_function(X_te_joint), temp=3.0) / n_folds

        fold_blend = (
            0.30 * oof_sgd[va_idx]
            + 0.25 * oof_svc[va_idx]
            + 0.20 * oof_ridge[va_idx]
            + 0.13 * oof_stylo_lr[va_idx]
            + 0.12 * oof_stylo_hgb[va_idx]
        )
        fold_acc = accuracy_score(y_va, np.argmax(fold_blend, axis=1))
        print(f"  Fold {fold + 1}/{n_folds} - Validation Accuracy: {fold_acc * 100:.2f}%")

    oof_ensemble = (
        0.30 * oof_sgd
        + 0.25 * oof_svc
        + 0.20 * oof_ridge
        + 0.13 * oof_stylo_lr
        + 0.12 * oof_stylo_hgb
    )
    test_ensemble = (
        0.30 * test_sgd
        + 0.25 * test_svc
        + 0.20 * test_ridge
        + 0.13 * test_stylo_lr
        + 0.12 * test_stylo_hgb
    )

    oof_preds = np.argmax(oof_ensemble, axis=1)
    overall_acc = accuracy_score(y, oof_preds)
    overall_f1 = f1_score(y, oof_preds, average="macro")

    print("\n" + "=" * 65)
    print(f"5-Fold OOF Ensemble Accuracy : {overall_acc * 100:.2f}%")
    print(f"5-Fold OOF Ensemble Macro-F1 : {overall_f1 * 100:.2f}%")
    print("=" * 65)

    return oof_ensemble, test_ensemble
