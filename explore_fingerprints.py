# explore_fingerprints.py
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.features import process_dataframe_features, to_syntactic_skeleton


def compute_style_feature_weights(X_tr, y_tr_model, y_tr_prompt, power_model=0.6, power_prompt=0.4):
    """
    Computes per-feature weights strictly on the training fold:
    Upweights features with high variance across models (F_model)
    and downweights features with high variance across prompts/topics (F_prompt).
    """
    f_model, _ = f_classif(X_tr, y_tr_model)
    f_prompt, _ = f_classif(X_tr, y_tr_prompt)

    f_model = np.nan_to_num(f_model, nan=0.0, posinf=0.0, neginf=0.0)
    f_prompt = np.nan_to_num(f_prompt, nan=1.0, posinf=100.0, neginf=1.0)

    # Excess F above 1.0 indicates genuine model signal
    signal = np.maximum(f_model - 0.8, 0.05) ** power_model
    topic_noise = (1.0 + np.maximum(f_prompt - 1.0, 0.0)) ** power_prompt
    weights = signal / topic_noise
    # Normalize mean weight to 1.0
    weights = weights / (np.mean(weights) + 1e-8)
    return weights


def main():
    print("=" * 75)
    print("PHASE 1 & 2: EXPANDED STYLOMETRIC & SYNTACTIC FINGERPRINT ANALYSIS")
    print("=" * 75)

    train_df = pd.read_csv("train.csv")
    models = sorted(train_df["model"].unique())
    encoder = LabelEncoder()
    y = encoder.fit_transform(train_df["model"])
    prompt_encoder = LabelEncoder()
    p_ids = prompt_encoder.fit_transform(train_df["prompt_id"])

    print("Extracting expanded stylometric features...")
    feat_df = process_dataframe_features(train_df)
    print(f"Total engineered stylometric features: {feat_df.shape[1]}")

    print("Generating topic-masked syntactic skeletons...")
    skeletons = [to_syntactic_skeleton(t) for t in train_df["text"]]
    print(f"Sample Skeleton:\n  Raw     : {train_df['text'].iloc[0][:140]}...\n  Skeleton: {skeletons[0][:140]}...")

    # Rank features by Model-to-Prompt F-ratio
    f_mod, _ = f_classif(feat_df.fillna(0), y)
    f_prm, _ = f_classif(feat_df.fillna(0), p_ids)
    f_mod = np.nan_to_num(f_mod, nan=0.0)
    f_prm = np.nan_to_num(f_prm, nan=1.0)

    f_rank = pd.DataFrame({
        "feature": feat_df.columns,
        "F_model": f_mod,
        "F_prompt": f_prm,
        "Style_to_Topic_Ratio": f_mod / np.maximum(f_prm, 1.0)
    }).sort_values("F_model", ascending=False)

    print("\n--- TOP 25 STYLOMETRIC FINGERPRINTS (MODEL F-SCORE VS PROMPT F-SCORE) ---")
    print(f_rank.head(25).to_string(index=False))

    # =========================================================================
    # PHASE 3: 5-FOLD STRATIFIED GROUP K-FOLD WITH STYLE-WEIGHTED VIEWS
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 3: 5-FOLD GROUP-AWARE CV WITH TOPIC-SUPPRESSED FEATURE WEIGHTING")
    print("=" * 75)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    n_samples = len(train_df)
    n_classes = len(models)

    oof_stylo_lr = np.zeros((n_samples, n_classes))
    oof_stylo_hgb = np.zeros((n_samples, n_classes))
    oof_skel_lr = np.zeros((n_samples, n_classes))
    oof_word_lr = np.zeros((n_samples, n_classes))
    oof_char_lr = np.zeros((n_samples, n_classes))
    oof_joint_lr = np.zeros((n_samples, n_classes))
    oof_joint_mlp = np.zeros((n_samples, n_classes))

    texts = train_df["text"].values
    skeletons_arr = np.array(skeletons)
    X_stylo_all = feat_df.fillna(0).values

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups=train_df["prompt_id"])):
        y_tr, y_va = y[train_idx], y[val_idx]
        p_tr = p_ids[train_idx]

        # --- View 1: Engineered Stylometric Features (209 features) ---
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_stylo_all[train_idx])
        X_va_s = scaler.transform(X_stylo_all[val_idx])
        X_tr_s = np.clip(X_tr_s, -8.0, 8.0)
        X_va_s = np.clip(X_va_s, -8.0, 8.0)

        w_stylo = compute_style_feature_weights(X_tr_s, y_tr, p_tr, power_model=0.5, power_prompt=0.4)
        X_tr_sw = X_tr_s * w_stylo
        X_va_sw = X_va_s * w_stylo

        clf_s_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        clf_s_lr.fit(X_tr_sw, y_tr)
        oof_stylo_lr[val_idx] = clf_s_lr.predict_proba(X_va_sw)

        clf_s_hgb = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=25, min_samples_leaf=15, l2_regularization=2.0, random_state=42
        )
        clf_s_hgb.fit(X_tr_sw, y_tr)
        oof_stylo_hgb[val_idx] = clf_s_hgb.predict_proba(X_va_sw)

        # --- View 2: Syntactic & Morphological Skeleton TF-IDF ---
        skel_vec = TfidfVectorizer(
            ngram_range=(1, 4),
            token_pattern=r"(?u)\b\w+\b|<NUM>|[—–;:,!?'’“”\"()-]",
            min_df=4,
            max_features=35000,
            sublinear_tf=True,
            lowercase=False
        )
        X_tr_skel = skel_vec.fit_transform(skeletons_arr[train_idx])
        X_va_skel = skel_vec.transform(skeletons_arr[val_idx])
        w_skel = compute_style_feature_weights(X_tr_skel, y_tr, p_tr, power_model=0.65, power_prompt=0.45)
        X_tr_skel_w = X_tr_skel.multiply(w_skel).tocsr()
        X_va_skel_w = X_va_skel.multiply(w_skel).tocsr()

        clf_skel = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
        clf_skel.fit(X_tr_skel_w, y_tr)
        oof_skel_lr[val_idx] = clf_skel.predict_proba(X_va_skel_w)

        # --- View 3: Word + Punctuation N-Grams (Topic-Suppressed) ---
        word_vec = TfidfVectorizer(
            ngram_range=(1, 3),
            token_pattern=r"(?u)\b\w+\b|[—–;:,!?'’“”\"()-]",
            min_df=4,
            max_features=35000,
            sublinear_tf=True,
            lowercase=False
        )
        X_tr_word = word_vec.fit_transform(texts[train_idx])
        X_va_word = word_vec.transform(texts[val_idx])
        w_word = compute_style_feature_weights(X_tr_word, y_tr, p_tr, power_model=0.7, power_prompt=0.5)
        X_tr_word_w = X_tr_word.multiply(w_word).tocsr()
        X_va_word_w = X_va_word.multiply(w_word).tocsr()

        clf_word = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
        clf_word.fit(X_tr_word_w, y_tr)
        oof_word_lr[val_idx] = clf_word.predict_proba(X_va_word_w)

        # --- View 4: Character N-Grams (Topic-Suppressed) ---
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=6,
            max_features=50000,
            sublinear_tf=True,
            lowercase=False
        )
        X_tr_char = char_vec.fit_transform(texts[train_idx])
        X_va_char = char_vec.transform(texts[val_idx])
        w_char = compute_style_feature_weights(X_tr_char, y_tr, p_tr, power_model=0.7, power_prompt=0.5)
        X_tr_char_w = X_tr_char.multiply(w_char).tocsr()
        X_va_char_w = X_va_char.multiply(w_char).tocsr()

        clf_char = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
        clf_char.fit(X_tr_char_w, y_tr)
        oof_char_lr[val_idx] = clf_char.predict_proba(X_va_char_w)

        # --- View 5: Joint Multi-View Representation ---
        X_tr_joint = hstack([
            csr_matrix(X_tr_sw * 0.12),
            X_tr_skel_w * 0.9,
            X_tr_word_w * 1.0,
            X_tr_char_w * 1.1
        ]).tocsr()
        X_va_joint = hstack([
            csr_matrix(X_va_sw * 0.12),
            X_va_skel_w * 0.9,
            X_va_word_w * 1.0,
            X_va_char_w * 1.1
        ]).tocsr()

        clf_joint = LogisticRegression(C=1.2, max_iter=1000, random_state=42)
        clf_joint.fit(X_tr_joint, y_tr)
        oof_joint_lr[val_idx] = clf_joint.predict_proba(X_va_joint)

        # Select top 4000 style features for Neural MLP
        f_joint, _ = f_classif(X_tr_joint, y_tr)
        f_joint = np.nan_to_num(f_joint, nan=0.0)
        top_k_idx = np.argsort(f_joint)[-4000:]
        clf_mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1.5,
            learning_rate_init=0.001,
            max_iter=35,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42 + fold
        )
        clf_mlp.fit(X_tr_joint[:, top_k_idx], y_tr)
        oof_joint_mlp[val_idx] = clf_mlp.predict_proba(X_va_joint[:, top_k_idx])

    def eval_oof(name, probs):
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")
        print(f"{name:50s} | OOF Acc: {acc*100:6.2f}% | Macro F1: {f1*100:6.2f}%")
        return preds

    print("\n--- 5-FOLD OUT-OF-FOLD (OOF) RESULTS ON UNSEEN PROMPTS ---")
    eval_oof("1. Expanded Stylometric Only (LogReg, 209 feats)", oof_stylo_lr)
    eval_oof("2. Expanded Stylometric Only (HistGBDT)", oof_stylo_hgb)
    eval_oof("3. Syntactic Skeleton TF-IDF (Topic-Masked)", oof_skel_lr)
    eval_oof("4. Topic-Suppressed Word+Punct TF-IDF", oof_word_lr)
    eval_oof("5. Topic-Suppressed Character TF-IDF", oof_char_lr)
    eval_oof("6. Joint Multi-View Linear (Stylo+Skel+Word+Char)", oof_joint_lr)
    eval_oof("7. Joint Multi-View Neural MLP (Top 4k Style Feats)", oof_joint_mlp)

    oof_ens = (
        0.10 * oof_stylo_lr
        + 0.12 * oof_stylo_hgb
        + 0.12 * oof_skel_lr
        + 0.13 * oof_word_lr
        + 0.15 * oof_char_lr
        + 0.23 * oof_joint_lr
        + 0.15 * oof_joint_mlp
    )
    ens_preds = eval_oof("8. Multi-View Ensemble Blend", oof_ens)

    print("\n--- PER-MODEL OOF ACCURACY (MULTI-VIEW ENSEMBLE) ---")
    cm = confusion_matrix(y, ens_preds)
    rows = []
    for i, m in enumerate(models):
        tot = cm[i].sum()
        corr = cm[i, i]
        r_copy = cm[i].copy()
        r_copy[i] = -1
        top_c = np.argmax(r_copy)
        rows.append((m, corr / tot * 100, corr, tot, models[top_c], r_copy[top_c]))
    rows.sort(key=lambda x: x[1], reverse=True)
    for m, acc_i, corr_i, tot_i, cm_m, cm_cnt in rows:
        print(f"- {m:24s}: {acc_i:6.2f}% ({corr_i:3d}/{tot_i:3d}) | Top confusion: {cm_m} ({cm_cnt})")


if __name__ == "__main__":
    main()
