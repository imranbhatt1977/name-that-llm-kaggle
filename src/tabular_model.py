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


# Sibling model clusters where fine-grained within-family disambiguation is applied
SIBLING_CLUSTERS = [
    ("Gemini Family", ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"], 0.35),
    ("Gemini Flash Trio", ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"], 0.35),
    ("GPT Family", ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"], 0.35),
    ("GPT-5.6 Trio", ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"], 0.35),
    ("Claude + GLM Group", ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "glm-5.3", "glm-5.3-flash"], 0.35),
    ("Unspaced Em-Dash Group", ["claude-fable-5", "claude-opus-4-8", "claude-sonnet-5", "glm-5.3", "glm-5.3-flash"], 0.35),
    ("GLM Pair", ["glm-5.3", "glm-5.3-flash"], 0.35),
    ("GPT-5.5 / Kimi / Qwen", ["gpt-5.5", "kimi-k3", "qwen3.8-max"], 0.30),
]


def _softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    scaled = logits * temp
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_v = np.exp(scaled)
    return exp_v / np.sum(exp_v, axis=1, keepdims=True)


def compute_style_feature_weights(X_tr, y_tr_model, y_tr_prompt=None, power_model=0.55, power_prompt=0.35):
    """
    Computes per-feature weights strictly inside each training fold:
    Upweights features that discriminate between models (F_model)
    and downweights features driven by prompt topic (F_prompt).
    """
    f_model, _ = f_classif(X_tr, y_tr_model)
    f_model = np.nan_to_num(f_model, nan=0.0, posinf=0.0, neginf=0.0)
    signal = np.maximum(f_model - 0.85, 0.05) ** power_model

    if y_tr_prompt is not None:
        f_prompt, _ = f_classif(X_tr, y_tr_prompt)
        f_prompt = np.nan_to_num(f_prompt, nan=1.0, posinf=100.0, neginf=1.0)
        topic_noise = (1.0 + np.maximum(f_prompt - 1.0, 0.0)) ** power_prompt
        weights = signal / topic_noise
    else:
        weights = signal

    return weights / (np.mean(weights) + 1e-8)


def calibrate_class_priors(probs: np.ndarray, n_iters: int = 8, strength: float = 0.25) -> np.ndarray:
    """
    Gently aligns marginal predicted class probabilities toward the uniform 1/18 prior
    when predicting on a batch of samples.
    """
    if len(probs) < 18 or strength <= 0.0:
        return probs
    p = np.clip(probs.copy(), 1e-8, 1.0)
    target_prior = 1.0 / p.shape[1]
    for _ in range(n_iters):
        col_means = p.mean(axis=0, keepdims=True)
        scale = (target_prior / np.maximum(col_means, 1e-8)) ** strength
        p = p * scale
        p = p / p.sum(axis=1, keepdims=True)
    return p


def train_predict_cluster_specialist(
    X_tr_stylo: np.ndarray,
    X_va_stylo: np.ndarray,
    X_te_stylo: np.ndarray,
    X_tr_text: csr_matrix,
    X_va_text: csr_matrix,
    X_te_text: csr_matrix,
    y_tr: np.ndarray,
    p_tr: np.ndarray,
    cluster_indices: list,
    seed: int = 42
):
    """
    Trains a within-cluster specialist ensemble exclusively on training rows belonging
    to `cluster_indices`, using within-cluster ANOVA F-scores to weight sibling-discriminating
    features. Returns conditional probabilities P(m | x, m in C) for validation and test sets.
    """
    mask_tr = np.isin(y_tr, cluster_indices)
    X_tr_s_sub = X_tr_stylo[mask_tr]
    X_tr_t_sub = X_tr_text[mask_tr]
    y_tr_sub_raw = y_tr[mask_tr]
    p_tr_sub = p_tr[mask_tr]

    idx_map = {c: i for i, c in enumerate(cluster_indices)}
    y_tr_sub = np.array([idx_map[c] for c in y_tr_sub_raw])

    w_s = compute_style_feature_weights(X_tr_s_sub, y_tr_sub, p_tr_sub, power_model=0.65, power_prompt=0.35)
    w_t = compute_style_feature_weights(X_tr_t_sub, y_tr_sub, p_tr_sub, power_model=0.70, power_prompt=0.45)

    Xs_tr_w = X_tr_s_sub * w_s
    Xs_va_w = X_va_stylo * w_s
    Xs_te_w = X_te_stylo * w_s

    Xt_tr_w = X_tr_t_sub.multiply(w_t).tocsr()
    Xt_va_w = X_va_text.multiply(w_t).tocsr()
    Xt_te_w = X_te_text.multiply(w_t).tocsr()

    X_tr_comb = hstack([csr_matrix(Xs_tr_w * 0.14), Xt_tr_w]).tocsr()
    X_va_comb = hstack([csr_matrix(Xs_va_w * 0.14), Xt_va_w]).tocsr()
    X_te_comb = hstack([csr_matrix(Xs_te_w * 0.14), Xt_te_w]).tocsr()

    # Specialist 1: Stylometric Logistic Regression
    clf_lr = LogisticRegression(C=0.4, max_iter=500, random_state=seed)
    clf_lr.fit(Xs_tr_w, y_tr_sub)
    va_lr = clf_lr.predict_proba(Xs_va_w)
    te_lr = clf_lr.predict_proba(Xs_te_w)

    # Specialist 2: Combined Sparse SGD (log-loss)
    clf_sgd = SGDClassifier(loss="log_loss", alpha=8e-5, max_iter=50, random_state=seed)
    clf_sgd.fit(X_tr_comb, y_tr_sub)
    va_sgd = clf_sgd.predict_proba(X_va_comb)
    te_sgd = clf_sgd.predict_proba(X_te_comb)

    # Specialist 3: Combined Margin LinearSVC
    clf_svc = LinearSVC(C=0.20, random_state=seed)
    clf_svc.fit(X_tr_comb, y_tr_sub)
    dec_va = clf_svc.decision_function(X_va_comb)
    dec_te = clf_svc.decision_function(X_te_comb)
    if dec_va.ndim == 1:
        dec_va = np.vstack([-dec_va, dec_va]).T
        dec_te = np.vstack([-dec_te, dec_te]).T
    va_svc = _softmax(dec_va, temp=2.2)
    te_svc = _softmax(dec_te, temp=2.2)

    # Specialist 4: Stylometric HistGBDT
    clf_hgb = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=3.0, random_state=seed
    )
    clf_hgb.fit(Xs_tr_w, y_tr_sub)
    va_hgb = clf_hgb.predict_proba(Xs_va_w)
    te_hgb = clf_hgb.predict_proba(Xs_te_w)

    va_spec = 0.35 * va_sgd + 0.30 * va_svc + 0.20 * va_lr + 0.15 * va_hgb
    te_spec = 0.35 * te_sgd + 0.30 * te_svc + 0.20 * te_lr + 0.15 * te_hgb
    return va_spec, te_spec


def apply_hierarchical_refinement(
    global_probs: np.ndarray,
    spec_probs: np.ndarray,
    cluster_indices: list,
    alpha: float = 0.35
) -> np.ndarray:
    """
    Refines within-cluster conditional probabilities while strictly preserving
    the global probability mass assigned to the cluster:
      P_refined(m | x) = P_global(C | x) * [(1 - alpha) * P_global(m | x, m in C) + alpha * P_spec(m | x, m in C)]
    """
    refined = global_probs.copy()
    cluster_mass = np.sum(refined[:, cluster_indices], axis=1, keepdims=True)
    cond_global = refined[:, cluster_indices] / np.maximum(cluster_mass, 1e-8)
    cond_blended = (1.0 - alpha) * cond_global + alpha * spec_probs
    refined[:, cluster_indices] = cluster_mass * cond_blended
    return refined


def train_and_predict_multiview_ensemble(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: np.ndarray,
    class_names: list = None,
    n_folds: int = 5,
    seed: int = 42
):
    """
    Trains a 5-fold StratifiedGroupKFold (grouped by prompt_id) Hierarchical Multi-View Ensemble:
      - Stage 1 (Global 18-Class Ensemble):
          Combines 236 Stylometric features, Word/Punct N-Grams (1-3), Character N-Grams (2-5),
          and Topic-Masked Syntactic Skeleton N-Grams (1-4) across raw and topic-suppressed views.
      - Stage 2 (Sibling Cluster Specialists):
          Refines conditional probabilities within 8 confused sibling groups using within-cluster
          ANOVA F-score feature weighting.
      - Stage 3 (Marginal Prior Calibration):
          Aligns marginal class distributions when evaluating on batches.
    """
    print("\n--- Extracting 236 Stylometric & Syntactic Skeleton Features ---")
    X_stylo_train = process_dataframe_features(train_df).fillna(0).values
    X_stylo_test = process_dataframe_features(test_df).fillna(0).values

    train_texts = train_df["text"].astype(str).values
    test_texts = test_df["text"].astype(str).values

    train_skels = np.array([to_syntactic_skeleton(t) for t in train_texts])
    test_skels = np.array([to_syntactic_skeleton(t) for t in test_texts])

    prompt_codes = pd.factorize(train_df["prompt_id"])[0]
    if class_names is None:
        class_names = sorted(train_df["model"].unique())
    name_to_idx = {m: i for i, m in enumerate(class_names)}
    num_classes = len(class_names)

    oof_global = np.zeros((len(train_df), num_classes))
    oof_refined = np.zeros((len(train_df), num_classes))
    test_refined = np.zeros((len(test_df), num_classes))

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
        va_stylo_lr = clf_stylo_lr.predict_proba(Xs_va_w)
        te_stylo_lr = clf_stylo_lr.predict_proba(Xs_te_w)

        clf_stylo_hgb = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.04,
            max_leaf_nodes=25,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=seed + fold
        )
        clf_stylo_hgb.fit(Xs_tr_w, y_tr)
        va_stylo_hgb = clf_stylo_hgb.predict_proba(Xs_va_w)
        te_stylo_hgb = clf_stylo_hgb.predict_proba(Xs_te_w)

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

        X_text_tr = hstack([Xw_tr, Xc_tr, Xcw_tr, Xsk_tr * 0.8]).tocsr()
        X_text_va = hstack([Xw_va, Xc_va, Xcw_va, Xsk_va * 0.8]).tocsr()
        X_text_te = hstack([Xw_te, Xc_te, Xcw_te, Xsk_te * 0.8]).tocsr()

        # Raw Joint Stack
        X_tr_joint = hstack([csr_matrix(Xs_tr_std * 0.10), X_text_tr]).tocsr()
        X_va_joint = hstack([csr_matrix(Xs_va_std * 0.10), X_text_va]).tocsr()
        X_te_joint = hstack([csr_matrix(Xs_te_std * 0.10), X_text_te]).tocsr()

        sgd = SGDClassifier(loss="log_loss", alpha=5e-5, max_iter=45, random_state=seed + fold)
        sgd.fit(X_tr_joint, y_tr)
        va_sgd = sgd.predict_proba(X_va_joint)
        te_sgd = sgd.predict_proba(X_te_joint)

        svc = LinearSVC(C=0.25, random_state=seed + fold)
        svc.fit(X_tr_joint, y_tr)
        va_svc = _softmax(svc.decision_function(X_va_joint), temp=2.5)
        te_svc = _softmax(svc.decision_function(X_te_joint), temp=2.5)

        ridge = RidgeClassifier(alpha=3.0, solver="lsqr", tol=1e-3, random_state=seed + fold)
        ridge.fit(X_tr_joint, y_tr)
        va_ridge = _softmax(ridge.decision_function(X_va_joint), temp=3.0)
        te_ridge = _softmax(ridge.decision_function(X_te_joint), temp=3.0)

        # Topic-Suppressed Joint Stack
        w_text = compute_style_feature_weights(X_text_tr, y_tr, p_tr, power_model=0.65, power_prompt=0.45)
        X_text_tr_w = X_text_tr.multiply(w_text).tocsr()
        X_text_va_w = X_text_va.multiply(w_text).tocsr()
        X_text_te_w = X_text_te.multiply(w_text).tocsr()

        X_tr_wjoint = hstack([csr_matrix(Xs_tr_w * 0.12), X_text_tr_w]).tocsr()
        X_va_wjoint = hstack([csr_matrix(Xs_va_w * 0.12), X_text_va_w]).tocsr()
        X_te_wjoint = hstack([csr_matrix(Xs_te_w * 0.12), X_text_te_w]).tocsr()

        sgd_w = SGDClassifier(loss="log_loss", alpha=8e-5, max_iter=45, random_state=142 + fold)
        sgd_w.fit(X_tr_wjoint, y_tr)
        va_sgd_w = sgd_w.predict_proba(X_va_wjoint)
        te_sgd_w = sgd_w.predict_proba(X_te_wjoint)

        svc_w = LinearSVC(C=0.20, random_state=142 + fold)
        svc_w.fit(X_tr_wjoint, y_tr)
        va_svc_w = _softmax(svc_w.decision_function(X_va_wjoint), temp=2.5)
        te_svc_w = _softmax(svc_w.decision_function(X_te_wjoint), temp=2.5)

        fold_va_global = (
            0.22 * va_sgd
            + 0.18 * va_svc
            + 0.14 * va_ridge
            + 0.18 * va_sgd_w
            + 0.14 * va_svc_w
            + 0.08 * va_stylo_lr
            + 0.06 * va_stylo_hgb
        )
        fold_te_global = (
            0.22 * te_sgd
            + 0.18 * te_svc
            + 0.14 * te_ridge
            + 0.18 * te_sgd_w
            + 0.14 * te_svc_w
            + 0.08 * te_stylo_lr
            + 0.06 * te_stylo_hgb
        )
        oof_global[va_idx] = fold_va_global

        # 3. Stage 2: Sibling Cluster Specialist Refinement
        fold_va_hier = fold_va_global.copy()
        fold_te_hier = fold_te_global.copy()
        for c_name, c_models, alpha in SIBLING_CLUSTERS:
            c_idx = [name_to_idx[m] for m in c_models]
            va_spec, te_spec = train_predict_cluster_specialist(
                Xs_tr_rob, Xs_va_rob, Xs_te_rob,
                X_text_tr, X_text_va, X_text_te,
                y_tr, p_tr,
                cluster_indices=c_idx,
                seed=seed + fold
            )
            fold_va_hier = apply_hierarchical_refinement(fold_va_hier, va_spec, c_idx, alpha=alpha)
            fold_te_hier = apply_hierarchical_refinement(fold_te_hier, te_spec, c_idx, alpha=alpha)

        oof_refined[va_idx] = fold_va_hier
        test_refined += fold_te_hier / n_folds

        g_acc = accuracy_score(y_va, np.argmax(fold_va_global, axis=1))
        h_acc = accuracy_score(y_va, np.argmax(fold_va_hier, axis=1))
        print(f"  Fold {fold + 1}/{n_folds} - Global Acc: {g_acc * 100:.2f}% -> +Sibling Specialists Acc: {h_acc * 100:.2f}%")

    global_acc = accuracy_score(y, np.argmax(oof_global, axis=1))
    global_f1 = f1_score(y, np.argmax(oof_global, axis=1), average="macro")

    hier_acc = accuracy_score(y, np.argmax(oof_refined, axis=1))
    hier_f1 = f1_score(y, np.argmax(oof_refined, axis=1), average="macro")

    print("\n" + "=" * 70)
    print(f"Stage 1 - 5-Fold Global Multi-View OOF   : Acc = {global_acc * 100:.2f}% | Macro-F1 = {global_f1 * 100:.2f}%")
    print(f"Stage 2 - + Sibling Cluster Specialists  : Acc = {hier_acc * 100:.2f}% | Macro-F1 = {hier_f1 * 100:.2f}%")
    print("=" * 70)

    return oof_refined, test_refined
