# 🤖 Name That LLM - Kaggle Competition Pipeline

Welcome to the repository for the **"Name That LLM"** Kaggle competition. The core task of this project is **authorship attribution (specifically, model attribution)**—identifying which specific frontier language model (out of 18 models across 6 major AI labs) generated a given text passage.

---

## 📁 Repository Architecture

```text
name-that-llm-kaggle/
│
├── src/                        # Core Python modules
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset for DeBERTa tokenization + stylometric fusion
│   ├── features.py             # 236 stylometric, typographic & syntactic skeleton features
│   ├── tabular_model.py        # 3-Stage Hierarchical Multi-View & Sibling Specialist Ensemble
│   ├── model.py                # HybridLLMClassifier (DeBERTa-v3 + Stylometric MLP Fusion)
│   └── utils.py                # Seed setting for reproducibility
│
├── train.py                    # Master 5-fold training, inference & submission validation script
├── kaggle_submission.ipynb     # Self-contained Jupyter Notebook for EDA & Kaggle submission
├── explore_fingerprints.py     # Statistical fingerprint analysis & multi-view CV benchmark
├── test_submission.py          # Automated local validation script for submission.parquet
├── submit.py                   # Automated Kaggle API upload script
├── train.csv                   # Training dataset (2,699 passages across 18 models and 150 prompts)
├── test.csv                    # Test dataset
├── sample_submission.parquet   # Sample submission file
├── requirements.txt            # Project Python dependencies
└── README.md                   # Project documentation
```

---

## 🔬 Key Model Fingerprints Discovered

Using **5-Fold `StratifiedGroupKFold` grouped by `prompt_id`** (preventing prompt-topic leakage across folds), we analyzed 236 engineered features and topic-demeaned n-grams across all 18 models:

1. **Hard Typographic Separators**:
   * **Anthropic (`claude-*`) & Zhipu (`glm-*`)**: `0.00%` curly typography across all 1,049 passages—they **exclusively emit straight ASCII apostrophes/quotes (`'`, `"`)** and heavily use em-dashes (`0.87–1.55` per 100 words).
   * **Google (`gemini-*`)**: **Almost never use em-dashes** (`0.00–0.01` per 100 words), have the lowest colon/semicolon densities (`0.16–0.28`), and have the highest `-ly` adverb rates (`2.61–2.81` per 100 words).
   * **OpenAI (`gpt-5.6-sol` & `gpt-5.6-luna`)**: Almost exclusively emit **curly Unicode apostrophes/quotes (`’`, `“`, `”`)** (`0.00–0.01` straight typography).
2. **Structural & Cadence Signatures**:
   * `claude-opus-4-6` & `claude-opus-5` uniquely prefer **spaced dashes** (`+1.32σ` and `+1.04σ`), while `claude-opus-5` uses the shortest average word length (`4.97` chars).
   * `kimi-k3` has the highest colon (`0.99`) and semicolon (`0.83`) densities and lowest `-ly` adverb density (`0.96`).
   * `qwen3.8-max` combines high comma density (`+0.77σ`) and participial `, ...ing` clauses (`2.01`) with near-zero em-dashes (`0.05`).

---

## 📊 5-Fold Cross-Validation Benchmark (Unseen Prompts)

| Pipeline Stage | 5-Fold OOF Accuracy | 5-Fold Macro F1 | Gain vs. Naive |
| :--- | :---: | :---: | :---: |
| Naive Baseline (Random / Majority) | `5.56%` | `5.56%` | `1.0x` |
| Initial 12 Stylometric Features (LogReg) | `31.49%` | `30.80%` | `5.7x` |
| Expanded 236 Stylometric Features Only (LogReg) | `35.01%` | `34.72%` | `6.3x` |
| Stage 1: Global Multi-View Joint Ensemble | `38.87%` | `38.07%` | `7.0x` |
| **Stage 2: + Hierarchical Sibling Cluster Specialists** | **`39.27%`** | **`38.49%`** | **`7.1x`** |

---

## 🚀 Quickstart

```bash
# 1. Run 5-fold Hierarchical Multi-View Ensemble training, test inference, and validation
python train.py

# 2. (Optional) Include 5-fold Hybrid DeBERTa-v3 + Stylometric Fusion on GPU
python train.py --use-deberta

# 3. Validate submission.parquet standalone
python test_submission.py

# 4. Submit to Kaggle
python submit.py
```
