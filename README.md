# name-that-llm-kaggle
The core task of this project is authorship attribution (specifically, model attribution).
# 🤖 Name That LLM - Kaggle Competition Pipeline

Welcome to the repository for the **"Name That LLM"** Kaggle competition. The goal of this challenge is to build a robust multi-class authorship attribution pipeline that identifies which specific frontier language model (out of 18 models across 6 major AI labs) generated a given text passage.

---

## 📁 Repository Architecture

```text
name-that-llm-kaggle/
│
├── data/                  # Dataset directory (train.csv, test.csv)
├── src/                   # Core Python modules
│   ├── __init__.py
│   ├── dataset.py         # PyTorch Dataset for tokenization
│   ├── features.py        # Stylistic & punctuation fingerprint extraction
│   ├── model.py           # Training loops, loss criteria, and validation
│   └── utils.py           # Seed setting for reproducibility
│
├── train.py               # Master training & cross-validation orchestration
├── test_submission.py     # Automated local validation script for submission.parquet
├── submit.py              # Automated Kaggle API upload script
├── requirements.txt       # Project Python dependencies
└── README.md              # Project documentation
