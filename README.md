# Customer Churn Prediction using Machine Learning

A modular, production-style Python project that predicts customer churn from the Telco Customer Churn dataset using multiple ML models with full evaluation, tuning, and interpretation.

---

## Project Structure

```
customer-churn-ml/
│
├── data/
│   ├── raw/                  ← Place the dataset CSV here
│   └── processed/            ← Auto-generated preprocessed files
│
├── notebooks/
│   └── exploration.ipynb     ← Optional interactive exploration
│
├── src/
│   ├── config.py             ← Paths, constants, model settings
│   ├── data_loader.py        ← Load raw data and inspect
│   ├── preprocessing.py      ← Clean, fix types, handle missing values
│   ├── eda.py                ← Exploratory Data Analysis plots
│   ├── feature_engineering.py← Encoding, scaling, SMOTE, train-test split
│   ├── train_models.py       ← Train all ML models
│   ├── evaluate.py           ← Metrics, confusion matrix, ROC-AUC
│   ├── tune_models.py        ← Hyperparameter tuning
│   ├── explain.py            ← Feature importance and SHAP
│   └── utils.py              ← Shared helper functions
│
├── outputs/
│   ├── figures/              ← Saved plots
│   ├── models/               ← Saved model files (.joblib)
│   ├── reports/              ← Classification reports
│   └── metrics/              ← CSV metrics summaries
│
├── main.py                   ← Run the full pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

Download the **Telco Customer Churn** dataset from Kaggle:

> https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the file at:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Setup

### 1. Clone or extract the project

```bash
cd customer-churn-ml
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Full pipeline (recommended)

```bash
python main.py
```

This runs every stage in order:
1. Load & inspect data
2. Preprocess (clean, fix types, encode target)
3. Exploratory Data Analysis → saves plots to `outputs/figures/`
4. Feature engineering (encode, scale, SMOTE)
5. Train all models
6. Evaluate all models → saves metrics to `outputs/metrics/`
7. Hyperparameter tuning on best models
8. Feature importance & SHAP analysis

### Run individual stages

Each `src/` file can also be run standalone:

```bash
python -m src.eda
python -m src.train_models
python -m src.evaluate
```

### Jupyter exploration

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Models Trained

| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Decision Tree | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting (XGBoost) | xgboost |
| Gradient Boosting (LightGBM) | lightgbm |

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Full Classification Report

---

## Outputs

| Location | Contents |
|---|---|
| `outputs/figures/` | EDA plots, confusion matrices, ROC curves, SHAP plots |
| `outputs/models/` | Saved `.joblib` model files |
| `outputs/metrics/` | `metrics_summary.csv` comparing all models |
| `outputs/reports/` | Per-model classification report `.txt` files |

---

## Class Imbalance

The pipeline uses **SMOTE** (Synthetic Minority Oversampling Technique) from `imbalanced-learn` to handle the imbalanced churn distribution. This can be toggled via `config.py`.

---

## Requirements

- Python 3.9+
- See `requirements.txt` for all packages
