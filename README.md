# Credit Risk Assessment & Explainable AI (XAI)

This repository demonstrates a production-level Data Science pipeline for assessing credit risk. Inspired by large-scale financial modeling workflows, it utilizes **XGBoost** for classification and **SHAP (SHapley Additive exPlanations)** to provide local and global interpretability.

## ðŸ“Š Key Features
- **Data Engineering:** Automated handling of missing values, categorical encoding, and feature scaling.
- **Model Training:** Optimized Gradient Boosted Trees with Hyperparameter tuning via Optuna.
- **Interpretability:** Detailed SHAP plots to explain why a specific credit application was approved or denied.
- **Metrics:** Precision-Recall curves and ROC-AUC optimized for imbalanced datasets.

## ðŸ›  Tech Stack
- **Python** (Pandas, Scikit-Learn, XGBoost)
- **Interpretability:** SHAP
- **Visualization:** Matplotlib, Seaborn

## ðŸš€ Usage
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline: `python model.py`
