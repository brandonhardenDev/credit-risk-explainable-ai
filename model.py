import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap

# 1. Generate Synthetic Credit Data
def load_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'income': np.random.normal(50000, 15000, n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'loan_amount': np.random.normal(10000, 5000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

# 2. Train Model
def train_pipeline():
    df = load_data()
    X = df.drop('default', axis=1)
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, preds))
    
    # 3. Explainability with SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    print("\nSHAP values calculated successfully.")
    return model, shap_values

if __name__ == "__main__":
    train_pipeline()
