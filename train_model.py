import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import os

def train():
    # 1. Load data
    data_path = 'heart.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # 2. Preprocessing
    # Handle missing values
    if df.isnull().values.any():
        print("Missing values found. Dropping rows with missing values.")
        df.dropna(inplace=True)
    else:
        print("No missing values found.")

    # Target: ensure it's binary (0 = No disease, 1 = Disease)
    if 'target' in df.columns:
        target_col = 'target'
    elif 'output' in df.columns:
        target_col = 'output'
    else:
        target_col = df.columns[-1]
        print(f"Warning: 'target' column not found. Using '{target_col}' as target.")

    df[target_col] = df[target_col].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split dataset (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model 1: Logistic Regression (with Regularization C=0.1 to prevent overtraining)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # C=0.1 provides stronger regularization than default C=1.0 to prevent overfitting
    lr_model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    # 4. Model 2: Decision Tree (Gini impurity)
    dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42) # Limited depth to prevent overtraining
    dt_model.fit(X_train, y_train)

    def evaluate(model, X_t, y_t, model_name, is_scaled=False, X_full=None):
        preds = model.predict(X_t)
        acc = accuracy_score(y_t, preds)
        cm = confusion_matrix(y_t, preds)
        prec = precision_score(y_t, preds)
        rec = recall_score(y_t, preds)
        f1 = f1_score(y_t, preds)
        
        # 5-Fold CV on full data
        if is_scaled:
            X_full_proc = scaler.transform(X_full)
        else:
            X_full_proc = X_full
            
        cv_scores = cross_val_score(model, X_full_proc, y, cv=5)
            
        print(f"\n--- {model_name} Evaluation ---")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"5-Fold CV Mean Accuracy: {cv_scores.mean():.4f}")
        
        return {
            'accuracy': acc,
            'f1': f1,
            'model': model,
            'name': model_name,
            'cv_mean': cv_scores.mean()
        }

    lr_results = evaluate(lr_model, X_test_scaled, y_test, "Logistic Regression", is_scaled=True, X_full=X)
    dt_results = evaluate(dt_model, X_test, y_test, "Decision Tree", is_scaled=False, X_full=X)

    # 5. Model Selection
    if lr_results['f1'] >= dt_results['f1']:
        best_model = lr_results['model']
        best_name = lr_results['name']
        best_accuracy = lr_results['accuracy']
        use_scaler = True
    else:
        best_model = dt_results['model']
        best_name = dt_results['name']
        best_accuracy = dt_results['accuracy']
        use_scaler = False

    print(f"\nBetter Model Selected: {best_name}")

    # 6. Save model and assets
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump({
        'model_name': best_name, 
        'use_scaler': use_scaler,
        'feature_names': X.columns.tolist(),
        'target_col': target_col,
        'test_accuracy': best_accuracy
    }, 'model_info.pkl')
    print("Model and metadata saved successfully.")

if __name__ == "__main__":
    train()
