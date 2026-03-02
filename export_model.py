import joblib
import json
import numpy as np

def export():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        info = joblib.load('model_info.pkl')
        
        # Extract Logistic Regression parameters
        # Logistic Regression formula: y = sigmoid(W*X + b)
        params = {
            "model_name": info['model_name'],
            "feature_names": info['feature_names'],
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist()
            },
            "test_accuracy": info['test_accuracy']
        }
        
        with open('model_params.json', 'wb') as f:
            f.write(json.dumps(params, indent=4).encode('utf-8'))
            
        print("Model parameters exported to model_params.json successfully.")
        
    except Exception as e:
        print(f"Error during export: {e}")

if __name__ == "__main__":
    export()
