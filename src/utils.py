"""
Utility functions module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

def create_submission_file(test_ids, predictions, filename="submission.csv"):
    """Create Kaggle submission file"""
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission_df.to_csv(filename, index=False)
    print(f"Submission file created: {filename}")
    print(f"File contains {len(submission_df)} predictions")
    
    return submission_df

def save_model(model, model_name, save_dir="models"):
    """Save trained model"""
    import joblib
    
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/{model_name}_{timestamp}.pkl"
    
    joblib.dump(model, filename)
    print(f"Model saved: {filename}")
    
    return filename

def load_model(filename):
    """Load saved model"""
    import joblib
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found: {filename}")
    
    model = joblib.load(filename)
    print(f"Model loaded: {filename}")
    
    return model

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance"""
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Top {top_n} Feature Importance")
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved: {save_path}")
    
    plt.show()
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    
    return importance_df.head(top_n)

def log_results(results, log_file="results/log.json"):
    """Log experiment results"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Read existing logs
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    # Add new results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results['timestamp'] = timestamp
    all_results.append(results)
    
    # Save logs
    with open(log_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results logged to: {log_file}")