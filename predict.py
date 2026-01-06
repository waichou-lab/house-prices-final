#!/usr/bin/env python3
"""
Kaggle House Price Prediction - Generate Submission File
Usage: python predict.py [model_version]
"""

import sys
import os
import numpy as np
import pandas as pd
import yaml
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from data_loader import DataLoader
from utils import load_model, create_submission_file

def predict_and_submit(model_path=None, submission_version=1):
    """
    Load model and generate submission file
    """
    print("=" * 60)
    print("Generating Kaggle Submission File")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Load model
    print("\n[1/3] Loading model...")
    
    if model_path is None:
        # Find latest model
        model_dir = Path("models")
        if not model_dir.exists():
            print("Error: No model directory found. Please run train_model.py first.")
            return
        
        model_files = list(model_dir.glob("*.pkl"))
        if not model_files:
            print("Error: No model files found.")
            return
        
        # Select most recent model
        model_path = max(model_files, key=os.path.getctime)
    
    model = load_model(str(model_path))
    print(f"Using model: {model_path.name}")
    
    # 2. Load processed test data
    print("\n[2/3] Loading test data...")
    
    test_processed_path = "data/X_test_processed.csv"
    test_ids_path = "data/test_ids.npy"
    
    if not os.path.exists(test_processed_path):
        print("Error: Processed test data not found. Please run train_model.py first.")
        return
    
    X_test = pd.read_csv(test_processed_path)
    
    # Load test IDs
    if os.path.exists(test_ids_path):
        test_ids = np.load(test_ids_path, allow_pickle=True)
    else:
        # Get IDs from original test data
        data_loader = DataLoader()
        _, test_df = data_loader.load_data()
        test_ids = test_df['Id']
        np.save(test_ids_path, test_ids)
    
    print(f"Test data shape: {X_test.shape}")
    
    # 3. Generate predictions
    print("\n[3/3] Generating predictions...")
    predictions_log = model.predict(X_test)
    
    # Convert log predictions back to original scale
    predictions = np.expm1(predictions_log)
    
    # Create submission file
    submission_filename = f"submissions/v{submission_version}_submission.csv"
    submission_df = create_submission_file(test_ids, predictions, submission_filename)
    
    # Show statistics
    print("\nPrediction Statistics:")
    print(f"  Min: ${predictions.min():,.2f}")
    print(f"  Max: ${predictions.max():,.2f}")
    print(f"  Mean: ${predictions.mean():,.2f}")
    print(f"  Median: ${np.median(predictions):,.2f}")
    
    print("\n" + "=" * 60)
    print("Submission file ready!")
    print(f"File path: {submission_filename}")
    print("Upload to Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submit")
    print("=" * 60)
    
    return submission_df

def main():
    parser = argparse.ArgumentParser(description='Generate Kaggle submission file')
    parser.add_argument('--model', type=str, help='Model file path')
    parser.add_argument('--version', type=int, default=1, help='Submission version number')
    
    args = parser.parse_args()
    
    predict_and_submit(args.model, args.version)

if __name__ == "__main__":
    main()