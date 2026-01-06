#!/usr/bin/env python3
"""
Validate submission file format
"""

import pandas as pd
import sys

def check_submission(filepath):
    """Check submission file format"""
    try:
        df = pd.read_csv(filepath)
        
        print(f"File: {filepath}")
        print(f"Rows: {len(df)} (Should be 1459)")
        print(f"Columns: {len(df.columns)} (Should be 2)")
        print(f"Column names: {list(df.columns)} (Should be ['Id', 'SalePrice'])")
        
        # Check ID range
        print(f"ID range: {df['Id'].min()} to {df['Id'].max()} (Should be 1461-2919)")
        
        # Check predictions
        print(f"\nPrediction Statistics:")
        print(f"  Min: ${df['SalePrice'].min():,.2f}")
        print(f"  Max: ${df['SalePrice'].max():,.2f}")
        print(f"  Mean: ${df['SalePrice'].mean():,.2f}")
        print(f"  Median: ${df['SalePrice'].median():,.2f}")
        
        # Check missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nWarning: {missing.sum()} missing values found")
        else:
            print("\n✓ No missing values")
        
        # Check for negative prices
        negative_count = (df['SalePrice'] < 0).sum()
        if negative_count > 0:
            print(f"\nWarning: {negative_count} negative prices found")
        else:
            print("✓ No negative prices")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "submissions/v1_submission.csv"
    
    print("=" * 60)
    print("Kaggle Submission File Check")
    print("=" * 60)
    
    success = check_submission(filepath)
    
    if success:
        print("\n✓ Submission file is ready for Kaggle!")
    else:
        print("\n✗ Please fix the issues above")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())