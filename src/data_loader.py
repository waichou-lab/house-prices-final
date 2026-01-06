"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
import yaml
import os
import chardet
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.train_path = self.config['data']['train_path']
        self.test_path = self.config['data']['test_path']
    
    def detect_file_encoding(self, file_path, sample_size=10000):
        """Detect file encoding"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)
            return result['encoding'], result['confidence']
    
    def read_csv_safe(self, file_path):
        """Safely read CSV file with multiple encoding attempts"""
        # List of encodings to try
        encodings_to_try = [
            'utf-8',
            'utf-8-sig',  # UTF-8 with BOM
            'latin1',
            'iso-8859-1',
            'cp1252',
            'cp1250',
            'cp950',
            'big5',
            'gbk',
            'gb2312',
            'ascii'
        ]
        
        # First try to detect encoding
        try:
            detected_encoding, confidence = self.detect_file_encoding(file_path)
            if detected_encoding and confidence > 0.7:
                print(f"Detected encoding: {detected_encoding} (confidence: {confidence:.2%})")
                encodings_to_try.insert(0, detected_encoding)
        except:
            pass
        
        # Try each encoding
        for encoding in encodings_to_try:
            try:
                print(f"Trying encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✓ Successfully read with {encoding}")
                return df
            except UnicodeDecodeError as e:
                continue
            except Exception as e:
                print(f"Error with {encoding}: {e}")
                continue
        
        # If all fail, try with error handling
        try:
            print("Trying with errors='replace'")
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
            return df
        except:
            raise ValueError(f"Could not read file {file_path} with any encoding")
    
    def load_data(self):
        """Load training and test data"""
        print(f"Loading training data: {self.train_path}")
        train_df = self.read_csv_safe(self.train_path)
        
        print(f"\nLoading test data: {self.test_path}")
        test_df = self.read_csv_safe(self.test_path)
        
        print(f"\nTraining shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        # Check for target column
        if 'SalePrice' not in train_df.columns:
            print("Warning: 'SalePrice' column not found in training data")
        
        return train_df, test_df
    
    def basic_preprocessing(self, df, is_train=True):
        """Basic data preprocessing"""
        df_copy = df.copy()
        
        # Save ID
        if 'Id' in df_copy.columns:
            ids = df_copy['Id']
            df_copy = df_copy.drop('Id', axis=1)
        else:
            ids = None
        
        # Handle missing values
        df_copy = self.handle_missing_values(df_copy)
        
        return df_copy, ids
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        df_copy = df.copy()
        
        # Numerical features: fill with median
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                median_val = df_copy[col].median()
                df_copy[col] = df_copy[col].fillna(median_val)
                if pd.isna(median_val):  # If all NaN, fill with 0
                    df_copy[col] = df_copy[col].fillna(0)
        
        # Categorical features: fill with mode or 'Missing'
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_copy[col].isnull().any():
                mode_val = df_copy[col].mode()
                if len(mode_val) > 0:
                    df_copy[col] = df_copy[col].fillna(mode_val[0])
                else:
                    df_copy[col] = df_copy[col].fillna('Missing')
        
        return df_copy
    
    def split_features_target(self, train_df):
        """Split features and target variable"""
        if 'SalePrice' not in train_df.columns:
            raise ValueError("No 'SalePrice' column in training data")
        
        X = train_df.drop('SalePrice', axis=1)
        y = train_df['SalePrice']
        
        return X, y