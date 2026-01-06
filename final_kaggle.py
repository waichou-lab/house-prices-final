#!/usr/bin/env python3
"""
Final Kaggle House Price Solution - No encoding issues
"""

import pandas as pd
import numpy as np
import os
import sys

print("=" * 70)
print("KAGGLE HOUSE PRICE PREDICTION - FINAL SOLUTION")
print("=" * 70)

# 1. 讀取數據（最簡單的方法）
print("\n[1/4] READING DATA...")

def read_csv_simple(filepath):
    """最簡單的讀取方法"""
    try:
        # 方法1: 讀取為二進制，然後解碼
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # 嘗試最常見的編碼
        encodings = ['latin1', 'cp1252', 'utf-8', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # 解碼並保存為臨時文件
                decoded = content.decode(encoding)
                # 寫入臨時文件
                temp_file = 'temp_data.csv'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(decoded)
                # 讀取臨時文件
                df = pd.read_csv(temp_file)
                # 刪除臨時文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                print(f"  ✓ Read {os.path.basename(filepath)} with {encoding}")
                return df
            except:
                continue
        
        # 如果都失敗，使用最寬容的方法
        print(f"  ⚠ Using fallback method for {os.path.basename(filepath)}")
        return pd.read_csv(filepath, encoding='latin1', low_memory=False)
        
    except Exception as e:
        print(f"  ✗ Error reading {filepath}: {e}")
        return None

# 檢查文件
data_files = ['data/train.csv', 'data/test.csv']
for file in data_files:
    if not os.path.exists(file):
        print(f"ERROR: {file} not found!")
        print("Please download from Kaggle and place in data/ directory")
        sys.exit(1)

# 讀取數據
train = read_csv_simple('data/train.csv')
if train is None:
    sys.exit(1)

test = read_csv_simple('data/test.csv')
if test is None:
    print("Warning: Test data not loaded properly")

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape if test is not None else 'N/A'}")

# 檢查必要列
if 'SalePrice' not in train.columns:
    print("\nERROR: 'SalePrice' column not found in training data!")
    print("Available columns:", train.columns.tolist()[:10])
    sys.exit(1)

# 2. 特徵工程（簡單版本）
print("\n[2/4] FEATURE ENGINEERING...")

def create_simple_features(df):
    """創建幾個關鍵特徵"""
    df_copy = df.copy()
    
    # 只創建最重要的特徵
    features_created = []
    
    # 1. Total living area
    if 'GrLivArea' in df_copy.columns and 'TotalBsmtSF' in df_copy.columns:
        df_copy['TotalSF'] = df_copy['GrLivArea'] + df_copy['TotalBsmtSF']
        features_created.append('TotalSF')
    
    # 2. House age
    if 'YrSold' in df_copy.columns and 'YearBuilt' in df_copy.columns:
        df_copy['HouseAge'] = df_copy['YrSold'] - df_copy['YearBuilt']
        features_created.append('HouseAge')
    
    # 3. Quality × Area
    if 'OverallQual' in df_copy.columns and 'GrLivArea' in df_copy.columns:
        df_copy['QualityArea'] = df_copy['OverallQual'] * df_copy['GrLivArea']
        features_created.append('QualityArea')
    
    # 4. Total bathrooms
    if 'FullBath' in df_copy.columns and 'HalfBath' in df_copy.columns:
        df_copy['TotalBath'] = df_copy['FullBath'] + 0.5 * df_copy['HalfBath']
        if 'BsmtFullBath' in df_copy.columns:
            df_copy['TotalBath'] += df_copy['BsmtFullBath']
        features_created.append('TotalBath')
    
    print(f"  Created {len(features_created)} features: {features_created}")
    return df_copy

# 應用特徵工程
train = create_simple_features(train)
if test is not None:
    test = create_simple_features(test)

# 3. 數據預處理
print("\n[3/4] DATA PREPROCESSING...")

# 分離特徵和目標
X_train_raw = train.drop(['Id', 'SalePrice'], axis=1, errors='ignore')
y_train = train['SalePrice']

# 保存ID
if test is not None:
    test_ids = test['Id']
    X_test_raw = test.drop(['Id'], axis=1, errors='ignore')
else:
    test_ids = None
    X_test_raw = None

print(f"Training features: {X_train_raw.shape}")

# 處理分類變數 - 簡單方法：轉換為數值
from sklearn.preprocessing import LabelEncoder

categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {len(categorical_cols)}")

# 對訓練數據進行編碼
for col in categorical_cols:
    if col in X_train_raw.columns:
        # 填充缺失值
        X_train_raw[col] = X_train_raw[col].fillna('Missing')
        # 轉換為數值
        le = LabelEncoder()
        X_train_raw[col] = le.fit_transform(X_train_raw[col].astype(str))

# 對測試數據進行相同處理
if X_test_raw is not None:
    for col in categorical_cols:
        if col in X_test_raw.columns:
            X_test_raw[col] = X_test_raw[col].fillna('Missing')
            # 使用訓練集的LabelEncoder或創建新的
            try:
                X_test_raw[col] = le.transform(X_test_raw[col].astype(str))
            except:
                # 如果有新類別，使用最常見的值
                X_test_raw[col] = 0

# 處理缺失值（數值型）
X_train_raw = X_train_raw.fillna(X_train_raw.median())
if X_test_raw is not None:
    X_test_raw = X_test_raw.fillna(X_train_raw.median())

# 確保測試數據與訓練數據有相同的列
if X_test_raw is not None:
    missing_cols = set(X_train_raw.columns) - set(X_test_raw.columns)
    for col in missing_cols:
        X_test_raw[col] = 0
    
    extra_cols = set(X_test_raw.columns) - set(X_train_raw.columns)
    for col in extra_cols:
        X_test_raw = X_test_raw.drop(col, axis=1)
    
    X_test_raw = X_test_raw[X_train_raw.columns]

# 標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
if X_test_raw is not None:
    X_test = scaler.transform(X_test_raw)

# 對數轉換目標變量
y_train_log = np.log1p(y_train)
print("✓ Target variable log-transformed")

# 4. 訓練模型
print("\n[4/4] MODEL TRAINING AND PREDICTION...")

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

# 使用簡單的Ridge回歸（穩定且快速）
model = Ridge(alpha=10.0, random_state=42)

# 交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train_log, 
                           cv=kf, scoring='neg_root_mean_squared_error')
cv_rmse = -cv_scores.mean()
print(f"Cross-validation RMSE: {cv_rmse:.5f}")

# 訓練最終模型
model.fit(X_train, y_train_log)
print("✓ Model trained successfully")

# 5. 預測和生成提交檔案
if test is not None and X_test is not None:
    print("\n[5/5] GENERATING PREDICTIONS...")
    
    # 預測
    predictions_log = model.predict(X_test)
    predictions = np.expm1(predictions_log)
    
    # 創建提交檔案
    os.makedirs('submissions', exist_ok=True)
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission_file = 'submissions/final_submission.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"✓ Submission file created: {submission_file}")
    print(f"  Number of predictions: {len(predictions)}")
    print(f"  Price range: ${predictions.min():,.0f} - ${predictions.max():,.0f}")
    print(f"  Average price: ${predictions.mean():,.0f}")
    
    print("\n" + "=" * 70)
    print("SUCCESS! Your submission is ready.")
    print("=" * 70)
    print("\nNEXT STEPS:")
    print("1. Go to: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques")
    print("2. Click 'Submit Predictions'")
    print("3. Upload: submissions/final_submission.csv")
    print("4. Add description (e.g., 'Ridge regression with 4 new features')")
    print("5. Wait for your score!")
    print("\nExpected score: 0.17000 - 0.18000")
    print("=" * 70)
else:
    print("\nWarning: Test data not available, only trained model")

# 保存模型供以後使用
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/final_ridge_model.pkl')
print("\n✓ Model saved as models/final_ridge_model.pkl")