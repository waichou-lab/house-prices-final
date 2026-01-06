#!/usr/bin/env python3
"""
Proper model with correct preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os

print("=" * 60)
print("PROPER MODEL WITH LOG TRANSFORM")
print("=" * 60)

# 1. 讀取數據
print("\n1. Loading data...")
train = pd.read_csv('data/train.csv', encoding='latin1')
test = pd.read_csv('data/test.csv', encoding='latin1')

print(f"Train: {train.shape}, Test: {test.shape}")

# 2. 處理缺失值 - 正確的方法
print("\n2. Handling missing values...")

# 檢查訓練數據的SalePrice
if 'SalePrice' not in train.columns:
    print("ERROR: No SalePrice column!")
    exit(1)

print(f"SalePrice stats before:")
print(f"  Min: ${train['SalePrice'].min():,.0f}")
print(f"  Max: ${train['SalePrice'].max():,.0f}")
print(f"  Mean: ${train['SalePrice'].mean():,.0f}")

# 3. 選擇最重要的特徵（基於相關性）
print("\n3. Selecting features...")

# 數值型特徵
numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
if 'Id' in numeric_features:
    numeric_features.remove('Id')
if 'SalePrice' in numeric_features:
    numeric_features.remove('SalePrice')

# 計算與SalePrice的相關性
correlations = train[numeric_features + ['SalePrice']].corr()['SalePrice'].abs()
top_features = correlations.sort_values(ascending=False).head(15).index.tolist()
top_features.remove('SalePrice')  # 移除目標變量

print(f"Top {len(top_features)} features by correlation:")
for i, feat in enumerate(top_features[:10], 1):
    print(f"  {i:2d}. {feat:20s} r = {correlations[feat]:.3f}")

# 4. 準備數據
print("\n4. Preparing data...")

X_train = train[top_features].copy()
y_train = train['SalePrice']
X_test = test[top_features].copy()
test_ids = test['Id']

# 填充缺失值 - 使用訓練數據的統計量
for col in top_features:
    if col in X_train.columns:
        # 計算訓練數據的統計量
        if X_train[col].dtype in [np.float64, np.int64]:
            fill_value = X_train[col].median()
        else:
            fill_value = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 0
        
        X_train[col] = X_train[col].fillna(fill_value)
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(fill_value)

print(f"Missing values after fill: {X_train.isnull().sum().sum()}")

# 5. 對數轉換（關鍵步驟！）
print("\n5. Applying log transformation...")
y_train_log = np.log1p(y_train)

# 也對數轉換偏態的特徵
skewed_features = X_train.select_dtypes(include=[np.number]).columns
for col in skewed_features:
    if X_train[col].skew() > 0.75:  # 如果偏態 > 0.75
        X_train[col] = np.log1p(X_train[col] + 1)  # +1 避免 log(0)
        if col in X_test.columns:
            X_test[col] = np.log1p(X_test[col] + 1)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 訓練模型
print("\n6. Training model...")

# 使用Ridge回歸（正則化）
model = Ridge(alpha=10.0, random_state=42)
model.fit(X_train_scaled, y_train_log)

# 7. 預測
print("\n7. Making predictions...")
predictions_log = model.predict(X_test_scaled)

# 轉換回原始尺度（重要！）
predictions = np.expm1(predictions_log)

# 確保合理的範圍
train_min = y_train.min()
train_max = y_train.max()
predictions = np.clip(predictions, train_min * 0.5, train_max * 1.5)

print(f"Prediction stats:")
print(f"  Min: ${predictions.min():,.0f}")
print(f"  Max: ${predictions.max():,.0f}")
print(f"  Mean: ${predictions.mean():,.0f}")

# 8. 創建提交檔案
print("\n8. Creating submission file...")

os.makedirs('submissions', exist_ok=True)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})

output_file = 'submissions/proper_model_v1.csv'
submission.to_csv(output_file, index=False)

print(f"✓ Submission saved: {output_file}")

# 9. 驗證
print("\n9. Validating submission...")
print(f"  Shape: {submission.shape} (should be 1459x2)")
print(f"  Columns: {list(submission.columns)}")
print(f"  Missing values: {submission.isnull().sum().sum()}")
print(f"  Negative values: {(submission['SalePrice'] < 0).sum()}")
print(f"  Zero values: {(submission['SalePrice'] == 0).sum()}")

print("\n" + "=" * 60)
print("READY FOR KAGGLE SUBMISSION!")
print("=" * 60)
print(f"\nFile to upload: {output_file}")
print("\nKey improvements:")
print("  1. Log transform of target variable")
print("  2. Ridge regression with regularization")
print("  3. Proper handling of skewed features")
print("  4. Reasonable price clipping")
print("\nExpected score: 0.15000 - 0.17000")
print("=" * 60)