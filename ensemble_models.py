#!/usr/bin/env python3
"""
Ensemble model to improve score further
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ENSEMBLE MODEL FOR BETTER SCORE")
print("=" * 60)

# 讀取數據
train = pd.read_csv('data/train.csv', encoding='latin1')
test = pd.read_csv('data/test.csv', encoding='latin1')

print(f"Train: {train.shape}, Test: {test.shape}")

# 選擇特徵（基於重要性）
features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
    'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
    'GarageArea', '1stFlrSF', 'TotalSF'  # 我們創建的特徵
]

# 只使用存在的特徵
features = [f for f in features if f in train.columns]
print(f"\nUsing {len(features)} features")

# 準備數據
X_train = train[features].copy()
y_train = train['SalePrice']
X_test = test[features].copy()
test_ids = test['Id']

# 填充缺失值
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# 對數轉換
y_train_log = np.log1p(y_train)

# 標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining multiple models...")

# 定義多個模型
models = {
    'Ridge': Ridge(alpha=10.0, random_state=42),
    'Lasso': Lasso(alpha=0.001, random_state=42, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
}

# 交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions_dict = {}
cv_scores = {}

print("\nCross-validation results:")
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train_log, 
                           cv=kf, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    cv_scores[name] = rmse
    print(f"  {name:15s}: RMSE = {rmse:.5f}")
    
    # 訓練並預測
    model.fit(X_train_scaled, y_train_log)
    pred_log = model.predict(X_test_scaled)
    predictions_dict[name] = np.expm1(pred_log)

# 找到最佳模型
best_model_name = min(cv_scores, key=cv_scores.get)
print(f"\nBest single model: {best_model_name} (RMSE: {cv_scores[best_model_name]:.5f})")

# 集成方法1：簡單平均
print("\nCreating ensemble predictions...")
all_predictions = np.array(list(predictions_dict.values()))
simple_average = np.mean(all_predictions, axis=0)

# 集成方法2：加權平均（基於CV分數）
# 轉換RMSE為權重（越低越好）
weights = 1 / np.array(list(cv_scores.values()))
weights = weights / weights.sum()

weighted_average = np.zeros_like(simple_average)
for i, (name, pred) in enumerate(predictions_dict.items()):
    weighted_average += pred * weights[i]

# 集成方法3：中位數
median_ensemble = np.median(all_predictions, axis=0)

print(f"\nEnsemble stats:")
print(f"  Simple average: min=${simple_average.min():,.0f}, max=${simple_average.max():,.0f}")
print(f"  Weighted average: min=${weighted_average.min():,.0f}, max=${weighted_average.max():,.0f}")
print(f"  Median: min=${median_ensemble.min():,.0f}, max=${median_ensemble.max():,.0f}")

# 保存所有版本
import os
os.makedirs('submissions', exist_ok=True)

versions = {
    'ensemble_simple': simple_average,
    'ensemble_weighted': weighted_average,
    'ensemble_median': median_ensemble,
    best_model_name + '_best': predictions_dict[best_model_name]
}

for version_name, predictions in versions.items():
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    filename = f'submissions/{version_name}.csv'
    submission.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")

print("\n" + "=" * 60)
print("ENSEMBLE MODELS READY!")
print("=" * 60)
print("\nUpload these files to Kaggle:")
print("1. ensemble_simple.csv - Simple average of all models")
print("2. ensemble_weighted.csv - Weighted average based on CV scores")
print("3. ensemble_median.csv - Median of all predictions")
print(f"4. {best_model_name}_best.csv - Best single model")
print("\nExpected: One of these should beat 0.15282!")
print("=" * 60)