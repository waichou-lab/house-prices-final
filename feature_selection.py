#!/usr/bin/env python3
"""
Ensemble model to improve score further - FIXED VERSION
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ENSEMBLE MODEL FOR BETTER SCORE - FIXED")
print("=" * 60)

# 讀取數據
train = pd.read_csv('data/train.csv', encoding='latin1')
test = pd.read_csv('data/test.csv', encoding='latin1')

print(f"Train: {train.shape}, Test: {test.shape}")

# 創建重要特徵
def create_features(df):
    df_copy = df.copy()
    
    # 總面積
    if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
        df_copy['TotalSF'] = df_copy['GrLivArea'] + df_copy['TotalBsmtSF']
    
    # 房屋年齡
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df_copy['HouseAge'] = df_copy['YrSold'] - df_copy['YearBuilt']
    
    # 品質×面積
    if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
        df_copy['QualityArea'] = df_copy['OverallQual'] * df_copy['GrLivArea']
    
    # 總浴室數
    if 'FullBath' in df.columns and 'HalfBath' in df.columns:
        df_copy['TotalBath'] = df_copy['FullBath'] + 0.5 * df_copy['HalfBath']
        if 'BsmtFullBath' in df.columns:
            df_copy['TotalBath'] += df_copy['BsmtFullBath']
    
    return df_copy

# 應用特徵工程
train = create_features(train)
test = create_features(test)

# 選擇最重要的特徵
features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
    'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
    'GarageArea', '1stFlrSF', 'TotalSF', 'HouseAge', 
    'QualityArea', 'TotalBath', 'MasVnrArea', 'Fireplaces'
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

# 對數轉換目標變量（關鍵！）
y_train_log = np.log1p(y_train)

# 標準化特徵
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
    try:
        scores = cross_val_score(model, X_train_scaled, y_train_log, 
                               cv=kf, scoring='neg_root_mean_squared_error')
        rmse = -scores.mean()
        cv_scores[name] = rmse
        print(f"  {name:15s}: RMSE = {rmse:.5f}")
        
        # 訓練並預測
        model.fit(X_train_scaled, y_train_log)
        pred_log = model.predict(X_test_scaled)
        predictions_dict[name] = np.expm1(pred_log)
    except Exception as e:
        print(f"  {name:15s}: Error - {e}")

if not predictions_dict:
    print("No models trained successfully!")
    exit(1)

# 找到最佳模型
best_model_name = min(cv_scores, key=cv_scores.get)
print(f"\nBest single model: {best_model_name} (RMSE: {cv_scores[best_model_name]:.5f})")

# 集成方法1：簡單平均
print("\nCreating ensemble predictions...")
all_predictions = np.array(list(predictions_dict.values()))
simple_average = np.mean(all_predictions, axis=0)

# 集成方法2：加權平均（基於CV分數）
# 轉換RMSE為權重（越低越好，權重越高）
weights = 1 / np.array(list(cv_scores.values()))
weights = weights / weights.sum()

weighted_average = np.zeros_like(simple_average)
for i, (name, pred) in enumerate(predictions_dict.items()):
    weighted_average += pred * weights[i]

# 集成方法3：中位數
median_ensemble = np.median(all_predictions, axis=0)

print(f"\nEnsemble stats:")
print(f"  Simple average: ${simple_average.min():,.0f} - ${simple_average.max():,.0f}")
print(f"  Weighted average: ${weighted_average.min():,.0f} - ${weighted_average.max():,.0f}")
print(f"  Median: ${median_ensemble.min():,.0f} - ${median_ensemble.max():,.0f}")

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
print("\nUpload these to Kaggle (try in this order):")
print("1. ensemble_weighted.csv - Weighted average (best chance)")
print("2. ensemble_simple.csv - Simple average")
print("3. ensemble_median.csv - Median")
print(f"4. {best_model_name}_best.csv - Best single model")
print("\nGoal: Beat 0.15282!")
print("=" * 60)