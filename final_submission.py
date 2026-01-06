#!/usr/bin/env python3
"""
Final optimized submission - Target: 0.149xx
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import os

print("=" * 60)
print("FINAL OPTIMIZED SUBMISSION")
print("=" * 60)

# 讀取數據
train = pd.read_csv('data/train.csv', encoding='latin1')
test = pd.read_csv('data/test.csv', encoding='latin1')

print(f"Train: {train.shape}, Test: {test.shape}")

# 創建所有重要特徵
def create_all_features(df):
    df_copy = df.copy()
    
    # 空間特徵
    if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
        df_copy['TotalSF'] = df_copy['GrLivArea'] + df_copy['TotalBsmtSF']
    
    if 'TotRmsAbvGrd' in df.columns and 'GrLivArea' in df.columns:
        df_copy['RoomDensity'] = df_copy['TotRmsAbvGrd'] / (df_copy['GrLivArea'] + 1)
    
    # 時間特徵
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df_copy['HouseAge'] = df_copy['YrSold'] - df_copy['YearBuilt']
    
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df_copy['RemodAge'] = df_copy['YrSold'] - df_copy['YearRemodAdd']
    
    # 品質特徵
    if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
        df_copy['QualityArea'] = df_copy['OverallQual'] * df_copy['GrLivArea']
    
    # 功能特徵
    if 'GarageArea' in df.columns and 'GarageCars' in df.columns:
        df_copy['GarageAreaPerCar'] = df_copy['GarageArea'] / (df_copy['GarageCars'] + 1)
    
    # 浴室特徵
    if 'FullBath' in df.columns and 'HalfBath' in df.columns:
        df_copy['TotalBath'] = df_copy['FullBath'] + 0.5 * df_copy['HalfBath']
        if 'BsmtFullBath' in df.columns:
            df_copy['TotalBath'] += df_copy['BsmtFullBath']
    
    return df_copy

# 應用特徵工程
train = create_all_features(train)
test = create_all_features(test)

# 選擇特徵（基於之前的成功）
features = [
    # 原始重要特徵
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
    'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
    'GarageArea', '1stFlrSF', 'MasVnrArea', 'Fireplaces',
    
    # 創建的新特徵
    'TotalSF', 'HouseAge', 'QualityArea', 'TotalBath',
    'GarageAreaPerCar'
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

# 對數轉換目標變量
y_train_log = np.log1p(y_train)

# 對數轉換偏態的特徵
for col in X_train.columns:
    if X_train[col].dtype in [np.float64, np.int64]:
        if X_train[col].skew() > 0.75:
            X_train[col] = np.log1p(X_train[col] + 1)
            X_test[col] = np.log1p(X_test[col] + 1)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining optimized models...")

# 訓練多個優化模型
models_predictions = {}

# 1. GradientBoosting (最佳單模型)
print("1. GradientBoosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=500,  # 增加樹的數量
    learning_rate=0.01,  # 降低學習率
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train_log)
gb_pred = np.expm1(gb_model.predict(X_test_scaled))
models_predictions['GB'] = gb_pred

# 2. Ridge回歸
print("2. Ridge regression...")
ridge_model = Ridge(alpha=15.0, random_state=42)  # 調整alpha
ridge_model.fit(X_train_scaled, y_train_log)
ridge_pred = np.expm1(ridge_model.predict(X_test_scaled))
models_predictions['Ridge'] = ridge_pred

# 3. 隨機森林
print("3. Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_log)
rf_pred = np.expm1(rf_model.predict(X_test_scaled))
models_predictions['RF'] = rf_pred

# 4. Lasso
print("4. Lasso...")
lasso_model = Lasso(alpha=0.0005, random_state=42, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train_log)
lasso_pred = np.expm1(lasso_model.predict(X_test_scaled))
models_predictions['Lasso'] = lasso_pred

# 創建集成預測
print("\nCreating ensemble predictions...")

# 加權平均（基於模型表現估計）
# 權重：GB: 0.4, Ridge: 0.3, RF: 0.2, Lasso: 0.1
weights = {'GB': 0.4, 'Ridge': 0.3, 'RF': 0.2, 'Lasso': 0.1}

final_prediction = np.zeros_like(gb_pred)
for name, pred in models_predictions.items():
    final_prediction += pred * weights[name]

# 混合：75% 集成 + 25% 最佳單模型
final_prediction = 0.75 * final_prediction + 0.25 * gb_pred

# 確保合理的範圍
train_prices = train['SalePrice']
q1 = train_prices.quantile(0.01)
q99 = train_prices.quantile(0.99)
final_prediction = np.clip(final_prediction, q1 * 0.8, q99 * 1.2)

print(f"\nFinal prediction stats:")
print(f"  Min: ${final_prediction.min():,.0f}")
print(f"  Max: ${final_prediction.max():,.0f}")
print(f"  Mean: ${final_prediction.mean():,.0f}")

# 保存最終提交
os.makedirs('submissions', exist_ok=True)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_prediction
})

output_file = 'submissions/final_optimized.csv'
submission.to_csv(output_file, index=False)

print(f"\n✓ Saved: {output_file}")

# 也保存各個模型版本
for name, pred in models_predictions.items():
    model_submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': pred
    })
    model_submission.to_csv(f'submissions/{name}_final.csv', index=False)
    print(f"✓ Saved: submissions/{name}_final.csv")

print("\n" + "=" * 60)
print("FINAL SUBMISSIONS READY!")
print("=" * 60)
print("\nBest to upload (in order):")
print("1. final_optimized.csv - Optimized ensemble (target: 0.149xx)")
print("2. GB_final.csv - GradientBoosting only")
print("3. Ridge_final.csv - Ridge regression")
print("4. RF_final.csv - Random Forest")
print("\nCurrent best: 0.15004")
print("Goal: Break 0.15000!")
print("=" * 60)