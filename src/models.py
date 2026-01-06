"""
Model definition and training module
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = float('inf')
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize all models"""
        models_config = {}
        
        # Linear models
        if 'ridge' in self.config['models']['use_models']:
            models_config['Ridge'] = Ridge(alpha=10.0, random_state=self.config['models']['random_state'])
        
        if 'lasso' in self.config['models']['use_models']:
            models_config['Lasso'] = Lasso(alpha=0.001, max_iter=10000, random_state=self.config['models']['random_state'])
        
        if 'elasticnet' in self.config['models']['use_models']:
            models_config['ElasticNet'] = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=self.config['models']['random_state'])
        
        # Tree models
        if 'random_forest' in self.config['models']['use_models']:
            models_config['RandomForest'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config['models']['random_state'],
                n_jobs=-1
            )
        
        if 'gradient_boosting' in self.config['models']['use_models']:
            models_config['GradientBoosting'] = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=5,
                random_state=self.config['models']['random_state']
            )
        
        # XGBoost
        if 'xgboost' in self.config['models']['use_models']:
            xgb_params = self.config['xgboost_params']
            models_config['XGBoost'] = XGBRegressor(
                n_estimators=xgb_params['n_estimators'],
                learning_rate=xgb_params['learning_rate'],
                max_depth=xgb_params['max_depth'],
                subsample=xgb_params['subsample'],
                colsample_bytree=xgb_params['colsample_bytree'],
                reg_alpha=xgb_params['reg_alpha'],
                reg_lambda=xgb_params['reg_lambda'],
                random_state=self.config['models']['random_state'],
                n_jobs=-1
            )
        
        self.models = models_config
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        
        return self.models
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Cross-validate all models"""
        print("\n" + "="*60)
        print("Starting Cross-Validation...")
        print("="*60)
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.config['models']['random_state'])
        results = {}
        
        for name, model in self.models.items():
            print(f"\nValidating {name}...")
            
            try:
                # Calculate cross-validation scores
                scores = cross_val_score(
                    model, X, y, 
                    cv=kf, 
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
                
                rmse_scores = -scores  # Convert to positive
                
                results[name] = {
                    'mean_rmse': rmse_scores.mean(),
                    'std_rmse': rmse_scores.std(),
                    'min_rmse': rmse_scores.min(),
                    'max_rmse': rmse_scores.max(),
                    'scores': rmse_scores
                }
                
                print(f"  RMSE: {rmse_scores.mean():.5f} (±{rmse_scores.std():.5f})")
                
                # Update best model
                if rmse_scores.mean() < self.best_score:
                    self.best_score = rmse_scores.mean()
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"  {name} validation failed: {e}")
                results[name] = None
        
        print("\n" + "="*60)
        print("Cross-Validation Results Summary:")
        print("="*60)
        
        for name, result in results.items():
            if result is not None:
                print(f"{name:20s}: {result['mean_rmse']:.5f}")
        
        print(f"\nBest Model: {self.best_model_name} (RMSE: {self.best_score:.5f})")
        
        return results
    
    def train_best_model(self, X, y):
        """Train the best model"""
        print(f"\nTraining Best Model: {self.best_model_name}")
        
        # Get best model
        model = self.models[self.best_model_name]
        
        # Train model
        model.fit(X, y)
        
        # Evaluate on training set
        y_pred = model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"Training RMSE: {train_rmse:.5f}")
        
        self.best_model = model
        
        return model
    
    def train_all_models(self, X, y):
        """Train all models"""
        print("\nTraining all models...")
        
        trained_models = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y)
            trained_models[name] = model
        
        return trained_models
    
    def predict(self, X, model_name=None):
        """Make predictions using model"""
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No trained model available. Please train the model first.")
            model = self.best_model
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model not found: {model_name}")
        
        return model.predict(X)