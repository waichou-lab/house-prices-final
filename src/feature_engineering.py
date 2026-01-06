"""
Feature engineering module - Create 17 new features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    def __init__(self, create_new_features=True):
        self.create_new_features = create_new_features
        self.scaler = None
        self.label_encoders = {}
        
    def create_all_features(self, df):
        """Create all new features"""
        df_copy = df.copy()
        
        if not self.create_new_features:
            return df_copy
        
        print("Creating new features...")
        
        # 1. Spatial features
        df_copy = self._create_spatial_features(df_copy)
        
        # 2. Temporal features
        df_copy = self._create_temporal_features(df_copy)
        
        # 3. Quality features
        df_copy = self._create_quality_features(df_copy)
        
        # 4. Functional features
        df_copy = self._create_functional_features(df_copy)
        
        # 5. Conditional features
        df_copy = self._create_conditional_features(df_copy)
        
        # 6. Bathroom features
        df_copy = self._create_bathroom_features(df_copy)
        
        print(f"Feature engineering complete. Original features: {len(df.columns)}, New features: {len(df_copy.columns)}")
        
        return df_copy
    
    def _create_spatial_features(self, df):
        """Create spatial features"""
        # TotalSF - Total living area
        if all(col in df.columns for col in ['GrLivArea', 'TotalBsmtSF']):
            df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
        
        # RoomDensity - Room density
        if all(col in df.columns for col in ['TotRmsAbvGrd', 'GrLivArea']):
            df['RoomDensity'] = df['TotRmsAbvGrd'] / (df['GrLivArea'] + 1e-10)
        
        # LivingAreaRatio - Land use ratio
        if all(col in df.columns for col in ['GrLivArea', 'LotArea']):
            df['LivingAreaRatio'] = df['GrLivArea'] / (df['LotArea'] + 1e-10)
        
        return df
    
    def _create_temporal_features(self, df):
        """Create temporal features"""
        # HouseAge - House age
        if all(col in df.columns for col in ['YrSold', 'YearBuilt']):
            df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        
        # RemodAge - Remodel age
        if all(col in df.columns for col in ['YrSold', 'YearRemodAdd']):
            df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        
        # IsRemodeled - Whether remodeled
        if all(col in df.columns for col in ['YearBuilt', 'YearRemodAdd']):
            df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
        
        return df
    
    def _create_quality_features(self, df):
        """Create quality features"""
        # QualityArea - Quality × Area
        if all(col in df.columns for col in ['OverallQual', 'GrLivArea']):
            df['QualityArea'] = df['OverallQual'] * df['GrLivArea']
        
        # QualityPerRoom - Quality per room
        if all(col in df.columns for col in ['OverallQual', 'TotRmsAbvGrd']):
            df['QualityPerRoom'] = df['OverallQual'] / (df['TotRmsAbvGrd'] + 1e-10)
        
        # Quality score conversion
        qual_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        
        for col in ['ExterQual', 'KitchenQual']:
            if col in df.columns:
                df[f'{col}_num'] = df[col].map(qual_map).fillna(3)
        
        # OverallQualityIndex - Overall quality index
        quality_cols = ['OverallQual']
        if 'ExterQual_num' in df.columns:
            quality_cols.append('ExterQual_num')
        if 'KitchenQual_num' in df.columns:
            quality_cols.append('KitchenQual_num')
        
        if len(quality_cols) > 1:
            df['OverallQualityIndex'] = df[quality_cols].mean(axis=1)
        
        return df
    
    def _create_functional_features(self, df):
        """Create functional features"""
        # FinishedBsmtRatio - Basement finish ratio
        if all(col in df.columns for col in ['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF']):
            total_finished = df['BsmtFinSF1'] + df['BsmtFinSF2']
            df['FinishedBsmtRatio'] = total_finished / (df['TotalBsmtSF'] + 1e-10)
        
        # GarageAreaPerCar - Garage area per car
        if all(col in df.columns for col in ['GarageArea', 'GarageCars']):
            df['GarageAreaPerCar'] = df['GarageArea'] / (df['GarageCars'] + 1e-10)
        
        # OutdoorSpace - Outdoor space
        if all(col in df.columns for col in ['WoodDeckSF', 'OpenPorchSF']):
            df['OutdoorSpace'] = df['WoodDeckSF'] + df['OpenPorchSF']
        
        return df
    
    def _create_conditional_features(self, df):
        """Create conditional features"""
        # HighQualLargeHouse - High quality large house
        if all(col in df.columns for col in ['OverallQual', 'GrLivArea']):
            df['HighQualLargeHouse'] = (
                (df['OverallQual'] >= 8) & 
                (df['GrLivArea'] > 2000)
            ).astype(int)
        
        # OldButRemodeled - Old but remodeled house
        if all(col in df.columns for col in ['HouseAge', 'IsRemodeled']):
            df['OldButRemodeled'] = (
                (df['HouseAge'] > 30) & 
                (df['IsRemodeled'] == 1)
            ).astype(int)
        
        return df
    
    def _create_bathroom_features(self, df):
        """Create bathroom features"""
        # TotalBath - Total bathrooms
        bath_cols = []
        if 'FullBath' in df.columns:
            bath_cols.append('FullBath')
        if 'HalfBath' in df.columns:
            bath_cols.append(0.5 * df['HalfBath'])
        if 'BsmtFullBath' in df.columns:
            bath_cols.append(df['BsmtFullBath'])
        
        if bath_cols:
            df['TotalBath'] = sum(bath_cols)
        
        # AmenityScore - Amenity score
        score = 0
        if 'Fireplaces' in df.columns:
            score += df['Fireplaces']
        if 'GarageCars' in df.columns:
            score += df['GarageCars']
        if 'PoolArea' in df.columns:
            score += (df['PoolArea'] > 0).astype(int)
        
        if score is not 0:
            df['AmenityScore'] = score
        
        return df
    
    def encode_categorical_features(self, X_train, X_test=None):
        """Encode categorical features"""
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy() if X_test is not None else None
        
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Train LabelEncoder
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
            
            # Save encoder
            self.label_encoders[col] = le
            
            # Encode test set
            if X_test_encoded is not None and col in X_test_encoded.columns:
                X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
        
        return X_train_encoded, X_test_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        # Select numerical features
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Scale training set
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        # Scale test set
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        return X_train_scaled, X_test_scaled