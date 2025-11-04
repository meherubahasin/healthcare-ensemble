import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from loguru import logger
from typing import Tuple, Dict
import joblib
import os

class Preprocessor:
    def __init__(self, scaler_path: str = None):
        self.scaler = StandardScaler()
        self.smote = None
        self.feature_names = None
        self.scaler_path = scaler_path or "models/scaler.pkl"

    def _clean_kidney_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'classification' in df.columns:
            df['classification'] = df['classification'].replace({
                'ckd\t': 'ckd', 'ckd': 'ckd', 'notckd': 'notckd'
            }).map({'ckd': 1, 'notckd': 0})
        return df

    def _preprocess_single(self, df: pd.DataFrame, target: str, disease_type: str) -> pd.DataFrame:
        df = df.copy()
        df = self._clean_kidney_target(df) if disease_type == 'kidney' else df

        # Impute
        for col in df.columns:
            if col == target:
                continue
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'missing', inplace=True)

        # Encode categorical
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.drop(target, errors='ignore')
        if len(cat_cols) > 0:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Scale numerical
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target, errors='ignore')
        if len(num_cols) > 0:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])

        df['disease_type'] = disease_type
        df.rename(columns={target: 'disease_outcome'}, inplace=True)
        return df

    def fit_transform_all(self, paths: Dict[str, str], targets: Dict[str, str]) -> pd.DataFrame:
        dfs = []
        for disease, path in paths.items():
            logger.info(f"Loading {disease} data from {path}")
            df = pd.read_csv(path)
            df = self._preprocess_single(df, targets[disease], disease)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        all_cols = sorted(set(col for df in dfs for col in df.columns))
        combined = combined.reindex(columns=all_cols, fill_value=0)

        self.feature_names = [c for c in all_cols if c not in ['disease_outcome', 'disease_type']]
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Preprocessed data shape: {combined.shape}")
        return combined

    def apply_smote(self, X: pd.DataFrame, y: pd.Series, k_neighbors: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        self.smote = SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))
        X_res, y_res = self.smote.fit_resample(X, y)
        return X_res, y_res