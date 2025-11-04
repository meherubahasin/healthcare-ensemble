import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from loguru import logger
from typing import Dict
import os

class DiseaseEnsemble:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.voting_classifiers = {}

    def train_single(self, X_train, y_train, disease: str):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        path = f"{self.model_dir}/{disease}_lr.pkl"
        joblib.dump(model, path)
        self.models[disease] = model
        logger.info(f"{disease.capitalize()} model trained and saved to {path}")

    def build_voting_ensemble(self, disease: str):
        if disease not in self.models:
            raise ValueError(f"Model for {disease} not trained")
        voting = VotingClassifier(
            estimators=[('lr', self.models[disease])],
            voting='soft'
        )
        self.voting_classifiers[disease] = voting
        path = f"{self.model_dir}/{disease}_ensemble.pkl"
        joblib.dump(voting, path)
        return voting

    def predict_proba(self, X: pd.DataFrame, disease: str) -> pd.Series:
        if disease not in self.voting_classifiers:
            self.build_voting_ensemble(disease)
        clf = joblib.load(f"{self.model_dir}/{disease}_ensemble.pkl")
        return pd.Series(clf.predict_proba(X)[:, 1], index=X.index)

    def predict(self, X: pd.DataFrame, disease: str) -> pd.Series:
        if disease not in self.voting_classifiers:
            self.build_voting_ensemble(disease)
        clf = joblib.load(f"{self.model_dir}/{disease}_ensemble.pkl")
        return pd.Series(clf.predict(X), index=X.index)