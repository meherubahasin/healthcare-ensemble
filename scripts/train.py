from src.preprocessing import Preprocessor
from src.models import DiseaseEnsemble
from config.settings import settings
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    preprocessor = Preprocessor()
    paths = {
        'diabetes': settings.DIABETES_PATH,
        'heart': settings.HEART_PATH,
        'kidney': settings.KIDNEY_PATH
    }
    targets = {
        'diabetes': 'Outcome',
        'heart': 'target',
        'kidney': 'classification'
    }

    combined_df = preprocessor.fit_transform_all(paths, targets)

    ensemble = DiseaseEnsemble()

    for disease in ['diabetes', 'heart', 'kidney']:
        df_disease = combined_df[combined_df['disease_type'] == disease]
        X = df_disease.drop(['disease_outcome', 'disease_type'], axis=1)
        y = df_disease['disease_outcome'].dropna()
        X = X.loc[y.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=y
        )

        k = 1 if disease == 'kidney' and y_train.value_counts().min() < 5 else 5
        X_train_res, y_train_res = preprocessor.apply_smote(X_train, y_train, k_neighbors=k)

        ensemble.train_single(X_train_res, y_train_res, disease)

        # Evaluate
        acc = (ensemble.predict(X_test, disease) == y_test).mean()
        print(f"{disease.capitalize()} Test Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()