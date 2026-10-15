
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/processed/train_features.parquet"
VALIDATION_PATH = ROOT / "data/processed/validation_features.parquet"
MODEL_PATH = ROOT / "models/lightgbm_tuned.joblib"


def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    y_train = train["in_study_repair"]
    y_val = (validation["class_label"] != 0).astype(int)

    drop_train = [
        "vehicle_id",
        "in_study_repair",
        "length_of_study_time_step",
    ]

    X_train = train.drop(
        columns=drop_train, errors="ignore"
    ).select_dtypes(include="number")

    X_val = validation.drop(
        columns=["vehicle_id", "class_label"],
        errors="ignore",
    ).select_dtypes(include="number")

    X_val = X_val.reindex(columns=X_train.columns)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    return X_train, y_train, X_val, y_val


def main():
    X_train, y_train, X_val, y_val = load_data()

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LGBMClassifier(
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )),
    ])

    parameter_grid = {
        "classifier__n_estimators": [150, 300, 500],
        "classifier__learning_rate": [0.01, 0.03, 0.05],
        "classifier__num_leaves": [15, 31, 63],
        "classifier__max_depth": [-1, 5, 10],
        "classifier__min_child_samples": [20, 50, 100],
        "classifier__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=8,
    scoring="average_precision",
    cv=cv,
    random_state=42,
    n_jobs=1,
    verbose=2,
    refit=True,
    error_score="raise",
    )

    print("Starting LightGBM tuning...")
    search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nBest CV average precision:")
    print(search.best_score_)

    best_model = search.best_estimator_
    probabilities = best_model.predict_proba(X_val)[:, 1]

    print("\nValidation evaluation:")
    print("ROC-AUC:", roc_auc_score(y_val, probabilities))
    print("PR-AUC:", average_precision_score(y_val, probabilities))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("\nSaved tuned model:", MODEL_PATH)


if __name__ == "__main__":
    main()
