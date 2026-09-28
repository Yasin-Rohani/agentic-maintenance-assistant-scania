from pathlib import Path
from lightgbm import LGBMClassifier
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_TABLES_DIR = Path("reports/tables")


TRAIN_FEATURES_PATH = PROCESSED_DATA_DIR / "train_features.parquet"
VALIDATION_FEATURES_PATH = PROCESSED_DATA_DIR / "validation_features.parquet"

MODEL_OUTPUT_PATH = MODELS_DIR / "random_forest_baseline.joblib"
VALIDATION_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "validation_predictions_baseline.csv"


ID_COLUMN = "vehicle_id"
TRAIN_TARGET_COLUMN = "in_study_repair"
VALIDATION_TARGET_COLUMN = "class_label"


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation feature tables."""
    print("Loading train features...")
    train_df = pd.read_parquet(TRAIN_FEATURES_PATH)

    print("Loading validation features...")
    validation_df = pd.read_parquet(VALIDATION_FEATURES_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")

    return train_df, validation_df


def prepare_train_data(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data.

    Target:
    - in_study_repair

    For the first baseline, only numeric features are used.
    Non-numeric specification columns are excluded.
    """
    y_train = train_df[TRAIN_TARGET_COLUMN]

    columns_to_drop = [
        ID_COLUMN,
        TRAIN_TARGET_COLUMN,
        "length_of_study_time_step",
    ]

    existing_columns_to_drop = [
        column for column in columns_to_drop if column in train_df.columns
    ]

    X_train = train_df.drop(columns=existing_columns_to_drop)

    # Keep only numeric columns for the first baseline
    X_train = X_train.select_dtypes(include=["number"])

    print(f"X_train shape after keeping numeric features only: {X_train.shape}")
    print("y_train distribution:")
    print(y_train.value_counts(normalize=True).sort_index())

    non_numeric_columns = train_df.drop(columns=existing_columns_to_drop).select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric_columns:
        print("\nExcluded non-numeric columns from baseline:")
        print(non_numeric_columns)

    return X_train, y_train


def prepare_validation_data(
    validation_df: pd.DataFrame,
    train_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare validation data.

    Validation labels contain multiple class_label values.
    For the first baseline, we convert the task into binary classification:

    - class_label = 0  -> no failure / normal
    - class_label != 0 -> failure-related class

    Only numeric features are used.
    """
    y_validation_binary = (validation_df[VALIDATION_TARGET_COLUMN] != 0).astype(int)

    vehicle_ids = validation_df[ID_COLUMN]

    columns_to_drop = [
        ID_COLUMN,
        VALIDATION_TARGET_COLUMN,
    ]

    existing_columns_to_drop = [
        column for column in columns_to_drop if column in validation_df.columns
    ]

    X_validation = validation_df.drop(columns=existing_columns_to_drop)

    # Keep only numeric columns
    X_validation = X_validation.select_dtypes(include=["number"])

    # Ensure validation has exactly the same feature columns as train
    X_validation = X_validation.reindex(columns=train_columns, fill_value=0)

    print(f"X_validation shape after keeping numeric features only: {X_validation.shape}")
    print("Binary validation target distribution:")
    print(y_validation_binary.value_counts(normalize=True).sort_index())

    return X_validation, y_validation_binary, vehicle_ids


def build_random_forest_model() -> Pipeline:
    """Create a baseline Random Forest pipeline."""
    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def build_lightgbm_model() -> Pipeline:
    """Create a baseline LightGBM pipeline."""
    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    num_leaves=31,
                    max_depth=-1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )

    return model


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    vehicle_ids: pd.Series,
) -> dict:
    """Evaluate model on validation data."""
    print("=" * 80)
    print(f"Evaluating model: {model_name}")
    print("Predicting validation probabilities...")

    validation_probabilities = model.predict_proba(X_validation)[:, 1]

    default_threshold = 0.5
    validation_predictions = (validation_probabilities >= default_threshold).astype(int)

    roc_auc = roc_auc_score(y_validation, validation_probabilities)
    pr_auc = average_precision_score(y_validation, validation_probabilities)

    print("\nValidation metrics:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print("\nConfusion matrix at threshold 0.50:")
    print(confusion_matrix(y_validation, validation_predictions))

    print("\nClassification report at threshold 0.50:")
    print(classification_report(y_validation, validation_predictions, digits=4))

    precision, recall, thresholds = precision_recall_curve(
        y_validation,
        validation_probabilities,
    )

    threshold_df = pd.DataFrame(
        {
            "threshold": list(thresholds),
            "precision": list(precision[:-1]),
            "recall": list(recall[:-1]),
        }
    )

    threshold_df["f1"] = (
        2
        * threshold_df["precision"]
        * threshold_df["recall"]
        / (threshold_df["precision"] + threshold_df["recall"])
    )

    threshold_df = threshold_df.dropna()

    best_row = threshold_df.sort_values("f1", ascending=False).iloc[0]

    print("\nBest validation threshold by F1:")
    print(best_row)

    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(
        {
            "vehicle_id": vehicle_ids,
            "true_label_binary": y_validation,
            "failure_probability": validation_probabilities,
            "prediction_threshold_0_50": validation_predictions,
        }
    )

    prediction_output_path = (
        REPORTS_TABLES_DIR / f"validation_predictions_{model_name}.csv"
    )

    threshold_output_path = (
        REPORTS_TABLES_DIR / f"validation_threshold_metrics_{model_name}.csv"
    )

    predictions_df.to_csv(prediction_output_path, index=False)
    threshold_df.to_csv(threshold_output_path, index=False)

    print(f"\nSaved validation predictions to: {prediction_output_path}")
    print(f"Saved threshold metrics to: {threshold_output_path}")

    return {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_row["threshold"],
        "best_precision": best_row["precision"],
        "best_recall": best_row["recall"],
        "best_f1": best_row["f1"],
    }


def save_model(model_name: str, model: Pipeline) -> None:
    """Save trained model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_output_path = MODELS_DIR / f"{model_name}.joblib"

    joblib.dump(model, model_output_path)

    print(f"Saved model to: {model_output_path}")

def main() -> None:
    train_df, validation_df = load_features()

    X_train, y_train = prepare_train_data(train_df)

    X_validation, y_validation, validation_vehicle_ids = prepare_validation_data(
        validation_df=validation_df,
        train_columns=X_train.columns.tolist(),
    )

    models = {
        "random_forest_baseline": build_random_forest_model(),
        "lightgbm_baseline": build_lightgbm_model(),
    }

    results = []

    for model_name, model in models.items():
        print("=" * 80)
        print(f"Training model: {model_name}")

        model.fit(X_train, y_train)

        model_result = evaluate_model(
            model_name=model_name,
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
            vehicle_ids=validation_vehicle_ids,
        )

        results.append(model_result)

        save_model(model_name=model_name, model=model)

    results_df = pd.DataFrame(results)

    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    comparison_output_path = REPORTS_TABLES_DIR / "baseline_model_comparison.csv"
    results_df.to_csv(comparison_output_path, index=False)

    print("=" * 80)
    print("Baseline model comparison:")
    print(results_df)

    print(f"\nSaved model comparison to: {comparison_output_path}")

    print("\nBaseline training completed successfully.")

if __name__ == "__main__":
    main()