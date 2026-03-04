cat > src/train.py <<'EOF'
import argparse
import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config


def build_pipeline(num_cols, model_params: dict) -> Pipeline:
    """Build a sklearn pipeline: scale numeric cols, passthrough the rest, then classifier."""
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)],
        remainder="passthrough",
    )
    clf = GradientBoostingClassifier(**model_params)
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn model.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")

    # Optional overrides
    parser.add_argument("--data", default=None, help="Override processed CSV path")
    parser.add_argument("--model-out", default=None, help="Override model output path")

    # Training overrides
    parser.add_argument("--test-size", type=float, default=None, help="Override test split size")
    parser.add_argument("--random-state", type=int, default=None, help="Override random seed")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = args.data or cfg["paths"]["processed_data"]
    model_path = args.model_out or cfg["paths"]["model"]

    num_cols = cfg["features"]["numerical"]

    test_size = args.test_size if args.test_size is not None else cfg["training"]["test_size"]
    random_state = args.random_state if args.random_state is not None else cfg["training"]["random_state"]

    # Model params from config + CLI random_state override
    model_params = dict(cfg["model"].get("params", {}))
    model_params["random_state"] = random_state

    print(f"[train] Reading processed data: {data_path}")
    df = pd.read_csv(data_path)

    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found in processed dataset.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline(num_cols=num_cols, model_params=model_params)

    # --- MLflow setup ---
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "churn-prediction")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with mlflow.start_run():
        # Log params
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model", cfg["model"]["name"])
        for k, v in model_params.items():
            mlflow.log_param(f"model_{k}", v)

        # Train
        print("[train] Training pipeline...")
        pipeline.fit(X_train, y_train)

        # Evaluate
        print("[train] Evaluating...")
        yhat = pipeline.predict(X_test)
        acc = accuracy_score(y_test, yhat)
        f1 = f1_score(y_test, yhat)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log model to MLflow
        mlflow.sklearn.log_model(pipeline, "model")

        # Save local pickle for evaluate/predict scripts
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        print(f"[train] Saved model to: {model_path}")
        print(f"[train] accuracy={acc:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    main()
EOF