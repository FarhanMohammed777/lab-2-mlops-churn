cat > src/evaluate.py <<'EOF'
import argparse
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate churn model.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--data", default=None, help="Override processed CSV path")
    parser.add_argument("--model", default=None, help="Override model pickle path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = args.data or cfg["paths"]["processed_data"]
    model_path = args.model or cfg["paths"]["model"]

    test_size = cfg["training"]["test_size"]
    random_state = cfg["training"]["random_state"]

    print(f"[evaluate] Loading data: {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[evaluate] Loading model: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    yhat = model.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    f1 = f1_score(y_test, yhat)

    print("\n[evaluate] Results")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, yhat))


if __name__ == "__main__":
    main()
EOF