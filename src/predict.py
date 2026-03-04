cat > src/predict.py <<'EOF'
import argparse
import os
import pickle

import pandas as pd

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make churn predictions.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--input", default=None, help="CSV to predict on (defaults to processed_data)")
    parser.add_argument("--model", default=None, help="Model pickle path (defaults to config paths.model)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV for predictions")
    parser.add_argument("--nrows", type=int, default=10, help="Predict only first N rows (for demo)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    input_path = args.input or cfg["paths"]["processed_data"]
    model_path = args.model or cfg["paths"]["model"]

    print(f"[predict] Loading input: {input_path}")
    df = pd.read_csv(input_path)

    # If target exists, drop it for prediction
    if "Churn" in df.columns:
        X = df.drop(columns=["Churn"])
    else:
        X = df

    X_small = X.head(args.nrows).copy()

    print(f"[predict] Loading model: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_small)

    out_df = X_small.copy()
    out_df["prediction"] = preds

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"[predict] Saved predictions to: {args.output}")
    print("[predict] First few predictions:")
    print(out_df[["prediction"]].head(10))


if __name__ == "__main__":
    main()
EOF