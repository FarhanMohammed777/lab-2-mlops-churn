cat > src/preprocess.py <<'EOF'
import argparse
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.config import load_config


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw churn CSV and perform basic cleaning."""
    df = pd.read_csv(path)

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric and handle missing
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove tenure = 0 rows (common in this dataset)
    if "tenure" in df.columns:
        df = df[df["tenure"] != 0].copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    # Map SeniorCitizen 0/1 to No/Yes (optional but matches tutorial style)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode object columns into integers."""
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw churn data.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Override raw CSV input path")
    parser.add_argument("--output", default=None, help="Override processed CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raw_path = args.input or cfg["paths"]["raw_data"]
    out_path = args.output or cfg["paths"]["processed_data"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[preprocess] Loading raw data from: {raw_path}")
    df = load_and_clean(raw_path)

    print("[preprocess] Encoding categorical columns...")
    df = encode_categoricals(df)

    df.to_csv(out_path, index=False)
    print(f"[preprocess] Done. Wrote {len(df)} rows to: {out_path}")


if __name__ == "__main__":
    main()
EOF