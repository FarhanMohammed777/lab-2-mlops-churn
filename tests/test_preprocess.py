cat > tests/test_preprocess.py <<'EOF'
import pandas as pd
from src.preprocess import load_and_clean, encode_categoricals


def _make_sample_csv(tmp_path):
    p = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "tenure": [1, 0, 5],
            "MonthlyCharges": [70.0, 80.0, 90.0],
            "TotalCharges": ["70", " ", "450.5"],
            "SeniorCitizen": [0, 1, 0],
            "Contract": ["Month-to-month", "Two year", "One year"],
            "Churn": [0, 1, 0],
        }
    )
    df.to_csv(p, index=False)
    return str(p)


def test_load_and_clean_drops_customer_id(tmp_path):
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    assert "customerID" not in df.columns


def test_load_and_clean_removes_tenure_zero(tmp_path):
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    assert (df["tenure"] != 0).all()


def test_totalcharges_is_numeric_after_clean(tmp_path):
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_encode_categoricals_converts_object_columns(tmp_path):
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    df2 = encode_categoricals(df)
    # Contract and SeniorCitizen become numeric after encoding (SeniorCitizen was mapped to string first)
    assert pd.api.types.is_numeric_dtype(df2["Contract"])
    assert pd.api.types.is_numeric_dtype(df2["SeniorCitizen"])
EOF