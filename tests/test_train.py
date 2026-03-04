cat > tests/test_train.py <<'EOF'
import numpy as np
import pandas as pd

from src.train import build_pipeline


def test_pipeline_fit_predict_length():
    rng = np.random.RandomState(42)

    X = pd.DataFrame(
        {
            "tenure": rng.randint(1, 72, size=30),
            "MonthlyCharges": rng.uniform(20, 120, size=30),
            "TotalCharges": rng.uniform(20, 8000, size=30),
            # some other already-encoded “categorical” ints
            "Contract": rng.randint(0, 3, size=30),
            "SeniorCitizen": rng.randint(0, 2, size=30),
        }
    )
    y = rng.randint(0, 2, size=30)

    pipeline = build_pipeline(
        num_cols=["tenure", "MonthlyCharges", "TotalCharges"],
        model_params={"random_state": 40},
    )
    pipeline.fit(X, y)
    yhat = pipeline.predict(X)
    assert len(yhat) == len(X)
EOF