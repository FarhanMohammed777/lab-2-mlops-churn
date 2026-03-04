icat > src/utils/config.py <<'EOF'
import yaml


def load_config(path: str = "config/default.yaml") -> dict:
    """Load YAML configuration into a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
EOF