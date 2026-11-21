"""Prediction of the claim amount with the trained model."""

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_NAME, MODEL_PATH
from .preprocessing import clean

REQUIRED_KEYS = ("model", "pipeline", "variables", "correction")


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model once and check that it is the expected one."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Only load a joblib file you trust: it can run code when opened.
    artifact = joblib.load(MODEL_PATH)

    missing = [key for key in REQUIRED_KEYS if key not in artifact]
    if missing:
        raise ValueError(f"Invalid model file, missing keys: {missing}")
    if artifact["model"] != MODEL_NAME:
        raise ValueError(f"Unexpected model: {artifact['model']}")

    return artifact


def predict_claim_amount(contract):
    """Return the expected claim amount, in euros, for one contract.

    `contract` is a dictionary with the fields of the training file.
    """
    artifact = load_model()

    contracts = clean(pd.DataFrame([contract]))

    missing = sorted(set(artifact["variables"]) - set(contracts.columns))
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    log_amount = float(artifact["pipeline"].predict(contracts[artifact["variables"]])[0])
    amount = float(np.exp(log_amount) * artifact["correction"])

    if not np.isfinite(amount) or amount < 0:
        raise ValueError("Invalid amount.")

    return amount
