"""Cleaning of contracts for the claim amount model.

The functions work on a DataFrame with any number of rows: the whole training
file, or a single contract sent to the API.
"""

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_CANDIDATES,
    DROPPED_COLUMNS,
    NUMERIC_CANDIDATES,
)

# Raw fields needed to build the candidate variables.
REQUIRED_COLUMNS = [
    *NUMERIC_CANDIDATES,
    *(v for v in CATEGORICAL_CANDIDATES if v != "departement"),
    "code_postal",
]

# Values that cannot be real: they are replaced by missing values.
IMPOSSIBLE_LICENCE_SENIORITY = 111


def clean(df):
    """Return a cleaned copy of the contracts, with the "departement" column."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    df = df.drop(columns=DROPPED_COLUMNS, errors="ignore")

    # Force numeric types. A missing value sent as None becomes NaN, which
    # XGBoost handles natively.
    for col in NUMERIC_CANDIDATES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # No second driver: his age and licence seniority do not exist.
    no_second_driver = (
        df["conducteur2"].astype("string").str.strip().str.upper().eq("NO")
    )
    df.loc[no_second_driver, ["age_conducteur2", "anciennete_permis2"]] = np.nan

    # Impossible values.
    for col in ["anciennete_permis1", "anciennete_permis2"]:
        df[col] = df[col].replace(IMPOSSIBLE_LICENCE_SENIORITY, np.nan)
    for col in ["poids_vehicule", "cylindre_vehicule"]:
        df[col] = df[col].replace(0, np.nan)

    return df.assign(departement=_department(df["code_postal"]))


def _department(postal_code):
    """First two characters of the postal code, padded to 5 characters."""
    code = postal_code.astype("string").str.strip().str.zfill(5)
    return code.str[:2].astype(object).where(code.notna(), np.nan)


def categorical_columns(variables):
    """Keep, in order, the selected variables that are categorical."""
    return [v for v in variables if v in CATEGORICAL_CANDIDATES]
