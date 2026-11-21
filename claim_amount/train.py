"""Training of the claim amount model.

Run from the project root:

    python -m claim_amount.train

The model is trained on claimants only and learns the logarithm of the
amount. The result is saved in one joblib file with everything the
prediction needs.
"""

from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from .config import (
    CV_FOLDS,
    MODEL_NAME,
    MODEL_PATH,
    P_VALUE_THRESHOLD,
    SEED,
    TARGET,
    TRAIN_PATH,
    XGB_PARAMS,
)
from .preprocessing import categorical_columns, clean
from .selection import select_variables


def build_pipeline(variables, params=XGB_PARAMS):
    """Encoder for the categorical variables, followed by XGBoost.

    Each class becomes one integer. A class never seen in training becomes -1.
    Numeric variables pass through unchanged.
    """
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    preprocessing = ColumnTransformer(
        [("categorical", encoder, categorical_columns(variables))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return make_pipeline(preprocessing, XGBRegressor(**params))


def duan_correction(x, y_log, variables, params=XGB_PARAMS):
    """Factor that corrects the bias of exp() when going back to euros.

    Computed on out-of-fold predictions, so it is not too optimistic.
    """
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    predictions = cross_val_predict(build_pipeline(variables, params), x, y_log, cv=folds)
    return float(np.exp(y_log - predictions).mean())


def train(raw_train, threshold=P_VALUE_THRESHOLD, params=XGB_PARAMS):
    """Train the model on raw contracts and return the artifact to save."""
    claims = clean(raw_train)
    claims = claims[claims[TARGET] > 0]

    variables = select_variables(claims, threshold)
    if not variables:
        raise ValueError(f"No variable has a p-value below {threshold}.")

    x = claims[variables]
    y_log = np.log(claims[TARGET])

    correction = duan_correction(x, y_log, variables, params)
    pipeline = build_pipeline(variables, params).fit(x, y_log)

    return {
        "model": MODEL_NAME,
        "pipeline": pipeline,
        "variables": variables,
        "correction": correction,
        "threshold": threshold,
        "params": params,
        "n_claims": len(claims),
        "trained_at": datetime.now(UTC).isoformat(),
    }


def main():
    raw_train = pd.read_csv(
        TRAIN_PATH, dtype={"code_postal": str, "modele_vehicule": str}
    )
    artifact = train(raw_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Claimants used : {artifact['n_claims']}")
    print(f"Variables kept ({len(artifact['variables'])}) : {artifact['variables']}")
    print(f"Duan correction factor : {artifact['correction']:.4f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
