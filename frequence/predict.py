
"""Prédiction avec le modèle XGBoost déjà entraîné."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessing import prepare_contract
from .monitoring import sauvegarder_prediction


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "artifacts" / "modele_frequence.joblib"


@lru_cache(maxsize=1)
def load_model():
    """Charge une seule fois le pipeline XGBoost entraîné."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    # Charger uniquement un fichier joblib de confiance.
    artifact = joblib.load(MODEL_PATH)

    if artifact.get("features") != "etendu":
        raise ValueError("Le modèle n'utilise pas le groupe 'etendu'.")

    if artifact.get("baseline"):
        raise ValueError("Le modèle utilise une baseline constante.")

    if artifact.get("calibration") != "aucune":
        raise ValueError("Cette version attend un modèle sans calibration.")

    if artifact.get("pipeline") is None:
        raise ValueError("Aucun pipeline XGBoost trouvé.")

    return artifact["pipeline"]


def predict_frequency(contract):
    """Retourne la probabilité d'au moins un sinistre."""

    model = load_model()

    # Préparer les 46 variables du contrat.
    x = prepare_contract(contract)

    # Respecter les colonnes apprises pendant l'entraînement.
    expected = list(model.feature_names_in_)

    missing = sorted(set(expected) - set(x.columns))
    extra = sorted(set(x.columns) - set(expected))

    if missing or extra:
        raise ValueError(
            f"Colonnes incohérentes : manquantes={missing}, en trop={extra}"
        )

    x = x[expected]

    # La colonne 1 correspond à P(sinistre).
    probability = float(model.predict_proba(x)[0, 1])

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probabilité invalide.")

    sauvegarder_prediction(contract, probability)

    return probability


if __name__ == "__main__":

    # Tester avec le premier contrat de test.csv.
    test_path = PROJECT_DIR / "data" / "test.csv"

    test = pd.read_csv(
        test_path,
        dtype={"code_postal": "string"},
        low_memory=False,
    )

    contract = test.iloc[0].to_dict()

    probability = predict_frequency(contract)

    print(f"Probabilité de sinistre : {probability:.6f} ({probability:.2%})")
