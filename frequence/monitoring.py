
"""Enregistrement des contrats envoyés en prédiction."""

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = PROJECT_DIR / "monitoring_data" / "predictions.jsonl"

WRITE_LOCK = Lock()


def convertir_valeur(value):
    """Convertit une valeur Pandas ou NumPy en valeur JSON."""

    if value is None or pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def sauvegarder_prediction(contract, probability):
    """Ajoute le contrat et sa prédiction au fichier historique."""

    # Conserver toutes les données reçues dans le contrat.
    donnees = {
        key: convertir_valeur(value)
        for key, value in contract.items()
    }

    enregistrement = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "modele": "xgboost_frequence_v2_etendu",
        "contrat": donnees,
        "probabilite_sinistre": float(probability),
    }

    # Créer le dossier de monitoring si nécessaire.
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Ajouter une ligne sans écraser les précédentes.
    with WRITE_LOCK:
        with LOG_FILE.open("a", encoding="utf-8") as fichier:
            fichier.write(
                json.dumps(enregistrement, ensure_ascii=False)
                + "\n"
            )
