"""Entraînement final du meilleur modèle sur tous les sinistrés, prédiction du test Kaggle et calcul de la prime.

Meilleurs réglages trouvés par recherche_avec_seuil.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from preprocess import CANDIDATES_QUALI, CANDIDATES_QUANTI, CIBLE, ROOT, charger_donnees, nettoyer, retour_en_euros
from recherche_avec_seuil import creer_pipeline

DOSSIER = Path(__file__).resolve().parent
SORTIE_GRAVITE = DOSSIER / "soumission_gravite_finale.csv"
SORTIE_KAGGLE = DOSSIER / "soumission_kaggle_finale_v2.csv"
FREQUENCE = ROOT / "results" / "soumission_frequence_xgboost-final.csv"

MEILLEURS_REGLAGES = {
    "preselection__seuil": 0.05,
    "xgbregressor__max_depth": 1,
    "xgbregressor__learning_rate": 0.02,
    "xgbregressor__n_estimators": 100,
    "xgbregressor__min_child_weight": 25,
}


if __name__ == "__main__":
    train, test = charger_donnees()
    index_test = test["index"]
    train, test = nettoyer(train), nettoyer(test)

    candidates = CANDIDATES_QUALI + CANDIDATES_QUANTI
    sinistres = train[train[CIBLE] > 0]
    x_train, y_train_log = sinistres[candidates], np.log(sinistres[CIBLE])

    modele = creer_pipeline().set_params(**MEILLEURS_REGLAGES)
    hors_echantillon = cross_val_predict(modele, x_train, y_train_log, cv=KFold(n_splits=5, shuffle=True, random_state=42))
    correction = np.exp(y_train_log - hors_echantillon).mean()

    modele.fit(x_train, y_train_log)
    montants = retour_en_euros(modele.predict(test[candidates]), correction)
    gravite = pd.DataFrame({"index": index_test, "montant": montants})
    gravite.to_csv(SORTIE_GRAVITE, index=False)

    frequence = pd.read_csv(FREQUENCE).rename(columns={"INDEX": "index", "pred": "probabilite"})
    prime = gravite.merge(frequence, on="index", how="inner", validate="one_to_one")
    assert len(prime) == len(gravite) == 50000
    prime["pred"] = prime["probabilite"] * prime["montant"]
    prime[["index", "pred"]].sort_values("index").to_csv(SORTIE_KAGGLE, index=False)

    print("Sinistrés d'entraînement :", len(sinistres))
    print("Variables retenues :", modele.named_steps["preselection"].variables_)
    print("Facteur de correction :", round(correction, 4))
    print("Montant prédit moyen :", round(montants.mean(), 2), "| min", round(montants.min(), 2), "| max", round(montants.max(), 2))
    print("Probabilité moyenne :", round(prime["probabilite"].mean(), 5))
    print("Prime moyenne :", round(prime["pred"].mean(), 2), "| min", round(prime["pred"].min(), 2), "| max", round(prime["pred"].max(), 2))
    print("Fichiers écrits :", SORTIE_GRAVITE.name, "et", SORTIE_KAGGLE.name)
