"""Modèle final de gravité : XGBoost version 5, entraîné sur tous les sinistrés, prédit le jeu test Kaggle."""
from pathlib import Path

import pandas as pd

from model import PARAMETRES, calculer_correction, creer_modele
from preprocess import preparer, retour_en_euros

SORTIE = Path(__file__).resolve().parent / "soumission_gravite_v5.csv"

PARAMETRES_FINAUX = PARAMETRES | {
    "max_depth": 1,
    "learning_rate": 0.035296682203324346,
    "n_estimators": 50,
    "min_child_weight": 66,
}


if __name__ == "__main__":
    x_train, y_train_log, x_test, index_test = preparer()

    correction = calculer_correction(x_train, y_train_log, PARAMETRES_FINAUX)
    modele = creer_modele(x_train, PARAMETRES_FINAUX).fit(x_train, y_train_log)
    predictions = retour_en_euros(modele.predict(x_test), correction)

    pd.DataFrame({"index": index_test, "pred": predictions}).to_csv(SORTIE, index=False)

    print("Entraînement :", x_train.shape, "| Test :", x_test.shape)
    print("Facteur de correction :", round(correction, 4))
    print("Montant prédit moyen :", round(predictions.mean(), 2))
    print("Prédictions écrites dans", SORTIE)
