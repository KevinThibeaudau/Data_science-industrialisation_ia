"""Teste l'effet des variables d'entrée : mêmes réglages que la version 5, avec une présélection plus stricte."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from model import PARAMETRES, calculer_correction, tracer_courbes
from preprocess import CIBLE, charger_donnees, creer_encodeur, nettoyer, preselectionner, retour_en_euros

GRAPHE_LOSS = Path(__file__).resolve().parent / "loss_xgboost_var.png"

SEUIL_REFERENCE = 0.20
SEUIL_TESTE = 0.01

PARAMETRES_V5 = PARAMETRES | {"max_depth": 1, "learning_rate": 0.013240682461037212, "n_estimators": 112, "min_child_weight": 12}


def scores(y, prediction):
    return {
        "MAE": mean_absolute_error(y, prediction),
        "RMSE": root_mean_squared_error(y, prediction),
        "total_predit / total_reel": prediction.sum() / y.sum(),
    }


def evaluer(entrainement, test_interne, seuil, chemin_graphe=None):
    """Entraîne sur les 80 % avec les variables sous le seuil, et évalue sur les 20 %."""
    variables = preselectionner(entrainement, seuil)
    x_train, x_test = entrainement[variables], test_interne[variables]
    y_train_log, y_test_log = np.log(entrainement[CIBLE]), np.log(test_interne[CIBLE])
    correction = calculer_correction(x_train, y_train_log, PARAMETRES_V5)

    encodeur = creer_encodeur(x_train)
    x_train_encode, x_test_encode = encodeur.fit_transform(x_train), encodeur.transform(x_test)
    modele = XGBRegressor(**PARAMETRES_V5).fit(
        x_train_encode, y_train_log, eval_set=[(x_train_encode, y_train_log), (x_test_encode, y_test_log)], verbose=False
    )
    if chemin_graphe:
        tracer_courbes(modele.evals_result(), y_train_log, y_test_log, chemin_graphe)

    log_predit = modele.predict(x_test_encode)
    y_test = test_interne[CIBLE]
    resultats = {
        f"p < {seuil}, sans correction": scores(y_test, retour_en_euros(log_predit)),
        f"p < {seuil}, avec correction": scores(y_test, retour_en_euros(log_predit, correction)),
    }
    return resultats, variables


if __name__ == "__main__":
    train, _ = charger_donnees()
    train = nettoyer(train)
    entrainement, test_interne = train_test_split(train[train[CIBLE] > 0], test_size=0.2, random_state=42)

    reference, variables_reference = evaluer(entrainement, test_interne, SEUIL_REFERENCE)
    teste, variables_testees = evaluer(entrainement, test_interne, SEUIL_TESTE, GRAPHE_LOSS)

    print(f"Variables p < {SEUIL_REFERENCE} ({len(variables_reference)}) :", variables_reference)
    print(f"Variables p < {SEUIL_TESTE} ({len(variables_testees)}) :", variables_testees)
    print(pd.DataFrame(reference | teste).T.round(3).to_string())
