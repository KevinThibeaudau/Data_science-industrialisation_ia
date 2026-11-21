"""Recherche aléatoire où le seuil de p-value de la présélection devient un hyperparamètre.

La présélection est refaite dans chaque pli, sur la partie entraînement seulement.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from model import PARAMETRES
from preprocess import CANDIDATES_QUALI, CANDIDATES_QUANTI, CIBLE, charger_donnees, nettoyer, preselectionner, retour_en_euros

N_TIRAGES = 60
SEUILS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.01]  # 1,01 : toutes les variables
LOIS = {
    "preselection__seuil": SEUILS,
    "xgbregressor__max_depth": list(range(1, 11)),
    "xgbregressor__n_estimators": list(range(50, 501, 50)),
    "xgbregressor__learning_rate": [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1],
    "xgbregressor__min_child_weight": [1] + list(range(5, 101, 5)),
}


class Preselection(BaseEstimator, TransformerMixin):
    """Garde les variables dont la p-value avec le montant est sous le seuil."""

    def __init__(self, seuil=0.2):
        self.seuil = seuil

    def fit(self, X, y):
        # Kruskal-Wallis et Spearman reposent sur les rangs : le log du montant donne les mêmes p-values
        self.variables_ = preselectionner(X.assign(**{CIBLE: np.asarray(y)}), self.seuil)
        return self

    def transform(self, X):
        return X[self.variables_]


def creer_pipeline():
    encodeur = ColumnTransformer(
        [("quali", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), make_column_selector(dtype_exclude="number"))],
        remainder="passthrough",
    )
    return make_pipeline(Preselection(), encodeur, XGBRegressor(**PARAMETRES))


def scores(y, prediction):
    return {
        "MAE": mean_absolute_error(y, prediction),
        "RMSE": root_mean_squared_error(y, prediction),
        "R2": r2_score(y, prediction),
        "total_predit / total_reel": prediction.sum() / y.sum(),
    }


if __name__ == "__main__":
    train, _ = charger_donnees()
    train = nettoyer(train)
    entrainement, test_interne = train_test_split(train[train[CIBLE] > 0], test_size=0.2, random_state=42)
    candidates = CANDIDATES_QUALI + CANDIDATES_QUANTI
    x_train, x_test = entrainement[candidates], test_interne[candidates]
    y_train_log = np.log(entrainement[CIBLE])

    combinaisons = np.prod([len(v) for v in LOIS.values()])
    print(f"Grille complète : {combinaisons} combinaisons, soit {combinaisons * 5} entraînements")
    print(f"Recherche aléatoire : {N_TIRAGES} combinaisons, soit {N_TIRAGES * 5} entraînements")

    plis = KFold(n_splits=5, shuffle=True, random_state=42)
    recherche = RandomizedSearchCV(creer_pipeline(), LOIS, n_iter=N_TIRAGES, scoring="neg_root_mean_squared_error",
                                   cv=plis, random_state=42).fit(x_train, y_train_log)
    meilleur = recherche.best_estimator_
    print("Meilleur RMSE du log en validation croisée :", round(-recherche.best_score_, 4))
    print("Meilleurs hyperparamètres :", {k.split("__")[1]: v for k, v in recherche.best_params_.items()})
    print("Variables retenues :", meilleur.named_steps["preselection"].variables_)

    tirages = pd.DataFrame(recherche.cv_results_)
    print("\nMeilleur score en validation croisée pour chaque seuil tiré :")
    print((-tirages.groupby("param_preselection__seuil")["mean_test_score"].max()).round(4).to_string())

    # Facteur de correction sur des prédictions hors échantillon, avec les meilleurs réglages
    hors_echantillon = cross_val_predict(recherche.best_estimator_, x_train, y_train_log, cv=plis)
    correction = np.exp(y_train_log - hors_echantillon).mean()
    log_predit = meilleur.predict(x_test)

    y_test = test_interne[CIBLE]
    resultats = pd.DataFrame({
        "Sans correction": scores(y_test, retour_en_euros(log_predit)),
        "Avec correction": scores(y_test, retour_en_euros(log_predit, correction)),
    }).T
    print("\nFacteur de correction :", round(correction, 4))
    print(resultats.round(4).to_string())
