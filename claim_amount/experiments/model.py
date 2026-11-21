"""Modèle XGBoost de gravité : apprend le log du montant sur les sinistrés, puis prédit le jeu test en euros."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from preprocess import CIBLE, charger_donnees, creer_encodeur, nettoyer, preparer, preselectionner, retour_en_euros

SORTIE = Path(__file__).resolve().parent / "predictions_gravite_v6.csv"
GRAPHE_LOSS = Path(__file__).resolve().parent / "loss_xgboost.png"
GRAPHE_LOSS_V3 = Path(__file__).resolve().parent / "loss_xgboost_v3.png"
GRAPHE_LOSS_V5 = Path(__file__).resolve().parent / "loss_xgboost_v5.png"
GRAPHE_LOSS_V6 = Path(__file__).resolve().parent / "loss_xgboost_v6.png"

PARAMETRES = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


GRILLE = {
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.1],
    "n_estimators": [50, 100, 200, 300],
    "min_child_weight": [5, 20],
}

N_TIRAGES = 60
DISTRIBUTIONS = {
    "max_depth": randint(1, 6),
    "learning_rate": loguniform(0.003, 0.1),
    "n_estimators": randint(50, 501),
    "min_child_weight": randint(5, 101),
}
DISTRIBUTIONS_V6 = DISTRIBUTIONS | {
    "max_depth": list(range(1, 11)),
    "n_estimators": list(range(50, 501, 50)),
}


def creer_modele(x, parametres=PARAMETRES):
    return make_pipeline(creer_encodeur(x), XGBRegressor(**parametres))


def calculer_correction(x, y_log, parametres=PARAMETRES):
    """Facteur de Duan pour le retour en euros, calculé sur des prédictions hors échantillon."""
    plis = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(creer_modele(x, parametres), x, y_log, cv=plis)
    return np.exp(y_log - predictions).mean()


def optimiser(x, y_log):
    """Cherche les hyperparamètres par validation croisée à 5 plis, sur le RMSE du log."""
    plis = KFold(n_splits=5, shuffle=True, random_state=42)
    grille = {f"xgbregressor__{nom}": valeurs for nom, valeurs in GRILLE.items()}
    recherche = GridSearchCV(creer_modele(x), grille, scoring="neg_root_mean_squared_error", cv=plis).fit(x, y_log)
    print("Meilleur RMSE du log en validation croisée :", round(-recherche.best_score_, 4))
    return PARAMETRES | {nom.split("__")[1]: valeur for nom, valeur in recherche.best_params_.items()}


def optimiser_aleatoire(x, y_log, lois=DISTRIBUTIONS):
    """Tire des combinaisons d'hyperparamètres au hasard, évaluées par validation croisée à 5 plis sur le RMSE du log."""
    plis = KFold(n_splits=5, shuffle=True, random_state=42)
    distributions = {f"xgbregressor__{nom}": loi for nom, loi in lois.items()}
    recherche = RandomizedSearchCV(
        creer_modele(x), distributions, n_iter=N_TIRAGES, scoring="neg_root_mean_squared_error", cv=plis, random_state=42
    ).fit(x, y_log)
    print("Recherche aléatoire, meilleur RMSE du log en validation croisée :", round(-recherche.best_score_, 4))
    return PARAMETRES | {nom.split("__")[1]: valeur for nom, valeur in recherche.best_params_.items()}


def tracer_courbes(courbes, y_train_log, y_test_log, chemin=GRAPHE_LOSS):
    """Trace la loss (RMSE du log) et le R² du log, l'équivalent de l'accuracy pour une régression."""
    loss_train, loss_val = np.array(courbes["validation_0"]["rmse"]), np.array(courbes["validation_1"]["rmse"])
    r2_train, r2_val = 1 - loss_train ** 2 / y_train_log.var(ddof=0), 1 - loss_val ** 2 / y_test_log.var(ddof=0)
    arbres = np.arange(1, len(loss_train) + 1)
    bleu, orange = "#1971C2", "#B4521A"

    graphes = [
        ("Loss : RMSE sur le log du montant", "RMSE", "loss", loss_train, loss_val),
        ("Équivalent de l'accuracy : R² sur le log du montant", "R²", "accuracy", r2_train, r2_val),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (titre, unite, metrique, train, val) in zip(axes, graphes):
        for nom, valeurs, couleur in [(f"training_{metrique}", train, bleu), (f"validation_{metrique}", val, orange)]:
            ax.plot(arbres, valeurs, color=couleur, linewidth=2, label=nom)
            autre = val if valeurs is train else train
            decalage = 6 if valeurs[-1] >= autre[-1] else -6  # évite que les deux étiquettes se chevauchent
            ax.annotate(f"{valeurs[-1]:.3f}", (arbres[-1], valeurs[-1]), xytext=(6, decalage), textcoords="offset points", va="center")
        ax.set_xlim(1, arbres[-1] * 1.12)
        ax.set_title(titre)
        ax.set_xlabel("Nombre d'arbres")
        ax.set_ylabel(unite)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(chemin, dpi=130)
    plt.close(fig)


def evaluer_80_20():
    """Entraîne sur 80 % des sinistrés et évalue sur les 20 % restants, avec tous les apprentissages faits sur les 80 %."""
    train, _ = charger_donnees()
    train = nettoyer(train)
    entrainement, test_interne = train_test_split(train[train[CIBLE] > 0], test_size=0.2, random_state=42)

    variables = preselectionner(entrainement)
    x_train, x_test = entrainement[variables], test_interne[variables]
    y_train_log, y_test_log = np.log(entrainement[CIBLE]), np.log(test_interne[CIBLE])
    correction = calculer_correction(x_train, y_train_log)

    encodeur = creer_encodeur(x_train)
    x_train_encode, x_test_encode = encodeur.fit_transform(x_train), encodeur.transform(x_test)
    modele = XGBRegressor(**PARAMETRES).fit(
        x_train_encode, y_train_log, eval_set=[(x_train_encode, y_train_log), (x_test_encode, y_test_log)], verbose=False
    )
    tracer_courbes(modele.evals_result(), y_train_log, y_test_log)

    y_test = test_interne[CIBLE]
    log_predit = modele.predict(x_test_encode)

    meilleurs_parametres = optimiser(x_train, y_train_log)
    correction_v3 = calculer_correction(x_train, y_train_log, meilleurs_parametres)
    modele_v3 = XGBRegressor(**meilleurs_parametres).fit(
        x_train_encode, y_train_log, eval_set=[(x_train_encode, y_train_log), (x_test_encode, y_test_log)], verbose=False
    )
    tracer_courbes(modele_v3.evals_result(), y_train_log, y_test_log, GRAPHE_LOSS_V3)
    log_predit_v3 = modele_v3.predict(x_test_encode)

    parametres_v5 = optimiser_aleatoire(x_train, y_train_log)
    correction_v5 = calculer_correction(x_train, y_train_log, parametres_v5)
    modele_v5 = XGBRegressor(**parametres_v5).fit(
        x_train_encode, y_train_log, eval_set=[(x_train_encode, y_train_log), (x_test_encode, y_test_log)], verbose=False
    )
    tracer_courbes(modele_v5.evals_result(), y_train_log, y_test_log, GRAPHE_LOSS_V5)
    log_predit_v5 = modele_v5.predict(x_test_encode)

    parametres_v6 = optimiser_aleatoire(x_train, y_train_log, DISTRIBUTIONS_V6)
    correction_v6 = calculer_correction(x_train, y_train_log, parametres_v6)
    modele_v6 = XGBRegressor(**parametres_v6).fit(
        x_train_encode, y_train_log, eval_set=[(x_train_encode, y_train_log), (x_test_encode, y_test_log)], verbose=False
    )
    tracer_courbes(modele_v6.evals_result(), y_train_log, y_test_log, GRAPHE_LOSS_V6)
    log_predit_v6 = modele_v6.predict(x_test_encode)

    predictions = {
        "Baseline moyenne": np.full(len(y_test), entrainement[CIBLE].mean()),
        "Baseline médiane": np.full(len(y_test), entrainement[CIBLE].median()),
        "Version 1 : XGBoost sans correction": retour_en_euros(log_predit),
        "Version 2 : XGBoost avec correction": retour_en_euros(log_predit, correction),
        "Version 3 : hyperparamètres optimisés, sans correction": retour_en_euros(log_predit_v3),
        "Version 3 : hyperparamètres optimisés, avec correction": retour_en_euros(log_predit_v3, correction_v3),
        "Version 5 : recherche aléatoire, sans correction": retour_en_euros(log_predit_v5),
        "Version 5 : recherche aléatoire, avec correction": retour_en_euros(log_predit_v5, correction_v5),
        "Version 6 : plages élargies, sans correction": retour_en_euros(log_predit_v6),
        "Version 6 : plages élargies, avec correction": retour_en_euros(log_predit_v6, correction_v6),
    }
    return pd.DataFrame({
        nom: {"MAE": mean_absolute_error(y_test, p), "RMSE": root_mean_squared_error(y_test, p), "total_predit / total_reel": p.sum() / y_test.sum()}
        for nom, p in predictions.items()
    }).T, len(entrainement), len(test_interne), meilleurs_parametres, parametres_v5, parametres_v6


if __name__ == "__main__":
    scores, n_entrainement, n_test, parametres_v3, parametres_v5, meilleurs_parametres = evaluer_80_20()
    print("Hyperparamètres version 3 :", {nom: parametres_v3[nom] for nom in GRILLE})
    print("Hyperparamètres version 5 :", {nom: parametres_v5[nom] for nom in GRILLE})
    print("Hyperparamètres version 6 :", {nom: meilleurs_parametres[nom] for nom in GRILLE})
    print(f"Entraînement : {n_entrainement} sinistrés | Test interne : {n_test} sinistrés")
    print(scores.round(3).to_string())
    print("Graphe des loss écrit dans", GRAPHE_LOSS)

    x_train, y_train_log, x_test, index_test = preparer()

    correction = calculer_correction(x_train, y_train_log, meilleurs_parametres)
    modele = creer_modele(x_train, meilleurs_parametres).fit(x_train, y_train_log)
    predictions = retour_en_euros(modele.predict(x_test), correction)

    pd.DataFrame({"index": index_test, "montant_sinistre": predictions}).to_csv(SORTIE, index=False)

    print("Facteur de correction :", round(correction, 4))
    print("Montant prédit moyen :", round(predictions.mean(), 2))
    print("Prédictions écrites dans", SORTIE)
