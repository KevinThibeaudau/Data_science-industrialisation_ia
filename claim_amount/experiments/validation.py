"""Validation croisée à 5 plis du modèle de gravité : découpage, baselines et stabilité de la présélection."""
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold

from preprocess import CIBLE, charger_donnees, nettoyer, preselectionner

N_PLIS = 5
GRAINE = 42


def charger_sinistres():
    train, _ = charger_donnees()
    train = nettoyer(train)
    return train[train[CIBLE] > 0].reset_index(drop=True)


def creer_plis(sinistres):
    return list(KFold(n_splits=N_PLIS, shuffle=True, random_state=GRAINE).split(sinistres))


def evaluer_baseline(sinistres, plis, statistique):
    """Prédit la moyenne ou la médiane du montant apprise sur la partie entraînement de chaque pli."""
    resultats = []
    for idx_train, idx_val in plis:
        y_train, y_val = sinistres[CIBLE].iloc[idx_train], sinistres[CIBLE].iloc[idx_val]
        prediction = y_train.agg(statistique)
        resultats.append({
            "MAE_train": mean_absolute_error(y_train, [prediction] * len(y_train)),
            "MAE_val": mean_absolute_error(y_val, [prediction] * len(y_val)),
            "RMSE_train": root_mean_squared_error(y_train, [prediction] * len(y_train)),
            "RMSE_val": root_mean_squared_error(y_val, [prediction] * len(y_val)),
        })
    return pd.DataFrame(resultats)


def stabilite_preselection(sinistres, plis):
    """Compte dans combien de plis chaque variable passe la présélection, apprise sur la partie entraînement."""
    selections = [preselectionner(sinistres.iloc[idx_train]) for idx_train, _ in plis]
    return pd.Series([var for selection in selections for var in selection]).value_counts()


if __name__ == "__main__":
    sinistres = charger_sinistres()
    plis = creer_plis(sinistres)
    print("Sinistrés :", len(sinistres), "| Taille des plis de validation :", [len(idx_val) for _, idx_val in plis])

    for statistique in ["mean", "median"]:
        scores = evaluer_baseline(sinistres, plis, statistique)
        print(f"\nBaseline {statistique} (en euros), moyenne et écart-type sur {N_PLIS} plis :")
        print(pd.DataFrame({"moyenne": scores.mean(), "ecart_type": scores.std()}).round(1).to_string())

    print(f"\nNombre de plis où chaque variable est présélectionnée (sur {N_PLIS}) :")
    print(stabilite_preselection(sinistres, plis).to_string())
