"""Prépare les données du modèle de gravité : nettoyage, puis présélection des variables."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]

CIBLE = "montant_sinistre"
SEUIL_P = 0.20

# sex_conducteur1 et sex_conducteur2 sont exclues : interdites pour la tarification dans l'UE depuis 2012
COLONNES_RETIREES = ["sex_conducteur1", "sex_conducteur2", "index", "id_client", "id_contrat", "id_vehicule"]

CANDIDATES_QUALI = [
    "type_contrat", "freq_paiement", "paiement", "utilisation", "conducteur2", "essence_vehicule",
    "type_vehicule", "marque_vehicule", "modele_vehicule", "departement",
]
CANDIDATES_QUANTI = [
    "bonus", "duree_contrat", "anciennete_info", "age_conducteur1", "age_conducteur2",
    "anciennete_permis1", "anciennete_permis2", "anciennete_vehicule", "cylindre_vehicule",
    "din_vehicule", "debut_vente_vehicule", "fin_vente_vehicule", "vitesse_vehicule",
    "prix_vehicule", "poids_vehicule",
]


def charger_donnees():
    types = {"code_postal": str, "modele_vehicule": str}
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype=types)
    test = pd.read_csv(ROOT / "data" / "test.csv", dtype=types)
    return train, test


def nettoyer(df):
    df = df.drop(columns=COLONNES_RETIREES, errors="ignore")

    # Pas de second conducteur : l'âge et l'ancienneté de permis n'existent pas
    df.loc[df["conducteur2"] == "No", ["age_conducteur2", "anciennete_permis2"]] = np.nan

    # Valeurs impossibles
    for col in ["anciennete_permis1", "anciennete_permis2"]:
        df[col] = df[col].replace(111, np.nan)
    for col in ["poids_vehicule", "cylindre_vehicule"]:
        df[col] = df[col].replace(0, np.nan)

    return df.assign(departement=df["code_postal"].str.zfill(5).str[:2])


def p_kruskal(df, variable):
    groupes = [g.values for _, g in df.groupby(variable)[CIBLE] if len(g) >= 2]
    return stats.kruskal(*groupes).pvalue


def p_spearman(df, variable):
    donnees = df[[variable, CIBLE]].dropna()
    return stats.spearmanr(donnees[variable], donnees[CIBLE]).pvalue


def preselectionner(df_sinistres, seuil=SEUIL_P):
    """Garde les variables dont la p-value avec le montant est sous le seuil."""
    p_values = {var: p_kruskal(df_sinistres, var) for var in CANDIDATES_QUALI}
    p_values |= {var: p_spearman(df_sinistres, var) for var in CANDIDATES_QUANTI}
    return [var for var, p in p_values.items() if p < seuil]


def creer_encodeur(x):
    """Encodage des variables qualitatives, à apprendre sur le jeu d'entraînement seulement.

    Chaque classe reçoit un nombre entier, dans une seule colonne. Une classe jamais vue à l'entraînement reçoit -1.
    Les variables quantitatives passent sans changement.
    """
    colonnes_quali = x.select_dtypes(exclude="number").columns.tolist()
    encodeur_quali = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer([("quali", encodeur_quali, colonnes_quali)], remainder="passthrough", verbose_feature_names_out=False)


def retour_en_euros(y_log, correction=1.0):
    """Repasse une prédiction en log à l'échelle d'origine. `correction` compense le biais de exp()."""
    return np.exp(y_log) * correction


def preparer():
    train, test = charger_donnees()
    index_test = test["index"]  # conservé pour la soumission Kaggle

    train = nettoyer(train)
    test = nettoyer(test)

    # Le modèle s'entraîne sur les sinistrés, mais prédit toutes les lignes du jeu test
    sinistres = train[train[CIBLE] > 0]
    variables = preselectionner(sinistres)

    # Le montant est très asymétrique : le modèle apprend son logarithme
    y_train_log = np.log(sinistres[CIBLE])

    return sinistres[variables], y_train_log, test[variables], index_test


if __name__ == "__main__":
    x_train, y_train_log, x_test, index_test = preparer()
    print("x_train :", x_train.shape, "| y_train_log :", y_train_log.shape, "| x_test :", x_test.shape)
    print("Asymétrie de la cible en log :", round(y_train_log.skew(), 2))
    print("Variables gardées :", x_train.columns.tolist())

    encodeur = creer_encodeur(x_train)
    x_train_encode = encodeur.fit_transform(x_train)
    x_test_encode = encodeur.transform(x_test)
    print("Après encodage, x_train :", x_train_encode.shape, "| x_test :", x_test_encode.shape)
