
"""Prétraitement du modèle de fréquence automobile.

Version figée pour le groupe de variables « etendu » (46 variables).
Les statistiques d'imputation et l'encodage sont appris lors du fit
et conservés dans le pipeline sauvegardé avec le modèle.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

EXCLUDE = [
    'index', 'id_client', 'id_vehicule', 'id_contrat', 'code_postal',
    'nombre_sinistres', 'montant_sinistre',
]

# Champs de description nécessaires pour créer les 46 variables.
REQUIRED = [
    'bonus', 'type_contrat', 'duree_contrat', 'anciennete_info',
    'freq_paiement', 'paiement', 'utilisation', 'code_postal', 'conducteur2',
    'age_conducteur1', 'age_conducteur2', 'sex_conducteur1',
    'sex_conducteur2', 'anciennete_permis1', 'anciennete_permis2',
    'anciennete_vehicule', 'cylindre_vehicule', 'din_vehicule',
    'essence_vehicule', 'marque_vehicule', 'modele_vehicule',
    'debut_vente_vehicule', 'fin_vente_vehicule', 'vitesse_vehicule',
    'type_vehicule', 'prix_vehicule', 'poids_vehicule',
]

NUMERIC = [
    'bonus', 'duree_contrat', 'anciennete_info',
    'age_conducteur1', 'age_conducteur2',
    'anciennete_permis1', 'anciennete_permis2',
    'anciennete_vehicule', 'cylindre_vehicule', 'din_vehicule',
    'debut_vente_vehicule', 'fin_vente_vehicule', 'vitesse_vehicule',
    'prix_vehicule', 'poids_vehicule',
]


def make_features(df):
    """Ajoute des informations métier sans apprendre de statistiques globales.

    Toutes les opérations ici s'appliquent indépendamment à chaque ligne :
    pas de médiane calculée sur tout le dataset avant validation.
    Les imputations apprises sur Train viendront dans le pipeline.
    """
    d = df.copy()  # Préserver le DataFrame brut.

    # Nettoyer les espaces superflus dans toutes les colonnes texte.
    for col in d.select_dtypes(include=['object', 'string']).columns:
        d[col] = d[col].astype('string').str.strip()

    # Standardiser YES/NO et identifier la présence d'un second conducteur.
    d['conducteur2'] = d['conducteur2'].str.upper().str.strip()
    present = d['conducteur2'].eq('YES').fillna(False)
    absent = d['conducteur2'].eq('NO').fillna(False)

    # Garantir que le sexe du conducteur 2 est une variable texte
    d['sex_conducteur2'] = d['sex_conducteur2'].astype('string')

    d.loc[absent, 'sex_conducteur2'] = 'Aucun'
    d['sex_conducteur2'] = d['sex_conducteur2'].fillna('Inconnu')
    
        # Même nettoyage du permis pour les conducteurs 1 et 2.
    for k in (1, 2):
        col = f'anciennete_permis{k}'  # Exemple : anciennete_permis1.
        age = f'age_conducteur{k}'     # Exemple : age_conducteur1.

        # La valeur 111 est suspecte : on mémorise sa présence avant suppression.
        d[col + '_code111'] = d[col].eq(111).astype('int8')
        d[col] = d[col].replace(111, np.nan)

        # Si âge - ancienneté < 15, l'âge estimé d'obtention est suspect.
        bad = d[col].notna() & (d[age] - d[col] < 15)
        if k == 2:
            bad &= present  # Ne contrôler le permis 2 que si conducteur 2 existe.
        d[f'permis{k}_incoherent'] = bad.astype('int8')
        d.loc[bad, col] = np.nan

        # Nouvelle feature : âge estimé d'obtention du permis.
        # Pour le conducteur 2 absent, le résultat n'a pas de sens => NaN.
        d[f'age_obtention_permis{k}'] = (
            (d[age] - d[col]).where(present) if k == 2 else (d[age] - d[col])
        )

    # Transformer la présence du conducteur 2 en nombre 0/1.
    d['conducteur2_present'] = present.astype('int8')

    # Différence d'âge en valeur absolue ; NaN si pas de 2e conducteur.
    d['ecart_age_conducteurs'] = (
        d['age_conducteur1'] - d['age_conducteur2']
    ).abs().where(present)

    # Puissance DIN par tonne (si poids en kilogrammes).
    # Les zéros au dénominateur deviennent NaN pour éviter une division par zéro.
    d['rapport_puissance_poids'] = (
        1000 * d['din_vehicule'] / d['poids_vehicule'].replace(0, np.nan)
    )
    d['prix_par_cheval'] = (
        d['prix_vehicule'] / d['din_vehicule'].replace(0, np.nan)
    )

    # Les codes sont géographiques (probablement codes INSEE, pas simplement CP).
    cp = d['code_postal'].astype('string').str.upper().str.strip().str.replace(
        r'\.0$', '', regex=True
    )
    # Compléter les codes numériques courts : « 1001 » devient « 01001 ».
    cp = cp.where(
        ~(cp.str.isnumeric().fillna(False) & cp.str.len().lt(5)),
        cp.str.zfill(5),
    )
    dept = cp.str[:2]  # Métropole : deux premiers caractères, dont 2A/2B.
    overseas = cp.str.startswith(('97', '98'), na=False)
    # Outre-mer : trois premiers caractères (971, 972, ...).
    d['departement'] = dept.mask(overseas, cp.str[:3]).fillna('Inconnu')

    # Clé catégorielle véhicule ; ex. « RENAULT|CLIO ».
    d['marque_modele'] = (
        d['marque_vehicule'].fillna('Inconnu').str.upper() + '|'
        + d['modele_vehicule'].fillna('Inconnu').str.upper()
    )

    # Âge minimum des conducteurs du contrat.
    # Sans 2e conducteur, utiliser l'âge du conducteur 1.
    d['age_min_conducteurs'] = np.where(
        present,
        np.minimum(d['age_conducteur1'], d['age_conducteur2']),
        d['age_conducteur1'],
    )

    # Ancienneté minimum des permis présents.
    # np.fmin ignore un NaN quand l'autre valeur existe.
    d['anciennete_min_permis'] = np.where(
        present,
        np.fmin(d['anciennete_permis1'], d['anciennete_permis2']),
        d['anciennete_permis1'],
    )

    # Autres candidats : véhicule ancien, transformation de prix, ratios, interactions.
    d['age_vehicule_collection'] = d['anciennete_vehicule'].ge(30).astype('int8')
    d['log_prix_vehicule'] = np.log1p(d['prix_vehicule'].where(d['prix_vehicule'].ge(0)))
    d['prix_par_poids'] = d['prix_vehicule'] / d['poids_vehicule'].replace(0, np.nan)
    d['din_par_cylindre'] = d['din_vehicule'] / d['cylindre_vehicule'].replace(0, np.nan)
    d['age1_x_din'] = d['age_conducteur1'] * d['din_vehicule']
    d['duree_par_info'] = d['duree_contrat'] / d['anciennete_info'].replace(0, np.nan)

    # XGBoost/pipeline gèrent les NaN, mais pas les infinis.
    return d.replace([np.inf, -np.inf], np.nan)


def x_for(df):
    """Conserve les 46 caractéristiques du groupe « etendu ».

    La colonne cible, les identifiants et le code géographique brut
    sont exclus. On garde le département construit dans make_features.
    """
    x = df.drop(columns=EXCLUDE, errors='ignore').copy()

    # Conversion compatible avec SimpleImputer et OneHotEncoder.
    for c in x.select_dtypes(exclude='number').columns:
        x[c] = x[c].astype('object').where(x[c].notna(), np.nan)

    return x


def preprocessor(x, min_frequency=60, max_categories=150):
    """Prépare les deux sortes de colonnes avant l'apprentissage.

    Numériques   : médiane + indicateur de valeur manquante.
    Catégorielles: catégorie « Inconnu » + encodage One-Hot.
    Le fit de ces transformations n'a lieu que sur les lignes d'entraînement
    du fold grâce à Pipeline : la validation ne fuit pas dans les médianes.
    """
    numeric = x.select_dtypes(include='number').columns.tolist()
    categorical = x.select_dtypes(exclude='number').columns.tolist()

    return ColumnTransformer([
        # Ex. NaN d'ancienneté -> médiane du Train de ce fold.
        ('num', SimpleImputer(
            strategy='median',
            add_indicator=True,
            keep_empty_features=True,
        ), numeric),
        # Ex. « DIESEL » -> colonne binaire DIESEL=1, autres=0.
        ('cat', Pipeline([
            ('fill', SimpleImputer(
                strategy='constant', fill_value='Inconnu', keep_empty_features=True,
            )),
            ('ohe', OneHotEncoder(
                handle_unknown='infrequent_if_exist',
                min_frequency=int(min_frequency),
                max_categories=int(max_categories),
            )),
        ]), categorical),
    ])


def prepare_contract(contract):
    """Convertit un contrat JSON (dict) en 46 variables non encore imputées.

    Les contrôles de types et de valeurs métier seront aussi réalisés par
    le schéma d'entrée de l'API. Ici, on garantit les types de colonnes
    nécessaires à la transformation, même quand une valeur vaut null.
    """
    if not isinstance(contract, dict):
        raise TypeError('Le contrat doit être un dictionnaire JSON')

    missing = sorted(set(REQUIRED) - set(contract))
    if missing:
        raise ValueError(f'Champs manquants : {missing}')

    df = pd.DataFrame([contract])
    df['code_postal'] = df['code_postal'].astype('string')

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='raise').astype(float)

    result = x_for(make_features(df))
    if result.shape[1] != 46:
        raise ValueError(f'46 variables attendues, {result.shape[1]} trouvées')
    return result
