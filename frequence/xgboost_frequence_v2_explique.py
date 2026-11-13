"""
XGBOOST FRÉQUENCE V2 — VERSION PÉDAGOGIQUE
========================================

QUESTION : pour chaque contrat, quelle est la probabilité d'au moins un sinistre ?
CIBLE    : nombre_sinistres, avec 0 = aucun sinistre, 1 = sinistre.
SORTIE   : resultats_frequence_v2/soumission_frequence_xgboost.csv
           Deux colonnes : INDEX,pred (pred est une probabilité entre 0 et 1).

Ce fichier conserve les mêmes opérations statistiques, paramètres, algorithmes
et fichiers de sortie que xgboost_frequence_v2.py.
Il est simplement réorganisé visuellement et commenté pour l'apprentissage.

PLAN :
    1. Définir les variables utilisables.
    2. Lire et contrôler les CSV.
    3. Construire les variables supplémentaires (feature engineering).
    4. Créer un pipeline : imputations + One-Hot + XGBoost.
    5. Réserver ~20 % des contrats comme test interne (holdout).
    6. Faire une validation croisée par client sur les ~80 % restants.
    7. Comparer les groupes de variables et optimiser avec Optuna.
    8. Calibrer les probabilités si cela améliore la Log Loss.
    9. Mesurer UNE FOIS la performance du test interne.
   10. Réentraîner sur toutes les données étiquetées et prédire test.csv.

IMPORTANT :
    - Le test interne est extrait de train.csv ; les vraies réponses y existent.
    - test.csv est le jeu EXTERNE sans réponses, utilisé pour la soumission.
    - On ne change PAS les hyperparamètres après avoir vu le test interne.
    - Pas de taux annuel sans durée d'exposition connue : la cible est binaire.

Commandes (depuis le dossier du projet contenant data/) :
    python xgboost_frequence_v2_explique.py --trials 2 --folds 3 --threads 4
    python xgboost_frequence_v2_explique.py --trials 80 --folds 5 --threads 4
"""

from __future__ import annotations  # Autorise notamment l'évaluation différée des annotations.

# Bibliothèques standard : arguments terminal, JSON, chemins de fichiers.
import argparse
import json
from pathlib import Path

# Données, entraînement, optimisation, métriques et sauvegarde.
import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# =============================================================================
# 1 — COLONNES ET PARAMÈTRES FIXES
# =============================================================================

# Graine aléatoire : rend les essais reproductibles à environnement identique.
SEED = 42

# Colonnes qui DOIVENT exister dans train.csv et test.csv.
# Ici, on ne liste pas les identifiants supplémentaires présents seulement en Train.
REQUIRED = [
    'index', 'bonus', 'type_contrat', 'duree_contrat', 'anciennete_info',
    'freq_paiement', 'paiement', 'utilisation', 'code_postal', 'conducteur2',
    'age_conducteur1', 'age_conducteur2', 'sex_conducteur1',
    'sex_conducteur2', 'anciennete_permis1', 'anciennete_permis2',
    'anciennete_vehicule', 'cylindre_vehicule', 'din_vehicule',
    'essence_vehicule', 'marque_vehicule', 'modele_vehicule',
    'debut_vente_vehicule', 'fin_vente_vehicule', 'vitesse_vehicule',
    'type_vehicule', 'prix_vehicule', 'poids_vehicule',
]

# Ces colonnes sont EXCLUES du modèle (mais certaines servent aux contrôles).
# nombre_sinistres = la cible ; montant_sinistre = une autre réponse, interdite en X.
# id_client = sert à séparer les clients, mais ne peut pas servir de feature.
EXCLUDE = [
    'index', 'id_client', 'id_vehicule', 'id_contrat', 'code_postal',
    'nombre_sinistres', 'montant_sinistre',
]

# Quatre variables supplémentaires de notre première sélection.
SELECTED = [
    'age_obtention_permis1', 'ecart_age_conducteurs',
    'prix_par_cheval', 'departement',
]

# Cinq autres variables candidates.
EXTRA = [
    'age_obtention_permis2', 'rapport_puissance_poids', 'conducteur2_present',
    'anciennete_permis1_code111', 'anciennete_permis2_code111',
]

# Onze variables supplémentaires ; elles ne sont PAS toutes automatiquement retenues.
EXTENDED = [
    'marque_modele', 'age_min_conducteurs', 'anciennete_min_permis',
    'age_vehicule_collection', 'log_prix_vehicule', 'prix_par_poids',
    'din_par_cylindre', 'age1_x_din', 'duree_par_info',
    'permis1_incoherent', 'permis2_incoherent',
]

# Les quatre expériences que le programme va comparer : 26, 30, 35 ou 46 features.
FEATURE_SETS = {
    'base': [],
    'selection': SELECTED,
    'complet': SELECTED + EXTRA,
    'etendu': SELECTED + EXTRA + EXTENDED,
}

# Première configuration raisonnable avant l'optimisation Optuna.
# Les valeurs ne sont PAS garanties optimales ; elles servent de point de départ.
DEFAULT = dict(
    n_estimators=350,          # Nombre d'arbres successifs.
    max_depth=3,              # Profondeur maximale de chaque arbre.
    learning_rate=0.045,      # Contribution de chaque nouvel arbre.
    min_child_weight=5,       # Contraint la création des feuilles de l'arbre.
    subsample=0.85,           # Fraction des contrats utilisée par arbre.
    colsample_bytree=0.85,    # Fraction des variables utilisée par arbre.
    reg_lambda=5.0,           # Régularisation L2.
    reg_alpha=0.0,            # Régularisation L1.
    gamma=0.0,                # Gain minimal pour autoriser une séparation.
    min_frequency=20,         # Regroupement des catégories rares (One-Hot).
    max_categories=70,        # Nombre maximal de modalités encodées.
)


# =============================================================================
# 2 — LECTURE ET CONTRÔLE DES DONNÉES
# =============================================================================

def read_data(train_path, test_path):
    """Charge les CSV et refuse toute erreur de schéma ou de cible.

    Retour : (train, test), avec un index Pandas propre (0, 1, 2, ...).
    Rien n'est encore entraîné. Le vrai test.csv ne contient pas la réponse.
    """
    # Le code géographique doit rester en texte : 01001, 2A004, 97101, etc.
    train = pd.read_csv(train_path, dtype={'code_postal': 'string'}, low_memory=False)
    test = pd.read_csv(test_path, dtype={'code_postal': 'string'}, low_memory=False)

    # Contrôles communs aux deux fichiers.
    for name, df in [('Train', train), ('Test', test)]:
        missing = sorted(set(REQUIRED) - set(df))  # Colonnes attendues non trouvées.
        if missing:
            raise ValueError(f'{name}: variables manquantes : {missing}')
        # INDEX doit exister pour chaque contrat et être unique dans son fichier.
        if df['index'].isna().any() or df['index'].duplicated().any():
            raise ValueError(f'{name}: INDEX manquant ou duplique')

    # Ces trois colonnes sont indispensables dans train.csv uniquement.
    for col in ('id_client', 'nombre_sinistres', 'montant_sinistre'):
        if col not in train:
            raise ValueError(f'Train: colonne absente {col}')

    # Impossible de garantir la séparation par client si l'identifiant manque.
    if train['id_client'].isna().any():
        raise ValueError('id_client manquant: pas de validation par client')

    # La fréquence est une CLASSIFICATION 0/1 ; ni 2 ni NaN ne sont acceptés.
    if train['nombre_sinistres'].isna().any() or not train['nombre_sinistres'].isin([0, 1]).all():
        raise ValueError('nombre_sinistres doit etre 0/1 sans NaN')

    # Garde-fou contre des fichiers vides ou trop petits.
    if len(train) < 100 or len(test) < 1:
        raise ValueError('Taille de donnees insuffisante')

    # Les colonnes de description communes doivent correspondre entre Train et Test.
    shared = set(train) - {
        'id_client', 'id_vehicule', 'id_contrat',
        'nombre_sinistres', 'montant_sinistre',
    }
    if shared != set(test):
        raise ValueError(f'Schema Train/Test different : {sorted(shared ^ set(test))}')

    # On ne supprime aucune observation ; on réinitialise seulement l'index Pandas.
    return train.reset_index(drop=True), test.reset_index(drop=True)


# =============================================================================
# 3 — FEATURE ENGINEERING (MÊME TRANSFORMATION POUR TRAIN ET TEST)
# =============================================================================

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

    # « Aucun » = pas de second conducteur ; « Inconnu » = donnée manquante.
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


def x_for(df, features):
    """Choisit l'un des quatre groupes de features à évaluer.

    « base » : seulement les 26 colonnes explicatives d'origine.
    « selection », « complet », « etendu » : ajout progressif de candidats.
    """
    x = df.drop(columns=EXCLUDE, errors='ignore').copy()

    # On retire toutes les nouvelles features non prévues pour le groupe choisi.
    unused = set(SELECTED + EXTRA + EXTENDED) - set(FEATURE_SETS[features])
    x = x.drop(columns=list(unused), errors='ignore')

    # Les colonnes catégorielles sont converties en objet/NaN compatible avec
    # SimpleImputer + OneHotEncoder. Cela ne change pas les valeurs apprises.
    for c in x.select_dtypes(exclude='number').columns:
        x[c] = x[c].astype('object').where(x[c].notna(), np.nan)

    return x


# =============================================================================
# 4 — CONSTRUIRE UN PIPELINE DE PRÉTRAITEMENT ET XGBOOST
# =============================================================================

def preprocessor(x, min_frequency, max_categories):
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


def make_model(x, params, threads, seed):
    """Assemble PRÉTRAITEMENT + CLASSIFIEUR dans un seul Pipeline.

    model.fit(X, y) apprend le prétraitement ET XGBoost sur X et y.
    model.predict_proba(X) retourne P(Y=0) et P(Y=1).
    """
    # Ces deux paramètres concernent le One-Hot, pas XGBoost.
    p = {k: v for k, v in params.items() if k not in ('min_frequency', 'max_categories')}

    clf = XGBClassifier(
        **p,                         # Paramètres optimisés (arbres, profondeur, etc.).
        objective='binary:logistic',# Classification binaire avec probabilités.
        eval_metric='logloss',      # Fonction d'évaluation interne du classifieur.
        tree_method='hist',         # Construction rapide des arbres par histogrammes.
        random_state=seed,          # Reproductibilité.
        n_jobs=threads,             # Nombre de threads CPU.
        scale_pos_weight=1.0,       # Ne pas rééquilibrer artificiellement la cible.
    )

    return Pipeline([
        ('preprocess', preprocessor(x, params['min_frequency'], params['max_categories'])),
        ('xgb', clf),
    ])


# =============================================================================
# 5 — CRÉER DES ENSEMBLES INDÉPENDANTS PAR CLIENT
# =============================================================================

def holdout_indices(train, seed):
    """Réserve environ 20 % de train.csv comme test interne indépendant.

    IMPORTANT : deux contrats du même client ne sont jamais des deux côtés.
    Le holdout est utilisé seulement APRÈS la sélection des hyperparamètres.
    """
    y = train['nombre_sinistres'].astype(int)
    groups = train['id_client']

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, hold = next(cv.split(train, y, groups))  # Premier des cinq découpages.

    # Contrôle : aucun identifiant client commun aux deux côtés.
    if set(groups.iloc[dev]) & set(groups.iloc[hold]):
        raise AssertionError('Fuite de client dans holdout')

    # ROC-AUC/PR-AUC nécessitent une représentation des deux classes.
    if y.iloc[dev].nunique() < 2 or y.iloc[hold].nunique() < 2:
        raise ValueError('Repartition sans deux classes')

    return dev, hold


def client_folds(y, groups, n, seed):
    """Découpe le développement en n folds : chacun sert une fois de validation.

    Exemple n=5 : 4/5 servent à entraîner, 1/5 à valider, à chaque tour.
    On garde tous les contrats d'un même client dans un seul côté du tour.
    """
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(y)), y, groups))

    for tr, va in splits:
        if set(groups.iloc[tr]) & set(groups.iloc[va]):
            raise AssertionError('Fuite client dans cross validation')
        if pd.Series(y).iloc[tr].nunique() < 2 or pd.Series(y).iloc[va].nunique() < 2:
            raise ValueError('Fold mono-classe, reduire --folds')

    return splits


def cv_pred(x, y, splits, params, threads, seed):
    """Produit une prédiction OOF pour chaque contrat du développement.

    OOF = Out-Of-Fold : la prédiction d'un contrat vient toujours d'un modèle
    qui n'a pas été entraîné avec ce contrat.

    Retour : (toutes_les_probabilites, logloss_OOF, dispersion_entre_folds).
    """
    result = np.full(len(y), np.nan)  # Une case vide par contrat de développement.
    per_fold = []                     # Log Loss calculée sur chaque fold séparément.

    for a, b in splits:
        model = make_model(x, params, threads, seed)
        model.fit(x.iloc[a], y.iloc[a])       # APPRENDRE sur les lignes a.
        p = model.predict_proba(x.iloc[b])[:, 1]  # PRÉDIRE P(sinistre) sur les lignes b.
        result[b] = p                        # Enregistrer la prédiction à la bonne place.
        per_fold.append(log_loss(y.iloc[b], p, labels=[0, 1]))

    # Chaque contrat doit avoir été prédit hors apprentissage exactement une fois.
    if not np.isfinite(result).all():
        raise ValueError('Probabilites OOF incompletes')

    # Le score global agrège TOUS les contrats, sans donner le même poids à
    # deux folds de tailles différentes. Plus la Log Loss est petite, mieux c'est.
    return (
        result,
        float(log_loss(y, result, labels=[0, 1])),
        float(np.std(per_fold)),
    )


def baseline_oof(y, splits):
    """Référence simple : prédire uniquement le taux moyen de sinistres du fold Train.

    XGBoost doit battre cette référence pour justifier sa complexité.
    """
    result = np.full(len(y), np.nan)
    for a, b in splits:
        result[b] = float(y.iloc[a].mean())  # Ex. prédire 5,8 % pour tous les contrats.
    return result


# =============================================================================
# 6 — OPTUNA : ESSAYER PLUSIEURS CONFIGURATIONS DE XGBOOST
# =============================================================================

def param_search(trial):
    """Définit les plages d'hyperparamètres autorisées pour un essai Optuna.

    Optuna propose une configuration, cv_pred mesure sa Log Loss ; Optuna
    utilise les résultats des essais passés pour en proposer de nouveaux.
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 150, 950, step=50),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.012, 0.13, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 30),
        'subsample': trial.suggest_float('subsample', 0.65, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 80.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 4.0),
        'min_frequency': trial.suggest_categorical('min_frequency', [5, 20, 60]),
        'max_categories': trial.suggest_categorical('max_categories', [30, 70, 150]),
    }


# =============================================================================
# 7 — CALIBRATION : AJUSTER LES PROBABILITÉS SI NÉCESSAIRE
# =============================================================================

def fit_calibrator(kind, p, y):
    """Entraîne un calibrateur sur des probabilités et les vraies réponses.

    - sigmoid  : régression logistique sur le logit(p).
    - isotonic : transformation croissante flexible.
    - aucune  : ne rien entraîner.
    """
    if kind == 'sigmoid':
        fit = LogisticRegression(C=5.0, max_iter=1000)
        fit.fit(logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1, 1), y)
        return fit

    if kind == 'isotonic':
        fit = IsotonicRegression(y_min=1e-6, y_max=1-1e-6, out_of_bounds='clip')
        fit.fit(p, y)
        return fit

    return None


def apply_calibrator(kind, cal, p):
    """Applique au tableau de probabilités un calibrateur déjà entraîné."""
    if kind == 'sigmoid':
        return cal.predict_proba(logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1]
    if kind == 'isotonic':
        return cal.predict(p)
    return p  # Type « aucune » : ne change PAS la probabilité.


def crossfit_calibration(raw, y, splits, kind):
    """Compare la calibration sur des prédictions OOF du développement.

    Chaque calibrateur est ajusté sur les probabilités du reste des folds,
    puis appliqué au fold tenu de côté : limite l'optimisme de calibration.
    """
    if kind == 'aucune':
        return raw  # Calibration absente : conserver les probabilités brutes.

    out = np.full(len(raw), np.nan)
    for a, b in splits:
        cal = fit_calibrator(kind, raw[a], y.iloc[a])
        out[b] = apply_calibrator(kind, cal, raw[b])

    # Évite log(0) lors du calcul de la Log Loss.
    return np.clip(out, 1e-6, 1-1e-6)


# =============================================================================
# 8 — MÉTRIQUES ET FICHIERS DE SORTIE
# =============================================================================

def report_metrics(name, y, p):
    """Calcule les principaux scores de fréquence sur des données étiquetées.

    - ROC-AUC : classe-t-on bien les contrats les plus risqués ? (plus grand = mieux)
    - PR-AUC  : qualité du classement des sinistres rares (plus grand = mieux)
    - Log Loss : qualité des probabilités (plus petit = mieux)
    - Brier    : erreur quadratique des probabilités (plus petit = mieux)
    """
    return {
        'ensemble': name,
        'contrats': len(y),
        'sinistres': int(np.sum(y)),
        'prevalence': float(np.mean(y)),
        'proba_moyenne': float(np.mean(p)),
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'brier': float(brier_score_loss(y, p)),
    }


def emit_json(filepath, obj):
    """Sauvegarde un petit rapport d'exécution au format JSON lisible."""
    Path(filepath).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )


def args_parser():
    """Récupère les réglages définis par l'utilisateur dans le terminal."""
    p = argparse.ArgumentParser(
        description='Frequence XGBoost v2 : optimisation groupée et calibration'
    )
    p.add_argument('--train', type=Path, default=Path('data/train.csv'))
    p.add_argument('--test', type=Path, default=Path('data/test.csv'))
    p.add_argument('--output', type=Path, default=Path('resultats_frequence_v2'))
    p.add_argument('--trials', type=int, default=60)  # Nombre d'essais Optuna.
    p.add_argument('--folds', type=int, default=4)   # Nombre de folds de validation.
    p.add_argument('--threads', type=int, default=4) # Threads processeur autorisés.
    p.add_argument('--seed', type=int, default=42)   # Graine de reproductibilité.
    p.add_argument(
        '--importance', action='store_true',
        help='Importance par permutation (long)',
    )
    return p.parse_args()


# =============================================================================
# 9 — MAIN : TOUT LE DÉROULEMENT DE L'EXPÉRIENCE
# =============================================================================

def main(args):
    """Orchestre la sélection, l'évaluation et la production finale.

    Comprendre main() revient à comprendre la logique statistique du projet.
    Les étapes 9.1 à 9.8 suivent exactement l'ordre du script V2 initial.
    """

    # -------------------------------------------------------------------------
    # 9.1 — Initialisation : arguments, lecture, cible, découpage indépendant
    # -------------------------------------------------------------------------
    if args.trials < 0 or args.folds < 2 or args.threads < 1:
        raise ValueError('trials>=0, folds>=2 et threads>=1')

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Moins de logs Optuna.
    args.output.mkdir(parents=True, exist_ok=True)        # Créer le dossier de sortie.

    train, test = read_data(args.train, args.test)
    y = train['nombre_sinistres'].astype(int)  # La réponse que l'on veut apprendre.
    groups = train['id_client']               # Servira uniquement aux découpages.

    # Réserver ~20 % des contrats en Holdout (intouchable pendant Optuna).
    dev_i, hold_i = holdout_indices(train, args.seed)

    # y_dev et y_hold gardent chacun les réponses correspondant à leurs indices.
    y_dev = y.iloc[dev_i].reset_index(drop=True)
    y_hold = y.iloc[hold_i].reset_index(drop=True)
    g_dev = groups.iloc[dev_i].reset_index(drop=True)

    # Construire les folds du DÉVELOPPEMENT (et non du test interne).
    splits = client_folds(y_dev, g_dev, args.folds, args.seed + 1)

    # Création de variables SANS statistiques apprises sur l'ensemble du dataset.
    enriched = make_features(train)
    external = make_features(test)

    # Préparer les quatre versions des données pour toutes les comparaisons.
    views = {name: x_for(enriched, name) for name in FEATURE_SETS}
    external_views = {name: x_for(external, name) for name in FEATURE_SETS}

    # Même ensemble ET même ordre de colonnes entre Train et Test externe.
    for name in views:
        if not views[name].columns.equals(external_views[name].columns):
            raise ValueError(f'Schema Train/Test inconsistent pour {name}')

    # -------------------------------------------------------------------------
    # 9.2 — Baseline : prédire le taux moyen de sinistres par fold
    # -------------------------------------------------------------------------
    baseline = baseline_oof(y_dev, splits)
    baseline_score = float(log_loss(y_dev, baseline, labels=[0, 1]))

    print(
        f'Frequence : train={len(train)}, dev={len(dev_i)}, '
        f'holdout={len(hold_i)}, sinistres={int(y.sum())}',
        flush=True,
    )
    print(f'Baseline frequence CV logloss={baseline_score:.6f}', flush=True)

    # -------------------------------------------------------------------------
    # 9.3 — Comparaison des 4 jeux de variables avec les MÊMES paramètres XGB
    # -------------------------------------------------------------------------
    comparisons = []  # Enregistre : groupe, nombre de variables, Log Loss, dispersion.

    for name in FEATURE_SETS:
        x = views[name].iloc[dev_i].reset_index(drop=True)
        _, score, sd = cv_pred(x, y_dev, splits, DEFAULT, args.threads, args.seed)
        comparisons.append({
            'features': name,
            'variables': x.shape[1],
            'cv_logloss': score,
            'cv_ecart_type': sd,
        })
        print(f'Features {name}: logloss={score:.6f} (+/- {sd:.6f})', flush=True)

    # Enregistrer les quatre scores, du meilleur au moins bon.
    pd.DataFrame(comparisons).sort_values('cv_logloss').to_csv(
        args.output / 'comparaison_features.csv', index=False
    )

    # Ne conserver que les DEUX meilleurs ensembles pour la recherche Optuna.
    leaders = sorted(comparisons, key=lambda r: r['cv_logloss'])[:2]
    candidates = [v['features'] for v in leaders]

    # Si Optuna n'améliore pas la référence, conserver la meilleure configuration.
    best = min(comparisons, key=lambda r: r['cv_logloss'])
    chosen = best['features']
    params = DEFAULT.copy()
    best_score = float(best['cv_logloss'])

    # -------------------------------------------------------------------------
    # 9.4 — Optimisation Optuna, uniquement sur le DÉVELOPPEMENT
    # -------------------------------------------------------------------------
    if args.trials:
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )

        def objective(trial):
            """Un essai Optuna : proposer des paramètres puis mesurer leur Log Loss."""
            feature = trial.suggest_categorical('features', candidates)
            cfg = param_search(trial)
            x = views[feature].iloc[dev_i].reset_index(drop=True)
            _, score, _ = cv_pred(x, y_dev, splits, cfg, args.threads, args.seed)
            print(
                f'Optuna frequence {trial.number+1}/{args.trials}: '
                f'features={feature}, logloss={score:.6f}',
                flush=True,
            )
            return score  # Valeur qu'Optuna cherche à MINIMISER.

        # Une étude contient plusieurs essais ; chaque essai fait n folds.
        study.optimize(objective, n_trials=args.trials, n_jobs=1)
        study.trials_dataframe().to_csv(args.output / 'essais_optuna.csv', index=False)

        # Ne retenir Optuna que s'il bat la meilleure configuration de départ.
        if study.best_value < best_score:
            chosen = study.best_params['features']
            params = {k: v for k, v in study.best_params.items() if k != 'features'}
            best_score = float(study.best_value)

    # Si aucun XGBoost ne fait mieux que prédire le taux moyen, garder le simple.
    use_constant = best_score >= baseline_score

    # Construire les X du groupe gagnant, pour développement et test interne.
    x_dev = views[chosen].iloc[dev_i].reset_index(drop=True)
    x_hold = views[chosen].iloc[hold_i].reset_index(drop=True)
    print(
        'Configuration retenue :', chosen,
        'baseline choisie=', use_constant,
        'logloss CV=', round(min(best_score, baseline_score), 6),
        flush=True,
    )

    # -------------------------------------------------------------------------
    # 9.5 — Choisir (ou non) la calibration et prédire le test interne
    # -------------------------------------------------------------------------
    calibration_scores = {}
    calibration = 'aucune'
    calibrator = None

    if use_constant:
        # Cas rare : modèle constant vainqueur ; pas de XGBoost à calibrer.
        oof = baseline
        p_hold = np.full(len(hold_i), float(y_dev.mean()))
    else:
        # Obtenir des probabilités OOF de développement avec le modèle gagnant.
        oof, _, _ = cv_pred(x_dev, y_dev, splits, params, args.threads, args.seed)

        # Comparer : brut / sigmoïde / isotonique, sans toucher au test interne.
        for kind in ['aucune', 'sigmoid', 'isotonic']:
            op = crossfit_calibration(oof, y_dev, splits, kind)
            calibration_scores[kind] = float(log_loss(y_dev, op, labels=[0, 1]))

        # Plus petite Log Loss = meilleure calibration candidate sur développement.
        calibration = min(calibration_scores, key=calibration_scores.get)

        # Si le gain est < 0.00005, conserver les probabilités brutes.
        if (
            calibration != 'aucune'
            and calibration_scores['aucune'] - calibration_scores[calibration] < 0.00005
        ):
            calibration = 'aucune'

        # Réentraîner UN modèle sur tout le développement.
        dev_model = make_model(x_dev, params, args.threads, args.seed)
        dev_model.fit(x_dev, y_dev)

        # PRÉDICTION sur le Holdout : il n'a jamais servi à entraîner ce modèle.
        raw_hold = dev_model.predict_proba(x_hold)[:, 1]

        # Ajuster le calibrateur sur les OOF du développement puis l'appliquer.
        calibrator = fit_calibrator(calibration, oof, y_dev)
        p_hold = apply_calibrator(calibration, calibrator, raw_hold)

    # Sauvegarder les prédictions de développement hors entraînement.
    # IMPORTANT pour l'évaluation du coût attendu avec le modèle montant.
    calibrated_development_oof = (
        baseline if use_constant
        else crossfit_calibration(oof, y_dev, splits, calibration)
    )
    pd.DataFrame({
        'INDEX': train.iloc[dev_i]['index'].to_numpy(),
        'reel': y_dev.to_numpy(),
        'pred': np.clip(calibrated_development_oof, 1e-6, 1-1e-6),
    }).to_csv(args.output / 'predictions_developpement_oof.csv', index=False)

    # -------------------------------------------------------------------------
    # 9.6 — ÉVALUER le test interne : il n'a servi à AUCUN réglage du modèle
    # -------------------------------------------------------------------------
    p_hold = np.clip(p_hold, 1e-6, 1-1e-6)

    # Comparer le modèle sélectionné à une prédiction constante de prévalence.
    rows = [
        report_metrics('modele_retenu', y_hold, p_hold),
        report_metrics('baseline_prevalence_dev', y_hold, np.full(len(hold_i), y_dev.mean())),
    ]
    pd.DataFrame(rows).to_csv(args.output / 'evaluation_holdout.csv', index=False)
    print(pd.DataFrame(rows).round(6).to_string(index=False), flush=True)

    # Garder les index, prédictions et vraies réponses du test interne.
    pd.DataFrame({
        'INDEX': train.iloc[hold_i]['index'].to_numpy(),
        'reel': y_hold.to_numpy(),
        'pred': p_hold,
    }).to_csv(args.output / 'predictions_holdout.csv', index=False)

    # Importance facultative et lente : mélanger une variable à la fois,
    # puis mesurer combien la Log Loss se dégrade.
    if args.importance and not use_constant:
        perm = permutation_importance(
            dev_model, x_hold, y_hold,
            scoring='neg_log_loss', n_repeats=3,
            random_state=args.seed, n_jobs=1,
        )
        pd.DataFrame({
            'variable': x_hold.columns,
            'importance': perm.importances_mean,
            'std': perm.importances_std,
        }).sort_values('importance', ascending=False).to_csv(
            args.output / 'importance_variables.csv', index=False
        )

    # -------------------------------------------------------------------------
    # 9.7 — PRODUIRE le modèle final sur les 50 000 contrats étiquetés
    # -------------------------------------------------------------------------
    final_model = None
    final_calibrator = None

    if use_constant:
        # Même baseline, mais recalculée sur les 50 000 contrats étiquetés.
        p_external = np.full(len(test), float(y.mean()))
    else:
        # Toutes les lignes du Train, avec le même groupe de features retenu.
        x_all = views[chosen].reset_index(drop=True)
        x_external = external_views[chosen].reindex(columns=x_all.columns)

        # Nouveau XGBoost entraîné sur TOUT le train étiqueté.
        final_model = make_model(x_all, params, args.threads, args.seed)
        final_model.fit(x_all, y)

        # Test.csv : 50 000 lignes sans les réponses, donc pas de score ici.
        p_external = final_model.predict_proba(x_external)[:, 1]

        # Si calibration choisie, reconstruire un calibrateur sur tout le Train.
        if calibration != 'aucune':
            final_splits = client_folds(y, groups, args.folds, args.seed + 1)
            full_oof, _, _ = cv_pred(
                x_all, y, final_splits, params, args.threads, args.seed
            )
            final_calibrator = fit_calibrator(calibration, full_oof, y)
            p_external = apply_calibrator(
                calibration, final_calibrator, p_external
            )

    # Éviter les p=0 ou p=1 exactement ; conserver les probabilités valides.
    p_external = np.clip(p_external, 1e-6, 1-1e-6)
    if len(p_external) != len(test) or not np.isfinite(p_external).all():
        raise ValueError('Probabilites finales manquantes ou invalides')

    # UNE ligne par contrat externe ; aucune multiplication par le montant ici.
    submission = pd.DataFrame({
        'INDEX': test['index'].to_numpy(),
        'pred': p_external,
    })
    submission.to_csv(args.output / 'soumission_frequence_xgboost.csv', index=False)

    # -------------------------------------------------------------------------
    # 9.8 — SAUVEGARDER le modèle, les décisions et les résultats
    # -------------------------------------------------------------------------
    # Un joblib contient le pipeline, le calibrateur et les réglages retenus.
    joblib.dump({
        'pipeline': final_model,
        'calibrateur': final_calibrator,
        'calibration': calibration,
        'baseline': use_constant,
        'features': chosen,
        'params': params,
    }, args.output / 'modele_frequence.joblib')

    # Ce fichier JSON décrit les résultats pour rendre l'exécution traçable.
    emit_json(args.output / 'rapport_execution.json', {
        'objectif': 'probabilite de sinistre, pas nombre annuel',
        'train': len(train),
        'test_externe': len(test),
        'dev': len(dev_i),
        'holdout': len(hold_i),
        'prevalence_train': float(y.mean()),
        'baseline_cv_logloss': baseline_score,
        'best_cv_logloss': best_score,
        'feature_set': chosen,
        'parametres': params,
        'calibration': calibration,
        'scores_calibration': calibration_scores,
        'modele_constant': use_constant,
        'test_interne': rows,
        'validation': 'StratifiedGroupKFold par id_client ; holdout commun avec montant',
        'soumission': str(args.output / 'soumission_frequence_xgboost.csv'),
    })

    print(
        'Fichier frequence :', args.output / 'soumission_frequence_xgboost.csv',
        flush=True,
    )


# Quand on exécute ce fichier dans un terminal, Python entre ici.
# Quand on l'importe depuis un autre script, il NE lance PAS tout l'entraînement.
if __name__ == '__main__':
    main(args_parser())
