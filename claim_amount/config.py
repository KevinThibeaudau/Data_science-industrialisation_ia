"""Settings of the claim amount model.

This is the only file to edit to change the model: the variable selection
threshold, the XGBoost parameters or the file locations.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_DIR / "data" / "train.csv"
MODEL_PATH = PROJECT_DIR / "artifacts" / "claim_amount_model.joblib"

MODEL_NAME = "xgboost_claim_amount_v1"

# Column to predict. The model learns its logarithm, on claimants only.
TARGET = "montant_sinistre"

# A variable is kept if its p-value against the claim amount is below this.
P_VALUE_THRESHOLD = 0.20

# Columns that are never used as inputs. The sex of the drivers is excluded
# because it is forbidden in insurance pricing in the EU since 2012.
DROPPED_COLUMNS = [
    "sex_conducteur1",
    "sex_conducteur2",
    "index",
    "id_client",
    "id_contrat",
    "id_vehicule",
]

# Candidate input variables, tested by the p-value selection.
# "departement" is built from "code_postal" during cleaning.
CATEGORICAL_CANDIDATES = [
    "type_contrat",
    "freq_paiement",
    "paiement",
    "utilisation",
    "conducteur2",
    "essence_vehicule",
    "type_vehicule",
    "marque_vehicule",
    "modele_vehicule",
    "departement",
]
NUMERIC_CANDIDATES = [
    "bonus",
    "duree_contrat",
    "anciennete_info",
    "age_conducteur1",
    "age_conducteur2",
    "anciennete_permis1",
    "anciennete_permis2",
    "anciennete_vehicule",
    "cylindre_vehicule",
    "din_vehicule",
    "debut_vente_vehicule",
    "fin_vente_vehicule",
    "vitesse_vehicule",
    "prix_vehicule",
    "poids_vehicule",
]

# Final XGBoost parameters (same as claim_amount/experiments/modele-final.py).
XGB_PARAMS = {
    "n_estimators": 50,
    "max_depth": 1,
    "learning_rate": 0.035296682203324346,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 66,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# Cross-validation used to estimate the Duan correction factor.
CV_FOLDS = 5
SEED = 42
