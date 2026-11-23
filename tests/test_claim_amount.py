import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.api import app
from claim_amount import predict as predict_module
from claim_amount import train as train_module
from claim_amount.preprocessing import clean
from claim_amount.selection import select_variables

client = TestClient(app)

CONTRAT_VALIDE = {
    "id_client": "A00000001",
    "id_vehicule": "V01",
    "id_contrat": "A00000001-V01",
    "bonus": 0.5,
    "type_contrat": "Maxi",
    "duree_contrat": 29,
    "anciennete_info": 9,
    "freq_paiement": "Biannual",
    "paiement": "No",
    "utilisation": "Retired",
    "code_postal": "36233",
    "conducteur2": "No",
    "age_conducteur1": 85,
    "age_conducteur2": 0,
    "sex_conducteur1": "M",
    "sex_conducteur2": None,
    "anciennete_permis1": 62,
    "anciennete_permis2": 0,
    "anciennete_vehicule": 10,
    "cylindre_vehicule": 1587,
    "din_vehicule": 98,
    "essence_vehicule": "Gasoline",
    "marque_vehicule": "PEUGEOT",
    "modele_vehicule": "306",
    "debut_vente_vehicule": 10,
    "fin_vente_vehicule": 9,
    "vitesse_vehicule": 182,
    "type_vehicule": "Tourism",
    "prix_vehicule": 20700,
    "poids_vehicule": 1210,
}


def make_contracts(n, seed=0):
    """Synthetic contracts. The amount depends on the bonus and the vehicle type."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id_client": [f"A{i:08d}" for i in range(n)],
            "id_vehicule": "V01",
            "id_contrat": [f"A{i:08d}-V01" for i in range(n)],
            "bonus": rng.uniform(0.5, 1.5, n),
            "type_contrat": rng.choice(["Maxi", "Mini"], n),
            "duree_contrat": rng.integers(1, 40, n),
            "anciennete_info": rng.integers(0, 20, n),
            "freq_paiement": rng.choice(["Monthly", "Biannual"], n),
            "paiement": rng.choice(["Yes", "No"], n),
            "utilisation": rng.choice(["Retired", "WorkPrivate"], n),
            "code_postal": rng.choice(["36233", "01001", "75001"], n),
            "conducteur2": rng.choice(["Yes", "No"], n),
            "age_conducteur1": rng.integers(20, 90, n),
            "age_conducteur2": rng.integers(20, 90, n),
            "sex_conducteur1": rng.choice(["M", "F"], n),
            "sex_conducteur2": rng.choice(["M", "F"], n),
            "anciennete_permis1": rng.integers(0, 60, n),
            "anciennete_permis2": rng.integers(0, 60, n),
            "anciennete_vehicule": rng.integers(0, 20, n),
            "cylindre_vehicule": rng.integers(900, 3000, n),
            "din_vehicule": rng.integers(50, 300, n),
            "essence_vehicule": rng.choice(["Gasoline", "Diesel"], n),
            "marque_vehicule": rng.choice(["PEUGEOT", "RENAULT"], n),
            "modele_vehicule": rng.choice(["306", "CLIO"], n),
            "debut_vente_vehicule": rng.integers(0, 20, n),
            "fin_vente_vehicule": rng.integers(0, 20, n),
            "vitesse_vehicule": rng.integers(120, 250, n),
            "type_vehicule": rng.choice(["Tourism", "Utility"], n),
            "prix_vehicule": rng.integers(8000, 60000, n),
            "poids_vehicule": rng.integers(800, 2000, n),
        }
    )
    log_amount = (
        6
        + 0.8 * (df["bonus"] - 1)
        + 0.5 * (df["type_vehicule"] == "Utility")
        + rng.normal(0, 0.3, n)
    )
    claims = rng.random(n) < 0.5
    df["montant_sinistre"] = np.where(claims, np.exp(log_amount), 0.0)
    return df


@pytest.fixture(scope="module")
def raw_train():
    return make_contracts(600)


@pytest.fixture
def trained_model(raw_train, tmp_path, monkeypatch):
    """Train on synthetic data and point the prediction code to the saved file."""
    artifact = train_module.train(raw_train)
    path = tmp_path / "claim_amount_model.joblib"
    joblib.dump(artifact, path)

    monkeypatch.setattr(predict_module, "MODEL_PATH", path)
    predict_module.load_model.cache_clear()
    yield artifact
    predict_module.load_model.cache_clear()


def test_clean_handles_one_contract_with_missing_values():
    contract = dict(CONTRAT_VALIDE, code_postal=None, anciennete_vehicule=None)

    cleaned = clean(pd.DataFrame([contract]))

    assert np.isnan(cleaned.loc[0, "anciennete_vehicule"])
    assert pd.isna(cleaned.loc[0, "departement"])
    assert "sex_conducteur1" not in cleaned.columns


def test_clean_second_driver_absent():
    cleaned = clean(pd.DataFrame([CONTRAT_VALIDE]))

    assert np.isnan(cleaned.loc[0, "age_conducteur2"])
    assert np.isnan(cleaned.loc[0, "anciennete_permis2"])
    assert cleaned.loc[0, "departement"] == "36"


def test_clean_impossible_values():
    contract = dict(CONTRAT_VALIDE, anciennete_permis1=111, poids_vehicule=0)

    cleaned = clean(pd.DataFrame([contract]))

    assert np.isnan(cleaned.loc[0, "anciennete_permis1"])
    assert np.isnan(cleaned.loc[0, "poids_vehicule"])


def test_clean_missing_field():
    contract = {k: v for k, v in CONTRAT_VALIDE.items() if k != "bonus"}

    with pytest.raises(ValueError, match="bonus"):
        clean(pd.DataFrame([contract]))


def test_select_variables_threshold(raw_train):
    claims = clean(raw_train)
    claims = claims[claims["montant_sinistre"] > 0]

    loose = select_variables(claims, 0.20)
    strict = select_variables(claims, 0.001)

    assert "bonus" in strict
    assert "type_vehicule" in strict
    assert set(strict) <= set(loose)


def test_train_artifact(trained_model):
    assert trained_model["variables"]
    assert trained_model["correction"] > 0
    assert "sex_conducteur1" not in trained_model["variables"]


def test_predict_claim_amount(trained_model):
    amount = predict_module.predict_claim_amount(CONTRAT_VALIDE)

    assert isinstance(amount, float)
    assert 0 < amount < 100_000


def test_predict_with_missing_values(trained_model):
    contract = dict(CONTRAT_VALIDE, anciennete_vehicule=None, code_postal=None)

    amount = predict_module.predict_claim_amount(contract)

    assert np.isfinite(amount)


def test_predict_unseen_category(trained_model):
    contract = dict(CONTRAT_VALIDE, marque_vehicule="UNKNOWN BRAND")

    amount = predict_module.predict_claim_amount(contract)

    assert np.isfinite(amount)


def test_predict_missing_field(trained_model):
    contract = {k: v for k, v in CONTRAT_VALIDE.items() if k != "bonus"}

    with pytest.raises(ValueError, match="bonus"):
        predict_module.predict_claim_amount(contract)


def test_predict_without_model_file(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "absent.joblib")
    predict_module.load_model.cache_clear()

    with pytest.raises(FileNotFoundError):
        predict_module.predict_claim_amount(CONTRAT_VALIDE)


def test_api_claim_amount_mock(monkeypatch):
    monkeypatch.setattr(api_module, "predict_claim_amount", lambda contrat: 1500.0)

    reponse = client.post("/predict_claim_amount", json=CONTRAT_VALIDE)

    assert reponse.status_code == 200
    assert reponse.json() == {"claim_amount": 1500.0}


def test_api_claim_amount_incomplete_contract():
    reponse = client.post("/predict_claim_amount", json={"id_client": "A1"})

    assert reponse.status_code == 422


def test_api_claim_amount_error_becomes_422(monkeypatch):
    def failing_prediction(contrat):
        raise ValueError("Missing fields: ['bonus']")

    monkeypatch.setattr(api_module, "predict_claim_amount", failing_prediction)

    reponse = client.post("/predict_claim_amount", json=CONTRAT_VALIDE)

    assert reponse.status_code == 422


def test_api_premium_mock(monkeypatch):
    monkeypatch.setattr(api_module, "predict_frequency", lambda contrat: 0.1)
    monkeypatch.setattr(api_module, "predict_claim_amount", lambda contrat: 2000.0)

    reponse = client.post("/predict_premium", json=CONTRAT_VALIDE)

    assert reponse.status_code == 200
    assert reponse.json() == pytest.approx(
        {"frequency": 0.1, "claim_amount": 2000.0, "premium": 200.0}
    )
