
import re

from fastapi.testclient import TestClient

from app.api import app
from pathlib import Path
import pytest

from app import api as api_module
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
        "poids_vehicule": 1210
    }

MODELE_FREQUENCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "modele_frequence.joblib"
)

def test_health():
    reponse = client.get("/health")

    assert reponse.status_code == 200
    assert reponse.json() == {"status": "ok"}


def test_contrat_incomplet():
    contrat = {
        "id_client": "A00000001"
    }

    reponse = client.post("/predict_frequency", json=contrat)

    assert reponse.status_code == 422

    erreurs = reponse.json()["detail"]

    assert any(
        erreur["loc"] == ["body", "bonus"]
        for erreur in erreurs
    )

@pytest.mark.skipif(
    not MODELE_FREQUENCE.exists(),
    reason="Modele XGBoost absent"
)
def test_prediction_frequence():


    reponse = client.post("/predict_frequency", json=CONTRAT_VALIDE)

    assert reponse.status_code == 200

    resultat = reponse.json()

    assert "frequency" in resultat
    assert isinstance(resultat["frequency"], float)
    assert 0 <= resultat["frequency"] <= 1


def test_prediction_frequence_mock(monkeypatch):

    # Fonction qui remplace temporairement XGBoost
    def fausse_prediction(contrat):
        assert contrat["bonus"] == 0.5
        return 0.25

    # Remplacer la fonction appelée par notre API
    monkeypatch.setattr(
        api_module,
        "predict_frequency",
        fausse_prediction
    )

    # Envoyer un contrat valide à FastAPI
    reponse = client.post(
        "/predict_frequency",
        json=CONTRAT_VALIDE
    )

    # Vérifier la réponse
    assert reponse.status_code == 200
    assert reponse.json() == {"frequency": 0.25}
