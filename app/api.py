
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from frequence.predict import predict_frequency


app = FastAPI(
    title="Tarification automobile",
    description="API de prédiction des sinistres automobiles",
    version="1.0.0",
)


# Structure attendue pour un contrat automobile
class Contrat(BaseModel):
    id_client: str
    id_vehicule: str
    id_contrat: str

    bonus: float
    type_contrat: str
    duree_contrat: float
    anciennete_info: float
    freq_paiement: str
    paiement: str
    utilisation: str
    code_postal: str
    conducteur2: str

    age_conducteur1: float
    age_conducteur2: float
    sex_conducteur1: str
    sex_conducteur2: str | None

    anciennete_permis1: float
    anciennete_permis2: float

    anciennete_vehicule: float | None
    cylindre_vehicule: float
    din_vehicule: float
    essence_vehicule: str
    marque_vehicule: str
    modele_vehicule: str

    debut_vente_vehicule: float
    fin_vente_vehicule: float
    vitesse_vehicule: float
    type_vehicule: str
    prix_vehicule: float
    poids_vehicule: float

class FrequenceResponse(BaseModel):
    frequency: float

@app.get("/health")
def health():
    """Vérifie que l'API répond."""
    return {"status": "ok"}


@app.post(
    "/predict_frequency",
    response_model=FrequenceResponse
)
def prediction_frequence(contrat: Contrat):
    """Prédit la probabilité d'au moins un sinistre."""

    # Transformer le modèle Pydantic en dictionnaire Python
    donnees = contrat.model_dump()

    try:
        # Appeler notre modèle XGBoost déjà entraîné
        probabilite = predict_frequency(donnees)

    except (ValueError, TypeError) as erreur:
        raise HTTPException(
            status_code=422,
            detail=str(erreur),
        ) from erreur

    return {"frequency": probabilite}
