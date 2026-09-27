import httpx
import logging
import pymysql

from database import ouvrir_connexion
from fastapi import FastAPI,HTTPException
# from services import generer_message
from pydantic import BaseModel, Field
from services import extraire_phrases_importantes
from llm_service import extraire_avec_ollama, ResultatExtraction
from document_service import enregistrer_document

app = FastAPI(
    title="API de gestion de données",
    version="0.1.0",
)


@app.get("/")
def accueil():
    return {"message": "Bienvenue dans notre API !"}


@app.get("/health")
def verifier_etat():
    return {"status": "ok"}

# @app.get("/bonjour/{nom}")
# def dire_bonjour(nom: str):
#     message = generer_message(nom)
#     return {"message": message}

class TexteAAnalyser(BaseModel):
    texte: str = Field(min_length=1)
    max_phrases: int = Field(default=15, ge=1, le=100)


@app.post("/textes/extraire")
def analyser_texte(donnees: TexteAAnalyser):
    return extraire_phrases_importantes(
        texte=donnees.texte,
        max_phrases=donnees.max_phrases,
    )

class DemandeExtractionIA(BaseModel):
    texte: str = Field(min_length=1)

class DocumentAEnregistrer(BaseModel):
    titre: str = Field(min_length=1, max_length=255)
    texte: str = Field(min_length=1)
    extraction: ResultatExtraction


@app.post("/textes/extraire-ia", response_model=ResultatExtraction)
def analyser_texte_avec_ia(donnees: DemandeExtractionIA):
    if not donnees.texte.strip():
        raise HTTPException(
            status_code=422,
            detail="Le texte ne doit pas être vide.",
        )

    try:
        return extraire_avec_ollama(donnees.texte)


    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Ollama a mis trop de temps à répondre.",
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Impossible de communiquer avec Ollama.",
        )

    except (ValueError, KeyError, TypeError) as erreur:
        print("Erreur d'extraction :", repr(erreur), flush=True)

        raise HTTPException(
            status_code=502,
            detail=str(erreur),
        )

    except (ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="La réponse du modèle est incomplète ou ne respecte pas le format attendu.",
        )

@app.get("/health/database")
def verifier_base():
    connexion = None

    try:
        connexion = ouvrir_connexion()

        with connexion.cursor() as curseur:
            curseur.execute(
                "SELECT DATABASE() AS base, VERSION() AS version"
            )
            resultat = curseur.fetchone()

        return {
            "status": "ok",
            "base": resultat["base"],
            "version": resultat["version"],
        }

    except pymysql.MySQLError:
        logging.exception("Échec de connexion à MySQL")

        raise HTTPException(
            status_code=503,
            detail="La base de données est inaccessible.",
        )

    finally:
        if connexion is not None:
            connexion.close()

@app.post("/documents", status_code=201)
def creer_document(donnees: DocumentAEnregistrer):
    if not donnees.titre.strip() or not donnees.texte.strip():
        raise HTTPException(
            status_code=422,
            detail="Le titre et le texte ne doivent pas être vides.",
        )

    try:
        return enregistrer_document(
            titre=donnees.titre.strip(),
            texte=donnees.texte,
            resultat=donnees.extraction,
        )

    except pymysql.MySQLError:
        logging.exception("Échec de l'enregistrement du document")

        raise HTTPException(
            status_code=503,
            detail="Impossible d'enregistrer le document.",
        )
# @app.post("/textes/test-ollama")
# def test_ollama(donnees: DemandeExtractionIA):
#     return tester_ollama(donnees.texte)