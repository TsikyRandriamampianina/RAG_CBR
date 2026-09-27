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
from embedding_service import vectoriser_information
from typing import Optional
from recherche_service import rechercher_informations
from rag_service import repondre_avec_rag

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

class DemandeRecherche(BaseModel):
    question: str = Field(min_length=1)
    limite: int = Field(default=5, ge=1, le=50)
    document_id: Optional[int] = Field(default=None, gt=0)

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

@app.post("/informations/{information_id}/embedding")
def creer_embedding(information_id: int):
    if information_id <= 0:
        raise HTTPException(
            status_code=422,
            detail="L'identifiant doit être positif.",
        )

    try:
        return vectoriser_information(information_id)

    except LookupError as erreur:
        raise HTTPException(status_code=404, detail=str(erreur))

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="La génération du vecteur a dépassé le délai.",
        )

    except httpx.HTTPStatusError:
        logging.exception("Erreur renvoyée par Ollama")
        raise HTTPException(
            status_code=502,
            detail=(
                "Ollama a refusé la vectorisation. "
                "Vérifie le modèle installé et la longueur du texte."
            ),
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Impossible de communiquer avec Ollama.",
        )

    except pymysql.MySQLError:
        logging.exception("Erreur MySQL pendant la vectorisation")
        raise HTTPException(
            status_code=503,
            detail="Impossible de lire ou d'enregistrer les données.",
        )

    except ValueError as erreur:
        raise HTTPException(status_code=502, detail=str(erreur))

@app.post("/recherche")
def rechercher(donnees: DemandeRecherche):
    if not donnees.question.strip():
        raise HTTPException(
            status_code=422,
            detail="La question ne doit pas être vide.",
        )

    try:
        return rechercher_informations(
            question=donnees.question,
            limite=donnees.limite,
            document_id=donnees.document_id,
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="La vectorisation de la question a dépassé le délai.",
        )

    except httpx.HTTPStatusError:
        logging.exception("Erreur Ollama pendant la recherche")
        raise HTTPException(
            status_code=502,
            detail="Ollama a refusé la vectorisation de la question.",
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Impossible de communiquer avec Ollama.",
        )

    except pymysql.MySQLError:
        logging.exception("Erreur MySQL pendant la recherche")
        raise HTTPException(
            status_code=503,
            detail="Impossible de consulter la base de données.",
        )

    except (ValueError, TypeError):
        logging.exception("Données vectorielles invalides")
        raise HTTPException(
            status_code=502,
            detail="Un vecteur ou une réponse du modèle est invalide.",
        )

@app.post("/rag/question")
def poser_question(donnees: DemandeRecherche):
    if not donnees.question.strip():
        raise HTTPException(
            status_code=422,
            detail="La question ne doit pas être vide.",
        )

    try:
        return repondre_avec_rag(
            question=donnees.question,
            limite=donnees.limite,
            document_id=donnees.document_id,
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Le traitement RAG a dépassé le délai.",
        )

    except httpx.HTTPStatusError:
        logging.exception("Erreur renvoyée par Ollama")
        raise HTTPException(
            status_code=502,
            detail="Ollama a refusé une étape du traitement.",
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Impossible de communiquer avec Ollama.",
        )

    except pymysql.MySQLError:
        logging.exception("Erreur MySQL pendant le RAG")
        raise HTTPException(
            status_code=503,
            detail="Impossible de consulter les informations.",
        )

    except (ValueError, TypeError) as erreur:
        logging.exception("Erreur pendant le RAG")
        raise HTTPException(
            status_code=502,
            detail=str(erreur),
        )