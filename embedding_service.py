import json
import math
from typing import List

import httpx

from database import ouvrir_connexion


MODELE_EMBEDDING = "embeddinggemma:300m"


def generer_vecteur(texte: str) -> List[float]:
    if not texte.strip():
        raise ValueError("Le texte à vectoriser est vide.")

    reponse = httpx.post(
        "http://127.0.0.1:11434/api/embed",
        json={
            "model": MODELE_EMBEDDING,
            "input": texte,
            "truncate": False,
            "keep_alive": "0",
        },
        timeout=httpx.Timeout(180.0, connect=5.0),
    )

    reponse.raise_for_status()
    donnees = reponse.json()

    vecteurs = donnees.get("embeddings")

    if not isinstance(vecteurs, list) or len(vecteurs) != 1:
        raise ValueError("Ollama n'a pas retourné un vecteur unique.")

    vecteur = vecteurs[0]

    if not isinstance(vecteur, list) or not vecteur:
        raise ValueError("Le vecteur reçu est vide ou invalide.")

    for valeur in vecteur:
        if (
            type(valeur) not in (int, float)
            or not math.isfinite(valeur)
        ):
            raise ValueError("Le vecteur contient une valeur invalide.")

    if not any(valeur != 0 for valeur in vecteur):
        raise ValueError("Le vecteur reçu est entièrement nul.")

    return [float(valeur) for valeur in vecteur]


def vectoriser_information(information_id: int) -> dict:
    # Lire l'information, puis fermer la connexion avant le calcul.
    connexion = ouvrir_connexion()

    try:
        with connexion.cursor() as curseur:
            curseur.execute(
                "SELECT contenu FROM informations WHERE id = %s",
                (information_id,),
            )
            information = curseur.fetchone()
    finally:
        connexion.close()

    if information is None:
        raise LookupError("Information introuvable.")

    contenu = information["contenu"]

    # Préfixe recommandé pour indexer un document avec EmbeddingGemma.
    texte_vectorise = "title: none | text: " + contenu

    vecteur = generer_vecteur(texte_vectorise)

    connexion = ouvrir_connexion()

    try:
        with connexion.cursor() as curseur:
            # Vérifier que le contenu n'a pas changé pendant le calcul.
            curseur.execute(
                """
                SELECT contenu
                FROM informations
                WHERE id = %s
                FOR UPDATE
                """,
                (information_id,),
            )
            actuelle = curseur.fetchone()

            if actuelle is None:
                raise LookupError("L'information a été supprimée.")

            if actuelle["contenu"] != contenu:
                raise ValueError(
                    "L'information a changé pendant le calcul. Réessaie."
                )

            # Un nouvel appel remplace le vecteur du même modèle.
            curseur.execute(
                """
                INSERT INTO embeddings (
                    information_id,
                    modele,
                    dimensions,
                    texte_vectorise,
                    vecteur
                )
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    dimensions = VALUES(dimensions),
                    texte_vectorise = VALUES(texte_vectorise),
                    vecteur = VALUES(vecteur)
                """,
                (
                    information_id,
                    MODELE_EMBEDDING,
                    len(vecteur),
                    texte_vectorise,
                    json.dumps(vecteur, allow_nan=False),
                ),
            )

        connexion.commit()

    except Exception:
        connexion.rollback()
        raise

    finally:
        connexion.close()

    return {
        "information_id": information_id,
        "modele": MODELE_EMBEDDING,
        "dimensions": len(vecteur),
        "enregistre": True,
        "apercu_vecteur": vecteur[:5],
    }