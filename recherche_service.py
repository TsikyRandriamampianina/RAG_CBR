import json
import math

from database import ouvrir_connexion
from embedding_service import generer_vecteur, MODELE_EMBEDDING


def similarite_cosinus(vecteur_a, vecteur_b):
    if len(vecteur_a) != len(vecteur_b):
        raise ValueError("Les vecteurs ont des dimensions différentes.")

    for vecteur in (vecteur_a, vecteur_b):
        if not vecteur or any(
            type(valeur) not in (int, float)
            or not math.isfinite(valeur)
            for valeur in vecteur
        ):
            raise ValueError("Un vecteur contient des valeurs invalides.")

    norme_a = math.sqrt(sum(x * x for x in vecteur_a))
    norme_b = math.sqrt(sum(x * x for x in vecteur_b))

    if norme_a == 0 or norme_b == 0:
        raise ValueError("Impossible de comparer un vecteur nul.")

    produit = sum(
        a * b for a, b in zip(vecteur_a, vecteur_b)
    )

    score = produit / (norme_a * norme_b)

    # Corriger de possibles écarts d'arrondi numérique.
    return max(-1.0, min(1.0, score))


def rechercher_informations(
    question,
    limite=5,
    document_id=None,
):
    question = question.strip()

    if not question:
        raise ValueError("La question ne doit pas être vide.")

    # Préfixe de recherche correspondant à EmbeddingGemma.
    texte_question = "task: search result | query: " + question
    vecteur_question = generer_vecteur(texte_question)

    connexion = ouvrir_connexion()

    try:
        with connexion.cursor() as curseur:
            requete = """
                SELECT
                    i.id AS information_id,
                    i.document_id,
                    i.contenu,
                    i.passage_source,
                    i.statut_validation,
                    d.titre,
                    e.dimensions,
                    e.vecteur
                FROM embeddings e
                JOIN informations i ON i.id = e.information_id
                JOIN documents d ON d.id = i.document_id
                WHERE e.modele = %s
                  AND e.dimensions = %s
                  AND BINARY e.texte_vectorise =
                      BINARY CONCAT('title: none | text: ', i.contenu)
            """

            parametres = [
                MODELE_EMBEDDING,
                len(vecteur_question),
            ]

            if document_id is not None:
                requete += " AND i.document_id = %s"
                parametres.append(document_id)

            curseur.execute(requete, tuple(parametres))
            lignes = curseur.fetchall()

    finally:
        connexion.close()

    resultats = []

    for ligne in lignes:
        vecteur = json.loads(ligne["vecteur"])

        score = similarite_cosinus(
            vecteur_question,
            vecteur,
        )

        resultats.append({
            "information_id": ligne["information_id"],
            "document_id": ligne["document_id"],
            "titre_document": ligne["titre"],
            "information": ligne["contenu"],
            "passage_source": ligne["passage_source"],
            "statut_validation": ligne["statut_validation"],
            "score": score,
        })

    # Les meilleurs scores arrivent en premier.
    resultats.sort(
        key=lambda element: (
            -element["score"],
            element["information_id"],
        )
    )

    selection = resultats[:limite]

    for element in selection:
        element["score"] = round(element["score"], 4)

    return {
        "question": question,
        "nombre_informations_comparees": len(resultats),
        "resultats": selection,
    }