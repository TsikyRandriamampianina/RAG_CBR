from database import ouvrir_connexion
from llm_service import ResultatExtraction


def enregistrer_document(
    titre: str,
    texte: str,
    resultat: ResultatExtraction,
) -> dict:
    connexion = ouvrir_connexion()

    try:
        with connexion.cursor() as curseur:
            curseur.execute(
                """
                INSERT INTO documents (titre, texte_original)
                VALUES (%s, %s)
                """,
                (titre, texte),
            )

            document_id = curseur.lastrowid
            information_ids = []

            for element in resultat.informations:
                curseur.execute(
                    """
                    INSERT INTO informations (
                        document_id,
                        contenu,
                        passage_source
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        document_id,
                        element.information,
                        element.passage_source,
                    ),
                )

                information_ids.append(curseur.lastrowid)

        connexion.commit()

        return {
            "document_id": document_id,
            "nombre_informations": len(information_ids),
            "information_ids": information_ids,
        }

    except Exception:
        connexion.rollback()
        raise

    finally:
        connexion.close()