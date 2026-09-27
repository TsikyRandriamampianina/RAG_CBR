import re
import unicodedata


def normaliser(texte: str) -> str:
    """Facilite les comparaisons sans modifier le texte original."""
    texte = texte.lower().replace("’", "'")

    # Exemple : "déclaration" devient "declaration".
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(
        caractere
        for caractere in texte
        if unicodedata.category(caractere) != "Mn"
    )

    return re.sub(r"\s+", " ", texte).strip()


CRITERES = {
    "situation": [
        "marie", "enfant", "charge", "foyer fiscal", "ans",
    ],
    "finances": [
        "revenu", "salaire", "rente", "pension", "euros",
        "impot", "epargne", "assurance vie", "emprunt",
        "locatif", "location", "meuble", "tresorerie",
    ],
    "objectifs": [
        "mon but", "mon probleme", "je voudrais",
        "j'aimerais", "reduire", "optimiser", "verifier",
    ],
    "contraintes": [
        "peur", "perdre", "capital", "prudente",
        "pas de tresorerie", "ne peux pas",
    ],
    "actions": [
        "envoyer", "envoyez", "envoie", "declaration",
        "avis d'imposition", "projection", "rendez-vous",
    ],
}


def extraire_phrases_importantes(
    texte: str,
    max_phrases: int = 15,
) -> dict:
    # Nettoyer les espaces et les retours à la ligne.
    texte = re.sub(r"\s+", " ", texte).strip()

    # Séparation simple basée sur la ponctuation.
    phrases = re.split(r"(?<=[.!?])\s+", texte)

    candidats = []
    deja_vues = set()

    for position, phrase in enumerate(phrases):
        phrase_normalisee = normaliser(phrase)
        cle = phrase_normalisee.strip(" .!?–—-")

        # Ignorer les éléments vides et les répétitions exactes.
        if not cle or cle in deja_vues:
            continue

        deja_vues.add(cle)

        score = 0
        themes = []

        for theme, expressions in CRITERES.items():
            correspondances = [
                expression
                for expression in expressions
                if re.search(
                    rf"\b{re.escape(expression)}\b",
                    phrase_normalisee,
                )
            ]

            if correspondances:
                themes.append(theme)
                score += len(correspondances)

        # Un chiffre renforce une phrase déjà liée à un thème.
        if score > 0 and re.search(r"\d", phrase):
            score += 2

        if score > 0:
            candidats.append({
                "position": position,
                "phrase": phrase,
                "score": score,
                "themes": themes,
            })

    # Choisir les phrases les mieux notées.
    selection = sorted(
        candidats,
        key=lambda element: (-element["score"], element["position"]),
    )[:max_phrases]

    # Revenir à l'ordre du texte pour faciliter la lecture.
    selection.sort(key=lambda element: element["position"])

    return {
        "nombre_phrases_initiales": len(phrases) if texte else 0,
        "nombre_phrases_retenues": len(selection),
        "texte_extrait": " ".join(
            element["phrase"] for element in selection
        ),
        "phrases": [
            {
                "phrase": element["phrase"],
                "score": element["score"],
                "themes": element["themes"],
            }
            for element in selection
        ],
    }