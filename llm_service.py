import re
import httpx
from pydantic import BaseModel, Field
from typing import List,Optional


class InformationExtraite(BaseModel):
    information: str = Field(min_length=1)
    passage_source: Optional[str] = None


class ResultatExtraction(BaseModel):
    informations: List[InformationExtraite]


CONSIGNE = """
Extrais les informations utiles du texte suivant.
Ignore les salutations et les répétitions.
Conserve les négations, les contraintes et les actions prévues.
N'invente rien. Réponds en français sous forme de liste.
Chaque élément doit être un extrait copié du texte, sans reformulation.
Conserve les pronoms d'origine et n'ajoute aucune mise en forme en gras.
"""

def tester_ollama(texte: str) -> dict:
    prompt = (
        "Extrais les informations utiles du texte suivant.\n"
        "Ignore les salutations et les répétitions.\n"
        "Conserve les négations, les contraintes et les actions prévues.\n"
        "N'invente rien. Réponds en français sous forme de liste.\n\n"
        f"Texte :\n{texte}"
    )

    reponse = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "qwen3:1.7b",
            "messages": [
                # On ajoute un message système pour interdire le mode "think" au niveau du modèle
                {"role": "system", "content": "Tu es un extracteur d'informations concis. Ne génère JAMAIS de balise <think> ou de texte de réflexion. Donne directement le résultat."},
                {"role": "user", "content": prompt}
            ],
            "think": False, # Option native Ollama
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
            },
        },
        timeout=180.0,
    )

    reponse.raise_for_status()
    donnees = reponse.json()
    contenu = donnees["message"]["content"]
    
    # Nettoyage de sécurité : supprime tout ce qui se trouve entre <think> et </think>
    # ainsi que les résidus textuels comme "Thinking Process:"
    contenu_propre = re.sub(r"<think>.*?</think>", "", contenu, flags=re.DOTALL)
    contenu_propre = re.sub(r"(Thinking Process:.*?\n\n)", "", contenu_propre, flags=re.DOTALL | re.IGNORECASE)
    
    return {"reponse": contenu_propre}

def decouper_texte(
    texte: str,
    taille_bloc: int = 2500,
    chevauchement: int = 300,
) -> List[str]:
    """
    Découpe le texte sans supprimer de contenu.
    Conserve un chevauchement pour partager du contexte entre blocs.
    """
    if not 0 <= chevauchement < taille_bloc:
        raise ValueError("Le chevauchement doit être inférieur à la taille du bloc.")

    blocs = []
    debut = 0

    while debut < len(texte):
        fin = min(debut + taille_bloc, len(texte))

        if fin < len(texte):
            # Chercher une fin de phrase dans la seconde moitié du bloc.
            milieu = debut + taille_bloc // 2
            coupures = list(
                re.finditer(r"[.!?]\s+", texte[milieu:fin])
            )

            if coupures:
                fin = milieu + coupures[-1].end()
            else:
                # Sinon, éviter autant que possible de couper un mot.
                espace = texte.rfind(" ", milieu, fin)

                if espace != -1:
                    fin = espace + 1

        bloc = texte[debut:fin]

        if bloc.strip():
            blocs.append(bloc)

        if fin >= len(texte):
            break

        prochain_debut = max(debut + 1, fin - chevauchement)

        # Éviter de reprendre au milieu d'un mot.
        while (
            prochain_debut > debut + 1
            and not texte[prochain_debut - 1].isspace()
        ):
            prochain_debut -= 1

        debut = prochain_debut

    return blocs

def retrouver_passage_original(texte: str, citation: str) -> str:
    citation = citation.strip()

    if not citation:
        raise ValueError("Le passage source est vide.")

    # Essayer d'abord la citation complète, puis sans sa ponctuation finale.
    variantes = [
        citation,
        citation.rstrip(".!?…").rstrip(),
    ]

    for variante in variantes:
        if not variante:
            continue

        # Ignorer uniquement la casse et les différences d'espaces.
        motif = r"\s+".join(
            re.escape(mot)
            for mot in variante.split()
        )

        # Éviter une correspondance à l'intérieur d'un mot ou d'un nombre.
        # Par exemple, "2007" ne doit pas correspondre à "20070".
        motif = r"(?<!\w)" + motif + r"(?!\w)"

        correspondance = re.search(
            motif,
            texte,
            flags=re.IGNORECASE,
        )

        if correspondance:
            return correspondance.group(0)

    raise ValueError(
        "Le passage source ne correspond pas au texte original."
    )

def extraire_avec_ollama(texte: str) -> ResultatExtraction:
    if not texte.strip():
        raise ValueError("Le texte ne doit pas être vide.")

    blocs = decouper_texte(texte)

    informations = []
    deja_vues = set()

    for numero, bloc in enumerate(blocs, start=1):
        print(
            f"Extraction du bloc {numero}/{len(blocs)} "
            f"({len(bloc)} caractères)",
            flush=True,
        )

        # Un seul appel à la fois.
        resultat = extraire_bloc_avec_ollama(bloc)

        for element in resultat.informations:
            # Retirer uniquement les doublons identiques après
            # normalisation des espaces.
            cle = (
                " ".join(element.information.split()),
                " ".join((element.passage_source or "").split()),
            )

            if cle not in deja_vues:
                deja_vues.add(cle)
                informations.append(element)

    return ResultatExtraction(informations=informations)

def extraire_bloc_avec_ollama(texte: str) -> ResultatExtraction:
    reponse = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "qwen3:1.7b",
            "messages": [
                {"role": "system", "content": CONSIGNE},
                {"role": "user", "content": texte},
            ],
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
            },
        },
        timeout=httpx.Timeout(180.0, connect=5.0),
    )

    reponse.raise_for_status()
    donnees = reponse.json()

    if donnees.get("done_reason") == "length":
        raise ValueError("La réponse du modèle a été interrompue.")

    message = donnees.get("message")

    if not isinstance(message, dict):
        raise ValueError("Ollama n'a pas renvoyé de message valide.")

    contenu = message.get("content")

    if not isinstance(contenu, str) or not contenu.strip():
        raise ValueError("Ollama a renvoyé une réponse vide.")

    # Ne pas traiter un raisonnement comme des informations extraites.
    if "<think>" in contenu or "</think>" in contenu:
        raise ValueError(
            "Le modèle a renvoyé du raisonnement dans sa réponse."
        )

    # Reconnaître les listes avec tirets, puces ou numéros.
    motif_puce = re.compile(
        r"^\s*(?:[-*•–]\s+|\d+[.)]\s+)(.+?)\s*$"
    )

    elements = []
    element_courant = None

    for ligne in contenu.splitlines():
        correspondance = motif_puce.match(ligne)

        if correspondance:
            if element_courant is not None:
                elements.append(element_courant)

            element_courant = correspondance.group(1).strip()

        elif ligne.strip() and element_courant is not None:
            # Accepter une continuation indentée d'une puce.
            if ligne.startswith((" ", "\t")):
                element_courant += " " + ligne.strip()
            else:
                elements.append(element_courant)
                element_courant = None

    if element_courant is not None:
        elements.append(element_courant)

    if not elements:
        raise ValueError(
            "Le modèle n'a pas renvoyé de liste à puces ou numérotée."
        )

    informations = []

    for information in elements:
        try:
            passage = retrouver_passage_original(texte, information)
        except ValueError:
            # Une reformulation n'est pas une citation exacte.
            passage = None

        informations.append(
            InformationExtraite(
                information=information,
                passage_source=passage,
            )
        )

    return ResultatExtraction(informations=informations)
