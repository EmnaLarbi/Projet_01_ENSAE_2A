"""Récupération du PIB par habitant en parité de pouvoir d'achat (PPA) par l'API de la Banque mondiale.

Indicateur : NY.GDP.PCAP.PP.KD, « GDP per capita, PPP (constant 2021 international $) ».
L'API est publique et ne demande aucune clé : https://api.worldbank.org/v2/

Le script :
    1. télécharge l'indicateur pour tous les pays et toutes les années demandées ;
    2. retire les agrégats (régions, groupes de revenu, monde…) pour ne garder que les pays ;
    3. enregistre une table au format long (une ligne = un pays × une année) dans
       Data/processed/pib_hab_ppa_banque_mondiale.csv.

Usage, depuis la racine du projet :
    python Src/data_upload.py
"""
from pathlib import Path

import pandas as pd
import requests

# ======================================================================
# PARAMÈTRES
# ======================================================================

URL_API = "https://api.worldbank.org/v2"
INDICATEUR = "NY.GDP.PCAP.PP.KD"       # PIB/hab en PPA, dollars internationaux constants de 2021
ANNEES = "1990:2025"                   # plage d'années demandée à l'API
PAR_PAGE = 20000                       # nombre de lignes par page de résultats
DELAI_MAX = 60                         # secondes avant d'abandonner une requête

RACINE = Path(__file__).resolve().parents[1]
CHEMIN_SORTIE = RACINE / "Data" / "processed" / "pib_hab_ppa_banque_mondiale.csv"

# Pays de l'analyse (codes ISO3), pour le contrôle final
PAYS_ANALYSE = ["AUT", "CHE", "DEU", "ESP", "FIN", "FRA", "HRV", "HUN",
                "ISL", "ITA", "LTU", "NOR", "SWE", "SVN", "SVK"]
ANNEE_CONTROLE = 2022


# ======================================================================
# FONCTIONS
# ======================================================================

def interroger_api(chemin: str, params: dict | None = None) -> tuple[dict, list[dict]]:
    """Interroge l'API page par page et renvoie (métadonnées de la 1re page, toutes les lignes)."""
    params = {"format": "json", "per_page": PAR_PAGE, **(params or {})}
    lignes, page, nb_pages, meta = [], 1, 1, {}
    while page <= nb_pages:
        reponse = requests.get(f"{URL_API}/{chemin}", params={**params, "page": page}, timeout=DELAI_MAX)
        reponse.raise_for_status()
        contenu = reponse.json()
        # En cas d'erreur, l'API renvoie un seul élément contenant un message au lieu de [métadonnées, données]
        if len(contenu) < 2:
            raise ValueError(f"Réponse inattendue de l'API pour {chemin} : {contenu}")
        if page == 1:
            meta = contenu[0]
            nb_pages = int(meta["pages"])
        lignes.extend(contenu[1] or [])
        page += 1
    return meta, lignes


def lister_agregats() -> set[str]:
    """Renvoie les codes ISO3 des agrégats (régions, groupes de revenu, monde…), qui ne sont pas des pays."""
    _, pays = interroger_api("country")
    return {p["id"] for p in pays if p["region"]["value"] == "Aggregates"}


def telecharger_indicateur(indicateur: str = INDICATEUR, annees: str = ANNEES) -> tuple[pd.DataFrame, str]:
    """Télécharge l'indicateur pour tous les pays et renvoie (table au format long, date de mise à jour)."""
    meta, lignes = interroger_api(f"country/all/indicator/{indicateur}", {"date": annees})
    df = pd.DataFrame({
        "iso3": [l["countryiso3code"] for l in lignes],
        "pays": [l["country"]["value"] for l in lignes],
        "annee": [int(l["date"]) for l in lignes],
        "pib_hab_ppa": [l["value"] for l in lignes],
    })
    return df, meta.get("lastupdated", "inconnue")


def nettoyer(df: pd.DataFrame, agregats: set[str]) -> pd.DataFrame:
    """Retire les agrégats, les lignes sans code pays et les valeurs manquantes, puis trie."""
    df = df[(df["iso3"] != "") & ~df["iso3"].isin(agregats)]
    df = df.dropna(subset=["pib_hab_ppa"])
    return df.sort_values(["iso3", "annee"]).reset_index(drop=True)


def controler(df: pd.DataFrame) -> None:
    """Vérifie que les pays de l'analyse ont une valeur pour l'année de contrôle."""
    annee = df[df["annee"] == ANNEE_CONTROLE].set_index("iso3")["pib_hab_ppa"]
    manquants = [p for p in PAYS_ANALYSE if p not in annee.index]
    if manquants:
        print(f" Pays de l'analyse sans valeur en {ANNEE_CONTROLE} : {manquants}")
    else:
        print(f" Les {len(PAYS_ANALYSE)} pays de l'analyse ont une valeur en {ANNEE_CONTROLE}.")
    print(annee.reindex(PAYS_ANALYSE).round(0).to_string())


# ======================================================================
# POINT D'ENTRÉE
# ======================================================================

def main() -> None:
    print(f"Téléchargement de {INDICATEUR} ({ANNEES}) depuis l'API de la Banque mondiale…")
    brut, mise_a_jour = telecharger_indicateur()
    propre = nettoyer(brut, lister_agregats())

    print(f"Dernière mise à jour des données Banque mondiale : {mise_a_jour}")
    print(f"{propre['iso3'].nunique()} pays, années {propre['annee'].min()}-{propre['annee'].max()}, "
          f"{len(propre)} lignes ({len(brut) - len(propre)} lignes retirées : agrégats ou valeurs manquantes)")
    controler(propre)

    CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    propre.to_csv(CHEMIN_SORTIE, index=False)
    print(f"💾 Fichier enregistré : {CHEMIN_SORTIE}")


if __name__ == "__main__":
    main()
