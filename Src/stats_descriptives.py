"""Statistiques descriptives par pays.

Construit, à partir de la table jointe (une ligne = un pays), un tableau qui présente
pour chaque pays la consommation de bœuf ou d'agneau des hommes et des femmes, l'écart
et le ratio entre les deux, la masculinité ressentie pondérée par sexe, l'écart de
masculinité et le PIB par habitant en PPA.

Entrée  : Data/processed/table_jointe.csv (produite par Src/merge.py)
Sortie  : sorties/tableau_pays.tex

Usage, depuis la racine du projet :
    python Src/stats_descriptives.py
"""
from pathlib import Path

import pandas as pd

# ======================================================================
# PARAMÈTRES
# ======================================================================

RACINE = Path(__file__).resolve().parents[1]
CHEMIN_DONNEES = RACINE / "Data" / "processed" / "table_jointe.csv"
DOSSIER_SORTIES = RACINE / "sorties"
CHEMIN_SORTIE = DOSSIER_SORTIES / "tableau_pays.tex"

NOMS_PAYS = {
    "AT": "Autriche", "CH": "Suisse", "DE": "Allemagne", "ES": "Espagne", "FI": "Finlande",
    "FR": "France", "HR": "Croatie", "HU": "Hongrie", "IS": "Islande", "IT": "Italie",
    "LT": "Lituanie", "NO": "Norvège", "SE": "Suède", "SI": "Slovénie", "SK": "Slovaquie",
}

# Colonnes du tableau final : {nom affiché: nombre de décimales}
DECIMALES = {
    "Viande H": 2, "Viande F": 2, "Écart viande": 2, "Ratio H/F": 2,
    "Masc. H": 2, "Masc. F": 2, "Écart masc.": 2, "PIB/hab PPA": 0,
}


# ======================================================================
# FONCTIONS
# ======================================================================

def construire_tableau(df: pd.DataFrame) -> pd.DataFrame:
    """Construit le tableau par pays, trié par écart de consommation décroissant.

    - Viande H / F : jours par semaine où l'on mange du bœuf ou de l'agneau (ISSP, pondéré) ;
    - Écart viande : Viande H − Viande F ; Ratio H/F : Viande H / Viande F ;
    - Masc. H / F : masculinité ressentie moyenne, échelle 0-6 (ESS, pondérée par anweight) ;
    - Écart masc. : Masc. H − Masc. F ;
    - PIB/hab PPA : dollars internationaux constants de 2021, année 2022.
    """
    tableau = pd.DataFrame({
        "Pays": df["iso2"].map(NOMS_PAYS).fillna(df["iso2"]),
        "Viande H": df["conso_homme"],
        "Viande F": df["conso_femme"],
        "Écart viande": df["ecart_h_f"],
        "Ratio H/F": df["conso_homme"] / df["conso_femme"],
        "Masc. H": df["mascfel_homme_w"],
        "Masc. F": df["mascfel_femme_w"],
        "Écart masc.": df["mascfel_homme_w"] - df["mascfel_femme_w"],
        "PIB/hab PPA": df["pib_hab"],
    })
    return tableau.sort_values("Écart viande", ascending=False).set_index("Pays")


def arrondir(tableau: pd.DataFrame) -> pd.DataFrame:
    """Arrondit chaque colonne au nombre de décimales prévu ; le PIB passe en entier."""
    tableau = tableau.round(DECIMALES)
    tableau["PIB/hab PPA"] = tableau["PIB/hab PPA"].astype(int)
    return tableau


def main() -> None:
    df = pd.read_csv(CHEMIN_DONNEES)
    tableau = arrondir(construire_tableau(df))
    print(tableau.to_string())

    DOSSIER_SORTIES.mkdir(exist_ok=True)
    # Sans nom d'index, pour éviter une ligne d'en-tête vide supplémentaire dans le .tex
    tableau.rename_axis(None).to_latex(CHEMIN_SORTIE, float_format="%.2f", encoding="utf-8")
    print(f"Tableau par pays enregistré : {CHEMIN_SORTIE}")


if __name__ == "__main__":
    main()
