from pathlib import Path
 
import numpy as np
import pandas as pd

"""Deuxième étape  - Agrégation par pays de la base ISSP nettoyée.
 
Pour chaque pays : consommation moyenne (pondérée) de bœuf/agneau des hommes
et des femmes, effectifs, et écart de consommation Homme - Femme.
 
Entrée  : data/processed/issp_clean.csv (produit par src/nettoyage_issp.py)
Sortie  : data/processed/issp_par_pays.csv (une ligne par pays)
 
Utilisation depuis la racine du projet (après le script de nettoyage) :
    python src/agregation_issp.py
"""
 
RACINE = Path(__file__).resolve().parents[1]
CHEMIN_PROPRE = RACINE / "Data" / "processed" / "issp_clean.csv"
CHEMIN_PAYS = RACINE / "Data" / "processed" / "issp_par_pays.csv"
 
 
def moyenne_ponderee(groupe: pd.DataFrame, var: str = "jours_viande_rouge",
                     poids: str = "poids") -> float:
    """Moyenne de `var` pondérée par `poids` au sein d'un groupe."""
    return np.average(groupe[var], weights=groupe[poids])
 
 
def agreger_par_pays(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège la base individuelle nettoyée au niveau pays.
 
    Retour
    ------
    DataFrame indexé par pays (country_iso2) avec les colonnes :
    conso_homme, conso_femme, n_homme, n_femme, ecart_h_f.
    """
    moyennes = (
        df.groupby(["country_iso2", "sexe"])[["jours_viande_rouge", "poids"]]
        .apply(moyenne_ponderee)
        .unstack("sexe")
        .rename(columns={"homme": "conso_homme", "femme": "conso_femme"})
    )
    effectifs = (
        df.groupby(["country_iso2", "sexe"]).size()
        .unstack("sexe")
        .rename(columns={"homme": "n_homme", "femme": "n_femme"})
    )
 
    pays = moyennes.join(effectifs)
    pays.columns.name = None
    return pays[["conso_homme", "conso_femme", "n_homme", "n_femme"]]
 
 
def ajouter_ecart_sexes(pays: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la variable ecart_h_f = consommation Homme - consommation Femme.
 
    Un écart positif signifie que les hommes déclarent manger du bœuf/agneau
    plus de jours par semaine que les femmes.
    """
    pays = pays.copy()
    pays["ecart_h_f"] = pays["conso_homme"] - pays["conso_femme"]
    return pays
 
 
def main() -> None:
    """Lit la base nettoyée, agrège par pays, calcule l'écart et sauvegarde."""
    issp = pd.read_csv(CHEMIN_PROPRE)
    pays = ajouter_ecart_sexes(agreger_par_pays(issp))
 
    pays.to_csv(CHEMIN_PAYS)
 
    print(f"{len(pays)} pays agrégés, sauvegardés dans {CHEMIN_PAYS}")
    print(pays.sort_values("ecart_h_f", ascending=False).round(2))
 
if __name__ == "__main__":
    main()

