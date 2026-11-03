import numpy as np
import pandas as pd
from pathlib import Path
 
############################
# Nettoyage de l'ISSP 2020
############################

"""Variable de consommation utilisée :
    v50 - Q17c « Typical week: number of days that you eat beef/lamb? » (0 à 7 jours)
 
Codes spéciaux de v50 dans le fichier Stata :
    -9 No answer, -8 Can't choose, -1 Not available (DK)  -> valeurs manquantes
    -4 NAP (vegetarian)                                    -> recodé à 0 jour
SEX : 1 = Homme, 2 = Femme, -9 = No answer (-> manquant)
 
Entrée  : data/raw/ISSP_2020.dta
Sortie  : data/processed/issp_clean.csv (une ligne par répondant)
 
Utilisation depuis la racine du projet :
    python src/nettoyage_issp.py
"""

RACINE = Path(__file__).resolve().parents[1]
CHEMIN_BRUT = RACINE / "Data" / "raw" / "ISSP_2020.dta"
CHEMIN_PROPRE = RACINE / "Data" / "processed" / "issp_clean.csv"
 
COLONNES_ISSP = ["country", "c_alphan", "SEX", "v50", "WEIGHT"]
CODE_VEGETARIEN = -4
CODES_SEXE = {1: "homme", 2: "femme"}
 
 
def charger_issp(chemin: str | Path = CHEMIN_BRUT) -> pd.DataFrame:
    """Charge uniquement les colonnes utiles de la base ISSP (fichier .dta).
 
    On garde les codes numériques (convert_categoricals=False) pour pouvoir
    traiter nous-mêmes les codes négatifs de non-réponse.
    """
    return pd.read_stata(chemin, columns=COLONNES_ISSP, convert_categoricals=False)
 

def nettoyer_issp(df: pd.DataFrame, vegetariens_a_zero: bool = True) -> pd.DataFrame:
    """Renomme les variables, recode les non-réponses et le sexe.
 
    Paramètres
    ----------
    df : base brute issue de `charger_issp`.
    vegetariens_a_zero : si True, les végétariens (code -4) sont recodés à 0 jour
        de consommation de bœuf/agneau ; sinon ils sont traités comme manquants.
 
    Retour
    ------
    DataFrame avec les colonnes : country_iso2, country_code, sexe,
    jours_viande_rouge, poids. Les lignes sans sexe, sans consommation
    ou sans poids sont supprimées.
    """
    out = df.rename(columns={
        "c_alphan": "country_iso2",
        "country": "country_code",
        "v50": "jours_viande_rouge",
        "WEIGHT": "poids",
    }).copy()
 
    conso = out["jours_viande_rouge"].astype(float)
    if vegetariens_a_zero:
        conso = conso.mask(conso == CODE_VEGETARIEN, 0)
    # Tout code encore négatif = non-réponse
    out["jours_viande_rouge"] = conso.where(conso >= 0)
 
    out["sexe"] = out["SEX"].map(CODES_SEXE)  # -9 -> NaN
    out["country_iso2"] = out["country_iso2"].str.strip()
 
    out = out.dropna(subset=["sexe", "jours_viande_rouge", "poids"])
    return out[["country_iso2", "country_code", "sexe", "jours_viande_rouge", "poids"]]
 
 
def main() -> None:
    """Charge la base brute, la nettoie et sauvegarde le résultat."""
    brut = charger_issp(CHEMIN_BRUT)
    propre = nettoyer_issp(brut)
 
    CHEMIN_PROPRE.parent.mkdir(parents=True, exist_ok=True)
    propre.to_csv(CHEMIN_PROPRE, index=False)
 
    print(f"Base brute   : {len(brut)} répondants")
    print(f"Base nettoyée: {len(propre)} répondants "
          f"({len(brut) - len(propre)} lignes supprimées)")
    print(f"Sauvegardée dans {CHEMIN_PROPRE}")
    print(propre)

if __name__ == "__main__":
    main()