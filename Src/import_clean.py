from pathlib import Path
import numpy as np
import pandas as pd

# Chemins de base globaux
RACINE = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path.cwd()
DOSSIER_PROCESSED = RACINE / "Data" / "processed"

# ==========================================
# 1. TRAITEMENT DE L'ESS 11
# ==========================================

CHEMIN_BRUT_ESS = RACINE / "Data" / "raw" / "ESS11e04_2.csv"
VARIABLES_ESS = [
    'cntry',
    'gndr', # sexe (1 = homme, 2 = femme)
    'mascfel',
    'anweight',  # pondération : poids d'analyse ESS (= pspwght × pweight)
]

CODES_SPECIAUX_ESS = {
    'gndr': [9],  # 9 = sexe non renseigné
    'mascfel': [7, 8, 9],
}


def charger_ess(chemin: str | Path = CHEMIN_BRUT_ESS) -> pd.DataFrame:
  """Charge la base brute ESS11."""
  return pd.read_csv(chemin)


def nettoyer_ess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Nettoie l'ESS, remplace les codes spéciaux, crée les variables et agrège par pays."""
  colonnes_presentes = [col for col in VARIABLES_ESS if col in df.columns]
  df_subset = df[colonnes_presentes].copy()
  lignes_initiales = len(df_subset)

  for col, codes in CODES_SPECIAUX_ESS.items():
    if col in df_subset.columns:
      df_subset[col] = df_subset[col].replace(codes, np.nan)

  df_propre = df_subset.dropna(subset=VARIABLES_ESS).copy()
  print(f'Nombre de lignes éliminées (ESS) : {lignes_initiales - len(df_propre)}')
  print(f'Nombre de lignes restantes (ESS) : {len(df_propre)}')

  df_propre['sexe'] = df_propre['gndr'].map({1: 'homme', 2: 'femme'})

  colonnes_non_numeriques = [
      col
      for col in df_propre.select_dtypes(include=['object']).columns
      if col not in ('cntry', 'sexe')
  ]
  if colonnes_non_numeriques:
    print(f'⚠️ ALERTE ESS : Colonnes non numériques : {colonnes_non_numeriques}')
  else:
    print('✅ TEST RÉUSSI ESS : Toutes les variables à agréger sont numériques.')

  # Moyennes globales par pays (hors identifiants/poids)
  cols_moyenne = [
      c
      for c in df_propre.select_dtypes(include=[np.number]).columns
      if c not in ('gndr', 'anweight')
  ]
  df_agrege = df_propre.groupby('cntry', as_index=False)[cols_moyenne].mean()

  # Effectifs et scores mascfel pondérés par anweight, par pays et sexe
  effectifs = (
      df_propre.groupby(['cntry', 'sexe'])
      .size()
      .unstack('sexe', fill_value=0)
      .rename(columns={'homme': 'n_ess_homme', 'femme': 'n_ess_femme'})
  )
  scores_w = (
      df_propre.groupby(['cntry', 'sexe'])
      .apply(lambda g: np.average(g['mascfel'], weights=g['anweight']))
      .unstack('sexe')
      .rename(columns={'homme': 'mascfel_homme_w', 'femme': 'mascfel_femme_w'})
  )

  stats_sexe = effectifs.join(scores_w).reset_index()
  df_agrege = df_agrege.merge(stats_sexe, on='cntry', how='left').rename(
      columns={'cntry': 'code_pays'}
  )

  return df_propre, df_agrege


def executer_ess():
  if not CHEMIN_BRUT_ESS.exists():
    print(
        f' Fichier ESS introuvable à l’emplacement : {CHEMIN_BRUT_ESS}. Étape'
        ' ignorée.'
    )
    return

  print('--- Début du traitement de l’ESS 11 ---')
  brut = charger_ess(CHEMIN_BRUT_ESS)
  propre, agrege = nettoyer_ess(brut)

  DOSSIER_PROCESSED.mkdir(parents=True, exist_ok=True)
  propre.to_csv(DOSSIER_PROCESSED / 'ess11_nettoye.csv', index=False)
  agrege.to_csv(DOSSIER_PROCESSED / 'ess11_agrege_pays.csv', index=False)
  print(f' Fichiers ESS sauvegardés dans {DOSSIER_PROCESSED}/\n')

# ==========================================
# 2. TRAITEMENT ET AGRÉGATION DE L'ISSP 2020
# ==========================================

CHEMIN_BRUT_ISSP = RACINE / "Data" / "raw" / "ISSP_2020.dta"
CHEMIN_PROPRE_ISSP = DOSSIER_PROCESSED / "issp_clean.csv"
CHEMIN_PAYS_ISSP = DOSSIER_PROCESSED / "issp_par_pays.csv"

COLONNES_ISSP = ["country", "c_alphan", "SEX", "v50", "WEIGHT"]
CODE_VEGETARIEN = -4
CODES_SEXE = {1: "homme", 2: "femme"}


def charger_issp(chemin: str | Path = CHEMIN_BRUT_ISSP) -> pd.DataFrame:
  """Charge uniquement les colonnes utiles de la base ISSP."""
  return pd.read_stata(chemin, columns=COLONNES_ISSP, convert_categoricals=False)


def nettoyer_issp(
    df: pd.DataFrame, vegetariens_a_zero: bool = True
) -> pd.DataFrame:
  """Renomme les variables, recode les non-réponses et le sexe."""
  out = df.rename(
      columns={
          "c_alphan": "country_iso2",
          "country": "country_code",
          "v50": "jours_viande_rouge",
          "WEIGHT": "poids",
      }
  ).copy()

  conso = out["jours_viande_rouge"].astype(float)
  if vegetariens_a_zero:
    conso = conso.mask(conso == CODE_VEGETARIEN, 0)
  out["jours_viande_rouge"] = conso.where(conso >= 0)

  out["sexe"] = out["SEX"].map(CODES_SEXE)
  out["country_iso2"] = out["country_iso2"].str.strip()

  out = out.dropna(subset=["sexe", "jours_viande_rouge", "poids"])
  return out[
      ["country_iso2", "country_code", "sexe", "jours_viande_rouge", "poids"]
  ]


def moyenne_ponderee(
    groupe: pd.DataFrame, var: str = "jours_viande_rouge", poids: str = "poids"
) -> float:
  """Moyenne de `var` pondérée par `poids` au sein d'un groupe."""
  return np.average(groupe[var], weights=groupe[poids])


def agreger_par_pays_issp(df: pd.DataFrame) -> pd.DataFrame:
  """Agrège la base individuelle nettoyée de l'ISSP au niveau pays."""
  moyennes = (
      df.groupby(["country_iso2", "sexe"])[["jours_viande_rouge", "poids"]]
      .apply(moyenne_ponderee)
      .unstack("sexe")
      .rename(columns={"homme": "conso_homme", "femme": "conso_femme"})
  )
  effectifs = (
      df.groupby(["country_iso2", "sexe"])
      .size()
      .unstack("sexe")
      .rename(columns={"homme": "n_homme", "femme": "n_femme"})
  )

  pays = moyennes.join(effectifs)
  pays.columns.name = None
  return pays[["conso_homme", "conso_femme", "n_homme", "n_femme"]]


def ajouter_ecart_sexes_issp(pays: pd.DataFrame) -> pd.DataFrame:
  """Ajoute la variable ecart_h_f = consommation Homme - consommation Femme."""
  pays = pays.copy()
  pays["ecart_h_f"] = pays["conso_homme"] - pays["conso_femme"]
  return pays


def executer_issp():
  if not CHEMIN_BRUT_ISSP.exists():
    print(
        f"Fichier ISSP introuvable à l’emplacement : {CHEMIN_BRUT_ISSP}."
        " Étape ignorée."
    )
    return

  print("--- Début du traitement et de l'agrégation de l'ISSP 2020 ---")
  brut = charger_issp(CHEMIN_BRUT_ISSP)
  propre = nettoyer_issp(brut)

  DOSSIER_PROCESSED.mkdir(parents=True, exist_ok=True)
  propre.to_csv(CHEMIN_PROPRE_ISSP, index=False)

  # Étape d'agrégation pays pour l'ISSP
  pays = ajouter_ecart_sexes_issp(agreger_par_pays_issp(propre))
  pays.to_csv(CHEMIN_PAYS_ISSP)

  print(f"Base brute (ISSP)    : {len(brut)} répondants")
  print(
      f"Base nettoyée (ISSP) : {len(propre)} répondants"
      f" ({len(brut) - len(propre)} lignes supprimées)"
  )
  print(f" Fichier ISSP propre sauvegardé dans {CHEMIN_PROPRE_ISSP}")
  print(
      f" Fichier ISSP agrégé par pays sauvegardé dans {CHEMIN_PAYS_ISSP}\n"
  )
  print(pays.sort_values("ecart_h_f", ascending=False).round(2))


# ==========================================
#  POINT D'ENTRÉE GLOBAL
# ==========================================
def main():
  executer_ess()
  executer_issp()
  print(" Traitement global de toutes les bases terminé avec succès !")


if __name__ == "__main__":
  main()