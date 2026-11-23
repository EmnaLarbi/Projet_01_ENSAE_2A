from pathlib import Path
import numpy as np
import pandas as pd

# Chemins de base globaux
RACINE = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path.cwd()
DOSSIER_PROCESSED = RACINE / "Data" / "processed"

# ==========================================
# 1. TRAITEMENT DE L'ESS 11
# ==========================================

# 1. TRAITEMENT DE L'ESS 11
# ==========================================
# Les lignes ajoutées sont marquées « # AJOUT ». Tout le reste est inchangé.

CHEMIN_BRUT_ESS = RACINE / "Data" / "raw" / "ESS11e04_2.csv"
VARIABLES_ESS = [
    'cntry',
    'gndr',  # AJOUT : sexe (1 = homme, 2 = femme)
    'bctprd',
    'pslplonl',
    'alcfreq',
    'mascfel',
    'impbemw',
    'polintr',
    'wrclmch',
    'ccrdprs',
    'ccnthum',
    'nwspol',
    'netustm',
    'impenva',
    'anweight',  # AJOUT (pondération) : poids d'analyse ESS (= pspwght × pweight)
]

# AJOUT (pondération) : seules ces variables déterminent la suppression d'une ligne
VARIABLES_CLES_ESS = ['cntry', 'gndr', 'mascfel', 'anweight']  # AJOUT (pondération)

CODES_SPECIAUX_ESS = {
    'gndr': [9],  # AJOUT : 9 = sexe non renseigné
    'bctprd': [7, 8, 9],
    'pslplonl': [7, 8, 9],
    'alcfreq': [77, 88, 99],
    'mascfel': [7, 8, 9],
    'impbemw': [66, 77, 88, 99],
    'polintr': [7, 8, 9],
    'wrclmch': [6, 7, 8, 9],
    'ccrdprs': [66, 77, 88, 99],
    'nwspol': [7777, 8888, 9999],
    'netustm': [7, 8, 9],
    'impenva': [66, 77, 88, 99],
}


def charger_ess(chemin: str | Path = CHEMIN_BRUT_ESS) -> pd.DataFrame:
  """Charge la base brute ESS11."""
  return pd.read_csv(chemin)


def nettoyer_ess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Nettoie l'ESS, gère les codes spéciaux, crée les variables et agrège par pays."""
  colonnes_presentes = [col for col in VARIABLES_ESS if col in df.columns]
  df_subset = df[colonnes_presentes].copy()
  lignes_initiales = len(df_subset)

  # Remplacement des codes spécifiques par NaN
  for col, codes in CODES_SPECIAUX_ESS.items():
    if col in df_subset.columns:
      df_subset[col] = df_subset[col].replace(codes, np.nan)

  # Gestion spécifique de ccnthum (climatosceptique)
  if 'ccnthum' in df.columns:
    df_subset['climatosceptique'] = np.where(df['ccnthum'] == 55, 1, 0)
    df_subset.loc[df['ccnthum'].isin([77, 88, 99]), 'climatosceptique'] = np.nan
    df_subset['ccnthum'] = df_subset['ccnthum'].replace(55, np.nan)

  # Suppression des lignes comportant des NaN
  # MODIF (pondération) : suppression limitée aux variables clés, avant : df_subset.dropna()
  df_propre = df_subset.dropna(subset=VARIABLES_CLES_ESS)  # MODIF (pondération)
  lignes_eliminees = lignes_initiales - len(df_propre)
  print(f'Nombre de lignes éliminées (ESS) : {lignes_eliminees}')
  print(f"Nombre de lignes restantes (ESS) : {len(df_propre)}")

  # AJOUT : libellé du sexe dans la table individuelle (ess11_nettoye.csv)
  df_propre = df_propre.assign(sexe=df_propre['gndr'].map({1: 'homme', 2: 'femme'}))  # AJOUT

  # Vérification des types avant agrégation
  colonnes_non_numeriques = df_propre.select_dtypes(
      include=['object']
  ).columns.tolist()
  colonnes_non_numeriques = [
      col for col in colonnes_non_numeriques if col not in ('cntry', 'sexe')  # AJOUT : 'sexe' est un libellé
  ]
  if colonnes_non_numeriques:
    print(f'⚠️ ALERTE ESS : Colonnes non numériques : {colonnes_non_numeriques}')
  else:
    print('✅ TEST RÉUSSI ESS : Toutes les variables à agréger sont numériques.')

  # Agrégation par pays
  df_agrege = df_propre.groupby('cntry').mean(numeric_only=True).reset_index()
  df_agrege = df_agrege.drop(columns='gndr')  # AJOUT : la moyenne de gndr n'a pas de sens
  df_agrege = df_agrege.drop(columns='anweight')  # AJOUT (pondération) : la moyenne des poids n'a pas de sens

  # AJOUT : mascfel moyen par sexe et effectifs, par pays
  mascfel_sexe = (  # AJOUT
      df_propre.groupby(['cntry', 'sexe'])['mascfel']  # AJOUT
      .agg(['mean', 'size'])  # AJOUT
      .unstack('sexe')  # AJOUT
  )  # AJOUT
  mascfel_sexe.columns = [  # AJOUT
      f"{'mascfel' if stat == 'mean' else 'n_ess'}_{sexe}"  # AJOUT : mascfel_homme, n_ess_femme…
      for stat, sexe in mascfel_sexe.columns  # AJOUT
  ]  # AJOUT
  df_agrege = df_agrege.merge(mascfel_sexe.reset_index(), on='cntry', how='left')  # AJOUT

  # AJOUT (pondération) : mascfel moyen par pays et par sexe, pondéré par anweight
  mascfel_sexe_w = (  # AJOUT (pondération)
      df_propre.groupby(['cntry', 'sexe'])  # AJOUT (pondération)
      .apply(lambda g: np.average(g['mascfel'], weights=g['anweight']))  # AJOUT (pondération)
      .unstack('sexe')  # AJOUT (pondération)
  )  # AJOUT (pondération)
  mascfel_sexe_w.columns = [f'mascfel_{sexe}_w' for sexe in mascfel_sexe_w.columns]  # AJOUT (pondération) : mascfel_homme_w, mascfel_femme_w
  df_agrege = df_agrege.merge(mascfel_sexe_w.reset_index(), on='cntry', how='left')  # AJOUT (pondération)

  df_agrege = df_agrege.rename(columns={'cntry': 'code_pays'})

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