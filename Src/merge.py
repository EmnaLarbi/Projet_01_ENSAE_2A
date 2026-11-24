from pathlib import Path
import country_converter as coco  # pip install country_converter
import pandas as pd

# Chemins directs basés sur l'emplacement fixe des fichiers
RACINE = Path(__file__).resolve().parent.parent
PROC = RACINE / "Data" / "processed"
print("Racine du projet :", RACINE)

CHEMIN_ESS = PROC / "ess11_agrege_pays.csv"
CHEMIN_ISSP = PROC / "issp_par_pays.csv"
# PIB par habitant en PPA (dollars internationaux constants de 2021), produit par Src/data_upload.py
# à partir de l'API de la Banque mondiale. Format long : iso3, pays, annee, pib_hab_ppa.
CHEMIN_GDP = RACINE / "Data" / "raw" / "pib_hab_ppa_banque_mondiale.csv"
CHEMIN_SORTIE = PROC / "table_jointe.csv"

ANNEE_PIB = 2022  # Année cible pour le PIB par habitant


def charger_pib() -> pd.DataFrame:
  """Charge le PIB par habitant en PPA ; le télécharge par l'API (Src/data_upload.py) s'il est absent."""
  if not CHEMIN_GDP.exists():
    print(f"{CHEMIN_GDP.name} absent : téléchargement par l'API de la Banque mondiale.")
    import data_upload  # Src/data_upload.py, dans le même dossier que ce script
    data_upload.main()
  return pd.read_csv(CHEMIN_GDP)


def charger_et_preparer_donnees() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Charge les tables ESS, ISSP et PIB et homogénéise les clés pays (iso2)."""
  ess = pd.read_csv(CHEMIN_ESS).rename(columns={"code_pays": "iso2"})
  issp = pd.read_csv(CHEMIN_ISSP).rename(columns={"country_iso2": "iso2"})
  gdp = charger_pib()
  return ess, issp, gdp


def fusionner_donnees(
    ess: pd.DataFrame, issp: pd.DataFrame, gdp: pd.DataFrame
) -> pd.DataFrame:
  """Réalise les contrôles, les jointures et l'ajout du code ISO3 pour le PIB."""
  # Contrôles avant jointure
  assert (
      ess["iso2"].is_unique and issp["iso2"].is_unique
  ), "⚠️ Doublons de pays détectés"
  print("Seulement dans ESS :", sorted(set(ess["iso2"]) - set(issp["iso2"])))
  print("Seulement dans ISSP:", sorted(set(issp["iso2"]) - set(ess["iso2"])))

  # 1. Fusion ESS + ISSP sur le code ISO2
  df = ess.merge(issp, on="iso2", how="inner", validate="one_to_one")

  # 2. Ajout du code ISO3, clé de la table PIB
  df["iso3"] = coco.convert(df["iso2"], src="ISO2", to="ISO3")

  # 3. Préparation et jointure du PIB par habitant en PPA (Banque Mondiale), année ANNEE_PIB
  gdp_subset = gdp.loc[gdp["annee"] == ANNEE_PIB, ["iso3", "pib_hab_ppa"]].rename(
      columns={"pib_hab_ppa": "pib_hab"}
  )
  df = df.merge(gdp_subset, on="iso3", how="left", validate="one_to_one")

  return df


def main() -> None:
  """Fonction principale pour exécuter le pipeline de fusion et l'export."""
  ess, issp, gdp = charger_et_preparer_donnees()
  df = fusionner_donnees(ess, issp, gdp)

  # Vérifications finales
  print(f"Dimensions de la table finale : {df.shape}")
  valeurs_manquantes = df.isna().sum()
  print("Valeurs manquantes par colonne (si > 0) :")
  print(valeurs_manquantes[valeurs_manquantes > 0])

  # Sauvegarde
  CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(CHEMIN_SORTIE, index=False)
  print(f"💾 Table jointe sauvegardée avec succès dans {CHEMIN_SORTIE}")


if __name__ == "__main__":
  main()