from pathlib import Path
import pandas as pd
import country_converter as coco   # pip install country_converter

NOM_PROJET = "Projet_01_ENSAE_2A"


def trouver_racine() -> Path:
    """Retrouve la racine du projet, quel que soit l'emplacement du script."""
    ici = Path(__file__).resolve().parent
    candidats = [
        ici,                        # script à la racine du projet
        ici / NOM_PROJET,           # script à côté du dossier du projet (ex. ~/work)
        *ici.parents,               # script dans un sous-dossier
        Path.home() / "work" / NOM_PROJET,
    ]
    for c in candidats:
        if (c / "Data").is_dir():
            return c
    raise FileNotFoundError(
        f"Dossier 'Data' introuvable. Chemins testés : {[str(c) for c in candidats]}"
    )


RACINE = trouver_racine()
PROC = RACINE / "Data" / "processed"
print("Racine du projet :", RACINE)

CHEMIN_ESS  = PROC / "ess_par_pays.csv"
CHEMIN_ISSP = PROC / "issp_par_pays.csv"
CHEMIN_GDP  = RACINE / "Data" / "raw" / "gdp_world_countries_percapita.csv"
CHEMIN_SORTIE = PROC / "table_jointe.csv"

ANNEE_PIB = "2022"   # à ajuster : ESS11 date de 2023, ISSP de 2020
SKIPROWS_GDP = 0     # mettre 4 si le CSV commence par des lignes de métadonnées

ess  = pd.read_csv(CHEMIN_ESS).rename(columns={"code_pays": "iso2"})
issp = pd.read_csv(CHEMIN_ISSP).rename(columns={"country_iso2": "iso2"})

# Contrôles avant jointure
assert ess["iso2"].is_unique and issp["iso2"].is_unique, "Doublons de pays"
print("Seulement dans ESS :", sorted(set(ess["iso2"]) - set(issp["iso2"])))
print("Seulement dans ISSP:", sorted(set(issp["iso2"]) - set(ess["iso2"])))

# 1. ESS + ISSP sur le code ISO2
df = ess.merge(issp, on="iso2", how="inner", validate="one_to_one")

# 2. Ajout du code ISO3, clé de la table PIB
df["iso3"] = coco.convert(df["iso2"], src="ISO2", to="ISO3")

# 3. PIB par habitant (Banque Mondiale)
gdp = pd.read_csv(CHEMIN_GDP, skiprows=SKIPROWS_GDP)
gdp = gdp[["Country Code", ANNEE_PIB]].rename(
    columns={"Country Code": "iso3", ANNEE_PIB: "pib_hab"}
)
df = df.merge(gdp, on="iso3", how="left", validate="one_to_one")

# Vérifications finales
print(df.shape)
print(df.isna().sum()[lambda s: s > 0])

CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CHEMIN_SORTIE, index=False)
print("Table sauvegardée dans", CHEMIN_SORTIE)