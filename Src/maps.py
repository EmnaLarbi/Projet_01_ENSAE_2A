"""
src/maps.py
===========

Carte choroplèthe interactive de l'Europe : fréquence déclarée de consommation
de viande rouge selon le sexe (ISSP 2020).

Projet : « Viande et masculinité : existe-t-il une corrélation entre les normes
de genre et la consommation de viande rouge en Europe ? »

Exécution (depuis la racine du projet) :

    python Src/maps.py                       # images PNG dans sorties/
    python Src/maps.py --geojson Data/raw/ne_50m_admin_0_countries.geojson   # sans Internet
    python Src/maps.py --strict              # échoue si le contrôle croisé détecte un écart

Sorties (dossier sorties/) : une image PNG par vue
    carte_viande_ecart.png, carte_viande_hommes.png, carte_viande_femmes.png, carte_viande_tous.png
Export des images : paquet `kaleido` (pip install kaleido ; puis plotly_get_chrome si besoin).

Principes statistiques
----------------------
* Toutes les moyennes sont PONDÉRÉES par la variable `poids` (jamais de moyenne simple).
* Les indicateurs par pays sont recalculés à partir de `issp_clean.csv` (nécessaire
  pour la vue « Tous les répondants ») puis CONTRÔLÉS contre `issp_par_pays.csv`.
* `jours_viande_rouge` est une FRÉQUENCE déclarée : aucune conversion en kg.
* Les intervalles de confiance affichés sont INDICATIFS (approximation de Kish,
  ne tenant pas compte du plan de sondage complet) : aucune conclusion de
  significativité n'est formulée par ce script.
"""

from __future__ import annotations

import argparse
import gettext
import html
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pycountry

# ---------------------------------------------------------------------------
# 0. PARAMÈTRES
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]          # racine du projet
DATA_DIR = ROOT / "Data" / "processed"
CLEAN_PATH = DATA_DIR / "issp_clean.csv"
AGG_PATH = DATA_DIR / "issp_par_pays.csv"
SORTIES = ROOT / "sorties"                          # dossier des sorties du projet
PREFIXE = "carte_viande"

COLONNES_CLEAN = ["country_iso2", "country_code", "sexe", "jours_viande_rouge", "poids"]
COLONNES_AGG = ["conso_homme", "conso_femme", "n_homme", "n_femme", "ecart_h_f"]

# Libellé de l'unité affiché dans la carte. À adapter à la modalité exacte de
# votre variable (ex. « jours par semaine »). Aucune conversion n'est faite.
LIBELLE_UNITE = "fréquence moyenne déclarée"

SEUIL_EFFECTIF_FAIBLE = 100   # effectif brut par sexe en dessous duquel on alerte
TOLERANCE_CONTROLE = 1e-3     # écart toléré entre recalcul et issp_par_pays.csv
Z_95 = 1.959964

# Pays retenus comme « Europe » (ISO 3166-1 alpha-2). Liste explicite pour que
# le périmètre soit un choix documenté et non un effet de bord.
# Exclus par choix : Turquie, Géorgie, Arménie, Azerbaïdjan, Kazakhstan, Chypre du Nord.
# Kosovo (XK) : pas de code ISO officiel → non cartographiable avec la géométrie Plotly.
EUROPE_ISO2 = {
    "AL", "AD", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK",
    "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LI",
    "LT", "LU", "LV", "MC", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT",
    "RO", "RS", "RU", "SE", "SI", "SK", "SM", "UA", "VA",
}

# Codes non-ISO fréquents dans les bases européennes (Eurostat, ISSP…).
ALIAS_ISO2 = {"UK": "GB", "EL": "GR"}

# Codages possibles de la variable sexe → libellé harmonisé.
CODAGE_SEXE = {
    "homme": "homme", "h": "homme", "m": "homme", "masculin": "homme",
    "male": "homme", "man": "homme", "1": "homme",
    "femme": "femme", "f": "femme", "féminin": "femme", "feminin": "femme",
    "female": "femme", "woman": "femme", "2": "femme",
}

# Palette (sobre, validée daltonisme) : divergente bleu ↔ rouge, milieu gris neutre.
COULEUR_NEG = "#2a78d6"     # femmes > hommes
COULEUR_NEUTRE = "#f0efec"  # ≈ 0
COULEUR_POS = "#e34948"     # hommes > femmes
ECHELLE_DIVERGENTE = [
    [0.0, "#1c4f8f"], [0.25, COULEUR_NEG], [0.5, COULEUR_NEUTRE],
    [0.75, COULEUR_POS], [1.0, "#9e2525"],
]
# Séquentielle mono-teinte (violet) pour les fréquences : distincte des deux
# pôles de l'écart afin d'éviter toute confusion entre les vues.
ECHELLE_SEQUENTIELLE = [[0.0, "#f1eef6"], [0.5, "#9e86c8"], [1.0, "#3f2a73"]]
GRIS_SANS_DONNEES = "#c8c7c1"  # nettement distinct du gris neutre (≈ 0)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("maps")


class ErreurDonnees(ValueError):
    """Erreur bloquante sur les données d'entrée."""


# ---------------------------------------------------------------------------
# 1. CHARGEMENT ET CONTRÔLES DES DONNÉES INDIVIDUELLES
# ---------------------------------------------------------------------------

def normaliser_iso2(code: object) -> str | None:
    """'de-w ' → 'DE' ; 'UK' → 'GB' ; NaN → None. Les sous-échantillons
    régionaux de l'ISSP (DE-W, BE-FLA, GB-GBN…) sont ramenés au pays."""
    if pd.isna(code):
        return None
    c = str(code).strip().upper().split("-")[0].split("_")[0]
    c = ALIAS_ISO2.get(c, c)
    return c if len(c) == 2 and c.isalpha() else None


def normaliser_sexe(valeur: object) -> str | None:
    if pd.isna(valeur):
        return None
    v = str(valeur).strip().lower()
    if v.endswith(".0"):            # 1.0 → "1" (CSV relu en float)
        v = v[:-2]
    return CODAGE_SEXE.get(v)


def charger_individuel(chemin: Path = CLEAN_PATH) -> pd.DataFrame:
    """Charge issp_clean.csv et applique tous les contrôles de cohérence."""
    if not chemin.exists():
        raise ErreurDonnees(f"Fichier introuvable : {chemin}")

    # Lecture en texte pour ne rien laisser deviner à pandas (codes pays, sexe).
    df = pd.read_csv(chemin, dtype=str)
    manquantes = [c for c in COLONNES_CLEAN if c not in df.columns]
    if manquantes:
        raise ErreurDonnees(f"Colonnes absentes de {chemin.name} : {manquantes}. "
                            f"Colonnes trouvées : {list(df.columns)}")
    n0 = len(df)
    log.info("issp_clean.csv : %d lignes lues", n0)

    # Types numériques : on signale toute valeur non convertible au lieu de l'ignorer.
    for col in ["jours_viande_rouge", "poids"]:
        brut = df[col]
        df[col] = pd.to_numeric(brut.str.replace(",", ".", regex=False), errors="coerce")
        non_num = brut.notna() & df[col].isna()
        if non_num.any():
            exemples = brut[non_num].unique()[:5]
            log.warning("%s : %d valeurs non numériques (ex. %s) → traitées comme manquantes",
                        col, non_num.sum(), list(exemples))

    # Valeurs manquantes
    for col in COLONNES_CLEAN:
        n_na = df[col].isna().sum()
        if n_na:
            log.warning("Valeurs manquantes dans %-20s : %d (%.1f %%)", col, n_na, 100 * n_na / n0)

    # Poids : doivent être strictement positifs
    poids_invalides = df["poids"].notna() & (df["poids"] <= 0)
    if poids_invalides.any():
        log.warning("poids ≤ 0 : %d lignes exclues", poids_invalides.sum())
    df = df[~poids_invalides]

    # Plage de la fréquence
    x = df["jours_viande_rouge"].dropna()
    log.info("jours_viande_rouge : min=%.2f  max=%.2f  modalités distinctes=%d",
             x.min(), x.max(), x.nunique())
    if (x < 0).any():
        raise ErreurDonnees("jours_viande_rouge contient des valeurs négatives : "
                            "codes de non-réponse ISSP non recodés en NA ?")

    # Sexe
    df["sexe_h"] = df["sexe"].map(normaliser_sexe)
    inconnus = df["sexe"].notna() & df["sexe_h"].isna()
    if inconnus.any():
        log.warning("Modalités de sexe non reconnues (%d lignes) : %s — exclues des vues "
                    "Hommes/Femmes, conservées dans « Tous »",
                    inconnus.sum(), sorted(df.loc[inconnus, "sexe"].unique())[:10])
    log.info("Répartition sexe : %s", df["sexe_h"].value_counts(dropna=False).to_dict())

    # Codes pays
    df["iso2"] = df["country_iso2"].map(normaliser_iso2)
    invalides = df["iso2"].isna()
    if invalides.any():
        log.warning("Codes ISO2 invalides ou manquants : %d lignes exclues (%s)",
                    invalides.sum(), sorted(df.loc[invalides, "country_iso2"].dropna().unique())[:10])
    fusions = (df.loc[~invalides]
                 .groupby("iso2")["country_iso2"]
                 .agg(lambda s: sorted(s.str.strip().str.upper().unique())))
    for iso2, bruts in fusions.items():
        if len(bruts) > 1:
            log.warning("Codes %s regroupés sous %s. Si ce sont des sous-échantillons "
                        "régionaux, vérifiez que `poids` est calibré au niveau NATIONAL.",
                        bruts, iso2)
        elif bruts[0] != iso2:
            log.info("Code %s harmonisé en %s", bruts[0], iso2)
    df = df[~invalides]

    controler_codes_pays(df)

    # Doublons : sans identifiant individuel on ne peut que signaler les lignes identiques.
    n_dup = df.duplicated(subset=COLONNES_CLEAN).sum()
    if n_dup:
        log.info("%d lignes strictement identiques (normal pour des réponses identiques ; "
                 "à vérifier s'il s'agit d'un double export)", n_dup)

    # Échantillon analytique : fréquence et poids renseignés
    avant = len(df)
    df = df.dropna(subset=["jours_viande_rouge", "poids"])
    log.info("Échantillon analytique : %d lignes (%d exclues pour fréquence ou poids manquant)",
             len(df), avant - len(df))
    return df


def controler_codes_pays(df: pd.DataFrame) -> None:
    """Contrôle explicite ISO2 ↔ pycountry ↔ country_code ISSP."""
    # 1) chaque ISO2 doit exister dans la norme ISO 3166-1
    inconnus = sorted(c for c in df["iso2"].unique() if pycountry.countries.get(alpha_2=c) is None)
    if inconnus:
        log.warning("ISO2 absents de la norme ISO 3166-1 (seront exclus de la carte) : %s", inconnus)

    # 2) correspondance ISO2 ↔ country_code
    paires = df[["iso2", "country_code"]].dropna().drop_duplicates()
    multi = paires.groupby("iso2")["country_code"].nunique()
    for iso2 in multi[multi > 1].index:
        log.warning("%s associé à plusieurs country_code : %s (sous-échantillons ?)",
                    iso2, sorted(paires.loc[paires.iso2 == iso2, "country_code"]))
    multi_inv = paires.groupby("country_code")["iso2"].nunique()
    for code in multi_inv[multi_inv > 1].index:
        raise ErreurDonnees(f"country_code {code} associé à plusieurs ISO2 : "
                            f"{sorted(paires.loc[paires.country_code == code, 'iso2'])}")

    # 3) si country_code ressemble au code numérique ISO (ce qui est l'usage ISSP),
    #    on le compare au code numérique de pycountry.
    discordances = []
    for iso2, code in paires.itertuples(index=False):
        pays = pycountry.countries.get(alpha_2=iso2)
        code_num = pd.to_numeric(code, errors="coerce")
        if pays is None or pd.isna(code_num) or code_num > 999:
            continue
        if int(code_num) != int(pays.numeric):
            discordances.append(f"{iso2}: ISSP={code} / ISO={pays.numeric}")
    if discordances:
        log.warning("country_code ≠ code numérique ISO pour : %s — vérifier la nomenclature "
                    "(ISSP peut utiliser ses propres codes)", discordances)
    else:
        log.info("Correspondance ISO2 ↔ country_code : aucune incohérence détectée")


# ---------------------------------------------------------------------------
# 2. AGRÉGATION PONDÉRÉE PAR PAYS
# ---------------------------------------------------------------------------

def _stats_ponderees(g: pd.DataFrame) -> pd.Series:
    """Moyenne pondérée, effectif brut, effectif efficace de Kish et erreur-type
    approchée de la moyenne pondérée."""
    w = g["poids"].to_numpy(float)
    x = g["jours_viande_rouge"].to_numpy(float)
    sw = w.sum()
    m = np.dot(w, x) / sw
    n_eff = sw ** 2 / np.dot(w, w)
    var = np.dot(w, (x - m) ** 2) / sw * (n_eff / (n_eff - 1)) if n_eff > 1 else np.nan
    return pd.Series({"conso": m, "n": len(g), "n_eff": n_eff,
                      "se": np.sqrt(var / n_eff) if n_eff > 1 else np.nan})


def agreger_par_pays(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par pays : moyennes pondérées hommes / femmes / tous + écart."""
    blocs = []
    for groupe, sous_df in [("homme", df[df.sexe_h == "homme"]),
                            ("femme", df[df.sexe_h == "femme"]),
                            ("tous", df)]:
        s = sous_df.groupby("iso2").apply(_stats_ponderees, include_groups=False)
        s.columns = [f"{c}_{groupe}" for c in s.columns]
        blocs.append(s)
    res = pd.concat(blocs, axis=1)
    for c in ["n_homme", "n_femme", "n_tous"]:
        res[c] = res[c].fillna(0).astype(int)

    res["ecart_h_f"] = res["conso_homme"] - res["conso_femme"]
    # IC 95 % INDICATIF de l'écart (échantillons H/F supposés indépendants)
    se_ecart = np.sqrt(res["se_homme"] ** 2 + res["se_femme"] ** 2)
    res["ic_bas"] = res["ecart_h_f"] - Z_95 * se_ecart
    res["ic_haut"] = res["ecart_h_f"] + Z_95 * se_ecart
    res.index.name = "iso2"
    return res.reset_index()


# ---------------------------------------------------------------------------
# 3. CONTRÔLE CROISÉ AVEC issp_par_pays.csv
# ---------------------------------------------------------------------------

def controler_fichier_agrege(recalc: pd.DataFrame, chemin: Path = AGG_PATH,
                             strict: bool = False) -> None:
    """Compare les moyennes recalculées à celles produites par le script de nettoyage."""
    if not chemin.exists():
        log.warning("%s introuvable : contrôle croisé impossible", chemin.name)
        return
    agg = pd.read_csv(chemin)
    manquantes = [c for c in COLONNES_AGG if c not in agg.columns]
    if manquantes:
        log.warning("%s : colonnes attendues absentes %s", chemin.name, manquantes)

    # Identifiant pays : colonne explicite, ou index sauvegardé sans nom.
    col_pays = next((c for c in ["country_iso2", "iso2", "pays", "country", "Unnamed: 0"]
                     if c in agg.columns), None)
    if col_pays is None:
        msg = (f"{chemin.name} ne contient pas d'identifiant pays : contrôle croisé impossible. "
               "Ajoutez `country_iso2` (ex. `.reset_index()` avant `to_csv`).")
        if strict:
            raise ErreurDonnees(msg)
        log.warning(msg)
        return

    agg["iso2"] = agg[col_pays].map(normaliser_iso2)
    dup = agg["iso2"][agg["iso2"].duplicated()].unique()
    if len(dup):
        log.warning("%s : pays en double après normalisation : %s", chemin.name, list(dup))

    # Cohérence interne : ecart_h_f == conso_homme − conso_femme
    if {"ecart_h_f", "conso_homme", "conso_femme"} <= set(agg.columns):
        diff = (agg.ecart_h_f - (agg.conso_homme - agg.conso_femme)).abs().max()
        if diff > TOLERANCE_CONTROLE:
            log.warning("%s : ecart_h_f ≠ conso_homme − conso_femme (max |Δ| = %.4f)",
                        chemin.name, diff)

    comp = agg.drop_duplicates("iso2").merge(recalc, on="iso2", how="outer",
                                             suffixes=("_fichier", ""), indicator=True)
    seuls = comp.loc[comp._merge != "both", ["iso2", "_merge"]]
    if not seuls.empty:
        log.warning("Pays présents dans un seul des deux fichiers : %s",
                    seuls.to_dict("records"))

    problemes = []
    for col in COLONNES_AGG:
        if f"{col}_fichier" not in comp.columns:
            continue
        d = (comp[f"{col}_fichier"] - comp[col]).abs()
        tol = 0.5 if col.startswith("n_") else TOLERANCE_CONTROLE
        mauvais = comp.loc[d > tol, "iso2"].tolist()
        if mauvais:
            problemes.append(f"{col} (max |Δ| = {d.max():.4f}) : {mauvais}")
    if problemes:
        msg = ("Divergences entre le recalcul pondéré et issp_par_pays.csv :\n  - "
               + "\n  - ".join(problemes)
               + "\n  Causes possibles : moyenne non pondérée, filtrage différent des NA, "
                 "sous-échantillons régionaux, n pondérés au lieu de n bruts.")
        if strict:
            raise ErreurDonnees(msg)
        log.warning(msg)
    else:
        log.info("Contrôle croisé avec %s : OK (tolérance %.0e)", chemin.name, TOLERANCE_CONTROLE)


# ---------------------------------------------------------------------------
# 4. PRÉPARATION GÉOGRAPHIQUE
# ---------------------------------------------------------------------------

def _traducteur_fr():
    try:
        return gettext.translation("iso3166-1", pycountry.LOCALES_DIR, languages=["fr"]).gettext
    except OSError:
        return lambda s: s


def ajouter_geographie(df: pd.DataFrame, europe_seulement: bool = True) -> pd.DataFrame:
    """Ajoute ISO3 et nom français ; restreint au périmètre européen.
    Chaque exclusion est journalisée (aucune perte silencieuse)."""
    fr = _traducteur_fr()
    iso3, noms = [], []
    for c in df["iso2"]:
        p = pycountry.countries.get(alpha_2=c)
        iso3.append(p.alpha_3 if p else None)
        noms.append(fr(getattr(p, "common_name", None) or p.name) if p else c)
    out = df.assign(iso3=iso3, nom=noms)

    sans_iso3 = out[out.iso3.isna()]
    if not sans_iso3.empty:
        log.warning("Exclus (pas de code ISO3) : %s", sans_iso3.iso2.tolist())
    out = out.dropna(subset=["iso3"])

    if europe_seulement:
        hors = out[~out.iso2.isin(EUROPE_ISO2)]
        if not hors.empty:
            log.info("Hors périmètre européen (non cartographiés) : %s", sorted(hors.iso2))
        out = out[out.iso2.isin(EUROPE_ISO2)]

    if out.iso3.duplicated().any():
        raise ErreurDonnees(f"ISO3 en double après jointure : "
                            f"{out.loc[out.iso3.duplicated(), 'iso3'].tolist()}")
    log.info("%d pays européens cartographiés : %s", len(out), " ".join(sorted(out.iso2)))
    return out.sort_values("nom").reset_index(drop=True)


def charger_geojson(chemin: Path, iso3_attendus: set[str]) -> tuple[dict, str]:
    """Charge un GeoJSON (ex. Natural Earth admin-0) et vérifie la jointure.
    Retourne (geojson, featureidkey). Sans GeoPandas : json suffit ici."""
    gj = json.loads(Path(chemin).read_text(encoding="utf-8"))
    props = gj["features"][0]["properties"]
    # Natural Earth : ISO_A3 vaut "-99" pour la France et la Norvège dans certaines
    # versions → on privilégie ISO_A3_EH puis ADM0_A3.
    cle = next((k for k in ["ISO_A3_EH", "ISO_A3", "ADM0_A3", "iso_a3", "ISO3", "iso3"]
                if k in props), None)
    if cle is None:
        raise ErreurDonnees(f"Aucune propriété ISO3 dans {chemin} : {list(props)}")
    disponibles = {f["properties"].get(cle) for f in gj["features"]}
    absents = sorted(iso3_attendus - disponibles)
    if absents:
        raise ErreurDonnees(f"Jointure géographique incomplète : {absents} absents du GeoJSON "
                            f"(propriété {cle})")
    # On n'embarque que les pays européens (données + fond gris des pays sans donnée).
    europe_iso3 = {pycountry.countries.get(alpha_2=c).alpha_3 for c in EUROPE_ISO2
                   if pycountry.countries.get(alpha_2=c)} | iso3_attendus
    gj = {"type": "FeatureCollection",
          "features": [f for f in gj["features"]
                       if f["properties"].get(cle) in europe_iso3]}
    log.info("GeoJSON : jointure complète sur la propriété %s", cle)
    return gj, f"properties.{cle}"


# ---------------------------------------------------------------------------
# 5. FIGURE PLOTLY
# ---------------------------------------------------------------------------

def _fmt(v: float, signe: bool = False) -> str:
    if pd.isna(v):
        return "n.d."
    s = f"{v:+.2f}" if signe else f"{v:.2f}"
    return s.replace(".", ",").replace("-", "−")


def _texte_survol(r: pd.Series) -> str:
    alerte = ""
    if min(r.n_homme, r.n_femme) < SEUIL_EFFECTIF_FAIBLE:
        alerte = f"<br><i>⚠ Effectif faible (< {SEUIL_EFFECTIF_FAIBLE} pour un sexe)</i>"
    return (f"<b>{html.escape(r.nom)}</b><br>"
            f"Hommes : {_fmt(r.conso_homme)}  (n = {_n(r.n_homme)})<br>"
            f"Femmes : {_fmt(r.conso_femme)}  (n = {_n(r.n_femme)})<br>"
            f"Ensemble : {_fmt(r.conso_tous)}  (n = {_n(r.n_tous)})<br>"
            f"<b>Écart H − F : {_fmt(r.ecart_h_f, True)}</b><br>"
            f"IC 95 % indicatif : [{_fmt(r.ic_bas, True)} ; {_fmt(r.ic_haut, True)}]"
            f"{alerte}")


def construire_figure(geo: pd.DataFrame, geojson: tuple[dict, str] | None = None) -> go.Figure:
    """Quatre couches (Écart, Hommes, Femmes, Tous) commutées par un menu."""
    geo = geo.assign(survol=geo.apply(_texte_survol, axis=1))

    # Bornes : symétriques pour l'écart (0 = couleur neutre),
    # communes aux trois vues de fréquence pour qu'elles soient comparables.
    borne_ecart = float(np.nanmax(np.abs(geo.ecart_h_f))) or 1.0
    freq = geo[["conso_homme", "conso_femme", "conso_tous"]].to_numpy()
    fmin, fmax = float(np.nanmin(freq)), float(np.nanmax(freq))

    couches = [
        ("ecart_h_f", "Écart hommes − femmes", ECHELLE_DIVERGENTE, -borne_ecart, borne_ecart,
         "Écart H − F<br><sup>rouge : H > F<br>bleu : F > H</sup>"),
        ("conso_homme", "Hommes", ECHELLE_SEQUENTIELLE, fmin, fmax, "Hommes<br><sup>" + LIBELLE_UNITE + "</sup>"),
        ("conso_femme", "Femmes", ECHELLE_SEQUENTIELLE, fmin, fmax, "Femmes<br><sup>" + LIBELLE_UNITE + "</sup>"),
        ("conso_tous", "Tous les répondants", ECHELLE_SEQUENTIELLE, fmin, fmax, "Ensemble<br><sup>" + LIBELLE_UNITE + "</sup>"),
    ]

    fig = go.Figure()
    n_fond = 0
    if geojson:
        # Fond gris (pays européens sans données) : la géométrie Plotly par défaut
        # n'est pas utilisée, donc rien n'est téléchargé à l'ouverture.
        cle = geojson[1].split(".", 1)[1]
        tous = [f["properties"][cle] for f in geojson[0]["features"]]
        fig.add_trace(go.Choropleth(
            geojson=geojson[0], featureidkey=geojson[1], locations=tous,
            locationmode="geojson-id",
            z=[0] * len(tous), colorscale=[[0, GRIS_SANS_DONNEES], [1, GRIS_SANS_DONNEES]],
            showscale=False, hoverinfo="skip", marker_line_color="white",
            marker_line_width=0.6, name="fond"))
        n_fond = 1
    for i, (col, nom, echelle, zmin, zmax, titre_legende) in enumerate(couches):
        d = geo.dropna(subset=[col])             # jamais de pays sans donnée
        kw = dict(locations=d.iso3, z=d[col], text=d.survol,
                  hovertemplate="%{text}<extra></extra>",
                  colorscale=echelle, zmin=zmin, zmax=zmax,
                  marker_line_color="white", marker_line_width=0.6,
                  colorbar=dict(title=dict(text=titre_legende, font_size=12),
                                thickness=12, len=0.65, x=0.98,
                                tickformat=".2f", outlinewidth=0),
                  name=nom, visible=(i == 0))
        if col == "ecart_h_f":
            kw["zmid"] = 0
        if geojson:
            kw.update(geojson=geojson[0], featureidkey=geojson[1], locationmode="geojson-id")
        else:
            kw["locationmode"] = "ISO-3"
        fig.add_trace(go.Choropleth(**kw))

    boutons = [dict(label=nom, method="update",
                    args=[{"visible": [True] * n_fond + [j == i for j in range(len(couches))]}])
               for i, (_, nom, *_rest) in enumerate(couches)]

    fig.update_layout(
        updatemenus=[dict(buttons=boutons, direction="down", x=0.01, y=0.99,
                          xanchor="left", yanchor="top", bgcolor="white",
                          bordercolor="#c9c8c2", font=dict(size=13))],
        margin=dict(l=0, r=0, t=10, b=0), height=640,
        paper_bgcolor="#fcfcfb",
        font=dict(family="Georgia, 'Times New Roman', serif", color="#0b0b0b"),
        hoverlabel=dict(bgcolor="white", bordercolor="#c9c8c2",
                        font=dict(family="Georgia, serif", size=13, color="#0b0b0b")),
    )
    if geojson:
        # visible=False coupe toutes les couches de fond Plotly → aucun téléchargement.
        geo_kw = dict(visible=False, bgcolor="#fcfcfb", projection_type="conic conformal",
                      projection_parallels=[43, 62], projection_rotation_lon=12,
                      lataxis_range=[34, 71.5], lonaxis_range=[-25, 42])
    else:
        # Géométrie Plotly intégrée (Natural Earth), chargée depuis cdn.plot.ly à l'ouverture.
        geo_kw = dict(scope="europe", resolution=50, showframe=False,
                      showcoastlines=False, showcountries=True, countrycolor="white",
                      showland=True, landcolor=GRIS_SANS_DONNEES, showocean=True,
                      oceancolor="#fcfcfb", showlakes=False, bgcolor="#fcfcfb",
                      projection_type="conic conformal",
                      lataxis_range=[34, 71.5], lonaxis_range=[-25, 42])
    fig.update_geos(**geo_kw)
    return fig


# ---------------------------------------------------------------------------
# 6. INDICATEURS ET EXPORT DES IMAGES
# ---------------------------------------------------------------------------

def indicateurs(geo: pd.DataFrame) -> dict:
    e = geo.dropna(subset=["ecart_h_f"])
    imax, imin = e.ecart_h_f.idxmax(), e.ecart_h_f.idxmin()
    return {
        "n_pays": len(e),
        "ecart_moyen": e.ecart_h_f.mean(),              # chaque pays compte pour 1
        "n_total": int(e.n_tous.sum()),
        "max": (e.at[imax, "nom"], e.at[imax, "ecart_h_f"]),
        "min": (e.at[imin, "nom"], e.at[imin, "ecart_h_f"]),
        "faibles": e.loc[e[["n_homme", "n_femme"]].min(axis=1) < SEUIL_EFFECTIF_FAIBLE, "nom"].tolist(),
    }


def _n(v: int) -> str:
    return f"{v:,}".replace(",", "\u202f")


# Fond de carte vide : avec --geojson, les contours viennent du fichier GeoJSON et
# plotly n'a pas besoin de son propre fond (téléchargé sinon depuis cdn.plot.ly).
_FOND_VIDE = json.dumps({"type": "Topology", "arcs": [], "objects": {
    k: {"type": "GeometryCollection", "geometries": []}
    for k in ["coastlines", "land", "ocean", "lakes", "rivers", "countries", "subunits"]}})


def exporter_png(fig: go.Figure, vues: list[tuple[str, str]], dossier: Path,
                 source: str, geometrie_embarquee: bool) -> list[Path]:
    """Exporte une image PNG par vue de la carte (sans menu, avec titre et source).
    `vues` : liste (nom de fichier sans extension, titre), dans l'ordre des couches."""
    if geometrie_embarquee:
        d = Path(tempfile.mkdtemp())
        for nom in ["world_110m", "world_50m", "europe_110m", "europe_50m"]:
            (d / f"{nom}.json").write_text(_FOND_VIDE, encoding="utf-8")
        pio.defaults.topojson = d.as_uri() + "/"

    dossier.mkdir(parents=True, exist_ok=True)
    n_fond = len(fig.data) - len(vues)
    chemins = []
    for i, (nom, titre) in enumerate(vues):
        f = go.Figure(fig)
        for k, trace in enumerate(f.data):
            trace.visible = k < n_fond or k == n_fond + i
        f.layout.updatemenus = ()                       # pas de menu sur une image fixe
        f.update_layout(
            width=1100, height=750, margin=dict(l=10, r=10, t=60, b=40),
            title=dict(text=titre, x=0.02, font_size=20),
            annotations=[dict(text=source, x=0.02, y=-0.04, xref="paper", yref="paper",
                              showarrow=False, xanchor="left",
                              font=dict(size=12, color="#52514e"))])
        chemin = dossier / f"{nom}.png"
        f.write_image(chemin, scale=2)
        chemins.append(chemin)
        log.info("Image écrite : %s", chemin)
    return chemins


# ---------------------------------------------------------------------------
# 7. POINT D'ENTRÉE
# ---------------------------------------------------------------------------

def generer_carte(clean_path: Path = CLEAN_PATH, agg_path: Path = AGG_PATH,
                  dossier: Path = SORTIES, geojson_path: Path | None = None,
                  strict: bool = False) -> list[Path]:
    df = charger_individuel(clean_path)
    par_pays = agreger_par_pays(df)
    controler_fichier_agrege(par_pays, agg_path, strict=strict)
    geo = ajouter_geographie(par_pays)
    if geo.empty:
        raise ErreurDonnees("Aucun pays européen à cartographier.")

    gj = charger_geojson(geojson_path, set(geo.iso3)) if geojson_path else None
    fig = construire_figure(geo, gj)

    k = indicateurs(geo)
    log.info("%d pays · écart moyen H − F %s · %s répondants · max : %s (%s) · min : %s (%s) · "
             "effectifs faibles : %s", k["n_pays"], _fmt(k["ecart_moyen"], True), _n(k["n_total"]),
             k["max"][0], _fmt(k["max"][1], True), k["min"][0], _fmt(k["min"][1], True),
             ", ".join(k["faibles"]) or "aucun")

    source = ("Source : ISSP 2020 (Environment IV), moyennes pondérées. "
              "Gris : pays non couverts ou hors périmètre.")
    vues = [(f"{PREFIXE}_ecart", "Écart de fréquence de consommation de viande rouge H - F - ISSP 2020"),
            (f"{PREFIXE}_hommes", "Fréquence de consommation de viande rouge - hommes - ISSP 2020"),
            (f"{PREFIXE}_femmes", "Fréquence de consommation de viande rouge - femmes - ISSP 2020"),
            (f"{PREFIXE}_tous", "Fréquence de consommation de viande rouge - ensemble - ISSP 2020")]
    chemins = exporter_png(fig, vues, dossier, source, gj is not None)

    return chemins


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cartes viande rouge × sexe (ISSP 2020), en PNG")
    p.add_argument("--clean", type=Path, default=CLEAN_PATH)
    p.add_argument("--agrege", type=Path, default=AGG_PATH)
    p.add_argument("--sorties", type=Path, default=SORTIES, help="dossier des images")
    p.add_argument("--geojson", type=Path, default=None,
                   help="GeoJSON des pays (ex. Natural Earth) : export sans Internet")
    p.add_argument("--strict", action="store_true",
                   help="échoue si le contrôle croisé avec issp_par_pays.csv diverge")
    a = p.parse_args(argv)
    try:
        generer_carte(a.clean, a.agrege, a.sorties, a.geojson, a.strict)
    except ErreurDonnees as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
