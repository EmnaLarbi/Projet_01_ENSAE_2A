"""
Src/carte_masculinite.py
========================

Carte n°2 : masculinité ressentie en Europe (ESS 11), exportée en images PNG.

Exécution (depuis la racine du projet) :

    python Src/carte_masculinite.py
    python Src/carte_masculinite.py --geojson Data/raw/ne_50m_admin_0_countries.geojson   # sans Internet

Entrée  : Data/processed/table_jointe.csv
Sorties (dossier sorties/) : une image PNG par vue
    carte_masculinite_ensemble.png, carte_masculinite_hommes.png,
    carte_masculinite_femmes.png, carte_masculinite_ecart.png
"""

from __future__ import annotations

import argparse
import gettext
import html
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pycountry

# Briques réutilisées de src/maps.py (même périmètre, même gestion du fond de carte)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from maps import (  # noqa: E402
    DATA_DIR, EUROPE_ISO2, GRIS_SANS_DONNEES, SORTIES, ErreurDonnees, _fmt,
    charger_geojson, exporter_png,
)

# ---------------------------------------------------------------------------
# 0. PARAMÈTRES
# ---------------------------------------------------------------------------

TABLE_PATH = DATA_DIR / "table_jointe.csv"
PREFIXE = "carte_masculinite"

SEUIL_EFFECTIF_FAIBLE = 100   # répondants ESS par sexe en dessous duquel on alerte

# Vues : colonne → (libellé, suffixe du fichier image)
VUES = {
    "mascfel": "Masculinité ressentie-ensemble",
    "mascfel_homme": "Masculinité ressentie-hommes",
    "mascfel_femme": "Masculinité ressentie-femmes",
    "ecart_mascfel": "Écart de masculinité ressentie H-F",
}
SUFFIXES = {"mascfel": "ensemble", "mascfel_homme": "hommes",
            "mascfel_femme": "femmes", "ecart_mascfel": "ecart"}

# Palette séquentielle mono-teinte (clair = faible, foncé = élevé)
ECHELLE = [[0.0, "#eef6f3"], [0.5, "#5fa596"], [1.0, "#0b3d33"]]
ENCRE, ENCRE_2, FOND = "#0b0b0b", "#52514e", "#fcfcfb"

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("carte_masculinite")


# ---------------------------------------------------------------------------
# 1. CHARGEMENT ET CONTRÔLES
# ---------------------------------------------------------------------------

def charger_table(chemin: Path = TABLE_PATH) -> pd.DataFrame:
    """Lit table_jointe.csv, vérifie les colonnes et les codes pays,
    calcule l'écart H − F et restreint au périmètre européen."""
    if not chemin.exists():
        raise ErreurDonnees(f"Fichier introuvable : {chemin}")
    df = pd.read_csv(chemin)

    if "iso2" not in df.columns:
        raise ErreurDonnees("Colonne `iso2` absente de table_jointe.csv")
    presentes = [c for c in ["mascfel", "mascfel_homme", "mascfel_femme"] if c in df.columns]
    if not presentes:
        raise ErreurDonnees("Aucune colonne mascfel dans table_jointe.csv")
    if "mascfel_homme" not in df.columns:
        log.warning("mascfel_homme / mascfel_femme absents : relancer clean.py puis la jointure")

    # Codes pays : nettoyage, doublons
    df["iso2"] = df["iso2"].astype(str).str.strip().str.upper()
    if df["iso2"].duplicated().any():
        raise ErreurDonnees(f"Pays en double : {df.loc[df.iso2.duplicated(), 'iso2'].tolist()}")

    # Écart de masculinité ressentie entre hommes et femmes
    if {"mascfel_homme", "mascfel_femme"} <= set(df.columns):
        df["ecart_mascfel"] = df["mascfel_homme"] - df["mascfel_femme"]

    df = ajouter_iso3_et_noms(df)

    # Périmètre européen
    hors = sorted(set(df.iso2) - EUROPE_ISO2)
    if hors:
        log.info("Hors périmètre européen (exclus) : %s", hors)
    df = df[df.iso2.isin(EUROPE_ISO2)].reset_index(drop=True)

    # Valeurs manquantes
    for col in VUES:
        if col in df.columns and df[col].isna().any():
            log.warning("%s : pas de valeur pour %s", col, df.loc[df[col].isna(), "iso2"].tolist())
    log.info("%d pays européens : %s", len(df), " ".join(sorted(df.iso2)))
    return df


def ajouter_iso3_et_noms(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcule l'ISO3 depuis l'ISO2 avec pycountry (sans faire confiance à la
    colonne iso3 existante) et ajoute le nom du pays en français."""
    try:
        fr = gettext.translation("iso3166-1", pycountry.LOCALES_DIR, languages=["fr"]).gettext
    except OSError:
        fr = lambda s: s  # noqa: E731

    iso3, noms = [], []
    for code in df["iso2"]:
        pays = pycountry.countries.get(alpha_2=code)
        iso3.append(pays.alpha_3 if pays else None)
        noms.append(fr(getattr(pays, "common_name", None) or pays.name) if pays else code)

    if "iso3" in df.columns:
        ecarts = [c for c, a, b in zip(df.iso2, df.iso3, iso3) if b and a != b]
        if ecarts:
            log.warning("Colonne iso3 incohérente pour %s : corrigée via pycountry", ecarts)
    inconnus = [c for c, b in zip(df.iso2, iso3) if b is None]
    if inconnus:
        log.warning("Codes ISO2 inconnus (exclus) : %s", inconnus)

    return df.assign(iso3=iso3, nom=noms).dropna(subset=["iso3"])


# ---------------------------------------------------------------------------
# 2. INFOBULLES
# ---------------------------------------------------------------------------

def texte_survol(r: pd.Series) -> str:
    """Texte affiché au survol d'un pays."""
    def v(col, signe=False):
        return _fmt(r[col], signe) if col in r.index and pd.notna(r[col]) else "n.d."

    lignes = [f"<b>{html.escape(r['nom'])}</b>",
              f"Ensemble : {v('mascfel')}"]
    if "mascfel_homme" in r.index:
        n_h = f"  (n = {int(r['n_ess_homme'])})" if pd.notna(r.get("n_ess_homme")) else ""
        n_f = f"  (n = {int(r['n_ess_femme'])})" if pd.notna(r.get("n_ess_femme")) else ""
        lignes += [f"Hommes : {v('mascfel_homme')}{n_h}",
                   f"Femmes : {v('mascfel_femme')}{n_f}",
                   f"<b>Écart H − F : {v('ecart_mascfel', True)}</b>"]
        n_min = min(r.get("n_ess_homme", np.inf), r.get("n_ess_femme", np.inf))
        if n_min < SEUIL_EFFECTIF_FAIBLE:
            lignes.append(f"<i>⚠ Effectif faible (< {SEUIL_EFFECTIF_FAIBLE} pour un sexe)</i>")
    return "<br>".join(lignes)


# ---------------------------------------------------------------------------
# 3. CARTE
# ---------------------------------------------------------------------------

def construire_carte(df: pd.DataFrame, geojson: tuple[dict, str] | None = None) -> go.Figure:
    """Une couche par vue, commutées par un menu déroulant."""
    df = df.assign(survol=df.apply(texte_survol, axis=1))
    vues = [(c, lib) for c, lib in VUES.items() if c in df.columns]

    fig = go.Figure()
    n_fond = 0
    if geojson:   # fond gris des pays européens sans donnée (mode hors ligne)
        cle = geojson[1].split(".", 1)[1]
        tous = [f["properties"][cle] for f in geojson[0]["features"]]
        fig.add_trace(go.Choropleth(
            geojson=geojson[0], featureidkey=geojson[1], locationmode="geojson-id",
            locations=tous, z=[0] * len(tous), showscale=False, hoverinfo="skip",
            colorscale=[[0, GRIS_SANS_DONNEES], [1, GRIS_SANS_DONNEES]],
            marker_line_color="white", marker_line_width=0.6))
        n_fond = 1

    for i, (col, lib) in enumerate(vues):
        d = df.dropna(subset=[col])          # aucun pays sans donnée n'est coloré
        kw = dict(
            locations=d.iso3, z=d[col], text=d.survol,
            hovertemplate="%{text}<extra></extra>",
            colorscale=ECHELLE, visible=(i == 0),
            marker_line_color="white", marker_line_width=0.6,
            colorbar=dict(title=dict(text=lib.replace(" — ", "<br>"), font_size=11),
                          thickness=12, len=0.6, x=0.98, tickformat=".2f", outlinewidth=0))
        if geojson:
            kw.update(geojson=geojson[0], featureidkey=geojson[1], locationmode="geojson-id")
        else:
            kw["locationmode"] = "ISO-3"
        fig.add_trace(go.Choropleth(**kw))

    menu = [dict(label=lib, method="update",
                 args=[{"visible": [True] * n_fond + [j == i for j in range(len(vues))]}])
            for i, (_, lib) in enumerate(vues)]
    fig.update_layout(
        updatemenus=[dict(buttons=menu, x=0.01, y=0.99, xanchor="left", yanchor="top",
                          bgcolor="white", bordercolor="#c9c8c2")],
        margin=dict(l=0, r=0, t=10, b=0), height=620, paper_bgcolor=FOND,
        font=dict(family="Georgia, serif", color=ENCRE),
        hoverlabel=dict(bgcolor="white", font=dict(family="Georgia, serif", size=13)))

    cadrage = dict(projection_type="conic conformal", bgcolor=FOND,
                   lataxis_range=[34, 71.5], lonaxis_range=[-25, 42])
    if geojson:
        fig.update_geos(visible=False, projection_parallels=[43, 62],
                        projection_rotation_lon=12, **cadrage)
    else:
        fig.update_geos(scope="europe", resolution=50, showframe=False,
                        showcoastlines=False, showcountries=True, countrycolor="white",
                        showland=True, landcolor=GRIS_SANS_DONNEES, showocean=True,
                        oceancolor=FOND, **cadrage)
    return fig


# ---------------------------------------------------------------------------
# 4. INDICATEURS ET POINT D'ENTRÉE
# ---------------------------------------------------------------------------

def indicateurs(df: pd.DataFrame) -> str:
    """Résumé affiché dans la console."""
    morceaux = [f"{int(df['mascfel'].notna().sum()) if 'mascfel' in df else len(df)} pays"]
    if "mascfel_homme" in df:
        morceaux += [f"hommes {_fmt(df.mascfel_homme.mean())}", f"femmes {_fmt(df.mascfel_femme.mean())}"]
        e = df.dropna(subset=["ecart_mascfel"])
        if not e.empty:
            imax, imin = e.ecart_mascfel.idxmax(), e.ecart_mascfel.idxmin()
            morceaux += [f"écart max : {e.at[imax, 'nom']} ({_fmt(e.at[imax, 'ecart_mascfel'], True)})",
                         f"écart min : {e.at[imin, 'nom']} ({_fmt(e.at[imin, 'ecart_mascfel'], True)})"]
    return " · ".join(morceaux)


def generer(table: Path = TABLE_PATH, dossier: Path = SORTIES,
            geojson_path: Path | None = None) -> list[Path]:
    df = charger_table(table)
    if df.empty:
        raise ErreurDonnees("Aucun pays européen à cartographier.")
    gj = charger_geojson(geojson_path, set(df.iso3)) if geojson_path else None
    fig = construire_carte(df, gj)
    log.info("%s", indicateurs(df))

    vues = [(f"{PREFIXE}_{SUFFIXES[c]}", f"{lib} — ESS 11") for c, lib in VUES.items() if c in df.columns]
    source = ("Source : ESS ERIC, European Social Survey Round 11. Chaque carte a sa propre "
              "échelle de couleurs. Gris : pays sans donnée.")
    return exporter_png(fig, vues, dossier, source, gj is not None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cartes de la masculinité ressentie (ESS 11), en PNG")
    p.add_argument("--table", type=Path, default=TABLE_PATH)
    p.add_argument("--sorties", type=Path, default=SORTIES, help="dossier des images")
    p.add_argument("--geojson", type=Path, default=None,
                   help="GeoJSON des pays (ex. Natural Earth) : export sans Internet")
    a = p.parse_args(argv)
    try:
        generer(a.table, a.sorties, a.geojson)
    except ErreurDonnees as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
    