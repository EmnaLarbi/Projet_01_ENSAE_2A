"""Écart de consommation de viande hommes - femmes et écart de ressenti de masculinité.

Estimation par MCO, au niveau pays, du modèle :
    Y = b0 + b1 * X + b2 * log(PIB) + e
avec
    Y : écart de consommation de viande hommes - femmes (ecart_h_f)
    X : écart de score de masculinité/féminité hommes - femmes (mascfel_h - masfcel_f)
    log(PIB) : logarithme du PIB par habitant (variable de contrôle)

Données : ISSP 2020 et ESS 11, nettoyées et jointes au niveau pays (une ligne = un pays).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ======================================================================
# 0. PARAMÈTRES
# Tous les noms de colonnes et chemins sont centralisés ici : c'est le seul
# bloc à modifier si la table change.
# ======================================================================

def trouver_racine_projet() -> Path:
    """Renvoie la racine du dépôt Projet_01_ENSAE_2A.

    Le script est rangé dans Src/ : la racine est donc le dossier parent de Src/.
    Dans un notebook (où __file__ n'existe pas), on prend le dossier courant.
    """
    try:
        dossier_script = Path(__file__).resolve().parent
        return dossier_script.parent if dossier_script.name.lower() == "src" else dossier_script
    except NameError:
        return Path.cwd()


RACINE = trouver_racine_projet()
CHEMIN_DONNEES = RACINE / "Data" / "processed" / "table_jointe.csv"
DOSSIER_SORTIES = RACINE / "sorties"

if not CHEMIN_DONNEES.exists():
    raise FileNotFoundError(f"Table introuvable : {CHEMIN_DONNEES}")
print(f"Table utilisée : {CHEMIN_DONNEES}")

COLONNES = {
    "pays": "iso2",                 # colonne identifiant le pays (code iso2)
    "ecart_viande": "ecart_h_f",    # Y : écart de conso de viande h − f (déjà calculé)
    "masc_h": "mascfel_homme",          # score de masculinité/féminité des hommes
    "masc_f": "mascfel_femme",          # score des femmes 
    "pib": "pib_hab",                   # PIB par habitant, en niveau
}

# Noms explicites des variables construites
Y = "ecart_viande_h_f"
X = "ecart_masc_h_f"
CONTROLE = "log_pib"

DOSSIER_SORTIES.mkdir(exist_ok=True)


# ======================================================================
# 1. CHARGEMENT ET VÉRIFICATION DES DONNÉES
# ======================================================================

def charger_donnees(chemin: Path, colonnes: dict) -> pd.DataFrame:
    """Charge la table pays et vérifie la présence des colonnes attendues.

    Paramètres
    ----------
    chemin : Path
        Chemin relatif vers le fichier CSV.
    colonnes : dict
        Correspondance {rôle: nom de colonne dans le fichier}.

    Retour
    ------
    pd.DataFrame
        La table brute.
    """
    df = pd.read_csv(chemin)
    manquantes = [nom for nom in colonnes.values() if nom not in df.columns]
    if manquantes:
        raise KeyError(
            f"Colonnes introuvables : {manquantes}.\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )
    return df


def verifier_donnees(df: pd.DataFrame, colonnes: dict) -> None:
    """Affiche les dimensions, les types et les valeurs manquantes des colonnes utiles."""
    utiles = list(colonnes.values())
    print(f"Dimensions de la table : {df.shape[0]} pays × {df.shape[1]} colonnes\n")
    print("Types des colonnes utilisées :")
    print(df[utiles].dtypes, "\n")
    print("Valeurs manquantes par colonne :")
    print(df[utiles].isna().sum(), "\n")


df_brut = charger_donnees(CHEMIN_DONNEES, COLONNES)
verifier_donnees(df_brut, COLONNES)


# ======================================================================
# 2. CONSTRUCTION DES VARIABLES
# ======================================================================

def construire_variables(df: pd.DataFrame, colonnes: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit Y, X et log(PIB), puis sépare les pays exploitables des pays exclus.

    Les pays exclus (valeur manquante ou PIB non strictement positif) sont
    renvoyés à part avec le motif d'exclusion, pour être documentés dans le rapport.

    Retour
    ------
    (df_analyse, df_exclus)
    """
    df = df.copy()
    df[Y] = df[colonnes["ecart_viande"]]
    df[X] = df[colonnes["masc_h"]] - df[colonnes["masc_f"]]

    pib = df[colonnes["pib"]]
    df[CONTROLE] = np.where(pib > 0, np.log(pib.where(pib > 0)), np.nan)

    # Motif d'exclusion, pays par pays
    motifs = pd.Series("", index=df.index)
    for role in ["ecart_viande", "masc_h", "masc_f", "pib"]:
        nom = colonnes[role]
        motifs[df[nom].isna()] += f"{nom} manquant; "
    motifs[(pib <= 0)] += "PIB ≤ 0; "

    exclus = motifs != ""
    df_exclus = df.loc[exclus, [colonnes["pays"]]].assign(motif=motifs[exclus].str.rstrip("; "))
    df_analyse = df.loc[~exclus].reset_index(drop=True)

    print(f"Pays retenus : {len(df_analyse)} | pays exclus : {len(df_exclus)}")
    if len(df_exclus):
        print(df_exclus.to_string(index=False))
    return df_analyse, df_exclus


df, df_exclus = construire_variables(df_brut, COLONNES)


# ======================================================================
# 3. STATISTIQUES DESCRIPTIVES
# ======================================================================

def statistiques_descriptives(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Renvoie N, moyenne, écart-type, min et max pour chaque variable."""
    stats = df[variables].describe().T[["count", "mean", "std", "min", "max"]]
    stats.columns = ["N", "Moyenne", "Écart-type", "Min", "Max"]
    stats["N"] = stats["N"].astype(int)
    return stats.round(3)


VARIABLES = [Y, X, CONTROLE]
tab_desc = statistiques_descriptives(df, VARIABLES)
print(tab_desc, "\n")

matrice_corr = df[VARIABLES].corr().round(3)
print("Matrice de corrélation :")
print(matrice_corr)

tab_desc.to_latex(DOSSIER_SORTIES / "statistiques_descriptives.tex")
matrice_corr.to_latex(DOSSIER_SORTIES / "correlations.tex")


# ======================================================================
# 4. VISUALISATION
# ======================================================================

def tracer_nuage(df: pd.DataFrame, col_pays: str, chemin_sortie: Path) -> None:
    """Nuage de points X vs Y, étiqueté par pays, avec droite des MCO simples."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[X], df[Y], color="tab:blue", alpha=0.8)

    for _, ligne in df.iterrows():
        ax.annotate(str(ligne[col_pays]), (ligne[X], ligne[Y]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)

    pente, ordonnee = np.polyfit(df[X], df[Y], deg=1)
    grille = np.linspace(df[X].min(), df[X].max(), 100)
    ax.plot(grille, ordonnee + pente * grille, color="tab:red",
            label=f"Droite MCO (pente = {pente:.2f})")

    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("Écart de score de masculinité/féminité (hommes − femmes)")   # préciser l'unité du score
    ax.set_ylabel("Écart de consommation de viande (hommes − femmes)")          # préciser l'unité de conso
    ax.set_title("Écart de consommation de viande et écart de masculinité, par pays")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin_sortie, dpi=300)
    plt.show()


tracer_nuage(df, COLONNES["pays"], DOSSIER_SORTIES / "nuage_ecarts.png")


# ======================================================================
# 5. RÉGRESSIONS
# Écarts-types robustes à l'hétéroscédasticité HC3, recommandés pour les
# petits échantillons (ici quelques dizaines de pays).
# ======================================================================

modele_1 = smf.ols(f"{Y} ~ {X}", data=df).fit(cov_type="HC3")
modele_2 = smf.ols(f"{Y} ~ {X} + {CONTROLE}", data=df).fit(cov_type="HC3")

print(modele_2.summary())


# ======================================================================
# 6. TABLEAU DE RÉSULTATS
# ======================================================================

def tableau_resultats(modeles: list, noms: list[str]):
    """Présente les modèles côte à côte : coefficients, erreurs types (HC3) entre
    parenthèses, étoiles (* 10 %, ** 5 %, *** 1 %), N, R² et R² ajusté."""
    return summary_col(
        modeles,
        model_names=noms,
        stars=True,
        float_format="%.3f",
        regressor_order=["Intercept", X, CONTROLE],
        info_dict={"N": lambda m: f"{int(m.nobs)}"},
    )


tableau = tableau_resultats([modele_1, modele_2], ["(1)", "(2)"])
print(tableau)
print("Erreurs types robustes HC3 entre parenthèses. * p<0,1 ; ** p<0,05 ; *** p<0,01")

(DOSSIER_SORTIES / "tableau_regressions.tex").write_text(tableau.as_latex(), encoding="utf-8")


# ======================================================================
# 7. DIAGNOSTICS
# ======================================================================

def diagnostics(modele, df: pd.DataFrame, col_pays: str, chemin_sortie: Path) -> pd.DataFrame:
    """Trace résidus vs valeurs ajustées et la distance de Cook par pays.

    Renvoie les pays dont la distance de Cook dépasse le seuil usuel 4/N.
    """
    cook = modele.get_influence().cooks_distance[0]
    seuil = 4 / modele.nobs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(modele.fittedvalues, modele.resid, alpha=0.8)
    ax1.axhline(0, color="red", ls="--", lw=1)
    ax1.set_xlabel("Valeurs ajustées")
    ax1.set_ylabel("Résidus")
    ax1.set_title("Résidus vs valeurs ajustées — modèle (2)")

    ax2.bar(df[col_pays].astype(str), cook)
    ax2.axhline(seuil, color="red", ls="--", lw=1, label=f"Seuil 4/N = {seuil:.2f}")
    ax2.set_ylabel("Distance de Cook")
    ax2.set_title("Pays influents — modèle (2)")
    ax2.tick_params(axis="x", rotation=90, labelsize=7)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(chemin_sortie, dpi=300)
    plt.show()

    influents = df.assign(cook=cook).loc[cook > seuil, [col_pays, "cook"]]
    return influents.sort_values("cook", ascending=False)


def calculer_vif(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    """VIF de chaque régresseur (constante incluse dans la matrice, mais non reportée)."""
    matrice = np.column_stack([np.ones(len(df)), df[variables].to_numpy()])
    return pd.Series(
        [variance_inflation_factor(matrice, i + 1) for i in range(len(variables))],
        index=variables, name="VIF",
    ).round(2)


pays_influents = diagnostics(modele_2, df, COLONNES["pays"], DOSSIER_SORTIES / "diagnostics.png")
print("Pays influents (Cook > 4/N) :")
print(pays_influents.to_string(index=False) if len(pays_influents) else "aucun")

vif = calculer_vif(df, [X, CONTROLE])
print("\nVIF :")
print(vif)   # au-delà de 5 à 10, colinéarité préoccupante