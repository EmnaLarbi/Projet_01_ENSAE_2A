"""
Analyse du registre viriliste — publicités viande ROUGE vs groupe témoin
-----------------------------------------------------------------------
Script placé dans Src/, à lancer après scraping_meta.py et scrape_control.py.

Corpus principal : publicités du fichier viande dont le TEXTE (ou l'annonceur)
mentionne du bœuf ou de l'agneau, quel que soit le mot-clé qui les a ramenées.
Le type de viande est recalculé ici avec detect_meats (règles actuelles du
scraper), sans se fier à la colonne meat_type du CSV.
  - analyse principale : viande rouge, y compris les publicités « mixtes »
    (rouge + blanche, ex. promotions de supermarché) ;
  - analyse de sensibilité : viande rouge seule (aucune mention de viande blanche).
Les publicités sans viande rouge (viande blanche seule, ou seulement bbq/grill)
sont écartées.

Corrections par rapport à la version précédente
1. Fichier témoin : nom aligné sur scrape_control.py (meta_ads_control_clean.csv).
   Plus de fichier de repli : l'ancien fichier, filtré avec d'anciennes règles,
   ne peut plus être utilisé sans qu'on le remarque.
2. Registres resserrés : les plats et charcuteries « du chasseur » (sauce chasseur,
   Jägerschnitzel, lovecká saláma, vadászkolbász…), « recette ancestrale »,
   « primal cuts » et les mentions « pour hommes » ne comptent plus comme virils.
   Les mentions « pour hommes / masculin » sont gardées dans un registre
   exploratoire séparé (mention_hommes), hors de has_virility.
3. Formes manquantes ajoutées (testosterona, testosteronas, testosterón…) et
   couverture complétée pour l'allemand, l'italien et l'espagnol.
4. Recouvrement des groupes : une publicité témoin identique à une publicité du
   groupe viande, ou qui parle de bbq / barbecue / grill, est retirée du témoin.
5. Même nettoyage du pied de page Meta pour les deux groupes.
6. Test entre pays : si un effectif attendu est < 5, p-value par permutation.
7. Cellule nulle : OR corrigé (Haldane-Anscombe) affiché à côté de l'OR brut,
   pour que l'estimation et l'IC 95 % soient cohérents.
8. Correction de Holm pour les tests multiples (test global + 3 registres).
9. Mots-clés réellement dédoublonnés lors du regroupement global.
10. Nouveau : analyse de sensibilité (rouge seule vs rouge + mixtes) et fichier
    de vérification manuelle des termes détectés.

Limite non corrigeable ici : les scrapers gardent chaque ad_id une seule fois,
avec le premier pays où il a été vu. La répartition par pays dépend donc de
l'ordre de collecte (AT avant DE, etc.).
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

# Mêmes règles que la collecte
from scrape import  detect_meats, is_valid_clean_ad

RED_MEATS = {"boeuf", "agneau"}

# -----------------------------------------------------------------------------
# 0. CHEMINS ET PARAMÈTRES
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MAIN_FILE = BASE_DIR / "Data" / "processed" / "meta_ads_meat_bbq_clean.csv"
CONTROL_FILE = BASE_DIR / "Data" / "processed" / "meta_ads_control_clean.csv"   # correction 1
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 10_000
SEED = 2026

# -----------------------------------------------------------------------------
# 1. REGISTRES (listes à faire relire par des locuteurs natifs si possible)
# -----------------------------------------------------------------------------

REGISTERS = {
    # correction 3 : une seule forme couvre testostérone, testosterona, testosteronas,
    # tesztoszteron, testósterón, testosteroni…
    "testosterone": [
        r"\bte(?:s|sz)t[oó](?:s|sz)t[eé]r[oóô]n\w*",
    ],
    # Virilité au sens propre (« vrai homme », virilité, masculinité, viril).
    # Les simples adjectifs « masculin / pour hommes » sont dans mention_hommes.
    "identite_masculine": [
        r"\bviril\w*",                       # viril, virilité, virilidad, virilità, Virilität, virility
        r"\bmasculinit\w*", r"\bmascolinit\w*", r"\bmaskulinit\w*", r"\bmaszkulinit\w*",
        r"\bmaskuliinisuu\w*", r"\bmasculinidad\w*", r"\bmännlichkeit\b",
        r"\bmanlighet\w*", r"\bmannlighet\w*", r"\bmandig\w*", r"\bkarlmennsk\w*",
        r"\bvyriškum\w*", r"\bmuževn\w*", r"\bmožat\w*", r"\bmužn\w*",
        r"\bférfias\w*", r"\bmiehekä\w*", r"\bmiehisyy\w*",
        # « vrai homme » et équivalents
        r"\bvrais?\s+hommes?\b", r"\breal\s+m[ae]n\b",
        r"\becht(?:e|er|en)\s+(?:m[aä]nn(?:er)?|kerle?)\b",
        r"\bver[oi]\s+uom(?:o|ini)\b", r"\bhombres?\s+de\s+verdad\b", r"\bverdaderos?\s+hombres?\b",
        r"\btosimie\w*", r"\bprav(?:i|og|ih)\s+muškar\w*", r"\bigazi\s+férfi\w*",
        r"\bsann(?:ur|an)\s+karlm\w*", r"\btikr(?:as|i)\s+vyr\w*", r"\bekte\s+m[ae]nn\b",
        r"\briktiga?\s+(?:man|män|karlar?)\b", r"\bpravi\s+moški\b", r"\bprav[ýí]\s+mu(?:ž|ži)\b",
    ],
    "ancestral_alpha": [
        # « alpha » seulement dans une expression (pas les marques « Alpha »)
        r"\b(?:alpha|alfa)[- ]?(?:male|males|mâle|männchen|maschio|macho|uros|hím|samec|mužjak|hanne|hane)\b",
        r"\b(?:mâle|maschio|macho)\s+(?:alpha|alfa)\b", r"\balfauros\w*",
        # « ancestral / primal » seulement dans une expression (pas « recette ancestrale »)
        r"\b(?:instincts?|force|nature)\s+(?:primale?s?|ancestrale?s?|primitive?s?)\b",
        r"\bprimal\s+(?:instinct|power|strength|nature)s?\b", r"\bur-?instinkt\w*",
        r"\bhommes?\s+des\s+cavernes\b", r"\bcavem[ae]n\b", r"\bhöhlenmensch\w*",
        r"\bcavernícola\w*", r"\buom(?:o|ini)\s+delle\s+caverne\b", r"\bluolamie\w*",
        r"\bősember\w*",
        # chasseurs (formes de nom seulement : « lovecká », « vadászkolbász » exclus)
        r"\bchasseurs?\b", r"\bhunters?\b", r"\bjäger(?:n|s|in)?\b", r"\bcacciator[ei]\b",
        r"\bcazador(?:es)?\b", r"\bmetsästäj(?:ä|än|ät|ien|iä|ille)\b", r"\blovac\w*",
        r"\bvadász(?:ok|ként|t|nak)?\b", r"\bveiðim[aeö]\w*", r"\bmedžiotoj\w*",
        r"\bjeger(?:e|en|ne)?\b", r"\bjägar(?:e|en|na)?\b", r"\blov(?:ec|ca|ci|cov|cu)\b",
        # guerriers
        r"\bwarriors?\b", r"\bguerri(?:er|ère)s?\b", r"\bkrieger(?:in|n)?\b",
        r"\bguerrier[oi]\b", r"\bguerrer[oa]s?\b", r"\bsotur\w*", r"\bratnik\w*",
        r"\bharcos\w*", r"\bstríðsm[aö]\w*", r"\bkar(?:ys|iai|ių)\b", r"\bkriger\w*",
        r"\bkrigar\w*", r"\bbojevnik\w*", r"\bbojovník\w*",
    ],
}

# Registre exploratoire : publicités qui s'adressent aux hommes, sans connotation
# virile forcée (« équipe masculine », « moška majica »). Hors de has_virility.
EXPLORATORY_REGISTERS = {
    "mention_hommes": [
        r"\bmasculin\w*", r"\bmännlich\w*", r"\bmascolin\w*", r"\bmaschil\w*", r"\bmaskulin\w*",
        r"\bmaskuliin\w*", r"\bmaszkulin\w*", r"\bmužsk\w*", r"\bmošk\w*", r"\bmušk\w*",
        r"\bvyrišk\w*", r"\bkarlm[ae]nn\w*", r"\bmannlig\w*", r"\bmanlig\w*",
        r"\bpour\s+(?:les\s+)?hommes?\b", r"\bfor\s+men\b", r"\bmen'?s\b", r"\bfür\s+(?:den\s+)?männer?\b",
        r"\bper\s+(?:l')?uom(?:o|ini)\b", r"\bpara\s+(?:el\s+)?hombres?\b",
    ],
}

# Contextes culinaires retirés du texte AVANT la recherche des registres
CONTEXT_EXCLUSIONS = [
    r"\b(?:sauce|poulet|lapin|escalope|veau|façon|à\s+la)\s+chasseur\b",
    r"\bjäger[- ]?(?:schnitzel|soße|sosse|sauce|art|meister)\w*", r"\bnach\s+jägerart\b",
    r"\bsalam[ei]\s+cacciatore\b", r"\bcacciatorin[oi]\b",
    r"\bhunter'?s?\s+(?:chicken|sauce|sausages?|stew)\b", r"\bprimal\s+cuts?\b",
    r"\bvadász\s*módra\b", r"\bmedžiotojų\s+dešr\w*",
]

BBQ_PATTERN = re.compile(r"\b(?:bbq|barbecue|grill)\b", flags=re.IGNORECASE)  # comme scraping_meta.py

META_FOOTER_REGEX = re.compile(
    r"(?:\s*(?:" + "|".join([
        r"État du système\s*API Ad Library.*", r"Ad Library API status.*",
        r"À propos des publicités et de l[’']utilisation des données.*", r"About ads and data use.*",
        r"Confidentialité\s*Conditions\s*Cookies.*", r"Privacy\s*Terms\s*Cookies.*",
    ]) + r"))\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _rx(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns), flags=re.IGNORECASE)


REGISTER_RX = {name: _rx(p) for name, p in {**REGISTERS, **EXPLORATORY_REGISTERS}.items()}
EXCLUSION_RX = _rx(CONTEXT_EXCLUSIONS)
REG_COLS = [f"reg_{r}" for r in REGISTERS]                       # comptent dans has_virility
ALL_REG_COLS = [f"reg_{r}" for r in REGISTER_RX]                 # + registre exploratoire

# -----------------------------------------------------------------------------
# 2. CHARGEMENT ET ANNOTATION
# -----------------------------------------------------------------------------


def clean_meta_footer(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    return META_FOOTER_REGEX.sub("", text).strip()


def _join_unique(values) -> str:
    """Correction 9 : découpe les listes déjà jointes avant de dédoublonner."""
    return ", ".join(sorted({k.strip() for v in values for k in str(v).split(",") if k.strip()}))


def load_clean_ads(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    for col in ["ad_id", "country", "lang", "advertiser", "ad_text", "keyword", "image_urls", "meat_detail"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()
    df["ad_text"] = df["ad_text"].map(clean_meta_footer)          # correction 5
    return df


def select_red_meat(df: pd.DataFrame) -> pd.DataFrame:
    """Garde les publicités valides qui mentionnent du bœuf ou de l'agneau."""
    valid = df.apply(lambda r: is_valid_clean_ad(r.to_dict()), axis=1)
    meats = [set(detect_meats(t, a, l)) for t, a, l in zip(df["ad_text"], df["advertiser"], df["lang"])]
    df = df.assign(meat_detail=[",".join(sorted(m)) for m in meats])
    red = pd.Series([bool(m & RED_MEATS) for m in meats], index=df.index)

    print("=== SÉLECTION DU CORPUS VIANDE ROUGE (lignes du CSV) ===")
    print(f"Total : {len(df)} | invalides selon les règles actuelles : {int((~valid).sum())} | "
          f"viande rouge retenues : {int((valid & red).sum())} | sans viande rouge (écartées) : {int((valid & ~red).sum())}\n")
    return df[valid & red].copy()

def annotate_unique_ads(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """Dédoublonne au niveau (pays, langue, annonceur, texte) et détecte chaque registre."""
    df_unique = (
        df.groupby(["country", "lang", "advertiser", "ad_text"], as_index=False, dropna=False)
        .agg(
            n_variants=("ad_id", "count"),
            keywords=("keyword", _join_unique),
            image_urls=("image_urls", "first"),
            meat_detail=("meat_detail", "first"),
        )
    )
    text = df_unique["ad_text"].map(lambda t: EXCLUSION_RX.sub(" ", t))   # correction 2
    for name, rx in REGISTER_RX.items():
        df_unique[f"reg_{name}"] = text.str.contains(rx, na=False)
        df_unique[f"terme_{name}"] = text.map(lambda t, rx=rx: (m.group(0) if (m := rx.search(t)) else ""))
    df_unique["has_virility"] = df_unique[REG_COLS].any(axis=1)
    df_unique["group"] = group_name
    return df_unique


def remove_overlap(main_by_locale: pd.DataFrame, ctrl_by_locale: pd.DataFrame) -> pd.DataFrame:
    """Correction 4 : retire du témoin ce qui appartient (ou pourrait appartenir) au groupe viande."""
    main_keys = set(zip(main_by_locale["advertiser"], main_by_locale["ad_text"]))
    in_main = [k in main_keys for k in zip(ctrl_by_locale["advertiser"], ctrl_by_locale["ad_text"])]
    is_bbq = ctrl_by_locale["ad_text"].str.contains(BBQ_PATTERN) | ctrl_by_locale["advertiser"].str.contains(BBQ_PATTERN)
    drop = pd.Series(in_main, index=ctrl_by_locale.index) | is_bbq
    print(f"Témoin : {int(pd.Series(in_main).sum())} publicités aussi présentes dans le groupe viande, "
          f"{int(is_bbq.sum())} mentionnant bbq/barbecue/grill → {int(drop.sum())} retirées.")
    return ctrl_by_locale[~drop].copy()


def deduplicate_global(df_unique: pd.DataFrame) -> pd.DataFrame:
    """Une même création diffusée dans plusieurs pays ou langues n'est comptée qu'une fois."""
    if df_unique.empty:
        return df_unique.copy()
    agg = {
        "n_variants": ("n_variants", "sum"),
        "countries": ("country", _join_unique),
        "langs": ("lang", _join_unique),
        "keywords": ("keywords", _join_unique),
        "image_urls": ("image_urls", "first"),
        "meat_detail": ("meat_detail", "first"),
        "has_virility": ("has_virility", "max"),
        "group": ("group", "first"),
    }
    for name in REGISTER_RX:
        agg[f"reg_{name}"] = (f"reg_{name}", "max")
        agg[f"terme_{name}"] = (f"terme_{name}", "first")
    out = df_unique.groupby(["advertiser", "ad_text"], as_index=False, dropna=False).agg(**agg)
    for col in ["has_virility", "mixte", *ALL_REG_COLS]:
        out[col] = out[col].astype(bool)
    return out

# -----------------------------------------------------------------------------
# 3. STATISTIQUES
# -----------------------------------------------------------------------------


def compute_or_and_ci95(a: int, b: int, c: int, d: int) -> dict:
    """OR brut + p de Fisher ; si une case vaut 0, OR et IC 95 % (Woolf) corrigés
    par Haldane-Anscombe (+0,5), pour que l'estimation et l'intervalle soient cohérents.
        a = viande + positif | b = viande + négatif
        c = témoin + positif | d = témoin + négatif
    """
    odds_ratio, p = fisher_exact([[a, b], [c, d]])
    haldane = min(a, b, c, d) == 0
    a_, b_, c_, d_ = (x + 0.5 for x in (a, b, c, d)) if haldane else (a, b, c, d)
    or_corr = (a_ * d_) / (b_ * c_)
    se = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    return {
        "odds_ratio": float(odds_ratio),
        "odds_ratio_haldane": or_corr if haldane else float("nan"),
        "ci95_low": math.exp(math.log(or_corr) - 1.96 * se),
        "ci95_high": math.exp(math.log(or_corr) + 1.96 * se),
        "haldane_anscombe": haldane,
        "p_value_fisher": float(p),
    }


def holm(pvals: list[float]) -> list[float]:
    """Correction 8 : p-values ajustées de Holm."""
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare_groups(main_df: pd.DataFrame, ctrl_df: pd.DataFrame, col: str) -> dict:
    a = int(main_df[col].sum()); b = len(main_df) - a
    c = int(ctrl_df[col].sum()); d = len(ctrl_df) - c
    return {"rouge_n": a, "rouge_total": len(main_df),
            "rouge_pct": round(100 * a / len(main_df), 2) if len(main_df) else 0.0,
            "control_n": c, "control_total": len(ctrl_df),
            "control_pct": round(100 * c / len(ctrl_df), 2) if len(ctrl_df) else 0.0,
            **compute_or_and_ci95(a, b, c, d)}


def country_heterogeneity_test(df: pd.DataFrame) -> None:
    """Correction 6 : Chi2 classique, ou p par permutation si effectifs attendus < 5."""
    ct = pd.crosstab(df["country"], df["has_virility"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print("\n[Info] Effectifs ou modalités insuffisants pour un test entre pays.")
        return
    chi2_obs, p_asym, dof, expected = chi2_contingency(ct)
    print(f"\nTest entre pays : chi2 = {chi2_obs:.4f} (ddl = {dof}), p asymptotique = {p_asym:.4e}")
    if expected.min() >= 5:
        return

    codes = pd.Categorical(df["country"]).codes
    y = df["has_virility"].to_numpy().astype(int)
    k = codes.max() + 1

    def stat(labels: np.ndarray) -> float:
        tab = np.zeros((k, 2))
        np.add.at(tab, (codes, labels), 1)
        exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        return float(((tab - exp) ** 2 / exp).sum())

    rng = np.random.default_rng(SEED)
    obs = stat(y)
    n_ge = sum(stat(rng.permutation(y)) >= obs - 1e-12 for _ in range(N_PERMUTATIONS))
    p_perm = (n_ge + 1) / (N_PERMUTATIONS + 1)
    print(f"  {int((expected < 5).sum())} case(s) avec effectif attendu < 5 : p asymptotique peu fiable.")
    print(f"  p par permutation ({N_PERMUTATIONS} tirages) = {p_perm:.4f}")


def compute_country_summary(df_unique: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ctry, sub in df_unique.groupby("country", dropna=False):
        row = {"country": ctry, "n_unique_ads": len(sub),
               "n_virility": int(sub["has_virility"].sum()),
               "pct_virility": round(sub["has_virility"].mean() * 100, 1)}
        for name in REGISTER_RX:
            row[f"pct_{name}"] = round(sub[f"reg_{name}"].mean() * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pct_virility", "n_unique_ads"], ascending=[False, False])

# -----------------------------------------------------------------------------
# 4. PROGRAMME PRINCIPAL
# -----------------------------------------------------------------------------


def main():
    if not MAIN_FILE.exists():
        raise FileNotFoundError(f"Fichier viande introuvable : {MAIN_FILE}")

    # 1. Corpus viande rouge
    main_by_locale = annotate_unique_ads(select_red_meat(load_clean_ads(MAIN_FILE)), "viande_rouge")
    main_by_locale.to_csv(RESULTS_DIR / "main_unique_annotated.csv", index=False, encoding="utf-8-sig")
    main_global = deduplicate_global(main_by_locale)
    main_global.to_csv(RESULTS_DIR / "main_global_unique_annotated.csv", index=False, encoding="utf-8-sig")

    print("=== SYNTHÈSE GLOBALE (VIANDE ROUGE — DÉDOUBLONNAGE GLOBAL) ===")
    print(f"Créations uniques (pays/langue) : {len(main_by_locale)} | (global) : {len(main_global)}")
    if len(main_global):
        print(f"Part avec registre viriliste : {main_global['has_virility'].mean():.1%} "
              f"({int(main_global['has_virility'].sum())}/{len(main_global)})")
        for name in REGISTER_RX:
            col = f"reg_{name}"
            note = "  (exploratoire, hors has_virility)" if name in EXPLORATORY_REGISTERS else ""
            print(f" - {name:<18} : {int(main_global[col].sum())} ({main_global[col].mean():.1%}){note}")

    # 2. Par pays
    df_country = compute_country_summary(main_by_locale)
    df_country.to_csv(RESULTS_DIR / "by_country_virility.csv", index=False, encoding="utf-8-sig")
    print("\n=== RÉPARTITION PAR PAYS (VIANDE ROUGE) ===")
    print(df_country.to_string(index=False))
    print("[Attention] Pays attribué = premier pays où l'ad_id a été vu au scraping.")
    country_heterogeneity_test(main_by_locale)

    if not CONTROL_FILE.exists():
        print(f"\n[Info] Groupe témoin absent ({CONTROL_FILE}).")
        return

    # 3. Groupe témoin, sans recouvrement avec le groupe viande
    ctrl_by_locale = annotate_unique_ads(load_clean_ads(CONTROL_FILE), "groupe_temoin")
    ctrl_by_locale = remove_overlap(main_by_locale, ctrl_by_locale)
    ctrl_by_locale.to_csv(RESULTS_DIR / "control_unique_annotated.csv", index=False, encoding="utf-8-sig")
    ctrl_global = deduplicate_global(ctrl_by_locale)
    ctrl_global.to_csv(RESULTS_DIR / "control_global_unique_annotated.csv", index=False, encoding="utf-8-sig")
    if main_global.empty or ctrl_global.empty:
        print("[Info] Un des deux groupes est vide : comparaison impossible.")
        return

    # 4. Comparaison globale + par registre, avec correction de Holm
    rows = [{"register": "has_virility (global)", **compare_groups(main_global, ctrl_global, "has_virility")}]
    rows += [{"register": name, **compare_groups(main_global, ctrl_global, f"reg_{name}")} for name in REGISTERS]
    p_holm = holm([r["p_value_fisher"] for r in rows])
    for r, p in zip(rows, p_holm):
        r["p_value_holm"] = p
    for name in EXPLORATORY_REGISTERS:          # exploratoire : p brute seulement
        rows.append({"register": f"{name} (exploratoire)",
                     **compare_groups(main_global, ctrl_global, f"reg_{name}"), "p_value_holm": float("nan")})

    df_reg = pd.DataFrame(rows)
    df_reg.to_csv(RESULTS_DIR / "by_register_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n=== VIANDE ROUGE VS TÉMOIN (CRÉATIONS UNIQUES GLOBALES) ===")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(df_reg[["register", "rouge_n", "rouge_pct", "control_n", "control_pct",
                      "odds_ratio", "odds_ratio_haldane", "ci95_low", "ci95_high",
                      "p_value_fisher", "p_value_holm"]].to_string(index=False))
    print("OR = inf ou 0 : une case vaut 0 → lire odds_ratio_haldane et son IC 95 %.")

    # 5. Fichier de vérification manuelle des détections
    verif = pd.concat([main_global, ctrl_global], ignore_index=True)
    verif = verif[verif[ALL_REG_COLS].any(axis=1)]
    cols = ["group", "meat_detail", "countries", "advertiser", *[f"terme_{n}" for n in REGISTER_RX], "ad_text"]
    verif[cols].to_csv(RESULTS_DIR / "verification_registres.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(verif)} publicités détectées à relire dans verification_registres.csv")


if __name__ == "__main__":
    main()