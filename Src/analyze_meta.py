import math
import re
from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

# Chemins d'entrée et de sortie (script placé dans Src/)
BASE_DIR = Path(__file__).resolve().parent.parent
MAIN_FILE = BASE_DIR / "Data" / "processed" / "meta_ads_meat_bbq_clean.csv"
FALLBACK_MAIN_FILE = BASE_DIR / "Data" / "processed" / "meta_ads_multilang_clean.csv"
CONTROL_FILE = BASE_DIR / "Data" / "processed" / "meta_ads_control_multilang_clean.csv"
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Registres lexicaux / symboliques strictement centrés sur la virilité et les archétypes associés
REGISTERS = {
    "testosterone": [
        r"\btestost[ée]rone?\b", r"\btestosterone\b", r"\btestosteron\b",
        r"\btestosteroni\w*\b", r"\btesztoszteron\w*\b", r"\btestósterón\w*\b",
    ],
    "identite_masculine": [
        r"\bvirilit[ée]\b", r"\bvirilidad\b", r"\bvirilità\b", r"\bvirilität\b",
        r"\bmasculin\w*\b", r"\bmascolin\w*\b", r"\bmännlich\w*\b",
        r"\bvrai\s+homme\b", r"\breal\s+man\b", r"\bechter\s+mann\b", r"\bvero\s+uomo\b",
        r"\bmaskuliin\w*\b", r"\btosimies\w*\b",
        r"\bmuževn\w*\b", r"\bpravi\s+muškarac\w*\b",
        r"\bmaszkulin\w*\b", r"\bférfias\w*\b", r"\bigazi\s+férfi\w*\b",
        r"\bkarlmann\w*\b", r"\bkarlmennsk\w*\b", r"\bsannur\s+maður\b",
        r"\bvyrišk\w*\b", r"\btikras\s+vyras\b",
        r"\bmaskulin\w*\b", r"\bmannlig\w*\b", r"\bekte\s+mann\b", r"\briktig\s+man\b",
        r"\bmošk\w*\b", r"\bpravi\s+moški\b",
        r"\bmaskulín\w*\b", r"\bmužsk\w*\b", r"\bpravý\s+muž\b",
    ],
    "ancestral_alpha": [
        r"\bancestral\w*\b", r"\bprimal\w*\b", r"\balpha\b",
        r"\bchasseur\w*\b", r"\bhunter\w*\b", r"\bwarrior\w*\b", r"\bguerrier\w*\b",
        r"\bmetsästäj\w*\b", r"\bsotur\w*\b",
        r"\blovac\w*\b", r"\bratnik\w*\b",
        r"\bvadász\w*\b", r"\bharcos\w*\b",
        r"\bveiðimaður\b",
        r"\bmedžiotoj\w*\b",
        r"\bjeger\w*\b", r"\bkriger\w*\b",
        r"\bjägar\w*\b", r"\bkrigar\w*\b",
        r"\blovec\w*\b", r"\bbojevnik\w*\b",
        r"\blovec\w*\b", r"\bbojovník\w*\b",
    ],
}


def resolve_main_file() -> Path:
    """Sélectionne le fichier viande/BBQ nettoyé, ou l'ancien fichier nettoyé en repli."""
    if MAIN_FILE.exists():
        return MAIN_FILE
    if FALLBACK_MAIN_FILE.exists():
        print(f"[Info] {MAIN_FILE.name} absent, utilisation de {FALLBACK_MAIN_FILE.name}.")
        return FALLBACK_MAIN_FILE
    raise FileNotFoundError(f"Aucun fichier principal trouvé ({MAIN_FILE} ou {FALLBACK_MAIN_FILE}).")


def load_clean_ads(path: Path) -> pd.DataFrame:
    """Charge un CSV nettoyé de manière robuste."""
    df = pd.read_csv(path, dtype={"ad_id": str}, encoding="utf-8-sig")
    for col in ["country", "lang", "advertiser", "ad_text", "keyword", "image_urls"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    if "ad_id" not in df.columns:
        df["ad_id"] = ""
    return df


def annotate_unique_ads(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """Dédoublonne au niveau (pays, langue, annonceur, texte) et détecte chaque registre."""
    df_unique = (
        df.groupby(["country", "lang", "advertiser", "ad_text"], as_index=False, dropna=False)
        .agg(
            n_variants=("ad_id", "count"),
            keywords=("keyword", lambda s: ", ".join(sorted({x for x in s if x}))),
            image_urls=("image_urls", "first"),
        )
    )

    reg_cols = []
    for reg_name, patterns in REGISTERS.items():
        rx = re.compile("|".join(patterns), flags=re.IGNORECASE)
        col_name = f"reg_{reg_name}"
        df_unique[col_name] = df_unique["ad_text"].str.contains(rx, regex=True, na=False)
        reg_cols.append(col_name)

    df_unique["has_virility"] = df_unique[reg_cols].any(axis=1)
    df_unique["group"] = group_name
    return df_unique


def deduplicate_global(df_unique: pd.DataFrame) -> pd.DataFrame:
    """
    Dédoublonne au niveau global (advertiser + ad_text) pour éviter qu'une même création
    diffusée dans plusieurs pays ou langues ne soit comptée plusieurs fois dans le test global.
    """
    if df_unique.empty:
        return df_unique.copy()

    reg_cols = [f"reg_{r}" for r in REGISTERS]
    agg_dict = {
        "n_variants": ("n_variants", "sum"),
        "countries": ("country", lambda s: ", ".join(sorted({x for x in s if x}))),
        "langs": ("lang", lambda s: ", ".join(sorted({x for x in s if x}))),
        "keywords": ("keywords", lambda s: ", ".join(sorted({x for x in s if x}))),
        "image_urls": ("image_urls", "first"),
        "has_virility": ("has_virility", "max"),
        "group": ("group", "first"),
    }
    for col in reg_cols:
        agg_dict[col] = (col, "max")

    df_global = (
        df_unique.groupby(["advertiser", "ad_text"], as_index=False, dropna=False)
        .agg(**agg_dict)
    )
    df_global["has_virility"] = df_global["has_virility"].astype(bool)
    for col in reg_cols:
        df_global[col] = df_global[col].astype(bool)
    return df_global


def compute_country_summary(df_unique: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs et taux de registres par pays."""
    rows = []
    for ctry, sub in df_unique.groupby("country", dropna=False):
        n_ads = len(sub)
        row = {
            "country": ctry,
            "n_unique_ads": n_ads,
            "n_virility": int(sub["has_virility"].sum()),
            "pct_virility": round(sub["has_virility"].mean() * 100, 1) if n_ads else 0.0,
        }
        for reg_name in REGISTERS:
            row[f"pct_{reg_name}"] = round(sub[f"reg_{reg_name}"].mean() * 100, 1) if n_ads else 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by=["pct_virility", "n_unique_ads"],
        ascending=[False, False]
    )


def compute_or_and_ci95(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float, bool]:
    """
    Calcule l'OR de Fisher, la p-value exacte et l'IC 95 % (méthode log-OR de Woolf,
    avec correction de Haldane-Anscombe +0.5 uniquement pour l'IC 95 % si une case vaut 0).

    Table attendue :
        a = viande_bbq + positif | b = viande_bbq + négatif
        c = temoin + positif     | d = temoin + négatif
    """
    table = [[a, b], [c, d]]
    odds_ratio, p_fisher = fisher_exact(table)
    used_haldane = False

    if min(a, b, c, d) > 0:
        a_c, b_c, c_c, d_c = float(a), float(b), float(c), float(d)
        or_for_ci = float(odds_ratio)
    else:
        used_haldane = True
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        or_for_ci = (a_c * d_c) / (b_c * c_c)

    if or_for_ci > 0:
        se_log_or = math.sqrt((1 / a_c) + (1 / b_c) + (1 / c_c) + (1 / d_c))
        log_or = math.log(or_for_ci)
        ci_low = math.exp(log_or - 1.96 * se_log_or)
        ci_high = math.exp(log_or + 1.96 * se_log_or)
    else:
        ci_low, ci_high = float("nan"), float("nan")

    return float(odds_ratio), float(p_fisher), ci_low, ci_high, used_haldane


def main():
    main_path = resolve_main_file()

    # 1. Analyse du corpus viande & BBQ (par pays/langue puis global)
    df_main = load_clean_ads(main_path)
    main_u_by_locale = annotate_unique_ads(df_main, "viande_bbq")
    main_u_by_locale.to_csv(RESULTS_DIR / "main_unique_annotated.csv", index=False, encoding="utf-8-sig")

    main_u_global = deduplicate_global(main_u_by_locale)
    main_u_global.to_csv(RESULTS_DIR / "main_global_unique_annotated.csv", index=False, encoding="utf-8-sig")

    print("=== SYNTHÈSE GLOBALE (CORPUS VIANDE & BBQ — DÉDOUBLONNAGE GLOBAL) ===")
    print(f"Source : {main_path.name}")
    print(f"Créations uniques (pays/langue) : {len(main_u_by_locale)} | Créations uniques (global) : {len(main_u_global)}")
    if len(main_u_global) > 0:
        print(
            f"Part avec registre viriliste : {main_u_global['has_virility'].mean():.1%} "
            f"({int(main_u_global['has_virility'].sum())}/{len(main_u_global)})"
        )
        for reg_name in REGISTERS:
            col = f"reg_{reg_name}"
            print(f" - {reg_name:<18} : {int(main_u_global[col].sum())} ({main_u_global[col].mean():.1%})")

    # 2. Analyse par pays et test d'hétérogénéité entre pays
    df_country = compute_country_summary(main_u_by_locale)
    df_country.to_csv(RESULTS_DIR / "by_country_virility.csv", index=False, encoding="utf-8-sig")

    print("\n=== RÉPARTITION PAR PAYS (VIANDE & BBQ) ===")
    print(df_country.to_string(index=False))

    ct_country = pd.crosstab(main_u_by_locale["country"], main_u_by_locale["has_virility"])
    if ct_country.shape[0] > 1 and ct_country.shape[1] > 1:
        chi2_c, p_c, dof_c, _ = chi2_contingency(ct_country)
        print(f"\nTest Chi2 entre pays : chi2 = {chi2_c:.4f} (ddl = {dof_c}), p-value = {p_c:.4e}")
    else:
        print("\n[Info] Effectifs ou modalités insuffisants pour un test Chi2 global entre pays.")

    # 3. Comparaison statistique globale avec le groupe témoin (si disponible)
    if CONTROL_FILE.exists():
        df_ctrl = load_clean_ads(CONTROL_FILE)
        ctrl_u_by_locale = annotate_unique_ads(df_ctrl, "groupe_temoin")
        ctrl_u_by_locale.to_csv(RESULTS_DIR / "control_unique_annotated.csv", index=False, encoding="utf-8-sig")

        ctrl_u_global = deduplicate_global(ctrl_u_by_locale)
        ctrl_u_global.to_csv(RESULTS_DIR / "control_global_unique_annotated.csv", index=False, encoding="utf-8-sig")

        combined_global = pd.concat([main_u_global, ctrl_u_global], ignore_index=True)

        ct_grp = (
            pd.crosstab(combined_global["group"], combined_global["has_virility"])
            .reindex(index=["viande_bbq", "groupe_temoin"], columns=[True, False], fill_value=0)
        )

        a = int(ct_grp.loc["viande_bbq", True])
        b = int(ct_grp.loc["viande_bbq", False])
        c = int(ct_grp.loc["groupe_temoin", True])
        d = int(ct_grp.loc["groupe_temoin", False])

        odds_ratio, p_fisher, ci_low, ci_high, used_haldane_global = compute_or_and_ci95(a, b, c, d)

        print("\n=== COMPARAISON GLOBALE VIANDE & BBQ VS GROUPE TÉMOIN (DÉDOUBLONNAGE GLOBAL) ===")
        print(ct_grp)

        if ct_grp.shape == (2, 2) and (ct_grp.values.sum(axis=0) > 0).all() and (ct_grp.values.sum(axis=1) > 0).all():
            chi2_g, p_chi2_g, dof_g, _ = chi2_contingency(ct_grp)
            print(f"Chi2 = {chi2_g:.4f} (ddl = {dof_g}), p-value = {p_chi2_g:.4e}")

        haldane_note_g = " [IC 95% avec Haldane-Anscombe +0.5]" if used_haldane_global else ""
        print(
            f"Odds Ratio global (viande_bbq vs témoin) = {odds_ratio:.2f} "
            f"[IC 95 % : {ci_low:.2f} - {ci_high:.2f}]{haldane_note_g}, Fisher p = {p_fisher:.4e}"
        )

        # Détail par registre (sur créations uniques globales)
        reg_rows = []
        for reg_name in REGISTERS:
            col = f"reg_{reg_name}"
            m_pos = int(main_u_global[col].sum())
            m_neg = int(len(main_u_global) - m_pos)
            t_pos = int(ctrl_u_global[col].sum())
            t_neg = int(len(ctrl_u_global) - t_pos)

            or_r, p_r, lo_r, hi_r, haldane_r = compute_or_and_ci95(m_pos, m_neg, t_pos, t_neg)
            reg_rows.append({
                "register": reg_name,
                "meat_bbq_n": m_pos,
                "meat_bbq_pct": round(main_u_global[col].mean() * 100, 2) if len(main_u_global) else 0.0,
                "control_n": t_pos,
                "control_pct": round(ctrl_u_global[col].mean() * 100, 2) if len(ctrl_u_global) else 0.0,
                "odds_ratio": round(or_r, 3),
                "ci95_low": round(lo_r, 3),
                "ci95_high": round(hi_r, 3),
                "haldane_anscombe_ci": haldane_r,
                "p_value_fisher": p_r,
            })

        df_reg = pd.DataFrame(reg_rows)
        df_reg.to_csv(RESULTS_DIR / "by_register_comparison.csv", index=False, encoding="utf-8-sig")
        print("\n=== DÉTAIL PAR REGISTRE ===")
        print(df_reg.to_string(index=False))

        # Résumé global par groupe (sur créations uniques globales)
        comp_rows = []
        for grp, sub_df in [("viande_bbq", main_u_global), ("groupe_temoin", ctrl_u_global)]:
            n_ads = len(sub_df)
            row = {
                "group": grp,
                "n_unique_ads": n_ads,
                "n_virility": int(sub_df["has_virility"].sum()) if n_ads else 0,
                "pct_virility": round(sub_df["has_virility"].mean() * 100, 2) if n_ads else 0.0,
                "odds_ratio_global": round(odds_ratio, 3) if grp == "viande_bbq" else 1.0,
                "ci95_low": round(ci_low, 3) if grp == "viande_bbq" else None,
                "ci95_high": round(ci_high, 3) if grp == "viande_bbq" else None,
                "haldane_anscombe_ci": used_haldane_global if grp == "viande_bbq" else False,
                "p_value_fisher": p_fisher if grp == "viande_bbq" else None,
            }
            for reg_name in REGISTERS:
                row[f"pct_{reg_name}"] = round(sub_df[f"reg_{reg_name}"].mean() * 100, 2) if n_ads else 0.0
            comp_rows.append(row)

        pd.DataFrame(comp_rows).to_csv(
            RESULTS_DIR / "comparison_main_vs_control.csv", index=False, encoding="utf-8-sig"
        )
    else:
        print(f"\n[Info] Groupe témoin non encore présent ({CONTROL_FILE}).")


if __name__ == "__main__":
    main()