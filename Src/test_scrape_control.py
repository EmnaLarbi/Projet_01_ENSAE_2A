"""
Inspection rapide des annonces scrappées :
- Comptage par pays
- Tirage aléatoire de 10 annonces pour vérification manuelle de pertinence

Exemples d'exécution :
    python Src/inspect_scraped_ads.py
    python Src/inspect_scraped_ads.py Data/processed/meta_ads_control_multilang_clean.csv
    python Src/inspect_scraped_ads.py Data/processed/meta_ads_meat_bbq_clean.csv --seed 42
"""

import argparse
from pathlib import Path
import pandas as pd


CANDIDATE_FILES = [
    Path("Data/processed/meta_ads_control_multilang_clean.csv"),
    Path("Data/raw/meta_ads_control_multilang_raw.csv"),
    Path("Data/test_control/test_control_clean.csv"),
    Path("Data/test_control/test_control_raw.csv"),
    Path("Data/processed/meta_ads_meat_bbq_clean.csv"),
    Path("Data/raw/meta_ads_meat_bbq_raw.csv"),
]


def resolve_csv_path(explicit_path: str | None, project_dir: Path) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        return p if p.is_absolute() else (project_dir / p)

    for rel in CANDIDATE_FILES:
        candidate = project_dir / rel
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Aucun CSV trouvé automatiquement. Passe le chemin du fichier en argument :\n"
        "  python Src/inspect_scraped_ads.py chemin/vers/fichier.csv"
    )


def print_counts_by_country(df: pd.DataFrame) -> None:
    print("\n=== Nombre d'annonces scrappées par pays ===")
    if "country" not in df.columns or df.empty:
        print("Colonne 'country' absente ou DataFrame vide.")
        return

    tmp = df.copy()
    tmp["country"] = tmp["country"].fillna("").astype(str).str.strip()

    if "ad_id" in tmp.columns:
        tmp["ad_id"] = tmp["ad_id"].fillna("").astype(str).str.strip()
        summary = (
            tmp.groupby("country", dropna=False)
            .agg(
                lignes=("country", "count"),
                ids_uniques=("ad_id", lambda s: s[s != ""].nunique()),
            )
            .sort_values(by=["lignes", "country"], ascending=[False, True])
        )
    else:
        summary = (
            tmp.groupby("country", dropna=False)
            .size()
            .to_frame(name="lignes")
            .sort_values(by=["lignes", "country"], ascending=[False, True])
        )

    print(summary.to_string())
    print("-" * 45)
    print(f"Total lignes : {len(tmp)}")
    if "ad_id" in tmp.columns:
        unique_ids = tmp["ad_id"][tmp["ad_id"] != ""].nunique()
        print(f"Total IDs uniques : {unique_ids}")


def print_random_samples(df: pd.DataFrame, n: int = 10, seed: int | None = None) -> None:
    print(f"\n=== {min(n, len(df))} annonces tirées au hasard pour vérification manuelle ===")
    if df.empty:
        print("Aucune annonce à afficher.")
        return

    sample_df = df.sample(n=min(n, len(df)), random_state=seed)

    for idx, (_, row) in enumerate(sample_df.iterrows(), start=1):
        country = str(row.get("country", "")).strip()
        lang = str(row.get("lang", "")).strip()
        keyword = str(row.get("keyword", "")).strip()
        advertiser = str(row.get("advertiser", "")).strip()
        ad_id = str(row.get("ad_id", "")).strip()
        ad_text = " ".join(str(row.get("ad_text", "")).split())

        print(f"\n[{idx}/{len(sample_df)}] Pays={country or '?'} | Lang={lang or '?'} | Mot-clé={keyword or '?'}")
        print(f"Annonceur : {advertiser or '(non renseigné)'} | ID : {ad_id or '(n/a)'}")
        print(f"Texte     : {ad_text if ad_text else '(vide)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compte les annonces par pays et tire 10 exemples aléatoires.")
    parser.add_argument("csv_path", nargs="?", default=None, help="Chemin vers le fichier CSV à inspecter")
    parser.add_argument("-n", "--samples", type=int, default=10, help="Nombre d'annonces à tirer (défaut : 10)")
    parser.add_argument("--seed", type=int, default=None, help="Graine aléatoire optionnelle pour reproductibilité")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    csv_file = resolve_csv_path(args.csv_path, project_dir)

    print(f"Fichier inspecté : {csv_file}")
    df = pd.read_csv(csv_file, dtype=str).fillna("")

    print_counts_by_country(df)
    print_random_samples(df, n=args.samples, seed=args.seed)


if __name__ == "__main__":
    main()