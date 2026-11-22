from pathlib import Path
import pandas as pd

REG_COLS = ["reg_testosterone", "reg_identite_masculine", "reg_ancestral_alpha"]


def load_positives(csv_path: Path, corpus_label: str) -> pd.DataFrame:
    """Extrait les créations uniques positives d'un fichier annoté et ajoute les registres détectés."""
    if not csv_path.exists():
        print(f"[Info] Fichier absent : {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    pos = df[df["has_virility"].str.lower() == "true"].copy()
    pos["corpus"] = corpus_label

    def get_registers(row: pd.Series) -> str:
        active = [
            col.replace("reg_", "")
            for col in REG_COLS
            if row.get(col, "").strip().lower() == "true"
        ]
        return ", ".join(active)

    pos["registers"] = pos.apply(get_registers, axis=1)
    pos["audit_status"] = ""  # À remplir à la main : ex. "ok", "faux_positif", "hors_sujet"
    pos["audit_note"] = ""    # Commentaire libre lors de la vérification

    preferred_cols = [
        "corpus",
        "countries",
        "langs",
        "advertiser",
        "registers",
        "audit_status",
        "audit_note",
        "ad_text",
    ]
    remaining_cols = [c for c in pos.columns if c not in preferred_cols]
    return pos[preferred_cols + remaining_cols]


def main() -> None:
    out_dir = Path("Results")
    out_dir.mkdir(parents=True, exist_ok=True)

    main_pos = load_positives(out_dir / "main_global_unique_annotated.csv", "viande_bbq")
    ctrl_pos = load_positives(out_dir / "control_global_unique_annotated.csv", "groupe_temoin")

    print(f"Annonces positives (viande_bbq) : {len(main_pos)}")
    print(f"Annonces positives (groupe_temoin) : {len(ctrl_pos)}")

    combined = pd.concat([main_pos, ctrl_pos], ignore_index=True)
    out_file = out_dir / "audit_positive_ads.csv"
    combined.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nCSV de vérification créé : {out_file} ({len(combined)} lignes)")


if __name__ == "__main__":
    main()