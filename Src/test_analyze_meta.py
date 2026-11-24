import shutil
from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".")
processed_dir = base_dir / "Data" / "processed"

clean_file = processed_dir / "meta_ads_meat_bbq_clean.csv"
backup_file = processed_dir / "meta_ads_meat_bbq_clean_avant_filtre.csv"

# 1. Copie de sauvegarde
shutil.copy2(clean_file, backup_file)

df = pd.read_csv(clean_file, dtype={"ad_id": str}, encoding="utf-8-sig")
print("Avant filtrage :", len(df), "lignes")
print(df["meat_type"].value_counts(dropna=False))

# 2. Garder uniquement 'red_meat' (ou remplace par ['red_meat', 'general_meat_bbq'] si tu veux garder le BBQ générique)
keep_types = ["red_meat"]
df_filtered = df[df["meat_type"].isin(keep_types)].copy()

# 3. Recalculer n_variants sur les annonces conservées
df_filtered["n_variants"] = (
    df_filtered.groupby(["advertiser", "ad_text"])["ad_id"]
    .transform("count")
    .astype(int)
)

df_filtered.to_csv(clean_file, index=False, encoding="utf-8-sig")
print("\nAprès filtrage :", len(df_filtered), "lignes sauvegardées dans", clean_file)