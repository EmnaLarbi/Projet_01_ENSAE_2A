from pathlib import Path
import pandas as pd

FILE_PATH = Path("Data/processed/meta_ads_meat_bbq_clean.csv")

if not FILE_PATH.exists():
    print("KO : fichier introuvable.")
else:
    df = pd.read_csv(FILE_PATH, dtype={"ad_id": str}, encoding="utf-8-sig").fillna("")
    usable_text = df[df["ad_text"].str.strip().str.len() >= 25]
    unique_creatives = usable_text.drop_duplicates(subset=["country", "lang", "advertiser", "ad_text"])

    print(f"Lignes totales          : {len(df)}")
    print(f"Textes exploitables     : {len(usable_text)}")
    print(f"Créations uniques       : {len(unique_creatives)}")
    print(f"Pays présents ({unique_creatives['country'].nunique()}) : {sorted(unique_creatives['country'].unique())}")
    print(f"Annonceurs renseignés   : {(unique_creatives['advertiser'].str.strip() != '').sum()}/{len(unique_creatives)}")

    print("\n--- 5 exemples à relire ---")
    for _, row in unique_creatives.sample(min(5, len(unique_creatives)), random_state=1).iterrows():
        print(f"[{row['country']} | {row['lang']}] {row['advertiser']} -> {row['ad_text'][:180]}")