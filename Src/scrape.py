import time, re
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30))

def safe_call(fn, tries=5, wait=20):
    """Réessaie avec attente croissante en cas de 429."""
    for i in range(tries):
        try:
            return fn()
        except TooManyRequestsError:
            time.sleep(wait * (i + 1))
    raise RuntimeError("Google Trends bloque : réessaie plus tard ou change d'IP")

# ---- 1. Test minimal : un seul appel ----
kws = ["meat", "masculinity", "barbecue", "beef"]
pytrends.build_payload(kws, timeframe="2023-01-01 2023-12-31", geo="")
df_world = safe_call(lambda: pytrends.interest_by_region(
    resolution="COUNTRY", inc_low_vol=True))
print(df_world.sort_values("meat", ascending=False).head(10))
df_world.to_csv("trends_world.csv")

# ---- 2. Requêtes associées avec filtre par mots entiers ----
related = safe_call(pytrends.related_queries)
pattern = re.compile(r"\b(?:man|men|masculin\w*|manly|manliness|woman|women|vegan)\b", re.I)
for kw in kws:
    top = related.get(kw, {}).get("top")
    if top is not None:
        print(f"\n{kw}:\n", top[top["query"].str.contains(pattern)])

# ---- 3. Indice par pays (à lancer seulement une fois 1 validé) ----
countries = ["FR", "DE", "ES", "IT", "GB", "PL", "SE"]
rows = {}
for geo in countries:
    pytrends.build_payload(kws, timeframe="2023-01-01 2023-12-31", geo=geo)
    d = safe_call(pytrends.interest_over_time)
    rows[geo] = d.drop(columns="isPartial", errors="ignore").mean()
    time.sleep(10)
df_trends = pd.DataFrame(rows).T
df_trends.to_csv("trends_by_country.csv")
print(df_trends)