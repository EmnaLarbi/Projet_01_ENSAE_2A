"""
Scraping de publicités Meta – groupe témoin
-------------------------------------------
Collecte complète sur tous les pays et mots-clés témoins avec Playwright.

Installation :
    pip install pandas playwright
    playwright install chromium

Lancement :
    python Src/scrape_control.py

Corrections par rapport à la version précédente
-----------------------------------------------
1. Animaux de compagnie : liste appliquée LANGUE PAR LANGUE (« pas » = chien en
   croate, mais négation en français : les publicités françaises n'étaient plus
   rejetées à tort).
2. Groupe témoin sans viande : les publicités qui parlent de viande (bœuf, agneau,
   volaille, porc, veau, dans leur langue ou en anglais) sont retirées du fichier
   nettoyé. Sinon le témoin recoupe le groupe « viande » et la comparaison est faussée.
3. Mêmes filtres que pour le groupe « viande » : les motifs sont importés de
   Src/scraping_meta.py (hors sujet, animaux, mots de viande), pour que les deux
   groupes soient nettoyés de la même façon.
5. Même méthode d'extraction que le groupe « viande » : repérage des cartes
   publicitaires, défilement et lecture des blocs importés de Src/scraping_meta.py.
4. Mots-clés ambigus remplacés (« plat » → écran plat, ventre plat ; « rätt » / « rett »
   / « réttur » → « correct », « droit » ; « annos » → dose ; « fogás » → prise ;
   « chod » → marche) et filtre des meubles de cuisine (« cuisine », « Küche »,
   « cucina », « cocina » ramènent surtout des cuisinistes).
"""

import re
import urllib.parse
from pathlib import Path
from typing import Optional

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Filtres ET méthode de collecte communs avec le groupe « viande » (même dossier Src/) :
# les deux groupes sont extraits et lus exactement de la même façon.
from scraping_meta import (
    ETIQUETTES_ID, JS_EXTRAIRE_CARTES, OFF_TOPIC_PATTERN, PAUSE_ENTRE_REQUETES,
    PET_PATTERNS, _ad_langs, _faire_defiler, _fermer_bandeau_cookies, detect_meats,
    parse_ad_block,
)


# ---------------------------------------------------------------------
# 1. PARAMÈTRES DE COLLECTE (MODE COMPLET)
# ---------------------------------------------------------------------

HEADLESS = True
MAX_SCROLLS = 8
CHECKPOINT_EVERY = 10

# None = tous les pays configurés dans TARGET_COUNTRIES_LANGS
COUNTRIES_TO_SCRAPE: Optional[list[str]] = None

# None = toutes les requêtes des pays sélectionnés
MAX_QUERIES: Optional[int] = None

# Désactivé pour l'exécution complète automatisée
PAUSE_AFTER_FIRST_PAGE = False

# Reprise automatique depuis le CSV brut existant si présent
RESUME_FROM_EXISTING_CSV = True


# ---------------------------------------------------------------------
# 2. MOTS-CLÉS ET PAYS
# ---------------------------------------------------------------------
# Les mots ambigus sont remplacés (ancien mot en commentaire). Listes à faire
# relire par des locuteurs natifs si possible.

CONTROL_KEYWORDS_BY_LANG = {
    "fr": ["alimentation", "aliment", "nourriture", "nutrition", "cuisine",
           "plat cuisiné",          # « plat » : écran plat, ventre plat…
           "recette", "repas"],
    "de": ["Ernährung", "Lebensmittel", "Nahrung", "Nahrungsmittel", "Küche",
           "Gericht", "Rezept", "Mahlzeit"],
    "it": ["alimentazione", "alimento", "cibo", "nutrizione", "cucina",
           "piatto", "ricetta", "pasto"],
    "es": ["alimentación", "alimento", "comida", "nutrición", "cocina",
           "plato", "receta", "menú"],
    "fi": ["ravinto", "elintarvike", "ruoka", "ravitsemus", "ruoanlaitto",
           "ruoka-annos",           # « annos » : aussi une dose de médicament
           "resepti", "ateria"],
    "hr": ["prehrana", "namirnica", "hrana", "nutricionizam", "kuhanje",
           "jelo", "recept", "obrok"],
    "hu": ["táplálkozás", "élelmiszer", "étel", "táplálkozástudomány", "főzés",
           "főétel",                # « fogás » : aussi une prise, une astuce
           "recept", "étkezés"],
    "is": ["næring", "matvæli", "matur", "mataræði", "eldamennska",
           "kvöldmatur",            # « réttur » : aussi « droit », « correct »
           "uppskrift", "máltíð"],
    "lt": ["mityba", "maistas", "maisto produktas", "mitybos", "gaminimas",
           "patiekalas", "receptas", "valgis"],
    "no": ["ernæring", "matvare", "mat", "kosthold", "matlaging",
           "middag",                # « rett » : aussi « droit », « correct »
           "oppskrift", "måltid"],
    "se": ["näring", "livsmedel", "mat", "kost", "matlagning",
           "middag",                # « rätt » : aussi « correct », « tribunal »
           "recept", "måltid"],
    "si": ["prehrana", "živilo", "hrana", "hranilna", "kuhanje",
           "jed", "recept", "obrok"],
    "sk": ["výživa", "potravina", "jedlo", "stravovanie", "varenie",
           "pokrm", "recept",
           "obed"],                 # « chod » : aussi « marche », « fonctionnement »
}

TARGET_COUNTRIES_LANGS = {
    "AT": ["de"], "CH": ["de", "fr", "it"], "DE": ["de"], "ES": ["es"], "FI": ["fi"],
    "FR": ["fr"], "HR": ["hr"], "HU": ["hu"], "IS": ["is"], "IT": ["it"], "LT": ["lt"],
    "NO": ["no"], "SE": ["se"], "SI": ["si"], "SK": ["sk"],
}

# Meubles et équipement de cuisine : hors sujet pour un témoin « alimentation »
KITCHEN_FURNITURE_PATTERN = re.compile("|".join([
    r"\bcuisiniste\w*\b", r"\bcuisines?\s+(?:équipée|aménagée|sur\s+mesure)s?\b", r"\bplan\s+de\s+travail\b",
    r"\beinbauküche\w*\b", r"\bküchenstudio\w*\b", r"\bküchenmöbel\w*\b", r"\barbeitsplatte\w*\b",
    r"\bcucine?\s+componibil\w*\b", r"\bmobili\s+(?:da|per)\s+cucina\b", r"\bpiano\s+di\s+lavoro\b",
    r"\bmuebles\s+de\s+cocina\b", r"\bcocinas?\s+a\s+medida\b", r"\bencimera\w*\b",
    r"\bkitchen\s+(?:cabinet|furniture|design|renovation)\w*\b", r"\bworktop\w*\b",
    r"\bkeittiökalust\w*\b", r"\bkjøkkeninnredning\w*\b", r"\bköksinredning\w*\b",
]), flags=re.IGNORECASE)


# ---------------------------------------------------------------------
# 3. CONSTRUCTION DU PLAN DE RECHERCHE
# ---------------------------------------------------------------------

def build_control_search_plan(
    countries: Optional[list[str]] = None,
    limit: Optional[int] = None
) -> list[tuple[str, str, str]]:
    """Construit des tuples (pays, langue, mot-clé)."""
    selected_countries = countries if countries is not None else list(TARGET_COUNTRIES_LANGS.keys())
    plan: list[tuple[str, str, str]] = []

    for country in selected_countries:
        langs = TARGET_COUNTRIES_LANGS.get(country, [])
        for lang in langs:
            for keyword in CONTROL_KEYWORDS_BY_LANG.get(lang, []):
                plan.append((country, lang, keyword))

    return plan[:limit] if limit is not None else plan


def build_ad_library_url(country: str, keyword: str) -> str:
    """Construit l'URL de recherche dans Meta Ad Library (sans exact phrase)."""
    encoded_keyword = urllib.parse.quote(keyword)
    return (
        "https://www.facebook.com/ads/library/"
        f"?active_status=all&ad_type=all&country={country}&q={encoded_keyword}"
    )


# ---------------------------------------------------------------------
# 4. EXTRACTION (même méthode que le groupe « viande »)
# ---------------------------------------------------------------------

AD_ID_PATTERN = re.compile(r"^\d{10,20}$")      # un identifiant Meta valide

META_FOOTER_PATTERNS = [
    r"État du système\s*API Ad Library.*",
    r"Ad Library API status.*",
    r"À propos des publicités et de l[’']utilisation des données.*",
    r"About ads and data use.*",
    r"Confidentialité\s*Conditions\s*Cookies.*",
    r"Privacy\s*Terms\s*Cookies.*",
]

META_FOOTER_REGEX = re.compile(
    r"(?:\s*(?:" + "|".join(META_FOOTER_PATTERNS) + r"))\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_meta_footer(text: str) -> str:
    """Supprime les mentions de pied de page Meta Ad Library en fin de texte."""
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaned = META_FOOTER_REGEX.sub("", text)
    return re.sub(r"\s*$", "", cleaned).strip()


def extract_ad_records(
    cards: list[dict],
    country: str,
    lang: str,
    keyword: str,
    seen_ids: set[str]
) -> tuple[list[dict], int]:
    """Transforme les cartes repérées sur la page en enregistrements.
    Chaque carte est lue par `parse_ad_block` du script viande."""
    records: list[dict] = []
    duplicates = 0

    for card in cards:
        parsed = parse_ad_block(card["texte"])
        ad_id = str(parsed.get("ad_id") or "").strip()
        if not AD_ID_PATTERN.match(ad_id):
            continue
        if ad_id in seen_ids:
            duplicates += 1
            continue

        parsed["ad_text"] = clean_meta_footer(parsed["ad_text"])
        parsed.update({
            "country": country,
            "lang": lang,
            "keyword": keyword,
            "ad_id": ad_id,
            "image_urls": " | ".join(card.get("images", [])),
        })
        seen_ids.add(ad_id)
        records.append(parsed)

    return records, duplicates


# ---------------------------------------------------------------------
# 5. FILTRES DU GROUPE TÉMOIN
# ---------------------------------------------------------------------

def control_rejection_reason(ad_text: str, advertiser: str, lang: str) -> Optional[str]:
    """Raison du rejet d'une publicité témoin, ou None si elle est retenue."""
    combined = f"{advertiser or ''} {ad_text or ''}"
    if len(ad_text or "") < 20:
        return "texte trop court"
    for l in _ad_langs(lang):                       # correction 1 : par langue
        rx = PET_PATTERNS.get(l)
        if rx is not None and rx.search(combined):
            return "animaux de compagnie"
    if OFF_TOPIC_PATTERN.search(combined):           # correction 3 : mêmes filtres
        return "hors sujet"
    if KITCHEN_FURNITURE_PATTERN.search(combined):   # correction 4
        return "meubles de cuisine"
    if detect_meats(ad_text, advertiser, lang):      # correction 2 : témoin sans viande
        return "mentionne de la viande"
    return None


# ---------------------------------------------------------------------
# 6. SAUVEGARDE ET CONFORMITÉ CSV
# ---------------------------------------------------------------------

RAW_COLUMNS = [
    "country", "lang", "keyword", "ad_id", "status",
    "start_date", "advertiser", "ad_text", "image_urls"
]

CLEAN_COLUMNS = [
    "country", "lang", "keyword", "ad_id", "status",
    "start_date", "advertiser", "ad_text", "image_urls", "n_variants"
]


def save_control_outputs(
    all_ads: list[dict],
    raw_file: Path,
    clean_file: Path
) -> None:
    """Sauvegarde et valide la conformité des fichiers brut et nettoyé."""
    if not all_ads:
        print("Aucune donnée à sauvegarder pour le moment.")
        return

    df_raw = pd.DataFrame(all_ads)

    for column in RAW_COLUMNS:
        if column not in df_raw.columns:
            df_raw[column] = ""

    df_raw = df_raw[RAW_COLUMNS].copy()
    for col in RAW_COLUMNS:
        df_raw[col] = df_raw[col].fillna("").astype(str).str.strip()

    df_raw = df_raw[df_raw["ad_id"] != ""]
    df_raw = df_raw.drop_duplicates(subset=["ad_id"], keep="first").reset_index(drop=True)
    df_raw.to_csv(raw_file, index=False, encoding="utf-8-sig")

    # Filtrage du fichier nettoyé (le fichier brut garde tout, pour traçabilité)
    df_clean = df_raw.copy()
    df_clean["ad_text"] = df_clean["ad_text"].map(clean_meta_footer)
    df_clean["rejet"] = [
        control_rejection_reason(r.ad_text, r.advertiser, r.lang) for r in df_clean.itertuples()
    ]
    print("Filtrage du groupe témoin :")
    print(df_clean["rejet"].fillna("retenue").value_counts().to_string())
    df_clean = df_clean[df_clean["rejet"].isna()].drop(columns="rejet")

    if not df_clean.empty:
        df_clean["n_variants"] = (
            df_clean.groupby(["advertiser", "ad_text"])["ad_id"]
            .transform("count")
            .astype(int)
        )
    else:
        df_clean["n_variants"] = pd.Series(dtype=int)

    df_clean = df_clean[CLEAN_COLUMNS].reset_index(drop=True)
    df_clean.to_csv(clean_file, index=False, encoding="utf-8-sig")

    assert list(pd.read_csv(raw_file, nrows=1).columns) == RAW_COLUMNS
    if len(df_clean) > 0:
        assert list(pd.read_csv(clean_file, nrows=1).columns) == CLEAN_COLUMNS

    print(f"CSV brut conforme sauvegardé : {raw_file} ({len(df_raw)} lignes)")
    print(f"CSV nettoyé conforme sauvegardé : {clean_file} ({len(df_clean)} lignes)")


# ---------------------------------------------------------------------
# 7. REPRISE ET COLLECTE
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "Data" / "raw" / "meta_ads_control_raw.csv"
CLEAN_FILE = BASE_DIR / "Data" / "processed" / "meta_ads_control_clean.csv"
DONE_FILE = BASE_DIR / "Data" / "raw" / "meta_ads_control_requetes_faites.txt"  # journal de reprise


def load_existing_data(raw_file: Path) -> tuple[list[dict], set[str]]:
    """Recharge le CSV brut existant pour reprendre une collecte."""
    if not raw_file.exists():
        return [], set()
    try:
        df = pd.read_csv(raw_file, dtype=str, encoding="utf-8-sig").fillna("")
    except Exception as e:
        print(f"Avertissement : lecture impossible de {raw_file} ({e}). Nouvelle collecte.")
        return [], set()
    df = df[df["ad_id"].str.match(r"^\d{10,20}$")]
    records = df.to_dict(orient="records")
    seen_ids = set(df["ad_id"])
    print(f"Reprise depuis {raw_file} : {len(records)} publicités déjà collectées.")
    return records, seen_ids


def scrape_control() -> None:
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)

    plan = build_control_search_plan(COUNTRIES_TO_SCRAPE, MAX_QUERIES)
    all_ads, seen_ids = load_existing_data(RAW_FILE) if RESUME_FROM_EXISTING_CSV else ([], set())

    # Requêtes déjà terminées (journal), même celles qui n'ont rien apporté de nouveau
    deja_faites = set()
    if RESUME_FROM_EXISTING_CSV and DONE_FILE.exists():
        deja_faites = {tuple(l.split("\t")) for l in DONE_FILE.read_text(encoding="utf-8").splitlines() if l}
    restantes = [q for q in plan if q not in deja_faites]
    print(f"Plan témoin : {len(plan)} requêtes, {len(plan) - len(restantes)} déjà faites, "
          f"{len(restantes)} restantes.")

    requetes_vides = 0
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=HEADLESS)       # navigateur standard, sans déguisement
            context = browser.new_context(locale="fr-FR", viewport={"width": 1440, "height": 900})
            page = context.new_page()

            for idx, (country, lang, keyword) in enumerate(restantes, start=1):
                print(f"\n[{idx}/{len(restantes)}] Pays : {country} ({lang}) | Mot-clé : '{keyword}'")
                try:
                    page.goto(build_ad_library_url(country, keyword),
                              wait_until="domcontentloaded", timeout=35000)
                    page.wait_for_timeout(3000)
                except PlaywrightTimeoutError:
                    print(" -> Délai de chargement dépassé.")
                    continue
                except Exception as e:
                    print(f" -> Erreur de chargement : {e}")
                    continue

                _fermer_bandeau_cookies(page)
                _faire_defiler(page, MAX_SCROLLS)
                cards = page.evaluate(JS_EXTRAIRE_CARTES, list(ETIQUETTES_ID))

                if PAUSE_AFTER_FIRST_PAGE and idx == 1:
                    input(" -> Vérifiez la page dans le navigateur puis appuyez sur Entrée…")

                if not cards:
                    requetes_vides += 1
                    print(" -> Aucune publicité affichée.")
                    # Plusieurs pages vides de suite : accès probablement bloqué
                    # (connexion demandée, limitation). On s'arrête au lieu d'insister.
                    if requetes_vides >= 5:
                        print("\n5 recherches vides consécutives : arrêt de la collecte. "
                              "Vérifier la page dans un navigateur.")
                        break
                    page.wait_for_timeout(PAUSE_ENTRE_REQUETES)
                    continue
                requetes_vides = 0

                records, duplicates = extract_ad_records(cards, country, lang, keyword, seen_ids)
                all_ads.extend(records)
                print(f" -> {len(cards)} cartes, {len(records)} nouvelles, {duplicates} déjà vues "
                      f"(total : {len(all_ads)})")

                with DONE_FILE.open("a", encoding="utf-8") as journal:
                    journal.write(f"{country}\t{lang}\t{keyword}\n")

                if idx % CHECKPOINT_EVERY == 0:
                    save_control_outputs(all_ads, RAW_FILE, CLEAN_FILE)

                page.wait_for_timeout(PAUSE_ENTRE_REQUETES)

            browser.close()
    finally:
        # Sauvegarde finale, y compris en cas d'interruption (Ctrl+C, erreur)
        save_control_outputs(all_ads, RAW_FILE, CLEAN_FILE)
        print("\nCollecte témoin terminée.")


if __name__ == "__main__":
    scrape_control()