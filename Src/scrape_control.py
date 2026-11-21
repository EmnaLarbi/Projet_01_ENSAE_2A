"""
Scraping de publicités Meta – groupe témoin
-------------------------------------------
Collecte complète sur tous les pays et mots-clés témoins avec Playwright.

Installation :
    pip install pandas playwright
    playwright install chromium

Lancement :
    python Src/scrape_control.py
"""

import re
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


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

PET_BLACKLIST = [
    # Français
    "chien", "chiens", "chiot", "chiots", "chat", "chats", "chaton", "chatons",
    "croquette", "croquettes", "animal de compagnie", "animaux de compagnie", "vétérinaire",
    # Anglais
    "dog", "dogs", "puppy", "puppies", "cat", "cats", "kitten", "kittens",
    "pet food", "dog food", "cat food", "kibble", "veterinary",
    # Allemand
    "hund", "hunde", "welpe", "welpen", "katze", "katzen", "kätzchen",
    "hundefutter", "katzenfutter", "tierfutter", "haustier", "haustiere",
    # Italien
    "cane", "cani", "cucciolo", "cuccioli", "gatto", "gatti", "gattino", "gattini",
    "crocchette", "cibo per cani", "cibo per gatti", "animali domestici",
    # Espagnol
    "perro", "perros", "cachorro", "cachorros", "gato", "gatos", "gatito", "gatitos",
    "croquetas", "comida para perros", "comida para gatos", "mascotas",
    # Finnois
    "koira", "koirat", "pentu", "pennut", "kissa", "kissat", "kissanpentu",
    "koiranruoka", "kissaruoka", "lemmikinruoka", "lemmikki", "lemmikit",
    # Croate
    "pas", "psi", "štene", "mačka", "mačke", "mačić",
    "hrana za pse", "hrana za mačke", "kućni ljubimci",
    # Hongrois
    "kutya", "kutyák", "kölyökkutya", "macska", "macskák", "kiscica",
    "kutyatáp", "macskatáp", "állateledel", "háziállat",
    # Islandais
    "hundur", "hundar", "hvolpur", "köttur", "kettir", "kettlingur",
    "hundamatur", "kattamatur", "gæludýr", "gæludýrafóður",
    # Lituanien
    "šuo", "šunys", "šuniukas", "katė", "katės", "kačiukas",
    "šunų maistas", "kačių maistas", "naminių gyvūnų",
    # Norvégien
    "hund", "hunder", "valp", "valper", "katt", "katter", "kattunge",
    "hundemat", "kattemat", "kjæledyr", "dyrefôr",
    # Suédois
    "hund", "hundar", "valp", "valpar", "katt", "katter", "kattunge",
    "hundmat", "kattmat", "husdjur", "djurfoder",
    # Slovène
    "pes", "psi", "kuža", "mačka", "mačke", "mucek",
    "hrana za pse", "hrana za mačke", "hišni ljubimci",
    # Slovak
    "pes", "psy", "šteňa", "mačka", "mačky", "mača",
    "krmivo pre psov", "krmivo pre mačky", "domáce zvieratá",
]

CONTROL_KEYWORDS_BY_LANG = {
    "fr": [
        "alimentation",
        "aliment",
        "nourriture",
        "nutrition",
        "cuisine",
        "plat",
        "recette",
        "repas",
        "santé",
        "bien-être",
    ],
    "de": [
        "Ernährung",
        "Lebensmittel",
        "Nahrung",
        "Nahrungsmittel",
        "Küche",
        "Gericht",
        "Rezept",
        "Mahlzeit",
        "Gesundheit",
        "Wohlbefinden",
    ],
    "it": [
        "alimentazione",
        "alimento",
        "cibo",
        "nutrizione",
        "cucina",
        "piatto",
        "ricetta",
        "pasto",
        "salute",
        "benessere",
    ],
    "es": [
        "alimentación",
        "alimento",
        "comida",
        "nutrición",
        "cocina",
        "plato",
        "receta",
        "menú",
        "salud",
        "bienestar",
    ],
    "fi": [
        "ravinto",
        "elintarvike",
        "ruoka",
        "ravitsemus",
        "ruoanlaitto",
        "annos",
        "resepti",
        "ateria",
        "terveys",
        "hyvinvointi",
    ],
    "hr": [
        "prehrana",
        "namirnica",
        "hrana",
        "nutricionizam",
        "kuhanje",
        "jelo",
        "recept",
        "obrok",
        "zdravlje",
        "blagostanje",
    ],
    "hu": [
        "táplálkozás",
        "élelmiszer",
        "étel",
        "táplálkozástudomány",
        "főzés",
        "fogás",
        "recept",
        "étkezés",
        "egészség",
        "jóllét",
    ],
    "is": [
        "næring",
        "matvæli",
        "matur",
        "mataræði",
        "eldamennska",
        "réttur",
        "uppskrift",
        "máltíð",
        "heilsa",
        "vellíðan",
    ],
    "lt": [
        "mityba",
        "maistas",
        "maisto produktas",
        "mitybos",
        "gaminimas",
        "patiekalas",
        "receptas",
        "valgis",
        "sveikata",
        "gerovė",
    ],
    "no": [
        "ernæring",
        "matvare",
        "mat",
        "kosthold",
        "matlaging",
        "rett",
        "oppskrift",
        "måltid",
        "helse",
        "velvære",
    ],
    "se": [
        "näring",
        "livsmedel",
        "mat",
        "kost",
        "matlagning",
        "rätt",
        "recept",
        "måltid",
        "hälsa",
        "välmående",
    ],
    "si": [
        "prehrana",
        "živilo",
        "hrana",
        "hranilna",
        "kuhanje",
        "jed",
        "recept",
        "obrok",
        "zdravje",
        "dobro počutje",
    ],
    "sk": [
        "výživa",
        "potravina",
        "jedlo",
        "stravovanie",
        "varenie",
        "pokrm",
        "recept",
        "chod",
        "zdravie",
        "pohoda",
    ],
}

# Langues locales uniquement (aligné avec la collecte viande/BBQ)
TARGET_COUNTRIES_LANGS = {
    "AT": ["de"],
    "CH": ["de", "fr", "it"],
    "DE": ["de"],
    "ES": ["es"],
    "FI": ["fi"],
    "FR": ["fr"],
    "HR": ["hr"],
    "HU": ["hu"],
    "IS": ["is"],
    "IT": ["it"],
    "LT": ["lt"],
    "NO": ["no"],
    "SE": ["se"],
    "SI": ["si"],
    "SK": ["sk"],
}


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
# 4. EXTRACTION ET PARSING DU TEXTE
# ---------------------------------------------------------------------

AD_ID_PATTERN = re.compile(
    r"(?:ID dans la bibliothèque|Library ID)\s*:\s*(\d{10,20})",
    flags=re.IGNORECASE
)


def parse_ad_block(block_text: str) -> dict:
    """Extrait les métadonnées et le texte d'un bloc d'annonce Meta."""
    lines = [line.strip() for line in block_text.splitlines() if line.strip()]

    status = ""
    start_date = ""
    advertiser = ""
    ad_body_lines: list[str] = []

    status_labels = {"actif", "inactive", "inactif", "active"}
    sponsored_labels = {"sponsorisé", "sponsorisee", "sponsored"}

    for i, line in enumerate(lines):
        normalized = line.casefold()

        if normalized in status_labels:
            status = line

        if "début de la diffusion" in normalized or "started running on" in normalized:
            start_date = (
                line.split("·")[0]
                .replace("Début de la diffusion le", "")
                .replace("Début de diffusion le", "")
                .replace("Started running on", "")
                .strip()
            )

        if normalized in sponsored_labels and i > 0:
            advertiser = lines[i - 1]

            for next_line in lines[i + 1:]:
                next_normalized = next_line.casefold()
                if (
                    next_normalized in status_labels
                    or "id dans la bibliothèque" in next_normalized
                    or "id de la bibliothèque" in next_normalized
                    or "library id" in next_normalized
                ):
                    break
                ad_body_lines.append(next_line)
            break

    return {
        "status": status,
        "start_date": start_date,
        "advertiser": advertiser,
        "ad_text": " ".join(ad_body_lines).strip(),
    }


def extract_ad_records(
    body_text: str,
    country: str,
    lang: str,
    keyword: str,
    seen_ids: set[str]
) -> tuple[list[dict], int]:
    """Extrait les nouvelles annonces à partir des identifiants détectés."""
    matches = list(AD_ID_PATTERN.finditer(body_text))
    records: list[dict] = []
    duplicates = 0

    for index, match in enumerate(matches):
        ad_id = match.group(1).strip()
        if not ad_id:
            continue

        if ad_id in seen_ids:
            duplicates += 1
            continue

        block_start = max(0, match.start() - 30)
        block_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else min(len(body_text), match.start() + 2500)
        )
        ad_block = body_text[block_start:block_end]

        parsed = parse_ad_block(ad_block)
        parsed.update({
            "country": country,
            "lang": lang,
            "keyword": keyword,
            "ad_id": ad_id,
            "image_urls": "",
        })

        seen_ids.add(ad_id)
        records.append(parsed)

    return records, duplicates


# ---------------------------------------------------------------------
# 5. SAUVEGARDE ET CONFORMITÉ CSV
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

    # Filtrage de conformité pour le fichier nettoyé
    df_clean = df_raw[df_raw["ad_text"].str.len() >= 20].copy()

    blacklist_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(term) for term in PET_BLACKLIST) + r")\b",
        flags=re.IGNORECASE
    )

    df_clean = df_clean[~df_clean["ad_text"].str.contains(blacklist_pattern, na=False)]
    df_clean = df_clean[~df_clean["advertiser"].str.contains(blacklist_pattern, na=False)]

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


def load_existing_data(raw_file: Path) -> tuple[list[dict], set[str]]:
    """Recharge le CSV brut existant pour reprendre une collecte."""
    if not RESUME_FROM_EXISTING_CSV or not raw_file.exists():
        return [], set()

    try:
        existing = pd.read_csv(raw_file, dtype=str).fillna("")
        for col in RAW_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[RAW_COLUMNS]
        existing["ad_id"] = existing["ad_id"].astype(str).str.strip()
        existing = existing[existing["ad_id"] != ""]
        records = existing.to_dict(orient="records")
        seen = set(existing["ad_id"])
        print(f"Reprise : {len(seen)} identifiants déjà présents.")
        return records, seen
    except Exception as exc:
        print(f"Impossible de relire le CSV existant : {exc}")
        return [], set()


# ---------------------------------------------------------------------
# 6. INSPECTION ET DÉFILEMENT
# ---------------------------------------------------------------------

def inspect_page(page, screenshot_path: Path, save_screenshot: bool = False) -> str:
    """Récupère le texte de la page et signale les alertes éventuelles."""
    try:
        body_text = page.locator("body").inner_text(timeout=15000)
    except PlaywrightTimeoutError:
        body_text = ""

    if save_screenshot:
        try:
            page.screenshot(path=str(screenshot_path), full_page=False)
        except Exception:
            pass

    lower_text = body_text.casefold()
    warning_terms = [
        "log in to facebook",
        "connectez-vous à facebook",
        "vous devez vous connecter",
        "security check",
        "checkpoint",
        "captcha",
        "something went wrong",
        "une erreur s'est produite",
    ]

    if any(term in lower_text for term in warning_terms):
        print(
            "ATTENTION : Meta affiche un message de connexion, "
            "de contrôle de sécurité ou d'erreur."
        )

    return body_text


def scroll_results(page, max_scrolls: int) -> None:
    """Fait défiler la page pour charger davantage d'annonces."""
    unchanged_rounds = 0
    previous_height = -1

    for scroll_number in range(max_scrolls):
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(3000)
            current_height = page.evaluate("document.body.scrollHeight")
        except Exception as exc:
            print(f"Erreur pendant le défilement : {exc}")
            break

        print(
            f"Défilement {scroll_number + 1}/{max_scrolls} "
            f"- hauteur page : {current_height}"
        )

        if current_height == previous_height:
            unchanged_rounds += 1
        else:
            unchanged_rounds = 0

        if unchanged_rounds >= 2:
            break

        previous_height = current_height


# ---------------------------------------------------------------------
# 7. SCRAPING PRINCIPAL
# ---------------------------------------------------------------------

def scrape_control_ads(
    max_scrolls: int = MAX_SCROLLS,
    checkpoint_every: int = CHECKPOINT_EVERY,
    limit: Optional[int] = MAX_QUERIES,
    countries: Optional[list[str]] = COUNTRIES_TO_SCRAPE
) -> None:
    """Exécute la collecte complète du groupe témoin."""
    project_dir = Path(__file__).resolve().parent.parent
    raw_dir = project_dir / "Data" / "raw"
    processed_dir = project_dir / "Data" / "processed"
    debug_dir = project_dir / "Data" / "debug"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / "meta_ads_control_multilang_raw.csv"
    clean_file = processed_dir / "meta_ads_control_multilang_clean.csv"

    plan = build_control_search_plan(countries=countries, limit=limit)
    countries_label = "TOUS" if countries is None else str(countries)

    print("=" * 65)
    print(f"Nombre de requêtes prévues : {len(plan)}")
    print(f"Pays sélectionnés : {countries_label}")
    print(f"Mode navigateur visible : {not HEADLESS} (HEADLESS={HEADLESS})")
    print("=" * 65)

    if not plan:
        print("Aucune requête dans le plan. Vérifiez les pays et les langues.")
        return

    all_ads, seen_ids = load_existing_data(raw_file)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context(
            locale="fr-FR",
            viewport={"width": 1440, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )

        page = context.new_page()
        page.set_default_timeout(15000)

        try:
            for idx, (country, lang, keyword) in enumerate(plan, start=1):
                url = build_ad_library_url(country, keyword)

                print("\n" + "-" * 65)
                print(f"[{idx}/{len(plan)}] Pays={country} | Langue={lang} | Mot-clé={keyword}")

                try:
                    response = page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )
                    page.wait_for_timeout(7000)

                    for btn_text in [
                        "Autoriser tous les cookies",
                        "Allow all cookies",
                        "Refuser les cookies optionnels"
                    ]:
                        try:
                            btn = page.get_by_role("button", name=btn_text).first
                            if btn.is_visible(timeout=800):
                                btn.click()
                                page.wait_for_timeout(1200)
                                break
                        except Exception:
                            pass

                    if response is not None:
                        print("Statut HTTP :", response.status)

                except Exception as exc:
                    print(f"Erreur de navigation : {repr(exc)}")
                    continue

                if idx == 1 and PAUSE_AFTER_FIRST_PAGE and not HEADLESS:
                    input("Appuyez sur Entrée pour continuer...")

                screenshot_path = debug_dir / f"meta_debug_{idx}.png"
                body_text = inspect_page(page, screenshot_path, save_screenshot=(idx <= 3))

                if not body_text.strip():
                    print("Page vide : aucune extraction pour cette requête.")
                    continue

                scroll_results(page, max_scrolls=max_scrolls)

                try:
                    body_text = page.locator("body").inner_text(timeout=15000)
                except Exception as exc:
                    print(f"Impossible de relire le texte : {exc}")
                    continue

                matches = list(AD_ID_PATTERN.finditer(body_text))
                print("Identifiants publicitaires détectés :", len(matches))

                if not matches:
                    print("Aucun identifiant reconnu pour cette requête.")
                    continue

                new_records, duplicate_count = extract_ad_records(
                    body_text=body_text,
                    country=country,
                    lang=lang,
                    keyword=keyword,
                    seen_ids=seen_ids
                )

                all_ads.extend(new_records)
                print("Nouvelles annonces extraites :", len(new_records))
                print("Identifiants déjà connus :", duplicate_count)

                if idx % max(1, checkpoint_every) == 0:
                    save_control_outputs(all_ads, raw_file, clean_file)

                time.sleep(2.5)

        finally:
            if all_ads:
                save_control_outputs(all_ads, raw_file, clean_file)
            context.close()
            browser.close()

    print("\nCollecte terminée.")
    print(f"Total d'identifiants uniques : {len(seen_ids)}")
    print(f"Fichier brut : {raw_file}")
    print(f"Fichier nettoyé : {clean_file}")


if __name__ == "__main__":
    scrape_control_ads()