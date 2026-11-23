import re
import time
import urllib.parse
from pathlib import Path
import pandas as pd
from playwright.sync_api import sync_playwright

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

# Liste noire / motifs d'exclusion : mentions "sans viande" ou alternatives végétales
MEAT_EXCLUSION_PATTERNS = [
    # Français
    r"\bsans\s+viande\b", r"\bsans\s+b[œoe]uf\b", r"\balternative\s+(?:à\s+la\s+viande|végétale|au\s+b[œoe]uf)\b",
    r"\b100\s*%\s*v[ée]g[ée]tal\b", r"\bv[ée]g[ée]tarien\w*\b", r"\bv[ée]gan\w*\b", r"\bplant[- ]based\b",
    # Anglais
    r"\bmeat[- ]?free\b", r"\bbeef[- ]?free\b", r"\bno\s+meat\b", r"\bplant[- ]based\b",
    r"\bmeat\s+alternative\w*\b", r"\bvegan\w*\b", r"\bvegetarian\w*\b",
    # Allemand
    r"\bfleischfrei\b", r"\bohne\s+fleisch\b", r"\bfleischalternative\w*\b",
    r"\bpflanzlich\w*\b", r"\bvegan\w*\b", r"\bvegetarisch\w*\b",
    # Italien
    r"\bsenza\s+carne\b", r"\balternativa\s+alla\s+carne\b", r"\bvegetal\w*\b",
    r"\bvegano?\b", r"\bvegetariano?\b", r"\ba\s+base\s+vegetale\b",
    # Espagnol
    r"\bsin\s+carne\b", r"\balternativa\s+a\s+la\s+carne\b", r"\ba\s+base\s+de\s+plantas\b",
    r"\bvegano?\b", r"\bvegetariano?\b",
    # Finnois
    r"\blihatek\w*\b", r"\bkasvipohj\w*\b", r"\bvegaan\w*\b", r"\bkasvis\w*\b",
    # Croate
    r"\bbez\s+mesa\b", r"\bbiljn\w*\b", r"\bvegan\w*\b", r"\bvegetarijan\w*\b",
    # Hongrois
    r"\bhúsmentes\b", r"\bnövényi\s+alapú\b", r"\bvegán\b", r"\bvegetáriánus\b",
    # Islandais
    r"\bkjötlaus\b", r"\bplöntubund\w*\b", r"\bvegan\w*\b", r"\bgrænmetis\w*\b",
    # Lituanien
    r"\bbe\s+mėsos\b", r"\baugalin\w*\b", r"\bvegan\w*\b", r"\bvegetar\w*\b",
    # Norvégien & Suédois
    r"\bkjøttfri\w*\b", r"\bköttfri\w*\b", r"\bväxtbaserad\b", r"\bplantebasert\b",
    r"\bvegan\w*\b", r"\bvegetarisk\w*\b",
    # Slovène & Slovaque
    r"\bbrez\s+mesa\b", r"\bbez\s+mäsa\b", r"\brastlinn\w*\b", r"\bvegan\w*\b", r"\bvegetarián\w*\b",
]

# Motifs hors sujet évidents (entretien/linge, non-alimentaire, édulcorants/confiserie)
OFF_TOPIC_PATTERNS = [
    r"\blavage\b", r"\blave[- ]linge\b", r"\bnettoyant\b", r"\bdétachant\b", r"\badoucissant\b",
    r"\bwaschmittel\b", r"\bweichspüler\b", r"\breiniger\b", r"\bhaushalt\w*\b",
    r"\bdetersivo\b", r"\bammorbidente\b", r"\bdetergente\b",
    r"\bne\s+maisto\s+prekių\b", r"\bnon[- ]food\b",
    r"\bérythritol\b", r"\beritritol\b", r"\bédulcorant\b", r"\bsüßstoff\b",
]

RED_MEAT_KEYWORDS = {
    "fr": ["viande rouge", "viande de bœuf", "bœuf", "boeuf", "steak", "entrecôte", "côte de bœuf", "rôti"],
    "en": ["red meat", "beef", "steak", "sirloin", "ribeye", "brisket", "ground beef"],
    "de": ["rotes Fleisch", "Rindfleisch", "Steak", "Rindersteak", "Braten", "Hackfleisch"],
    "it": ["carne rossa", "carne di manzo", "bistecca", "manzo", "costata", "fiorentina", "arrosto"],
    "es": ["carne roja", "carne de res", "filete", "ternera", "chuletón", "solomillo", "asado"],
    "fi": ["punainen liha", "naudanliha", "pihvi", "nauta", "paisti", "jauheliha"],
    "hr": ["crveno meso", "govedina", "odrezak", "junetina", "biftek", "pečenje"],
    "hu": ["vörös hús", "marhahús", "steak", "marha", "bélszín", "sült"],
    "is": ["rautt kjöt", "nautakjöt", "steik", "nautakjötssteik", "steiktur"],
    "lt": ["raudona mėsa", "jautiena", "kepsnys", "jautienos kepsnys", "kepta mėsa"],
    "no": ["rødt kjøtt", "storfekjøtt", "biff", "entrecôte", "stek", "kjøttdeig"],
    "se": ["rött kött", "nötkött", "biff", "entrecôte", "stek", "köttfärs"],
    "si": ["rdeče meso", "govedina", "zrezek", "pljučni file", "pečenka"],
    "sk": ["červené mäso", "hovädzie mäso", "steak", "roštenka", "pečienka", "mleté mäso"],
}

WHITE_MEAT_KEYWORDS = {
    "fr": ["viande blanche", "poulet", "volaille", "dinde", "porc", "veau", "escalope", "filet de poulet"],
    "en": ["white meat", "chicken", "poultry", "turkey", "pork", "veal", "chicken breast"],
    "de": ["weißes Fleisch", "Hähnchen", "Hühnchen", "Geflügel", "Pute", "Schweinefleisch", "Kalbfleisch", "Schnitzel"],
    "it": ["carne bianca", "pollo", "pollame", "tacchino", "maiale", "vitello", "petto di pollo"],
    "es": ["carne blanca", "pollo", "ave", "pavo", "cerdo", "pechuga de pollo"],
    "fi": ["vaalea liha", "kana", "broileri", "siipikarja", "kalkkuna", "sianliha", "vasikanliha"],
    "hr": ["bijelo meso", "piletina", "perad", "puretina", "svinjetina", "teletina"],
    "hu": ["fehér hús", "csirke", "csirkehús", "baromfi", "pulyka", "sertéshús", "borjúhús"],
    "is": ["hvítt kjöt", "kjúklingur", "alifuglakjöt", "kalkúnn", "svínakjöt", "kálfakjöt"],
    "lt": ["balta mėsa", "vištiena", "paukštiena", "kalakutiena", "kiauliena", "veršiena"],
    "no": ["hvitt kjøtt", "kylling", "fjærfe", "kalkun", "svinekjøtt", "kalvekjøtt"],
    "se": ["vitt kött", "kyckling", "fågel", "kalkon", "fläskkött", "kalvkött"],
    "si": ["belo meso", "piščanec", "perutnina", "puran", "svinjina", "teletina"],
    "sk": ["biele mäso", "kuracie mäso", "kura", "hydina", "morčacie mäso", "bravčové mäso", "teľacie mäso"],
}

KEYWORDS_BY_LANG = {
    lang: sorted(set(RED_MEAT_KEYWORDS.get(lang, [])) | set(WHITE_MEAT_KEYWORDS.get(lang, [])))
    for lang in set(RED_MEAT_KEYWORDS) | set(WHITE_MEAT_KEYWORDS)
}

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

PET_PATTERN = re.compile(r"\b(?:" + "|".join(map(re.escape, PET_BLACKLIST)) + r")\b", flags=re.IGNORECASE)
EXCLUSION_PATTERN = re.compile("|".join(MEAT_EXCLUSION_PATTERNS), flags=re.IGNORECASE)
OFF_TOPIC_PATTERN = re.compile("|".join(OFF_TOPIC_PATTERNS), flags=re.IGNORECASE)

UNIVERSAL_RELEVANCE_TERMS = ["bbq", "barbecue", "grill", "steak", "bacon"]
RED_MEAT_PATTERN = re.compile(
    r"\b(?:" + "|".join(map(re.escape, sorted(set(RED_MEAT_KEYWORDS["fr"] + [t for v in RED_MEAT_KEYWORDS.values() for t in v]), key=len, reverse=True))) + r")\b",
    flags=re.IGNORECASE,
)
WHITE_MEAT_PATTERN = re.compile(
    r"\b(?:" + "|".join(map(re.escape, sorted(set([t for v in WHITE_MEAT_KEYWORDS.values() for t in v]), key=len, reverse=True))) + r")\b",
    flags=re.IGNORECASE,
)


def _build_relevance_regexes() -> dict[str, re.Pattern]:
    compiled: dict[str, re.Pattern] = {}
    all_terms = set(UNIVERSAL_RELEVANCE_TERMS)
    for pats in KEYWORDS_BY_LANG.values():
        all_terms.update(pats)

    for lang, pats in KEYWORDS_BY_LANG.items():
        lang_terms = sorted(set(pats) | set(UNIVERSAL_RELEVANCE_TERMS), key=len, reverse=True)
        compiled[lang] = re.compile(r"\b(?:" + "|".join(map(re.escape, lang_terms)) + r")\b", flags=re.IGNORECASE)

    fallback_terms = sorted(all_terms, key=len, reverse=True)
    compiled["_fallback"] = re.compile(r"\b(?:" + "|".join(map(re.escape, fallback_terms)) + r")\b", flags=re.IGNORECASE)
    return compiled


RELEVANCE_REGEXES = _build_relevance_regexes()


def has_meat_bbq_relevance(text: str, lang: str = "", advertiser: str = "") -> bool:
    combined = f"{advertiser or ''} {text or ''}".strip()
    if not combined:
        return False
    rx = RELEVANCE_REGEXES.get(str(lang or "").lower().strip(), RELEVANCE_REGEXES["_fallback"])
    return bool(rx.search(combined) or RELEVANCE_REGEXES["_fallback"].search(combined))


def build_search_plan() -> list[tuple[str, str, str]]:
    plan = []
    for country, langs in TARGET_COUNTRIES_LANGS.items():
        for lang in langs:
            for kw in KEYWORDS_BY_LANG.get(lang, []):
                plan.append((country, lang, kw))
    return plan


def parse_ad_block(block_text: str) -> dict:
    lines = [line.strip() for line in block_text.split("\n") if line.strip()]
    status = ""
    ad_id = ""
    start_date = ""
    advertiser = ""
    ad_body_lines = []

    for i, line in enumerate(lines):
        if "ID dans la bibliothèque" in line or "Library ID" in line:
            ad_id = line.split(":")[-1].strip()
            if i > 0 and lines[i - 1] in ("Actif", "Inactif", "Active", "Inactive"):
                status = lines[i - 1]
        elif "Début de la diffusion" in line or "Started running on" in line:
            start_date = (
                line.split("·")[0]
                .replace("Début de la diffusion le", "")
                .replace("Started running on", "")
                .strip()
            )
        elif line in ("Sponsorisé", "Sponsored") and i > 0:
            advertiser = lines[i - 1]
            for next_line in lines[i + 1:]:
                if (
                    next_line in ("Actif", "Inactif", "Active", "Inactive")
                    or "ID dans la bibliothèque" in next_line
                    or "Library ID" in next_line
                ):
                    break
                ad_body_lines.append(next_line)
            break

    return {
        "ad_id": ad_id,
        "status": status,
        "start_date": start_date,
        "advertiser": advertiser,
        "ad_text": " ".join(ad_body_lines),
    }


def is_valid_clean_ad(ad: dict) -> bool:
    text = str(ad.get("ad_text") or "").strip()
    advertiser = str(ad.get("advertiser") or "").strip()
    lang = str(ad.get("lang") or "").strip()

    if len(text) < 20 or not advertiser:
        return False
    if PET_PATTERN.search(text) or PET_PATTERN.search(advertiser):
        return False
    if EXCLUSION_PATTERN.search(text) or EXCLUSION_PATTERN.search(advertiser):
        return False
    if OFF_TOPIC_PATTERN.search(text) or OFF_TOPIC_PATTERN.search(advertiser):
        return False
    if not has_meat_bbq_relevance(text, lang, advertiser):
        return False

    return True


def detect_meat_type(text: str, advertiser: str = "", keyword: str = "") -> str:
    combined = f"{advertiser or ''} {keyword or ''} {text or ''}".strip()
    has_red = bool(RED_MEAT_PATTERN.search(combined))
    has_white = bool(WHITE_MEAT_PATTERN.search(combined))
    if has_red and has_white:
        return "both"
    if has_red:
        return "red_meat"
    if has_white:
        return "white_meat"
    return "general_meat_bbq"


def save_clean_output(clean_ads: list[dict], clean_file: Path):
    if not clean_ads:
        return

    cols = [
        "country", "lang", "keyword", "meat_type", "ad_id", "status",
        "start_date", "advertiser", "ad_text", "image_urls", "n_variants"
    ]
    df_clean = pd.DataFrame(clean_ads)
    for c in cols:
        if c not in df_clean.columns:
            df_clean[c] = ""

    df_clean["ad_id"] = df_clean["ad_id"].astype(str)
    df_clean = df_clean.drop_duplicates(subset=["ad_id"])
    df_clean = df_clean[df_clean.apply(lambda r: is_valid_clean_ad(r.to_dict()), axis=1)].copy()

    if len(df_clean) == 0:
        return

    df_clean["meat_type"] = df_clean.apply(
        lambda r: detect_meat_type(r.get("ad_text", ""), r.get("advertiser", ""), r.get("keyword", "")),
        axis=1,
    )
    df_clean["n_variants"] = df_clean.groupby(["advertiser", "ad_text"])["ad_id"].transform("count")
    df_clean = df_clean[cols]
    df_clean.to_csv(clean_file, index=False, encoding="utf-8-sig")


def scrape_meta_ads(max_scrolls: int = 8, checkpoint_every: int = 10):
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "Data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    clean_file = processed_dir / "meta_ads_meat_bbq_clean.csv"

    plan = build_search_plan()
    print(f"Plan viande & barbecue : {len(plan)} requêtes sur {len(TARGET_COUNTRIES_LANGS)} pays.")

    clean_ads = []
    seen_ids = set()

    if clean_file.exists():
        try:
            existing = pd.read_csv(clean_file, dtype={"ad_id": str}, encoding="utf-8-sig")
            existing_valid = existing[existing.apply(lambda r: is_valid_clean_ad(r.to_dict()), axis=1)].copy()
            existing_valid["meat_type"] = existing_valid.apply(
                lambda r: detect_meat_type(r.get("ad_text", ""), r.get("advertiser", ""), r.get("keyword", "")),
                axis=1,
            )
            clean_ads = existing_valid.to_dict(orient="records")
            seen_ids = set(existing_valid["ad_id"].dropna().astype(str))
            print(
                f"Reprise depuis {clean_file} ({len(seen_ids)} IDs valides conservés "
                f"sur {len(existing)} lignes existantes)."
            )
        except Exception as e:
            print(f"Avertissement reprise depuis {clean_file} : {e}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = browser.new_context(
                locale="fr-FR",
                viewport={"width": 1440, "height": 900},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            for idx, (country, lang, kw) in enumerate(plan, start=1):
                encoded_kw = urllib.parse.quote(kw)
                url = (
                    f"https://www.facebook.com/ads/library/"
                    f"?active_status=all&ad_type=all&country={country}&q={encoded_kw}"
                )
                print(f"\n[{idx}/{len(plan)}] Pays: {country} ({lang}) | Mot-clé: '{kw}'")
                try:
                    page.goto(url, wait_until="networkidle", timeout=35000)
                except Exception:
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=35000)
                    except Exception as e:
                        print(f" -> Erreur chargement page : {e}")
                        continue

                page.wait_for_timeout(2500)

                for btn_text in ["Autoriser tous les cookies", "Allow all cookies", "Refuser les cookies optionnels"]:
                    try:
                        btn = page.get_by_role("button", name=btn_text).first
                        if btn.is_visible(timeout=800):
                            btn.click()
                            page.wait_for_timeout(1200)
                            break
                    except Exception: