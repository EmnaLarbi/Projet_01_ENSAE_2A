import re
import time
import urllib.parse
import pandas as pd
from pathlib import Path
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

# Mots-clés centrés uniquement sur la viande (bœuf, viande rouge, pièces) et le barbecue/grill
KEYWORDS_BY_LANG = {
    "fr": [
        "viande rouge", "viande de bœuf", "viande", "bœuf", "boeuf",
        "steak", "entrecôte", "côte de bœuf", "rôti",
        "barbecue", "bbq", "grill"
    ],
    "en": [
        "red meat", "beef", "steak", "meat",
        "sirloin", "ribeye", "brisket", "ground beef",
        "barbecue", "bbq", "grill"
    ],
    "de": [
        "rotes Fleisch", "Rindfleisch", "Steak", "Fleisch",
        "Rindersteak", "Braten", "Hackfleisch",
        "Grill", "grillen", "Barbecue", "BBQ"
    ],
    "it": [
        "carne rossa", "carne di manzo", "bistecca", "carne",
        "manzo", "costata", "fiorentina", "arrosto",
        "griglia", "grigliata", "barbecue", "bbq"
    ],
    "es": [
        "carne roja", "carne de res", "filete", "carne",
        "ternera", "chuletón", "solomillo", "asado",
        "barbacoa", "parrilla", "bbq", "grill"
    ],
    "fi": [
        "punainen liha", "naudanliha", "pihvi", "liha",
        "nauta", "paisti", "jauheliha",
        "grilli", "grillaus", "bbq", "barbecue"
    ],
    "hr": [
        "crveno meso", "govedina", "odrezak", "meso",
        "junetina", "biftek", "pečenje",
        "roštilj", "bbq", "barbecue"
    ],
    "hu": [
        "vörös hús", "marhahús", "steak", "hús",
        "marha", "bélszín", "sült",
        "grill", "grillezes", "bbq", "barbecue"
    ],
    "is": [
        "rautt kjöt", "nautakjöt", "steik", "kjöt",
        "nautakjötssteik", "steiktur",
        "grill", "bbq", "barbecue"
    ],
    "lt": [
        "raudona mėsa", "jautiena", "kepsnys", "mėsa",
        "jautienos kepsnys", "kepta mėsa",
        "grilis", "kepsninė", "bbq", "barbecue"
    ],
    "no": [
        "rødt kjøtt", "storfekjøtt", "biff", "kjøtt",
        "entrecôte", "stek", "kjøttdeig",
        "grill", "grilling", "bbq", "barbecue"
    ],
    "se": [
        "rött kött", "nötkött", "biff", "kött",
        "entrecôte", "stek", "köttfärs",
        "grill", "grilla", "bbq", "barbecue"
    ],
    "si": [
        "rdeče meso", "govedina", "zrezek", "meso",
        "pljučni file", "pečenka",
        "žar", "bbq", "barbecue"
    ],
    "sk": [
        "červené mäso", "hovädzie mäso", "steak", "mäso",
        "roštenka", "pečienka", "mleté mäso",
        "gril", "grilovanie", "bbq", "barbecue"
    ],
}

# Langues locales uniquement (sans "en" systématique)
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


def _build_relevance_regexes() -> dict[str, re.Pattern]:
    """Compile un motif par langue exigeant au moins un mot viande/BBQ dans le texte."""
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
    """Vérifie que le texte ou l'annonceur contient un terme viande/BBQ."""
    combined = f"{advertiser or ''} {text or ''}".strip()
    if not combined:
        return False
    rx = RELEVANCE_REGEXES.get(str(lang or "").lower().strip(), RELEVANCE_REGEXES["_fallback"])
    if rx.search(combined):
        return True
    return bool(RELEVANCE_REGEXES["_fallback"].search(combined))


def build_search_plan() -> list[tuple[str, str, str]]:
    """Retourne une liste de tuples (country, lang, keyword)."""
    plan = []
    for country, langs in TARGET_COUNTRIES_LANGS.items():
        for lang in langs:
            for kw in KEYWORDS_BY_LANG.get(lang, []):
                plan.append((country, lang, kw))
    return plan


def parse_ad_block(block_text: str) -> dict:
    """Découpe proprement le texte brut d'une carte d'annonce Meta."""
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
    """Vérifie qu'une annonce est valide, non hors-sujet et contient un terme viande/BBQ."""
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


def save_clean_output(clean_ads: list[dict], clean_file: Path):
    """Nettoie, filtre les hors-sujet, dédoublonne et sauvegarde meta_ads_meat_bbq_clean.csv."""
    if not clean_ads:
        return

    cols = [
        "country", "lang", "keyword", "ad_id", "status",
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

    # Reprise depuis le fichier nettoyé existant en cas d'arrêt ou d'erreur précédente
    if clean_file.exists():
        try:
            existing = pd.read_csv(clean_file, dtype={"ad_id": str}, encoding="utf-8-sig")
            existing_valid = existing[existing.apply(lambda r: is_valid_clean_ad(r.to_dict()), axis=1)]
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
                        pass

                last_height = 0
                for _ in range(max_scrolls):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(2000)
                    new_height = page.evaluate("document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height

                body_text = page.inner_text("body")
                pattern = re.compile(r"(?:ID dans la bibliothèque|Library ID)\s*:\s*(\d{10,20})")
                matches = list(pattern.finditer(body_text))

                new_count = 0
                for m_idx, match in enumerate(matches):
                    ad_id = match.group(1)
                    if ad_id in seen_ids:
                        continue
                    seen_ids.add(ad_id)

                    start_pos = max(0, match.start() - 30)
                    end_pos = matches[m_idx + 1].start() if m_idx + 1 < len(matches) else match.start() + 2500
                    ad_block = body_text[start_pos:end_pos]

                    parsed = parse_ad_block(ad_block)
                    parsed["ad_id"] = ad_id
                    parsed["country"] = country
                    parsed["lang"] = lang
                    parsed["keyword"] = kw

                    if not is_valid_clean_ad(parsed):
                        continue

                    img_urls = []
                    try:
                        card_container = page.locator(f"xpath=//*[contains(text(), '{ad_id}')]/ancestor::div[5]").first
                        for img in card_container.locator("img").all():
                            src = img.get_attribute("src") or ""
                            if "scontent" in src:
                                w_raw = img.get_attribute("width")
                                try:
                                    if not w_raw or int(w_raw) > 80:
                                        img_urls.append(src)
                                except ValueError:
                                    img_urls.append(src)
                    except Exception:
                        pass

                    parsed["image_urls"] = "; ".join(dict.fromkeys(img_urls))
                    clean_ads.append(parsed)
                    new_count += 1

                print(f" -> {new_count} nouvelles annonces propres retenues.")

                if idx % checkpoint_every == 0:
                    save_clean_output(clean_ads, clean_file)
                    print(f" [Checkpoint] Sauvegarde dans {clean_file} ({len(clean_ads)} annonces propres).")

                time.sleep(1.5)

            browser.close()

    except Exception as e:
        print(f"\nInterruption ou erreur inattendue : {e}")
        save_clean_output(clean_ads, clean_file)
        print(f"Sauvegarde de sécurité effectuée dans {clean_file}.")
        raise

    if not clean_ads:
        print("\nAucune annonce propre extraite.")
        return

    save_clean_output(clean_ads, clean_file)
    print(f"\n[Propre] {clean_file} ({len(clean_ads)} annonces propres enregistrées au total).")


if __name__ == "__main__":
    scrape_meta_ads(max_scrolls=8, checkpoint_every=10)