import re
import urllib.parse
from pathlib import Path
import pandas as pd
from playwright.sync_api import sync_playwright

# =============================================================================
# SCRAPING DE LA BIBLIOTHÈQUE PUBLICITAIRE META — viande rouge.
# Les noms et signatures des fonctions sont inchangés : la boucle de collecte
# `scrape_meta_ads` fonctionne sans modification.

# Viande ROUGE = bœuf et agneau uniquement ; 
# =============================================================================

# -----------------------------------------------------------------------------
# 1. MOTS-CLÉS DE VIANDE PAR LANGUE ET PAR TYPE
#    Codes de langue internes inchangés (« se » = suédois, « si » = slovène, « no » = norvégien).
# -----------------------------------------------------------------------------

MEAT_KEYWORDS = {
    "fr": {"boeuf": ["bœuf", "boeuf", "viande bovine", "entrecôte", "côte de bœuf", "bavette", "steak haché"],
           "agneau": ["agneau", "gigot", "mouton"],
           },
    "en": {"boeuf": ["beef", "steak", "sirloin", "ribeye", "brisket"],
           "agneau": ["lamb", "mutton"],
           },
    "de": {"boeuf": ["Rindfleisch", "Rind", "Rindersteak", "Steak"],
           "agneau": ["Lammfleisch", "Lamm"],
           },
    "it": {"boeuf": ["manzo", "carne di manzo", "carne bovina", "bistecca", "fiorentina"],
           "agneau": ["agnello", "carne ovina"],
           },
    "es": {"boeuf": ["carne de vacuno", "vacuno", "ternera", "buey", "chuletón"],  # « ternera » = bœuf en Espagne
           "agneau": ["cordero"],
           },
    "fi": {"boeuf": ["naudanliha", "nauta"],
           "agneau": ["lammas", "lampaanliha"],
           },
    "hr": {"boeuf": ["govedina", "junetina"],
           "agneau": ["janjetina", "ovčetina"],
           },
    "hu": {"boeuf": ["marhahús", "marha"],
           "agneau": ["bárányhús", "bárány"],
           },
    "is": {"boeuf": ["nautakjöt"],
           "agneau": ["lambakjöt", "lambalæri"],
           },
    "lt": {"boeuf": ["jautiena"],
           "agneau": ["ėriena", "aviena"],
           },
    "no": {"boeuf": ["storfekjøtt", "oksekjøtt", "biff"],
           "agneau": ["lammekjøtt", "fårikål"],
           },
    "se": {"boeuf": ["nötkött", "oxfilé"],
           "agneau": ["lammkött", "lamm"],
           },
    "si": {"boeuf": ["govedina", "goveje meso"],
           "agneau": ["jagnjetina"],
           },
    "sk": {"boeuf": ["hovädzie mäso", "hovädzie"],
           "agneau": ["jahňacie mäso", "jahňacie"],
           },
}

# Point 1 : seuls le bœuf et l'agneau sont des viandes rouges
MEAT_CATEGORY = {"boeuf": "red", "agneau": "red"}

# Rétrocompatibilité : mêmes noms qu'avant
RED_MEAT_KEYWORDS = {l: [t for d, ts in v.items() if MEAT_CATEGORY[d] == "red" for t in ts]
                     for l, v in MEAT_KEYWORDS.items()}
KEYWORDS_BY_LANG = {l: sorted(set(RED_MEAT_KEYWORDS[l])) for l in MEAT_KEYWORDS}

TARGET_COUNTRIES_LANGS = {
    "AT": ["de"], "CH": ["de", "fr", "it"], "DE": ["de"], "ES": ["es"], "FI": ["fi"],
    "FR": ["fr"], "HR": ["hr"], "HU": ["hu"], "IS": ["is"], "IT": ["it"], "LT": ["lt"],
    "NO": ["no"], "SE": ["se"], "SI": ["si"], "SK": ["sk"],
}

UNIVERSAL_RELEVANCE_TERMS = ["bbq", "barbecue", "grill"]
# Langues à déclinaisons : la fin des mots change selon la fonction grammaticale
INFLECTED_LANGS = {"fi", "hu", "lt", "hr", "si", "sk", "is"}

# -----------------------------------------------------------------------------
# 2. FILTRES D'EXCLUSION
# -----------------------------------------------------------------------------

# Point 3 : animaux de compagnie, PAR LANGUE
PET_BY_LANG = {
    "fr": ["chien", "chiens", "chiot", "chiots", "chat", "chats", "chaton", "chatons", "croquette",
           "croquettes", "animal de compagnie", "animaux de compagnie", "vétérinaire"],
    "en": ["dog", "dogs", "puppy", "puppies", "cat", "cats", "kitten", "kittens", "pet food",
           "dog food", "cat food", "kibble", "veterinary"],
    "de": ["hund", "hunde", "welpe", "welpen", "katze", "katzen", "kätzchen", "hundefutter",
           "katzenfutter", "tierfutter", "haustier", "haustiere"],
    "it": ["cane", "cani", "cucciolo", "cuccioli", "gatto", "gatti", "gattino", "gattini",
           "crocchette", "cibo per cani", "cibo per gatti", "animali domestici"],
    "es": ["perro", "perros", "cachorro", "cachorros", "gato", "gatos", "gatito", "gatitos",
           "croquetas", "comida para perros", "comida para gatos", "mascotas"],
    "fi": ["koira", "koirat", "pentu", "pennut", "kissa", "kissat", "kissanpentu", "koiranruoka",
           "kissaruoka", "lemmikinruoka", "lemmikki", "lemmikit"],
    "hr": ["pas", "psi", "štene", "mačka", "mačke", "mačić", "hrana za pse", "hrana za mačke",
           "kućni ljubimci"],
    "hu": ["kutya", "kutyák", "kölyökkutya", "macska", "macskák", "kiscica", "kutyatáp",
           "macskatáp", "állateledel", "háziállat"],
    "is": ["hundur", "hundar", "hvolpur", "köttur", "kettir", "kettlingur", "hundamatur",
           "kattamatur", "gæludýr", "gæludýrafóður"],
    "lt": ["šuo", "šunys", "šuniukas", "katė", "katės", "kačiukas", "šunų maistas",
           "kačių maistas", "naminių gyvūnų"],
    "no": ["hund", "hunder", "valp", "valper", "katt", "katter", "kattunge", "hundemat",
           "kattemat", "kjæledyr", "dyrefôr"],
    "se": ["hund", "hundar", "valp", "valpar", "katt", "katter", "kattunge", "hundmat",
           "kattmat", "husdjur", "djurfoder"],
    "si": ["pes", "psi", "kuža", "mačka", "mačke", "mucek", "hrana za pse", "hrana za mačke",
           "hišni ljubimci"],
    "sk": ["pes", "psy", "šteňa", "mačka", "mačky", "mača", "krmivo pre psov",
           "krmivo pre mačky", "domáce zvieratá"],
}

# Mentions « sans viande » / alternatives végétales (inchangées)
MEAT_EXCLUSION_PATTERNS = [
    r"\bsans\s+viande\b", r"\bsans\s+b[œoe]uf\b", r"\balternative\s+(?:à\s+la\s+viande|végétale|au\s+b[œoe]uf)\b",
    r"\b100\s*%\s*v[ée]g[ée]tal\b", r"\bv[ée]g[ée]tarien\w*\b", r"\bv[ée]gan\w*\b", r"\bplant[- ]based\b",
    r"\bmeat[- ]?free\b", r"\bbeef[- ]?free\b", r"\bno\s+meat\b", r"\bplant[- ]based\b",
    r"\bmeat\s+alternative\w*\b", r"\bvegan\w*\b", r"\bvegetarian\w*\b",
    r"\bfleischfrei\b", r"\bohne\s+fleisch\b", r"\bfleischalternative\w*\b",
    r"\bpflanzlich\w*\b", r"\bvegan\w*\b", r"\bvegetarisch\w*\b",
    r"\bsenza\s+carne\b", r"\balternativa\s+alla\s+carne\b", r"\bvegetal\w*\b",
    r"\bvegano?\b", r"\bvegetariano?\b", r"\ba\s+base\s+vegetale\b",
    r"\bsin\s+carne\b", r"\balternativa\s+a\s+la\s+carne\b", r"\ba\s+base\s+de\s+plantas\b",
    r"\bvegano?\b", r"\bvegetariano?\b",
    r"\blihatek\w*\b", r"\bkasvipohj\w*\b", r"\bvegaan\w*\b", r"\bkasvis\w*\b",
    r"\bbez\s+mesa\b", r"\bbiljn\w*\b", r"\bvegan\w*\b", r"\bvegetarijan\w*\b",
    r"\bhúsmentes\b", r"\bnövényi\s+alapú\b", r"\bvegán\b", r"\bvegetáriánus\b",
    r"\bkjötlaus\b", r"\bplöntubund\w*\b", r"\bvegan\w*\b", r"\bgrænmetis\w*\b",
    r"\bbe\s+mėsos\b", r"\baugalin\w*\b", r"\bvegan\w*\b", r"\bvegetar\w*\b",
    r"\bkjøttfri\w*\b", r"\bköttfri\w*\b", r"\bväxtbaserad\b", r"\bplantebasert\b",
    r"\bvegan\w*\b", r"\bvegetarisk\w*\b",
    r"\bbrez\s+mesa\b", r"\bbez\s+mäsa\b", r"\brastlinn\w*\b", r"\bvegan\w*\b", r"\bvegetarián\w*\b",
]

# Hors sujet (inchangé)
OFF_TOPIC_PATTERNS = [
    r"\blavage\b", r"\blave[- ]linge\b", r"\bnettoyant\b", r"\bdétachant\b", r"\badoucissant\b",
    r"\bwaschmittel\b", r"\bweichspüler\b", r"\breiniger\b", r"\bhaushalt\w*\b",
    r"\bdetersivo\b", r"\bammorbidente\b", r"\bdetergente\b",
    r"\bne\s+maisto\s+prekių\b", r"\bnon[- ]food\b",
    r"\bérythritol\b", r"\beritritol\b", r"\bédulcorant\b", r"\bsüßstoff\b",
]

EXCLUSION_PATTERN = re.compile("|".join(MEAT_EXCLUSION_PATTERNS), flags=re.IGNORECASE)
OFF_TOPIC_PATTERN = re.compile("|".join(OFF_TOPIC_PATTERNS), flags=re.IGNORECASE)

# -----------------------------------------------------------------------------
# 3. MOTIFS COMPILÉS
# -----------------------------------------------------------------------------


def _stem(term: str) -> str:
    """Langues à déclinaisons : retire la voyelle finale du dernier mot
    (govedina → govedin, qui reconnaît govedinom, govedine…)."""
    words = term.split()
    last = words[-1]
    if len(last) > 5 and last[-1].lower() in "aeiouyäöüáéíóúýåæø":
        words[-1] = last[:-1]
    return " ".join(words)


def _compile(terms: list[str], inflected: bool = False) -> re.Pattern | None:
    if inflected:
        terms = [_stem(t) for t in terms]
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    if not terms:
        return None
    end = r"\w*" if inflected else r"\b"
    return re.compile(r"\b(?:" + "|".join(map(re.escape, terms)) + r")" + end, flags=re.IGNORECASE)


MEAT_PATTERNS = {   # langue → type de viande → motif
    lang: {meat: _compile(terms, lang in INFLECTED_LANGS) for meat, terms in types.items() if terms}
    for lang, types in MEAT_KEYWORDS.items()
}
PET_PATTERNS = {lang: _compile(terms) for lang, terms in PET_BY_LANG.items()}
UNIVERSAL_PATTERN = _compile(UNIVERSAL_RELEVANCE_TERMS)


def _ad_langs(lang: str) -> list[str]:
    """Langues dans lesquelles on lit une publicité : celle de la requête + l'anglais,
    très présent dans les publicités. Jamais toutes les langues à la fois (point 5)."""
    lang = str(lang or "").lower().strip()
    return sorted({lang, "en"} & set(MEAT_KEYWORDS)) if lang in MEAT_KEYWORDS else sorted(MEAT_KEYWORDS)


def detect_meats(text: str, advertiser: str = "", lang: str = "") -> list[str]:
    """Types de viande présents (boeuf, agneau)."""
    combined = f"{advertiser or ''} {text or ''}"
    found = {meat for l in _ad_langs(lang) for meat, rx in MEAT_PATTERNS.get(l, {}).items()
             if rx.search(combined)}
    return sorted(found)


# -----------------------------------------------------------------------------
# 4. FONCTIONS (mêmes noms et signatures qu'avant)
# -----------------------------------------------------------------------------


def has_meat_bbq_relevance(text: str, lang: str = "", advertiser: str = "") -> bool:
    """Point 5 : pertinente si elle contient un mot de viande DANS SA LANGUE (ou en anglais),
    ou un terme universel (bbq, barbecue, grill)."""
    combined = f"{advertiser or ''} {text or ''}".strip()
    if not combined:
        return False
    return bool(detect_meats(text, advertiser, lang) or UNIVERSAL_PATTERN.search(combined))


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
    combined = f"{advertiser} {text}"

    if len(text) < 20 or not advertiser:
        return False
    # Point 3 : animaux de compagnie dans la langue de la publicité (+ anglais)
    for l in _ad_langs(lang):
        rx = PET_PATTERNS.get(l)
        if rx is not None and rx.search(combined):
            return False
    if EXCLUSION_PATTERN.search(combined):
        return False
    if OFF_TOPIC_PATTERN.search(combined):
        return False
    if not has_meat_bbq_relevance(text, lang, advertiser):
        return False
    return True


def detect_meat_type(text: str, advertiser: str = "", keyword: str = "", lang: str = "") -> str:
    """Point 2 : le mot-clé de recherche est IGNORÉ (argument conservé pour compatibilité).
    Renvoie red_meat ou general_meat_bbq."""
    cats = {MEAT_CATEGORY[m] for m in detect_meats(text, advertiser, lang)}
    if cats == {"red"}:
        return "red_meat"
    return "general_meat_bbq"


def save_clean_output(clean_ads: list[dict], clean_file: Path):
    if not clean_ads:
        return

    cols = [
        "country", "lang", "keyword", "meat_type", "meat_detail", "ad_id", "status",
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
        lambda r: detect_meat_type(r.get("ad_text", ""), r.get("advertiser", ""), lang=r.get("lang", "")),
        axis=1,
    )
    df_clean["meat_detail"] = df_clean.apply(   # bœuf, agneau
        lambda r: ",".join(detect_meats(r.get("ad_text", ""), r.get("advertiser", ""), r.get("lang", ""))),
        axis=1,
    )
    df_clean["n_variants"] = df_clean.groupby(["advertiser", "ad_text"])["ad_id"].transform("count")
    df_clean = df_clean[cols]
    df_clean.to_csv(clean_file, index=False, encoding="utf-8-sig")


# =============================================================================
# Collecte : fonction scrape_meta_ads ci-dessous.
# =============================================================================


# -----------------------------------------------------------------------------
# 5. COLLECTE
# -----------------------------------------------------------------------------

PAUSE_ENTRE_REQUETES = 5000    # millisecondes entre deux recherches (limite la charge sur le site)
PAUSE_DEFILEMENT = 2000        # millisecondes après chaque défilement
ETIQUETTES_ID = ("ID dans la bibliothèque", "Library ID")

# Repère chaque carte publicitaire sans dépendre des classes CSS de Facebook :
# une carte est le plus grand bloc qui contient UNE SEULE mention « ID dans la
# bibliothèque » et dont le parent en contient plusieurs (la liste des cartes).
JS_EXTRAIRE_CARTES = """
(etiquettes) => {
  const compter = (txt) => etiquettes.reduce((n, e) => n + (txt.split(e).length - 1), 0);
  const cartes = [];
  for (const div of document.querySelectorAll('div')) {
    const txt = div.innerText || '';
    if (compter(txt) !== 1) continue;
    const parent = div.parentElement;
    if (!parent || compter(parent.innerText || '') < 2) continue;
    const images = [...div.querySelectorAll('img')]
      .filter(img => (img.naturalWidth || img.width) >= 150)
      .map(img => img.src);
    cartes.push({texte: txt, images: [...new Set(images)]});
  }
  // Une seule publicité sur la page : aucun parent n'en contient deux.
  if (cartes.length === 0 && compter(document.body.innerText || '') === 1) {
    cartes.push({texte: document.body.innerText, images: []});
  }
  return cartes;
}
"""


def _nombre_cartes(page) -> int:
    texte = page.inner_text("body")
    return sum(texte.count(e) for e in ETIQUETTES_ID)


def _fermer_bandeau_cookies(page) -> None:
    for btn_text in ["Refuser les cookies optionnels", "Decline optional cookies",
                     "Autoriser tous les cookies", "Allow all cookies"]:
        try:
            btn = page.get_by_role("button", name=btn_text).first
            if btn.is_visible(timeout=800):
                btn.click()
                page.wait_for_timeout(1200)
                return
        except Exception:
            continue


def _faire_defiler(page, max_scrolls: int) -> None:
    """Fait défiler la page pour charger plus de publicités ; s'arrête dès que rien
    de nouveau n'apparaît."""
    precedent = _nombre_cartes(page)
    for _ in range(max_scrolls):
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(PAUSE_DEFILEMENT)
        actuel = _nombre_cartes(page)
        if actuel == precedent:
            break
        precedent = actuel


def scrape_meta_ads(max_scrolls: int = 8, checkpoint_every: int = 10, start_index: int = 1):
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "Data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    clean_file = processed_dir / "meta_ads_meat_bbq_clean.csv"

    plan = build_search_plan()
    total_plan = len(plan)
    if start_index > 1:
        plan = plan[start_index - 1:]
    print(f"Plan viande & barbecue : {len(plan)} requêtes restantes (à partir de {start_index}/{total_plan}).")

    clean_ads: list[dict] = []
    seen_ids: set[str] = set()

    # Reprise : on repart des publicités déjà enregistrées
    if clean_file.exists():
        try:
            existing = pd.read_csv(clean_file, dtype={"ad_id": str}, encoding="utf-8-sig")
            existing_valid = existing[existing.apply(lambda r: is_valid_clean_ad(r.to_dict()), axis=1)].copy()
            clean_ads = existing_valid.to_dict(orient="records")
            seen_ids = set(existing_valid["ad_id"].dropna().astype(str))
            print(f"Reprise depuis {clean_file} ({len(seen_ids)} publicités valides conservées "
                  f"sur {len(existing)} lignes).")
        except Exception as e:
            print(f"Avertissement reprise depuis {clean_file} : {e}")

    requetes_vides = 0
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)          # navigateur standard, sans déguisement
            context = browser.new_context(locale="fr-FR", viewport={"width": 1440, "height": 900})
            page = context.new_page()

            for offset, (country, lang, kw) in enumerate(plan, start=1):
                idx = start_index + offset -1
                url = (
                    "https://www.facebook.com/ads/library/"
                    f"?active_status=all&ad_type=all&country={country}&q={urllib.parse.quote(kw)}"
                )
                print(f"\n[{idx}/{total_plan}] Pays : {country} ({lang}) | Mot-clé : '{kw}'")

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=35000)
                    page.wait_for_timeout(3000)
                except Exception as e:
                    print(f" -> Erreur de chargement : {e}")
                    continue

                _fermer_bandeau_cookies(page)
                _faire_defiler(page, max_scrolls)

                cartes = page.evaluate(JS_EXTRAIRE_CARTES, list(ETIQUETTES_ID))
                if not cartes:
                    requetes_vides += 1
                    print(" -> Aucune publicité affichée.")
                    # Plusieurs pages vides de suite : le site bloque probablement l'accès
                    # (connexion demandée, limitation). On s'arrête au lieu d'insister.
                    if requetes_vides >= 5:
                        print("\n5 recherches vides consécutives : arrêt de la collecte. "
                              "Vérifier la page dans un navigateur.")
                        break
                    page.wait_for_timeout(PAUSE_ENTRE_REQUETES)
                    continue
                requetes_vides = 0

                nouvelles = retenues = 0
                for carte in cartes:
                    ad = parse_ad_block(carte["texte"])
                    if not ad["ad_id"] or ad["ad_id"] in seen_ids:
                        continue
                    seen_ids.add(ad["ad_id"])
                    nouvelles += 1
                    ad.update({
                        "country": country,
                        "lang": lang,
                        "keyword": kw,
                        "image_urls": " | ".join(carte.get("images", [])),
                    })
                    if is_valid_clean_ad(ad):
                        ad["meat_type"] = detect_meat_type(ad["ad_text"], ad["advertiser"], lang=lang)
                        ad["meat_detail"] = ",".join(detect_meats(ad["ad_text"], ad["advertiser"], lang))
                        clean_ads.append(ad)
                        retenues += 1

                print(f" -> {len(cartes)} cartes, {nouvelles} nouvelles, {retenues} retenues "
                      f"(total : {len(clean_ads)})")

                if idx % checkpoint_every == 0:
                    save_clean_output(clean_ads, clean_file)
                    print(f" -> Sauvegarde intermédiaire : {clean_file.name}")

                page.wait_for_timeout(PAUSE_ENTRE_REQUETES)

            browser.close()
    finally:
        # Sauvegarde finale, y compris en cas d'interruption (Ctrl+C, erreur)
        save_clean_output(clean_ads, clean_file)
        print(f"\nCollecte terminée : {len(clean_ads)} publicités enregistrées dans {clean_file}")


if __name__ == "__main__":
    scrape_meta_ads(start_index=1)