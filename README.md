

requirements : 
python3 -m pip install playwright pandas
playwright install chromium
python3 -m playwright install-deps chromium
python3 -m pip install scipy

# Viande et masculinité

**Existe-t-il une corrélation entre les normes de genre et la consommation de viande rouge en Europe ?**

Projet de data science — ENSAE Paris, 2A.

---

## Démarche

1. **Données d'enquête** : fréquence de consommation de bœuf et d'agneau (ISSP 2020) et masculinité ressentie (ESS 11), agrégées par pays et par sexe.
2. **Cartographie** : écart de consommation entre hommes et femmes, et écart de masculinité ressentie.
3. **Régressions** au niveau des pays : écart de consommation H − F expliqué par l'écart de masculinité H − F, avec contrôle du PIB par habitant (en logarithme).


## Données

| Source | Contenu utilisé | Accès |
|---|---|---|
| ISSP 2020 — *Environment IV* (GESIS, ZA7650) | « Au cours d'une semaine habituelle, combien de jours mangez-vous du bœuf, de l'agneau ou des produits qui en contiennent ? » (0 à 7 jours) | inscription gratuite sur [gesis.org](https://www.gesis.org/en/issp/data-and-documentation/environment/2020) |
| European Social Survey, round 11 | masculinité ressentie (`mascfel`), attitudes | inscription gratuite sur [europeansocialsurvey.org](https://www.europeansocialsurvey.org/) |
| Banque mondiale | PIB par habitant, 2022 | téléchargement libre |
| Bibliothèque publicitaire Meta | textes de publicités | collecte par scraping (voir Limites) |

Les fichiers bruts ne sont pas versionnés dans le dépôt. Les placer dans `Data/raw/` :
`ISSP_2020.dta`, `ESS11e04_2.csv`, `gdp_world_countries_percapita.csv`.

## Arborescence

```
Projet_01_ENSAE_2A/
├── Data/
│   ├── raw/            données brutes 
│   └── processed/      données nettoyées, agrégées et jointes
├── Src/
│   ├── import.clean.py              nettoyage ISSP et ESS, agrégation par pays et par sexe
│   ├── merge.py           jointure ISSP × ESS × PIB → table_jointe.csv
│   ├── maps.py               cartes ISSP (consommation par sexe, écart H − F)
│   ├── carte_masculinite.py  cartes ESS (masculinité ressentie)
│   ├── regression.py         statistiques descriptives, corrélations, régressions 
│   └── lister_paquets.py     liste des paquets utilisés et de leurs versions
├── sorties/            tableaux (.tex) et figures (.png) appelés par le notebook
├── notebook.ipynb             notebook de présentation des résultats
├── requirements.txt
└── README.md
```

## Installation

Environnement testé : **Python 3.14.7** (service VSCode-Python d'Onyxia, SSP Cloud).

```bash
pip install -r requirements.txt
```

Versions utilisées :

| Paquet | Version |
|---|---|
| pandas | 3.0.6 |
| numpy | 2.5.5 |
| statsmodels | 0.15.0 |
| plotly | 7.1.0 |
| pycountry | 26.2.16 |
| matplotlib | 3.11.2 |
| kaleido | 1.5.2 |

La jointure demandent aussi `playwright` et `country_converter`. Pour régénérer la liste complète à partir des imports du projet :

```bash
python Src/lister_paquets.py > requirements.txt   # puis ajouter kaleido à la main
```

### Dépendances supplémentaires

L'export des cartes en PNG (Kaleido) utilise Chrome :

```bash
plotly_get_chrome             # Chrome pour l'export des cartes
```

Sur Onyxia, Chrome a aussi besoin de bibliothèques système (à réinstaller à chaque nouveau service) :

```bash
sudo apt-get update && sudo apt-get install -y libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
  libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
  libpango-1.0-0 libcairo2 libasound2t64 fonts-liberation
```

## Exécution

Depuis la racine du projet, dans cet ordre :

```bash
python Src/import_clean.py               # → Data/processed/issp_*.csv, ess11_*.csv
python Src/merge.py            # → Data/processed/table_jointe.csv
python Src/maps.py                # → sorties/carte_viande_*.png
python Src/carte_masculinite.py   # → sorties/carte_masculinite_*.png
python Src/regression.py          # → sorties/*.tex, *.png
```

Le notebook lit ensuite les fichiers de `Data/processed/` et de `sorties/`. Avec `RECALCULER = True` dans sa cellule de configuration, il relance lui-même les scripts.

## Autrices 

Solène Pinguet - Noriane Ménard - Emna Larbi