# Viande et masculinité

**Existe-t-il une corrélation entre les normes de genre et la consommation de viande rouge en Europe ?**

Projet de data science — ENSAE Paris, 2A.

---

## Démarche

1. **Données d'enquête** : fréquence de consommation de bœuf et d'agneau (ISSP 2020) et masculinité ressentie (ESS 11), agrégées par pays et par sexe.
2. **Cartographie** : écart de consommation entre hommes et femmes, et écart de masculinité ressentie.
3. **Régressions** au niveau des pays : écart de consommation H − F expliqué par l'écart de masculinité H − F, avec contrôle du PIB par habitant en parité de pouvoir d'achat (en logarithme).


## Données

| Source | Contenu utilisé | Accès |
|---|---|---|
| ISSP 2020 — *Environment IV* (GESIS, ZA7650) | « Au cours d'une semaine habituelle, combien de jours mangez-vous du bœuf, de l'agneau ou des produits qui en contiennent ? » (0 à 7 jours) | inscription gratuite sur [gesis.org](https://www.gesis.org/en/issp/data-and-documentation/environment/2020) |
| European Social Survey, round 11 | masculinité ressentie (`mascfel`) | inscription gratuite sur [europeansocialsurvey.org](https://www.europeansocialsurvey.org/) |
| Banque mondiale, *World Development Indicators* | PIB par habitant en PPA (`NY.GDP.PCAP.PP.KD`), 2022 | API publique, sans clé, interrogée par `Src/data_upload.py` |

Les deux fichiers d'enquête sont fournis dans `Data/raw/` : `ISSP_2020.dta` et `ESS11e04_2.csv`.
Le PIB n'est pas un fichier brut à télécharger à la main : `Src/data_upload.py` le récupère par l'API de la
Banque mondiale et l'enregistre dans `Data/processed/pib_hab_ppa_banque_mondiale.csv` (connexion internet requise).

## Arborescence

```
Projet_01_ENSAE_2A/
├── Data/
│   ├── raw/                  données brutes des enquêtes (ISSP, ESS)
│   └── processed/            données nettoyées, agrégées et jointes, PIB téléchargé
├── Src/
│   ├── data_upload.py        récupération du PIB par habitant en PPA par l'API de la Banque mondiale
│   ├── import_clean.py       nettoyage ISSP et ESS, agrégation par pays et par sexe
│   ├── merge.py              jointure ISSP × ESS × PIB → table_jointe.csv
│   ├── stats_descriptives.py tableau descriptif par pays → sorties/tableau_pays.tex
│   ├── maps.py               cartes ISSP (consommation par sexe, écart H − F)
│   ├── carte_masculinite.py  cartes ESS (masculinité ressentie)
│   └── regression.py         statistiques descriptives, corrélations, régressions, diagnostics
├── sorties/                  tableaux (.tex) et figures (.png) appelés par le notebook
├── notebook.ipynb            notebook de présentation des résultats
├── requirements.txt
└── README.md
```

## Installation

Environnement testé : **Python 3.14.7** (service VSCode-Python d'Onyxia, SSP Cloud).

```bash
pip install -r requirements.txt
```

Versions utilisées (toutes fixées dans `requirements.txt`) :

| Paquet | Version |
|---|---|
| pandas | 3.0.6 |
| numpy | 2.5.5 |
| statsmodels | 0.15.0 |
| plotly | 7.1.0 |
| pycountry | 26.2.16 |
| matplotlib | 3.11.2 |
| kaleido | 1.5.2 |
| country_converter | 1.3.2 |
| requests | 2.34.2 |
| jinja2 | 3.1.6 |

`country_converter` sert à la jointure (codes pays ISO2 → ISO3), `requests` à l'appel de l'API de la Banque
mondiale, et `jinja2` à l'export des tableaux en LaTeX.

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
python Src/data_upload.py         # → Data/processed/pib_hab_ppa_banque_mondiale.csv (API Banque mondiale)
python Src/import_clean.py        # → Data/processed/issp_*.csv, ess11_*.csv
python Src/merge.py               # → Data/processed/table_jointe.csv
python Src/stats_descriptives.py  # → sorties/tableau_pays.tex
python Src/maps.py                # → sorties/carte_viande_*.png
python Src/carte_masculinite.py   # → sorties/carte_masculinite_*.png
python Src/regression.py          # → sorties/*.tex, nuage_ecarts.png, diagnostics.png
```

Le notebook lit ensuite les fichiers de `Data/processed/` et de `sorties/`. Avec `RECALCULER = True` dans sa cellule de configuration, il relance lui-même les scripts, dans le même ordre.

## Autrices

Solène Pinguet - Noriane Ménard - Emna Larbi
