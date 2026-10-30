import pandas as pd
from pytrends.request import TrendReq

########################################################
# Trouver le nombre de recherche sur chaque mot par pays 
########################################################

# 1. Initialiser pytrends (langue anglaise, fuseau horaire UTC)
pytrends = TrendReq(hl='en-US', tz=360)

# 2. Définir tes mots-clés liés à la virilité et à la viande (à adapter ou traduire selon les pays)
# Exemple de requêtes combinées ou séparées
keywords = [
    'barbecue',
    'meat',
    'masculinity',
    'manliness',
    'carnism',
]
# 3. Construire le payload (on peut cibler des pays spécifiques via leur code ISO, ex: 'FR', 'DE', 'US')
# Note : Pour un comparatif mondial par pays, Google Trends permet de récupérer l'intérêt par pays (geo='').
pytrends.build_payload(
    keywords, cat=0, timeframe='2023-01-01 2023-12-31', geo=''
)

# 4. Récupérer l'intérêt par pays (Interest by Region)
df_trends_pays = pytrends.interest_by_region(
    resolution='COUNTRY', inc_low_vol=True
)

print(df_trends_pays.head(10))

########################################################
# Trouver les mots associées aux mots relatifs à la viande par pays 
########################################################

# 2. Définir ta liste de mots-clés (maximum 5 par requête pour l'API Google Trends)
keywords = ['beef', 'barbecue', 'meat', 'grill']

# 3. Construire le payload avec tous les mots-clés
pytrends.build_payload(
    keywords, cat=0, timeframe='2023-01-01 2023-12-31', geo=''
)

# 4. Récupérer toutes les requêtes associées d'un coup
related_queries_dict = pytrends.related_queries()

# 5. Parcourir les résultats pour chaque mot-clé
for kw in keywords:
  print(f'\n============================')
  print(f'🔍 Top recherches associées pour : "{kw}"')
  print(f'============================')

  if (
      related_queries_dict
      and kw in related_queries_dict
      and related_queries_dict[kw]['top'] is not None
  ):
    df_top = related_queries_dict[kw]['top']

    # Afficher le Top 10 des mots associés
    print(df_top)

    # (Optionnel) Filtrer automatiquement pour voir si des mots-clés cibles ressortent
    # Ex: chercher des termes liés au genre, aux hommes/femmes ou aux alternatives
    mots_cibles = ['man', 'men', 'masculinity', 'women', 'manliness']
    mas_filter = df_top['query'].str.contains(
        '|'.join(mots_cibles), case=False, na=False
    )
    resultats_filtres = df_top[mas_filter]

    if not resultats_filtres.empty:
      print(
          '   🎯 [Bingo] Termes en lien avec le genre ou les alternatives'
          ' trouvés :'
      )
      print(resultats_filtres)
    else:
      print('   (Aucun terme direct de cette liste dans le Top 10)')
  else:
    print(f'Pas de données disponibles pour {kw}')