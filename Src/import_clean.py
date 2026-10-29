import numpy as np
import pandas as pd

####################
# Nettoyage de l'ESS
####################

# 1. Charger la base intégrée ESS11
df_ess = pd.read_csv('Data/raw/ESS11e04_2.csv')

# 2. Liste des variables sélectionnées
variables_ess = [
    'cntry',
    'bctprd',
    'pslplonl',
    'alcfreq',
    'mascfel',
    'impbemw',
    'polintr',
    'wrclmch',
    'ccrdprs',
    'ccnthum',
    'nwspol',
    'netustm',
    'impenva',
]

colonnes_presentes = [col for col in variables_ess if col in df_ess.columns]
df_ess_subset = df_ess[colonnes_presentes].copy()

# Compter le nombre de lignes initiales avant nettoyage
lignes_initiales = len(df_ess_subset)

# 3. Dictionnaire des codes de non-réponse / refus / valeurs à éliminer par variable
codes_speciaux = {
    'bctprd': [7, 8, 9],  # Refus, Don't know, No answer
    'pslplonl': [7, 8, 9],  # Refus, Don't know, No answer
    'alcfreq': [77, 88, 99],  # Refus, Don't know, No answer
    'mascfel': [7, 8, 9],  # Refus, Don't know, No answer
    'impbemw': [66, 77, 88, 99],  # Not applicable, Refus, Don't know, No answer
    'polintr': [7, 8, 9],  # Refus, Don't know, No answer
    'wrclmch': [6, 7, 8, 9],  # Not applicable, Refus, Don't know, No answer
    'ccrdprs': [66, 77, 88, 99],  # Not applicable, Refus, Don't know, No answer
    'nwspol': [7777, 8888, 9999],  # Refus, Don't know, No answer
    'netustm': [7, 8, 9],  # Refus, Don't know, No answer
    'impenva': [66, 77, 88, 99],  # Not applicable, Refus, Don't know, No answer
}

# Remplacer ces codes spécifiques par NaN pour qu'ils soient traités comme des valeurs manquantes
for col, codes in codes_speciaux.items():
  if col in df_ess_subset.columns:
    df_ess_subset[col] = df_ess_subset[col].replace(codes, np.nan)

# On isole ccnthum pour créer une variable binaire "Climato-sceptique" avant de nettoyer le reste
if 'ccnthum' in df_ess.columns:
  # 1 si la personne pense que ça n'arrive pas (55), 0 sinon (mais on garde les NaN pour les vrais refus)
  df_ess['climatosceptique'] = np.where(df_ess['ccnthum'] == 55, 1, 0)
  df_ess.loc[df_ess['ccnthum'].isin([77, 88, 99]), 'climatosceptique'] = np.nan

# Pour ccnthum classique, si on veut garder l'échelle de 1 à 5 en excluant le 55 des moyennes continues :
# On transforme 55 en NaN pour ne pas fausser la moyenne sur les causes humaines/naturelles
df_ess['ccnthum'] = df_ess['ccnthum'].replace(55, np.nan)

# 4. Éliminer toutes les lignes contenant des NaN sur ces colonnes
df_ess_propre = df_ess_subset.dropna()
print(df_ess_propre.head(10))

# Calculer et afficher le nombre de lignes éliminées
lignes_eliminees = lignes_initiales - len(df_ess_propre)
print(f'Nombre de lignes éliminées suite au nettoyage : {lignes_eliminees}')
print(f"Nombre de lignes restantes pour l'analyse : {len(df_ess_propre)}")

# ==========================================
# 🔍 TEST DE VÉRIFICATION DES TYPES AVANT AGRÉGATION
# ==========================================
print('\n--- [TEST] Vérification des types de données ---')
print(df_ess_propre.dtypes)

# Détecter s'il reste des colonnes non numériques (hors 'cntry' qui est le code pays textuel)
colonnes_non_numeriques = df_ess_propre.select_dtypes(
    include=['object']
).columns.tolist()
# On retire 'cntry' de la liste car c'est normal qu'il soit en texte (ex: 'FR', 'DE')
colonnes_non_numeriques = [col for col in colonnes_non_numeriques if col != 'cntry']

if colonnes_non_numeriques:
  print(
      f'⚠️ ALERTE : Les colonnes suivantes contiennent du texte/non-numérique'
      f' : {colonnes_non_numeriques}'
  )
else:
  print(
      '✅ TEST RÉUSSI : Toutes les variables à agréger sont bien au format'
      ' numérique (hors identifiant pays).'
  )
print('------------------------------------------------\n')

# 5. Agréger par pays pour obtenir les scores moyens par nation (Étape 2)
df_ess_agrege = df_ess_propre.groupby('cntry').mean(numeric_only=True).reset_index()

# 6. Renommer 'cntry' en 'code_pays' pour préparer la jointure macroscopique (Étape 3)
df_ess_agrege = df_ess_agrege.rename(columns={'cntry': 'code_pays'})

# Afficher un aperçu des 10 premières lignes du résultat final agrégé
print(df_ess_agrege.head(10))