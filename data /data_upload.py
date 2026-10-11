import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

url_1 = "https://www.efsa.europa.eu/en/data-report/food-consumption-data"
data_alimentation = pd.read_csv(url_1)
data_alimentation.head(5)

url_2 = #faut se login pour télécharger les données ESS, on l'avait fait pour R l'année dernière mais 
#sais plus comment faire 
data_masculine = pd.read_csv(url_2)

url_3 = #j'ai trouve ce site mais faut un accès https://statbase.org/datasets/economics/gdp/?syear=2023&filter=&sort=
#sinon td1 macro l'année dernière on l'avait le csv mais c'est sur ma session ensae donc j'essaierai
#de faire à l'ensae
data_PIB = pd.read_csv(url_3)

