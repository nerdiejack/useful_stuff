import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from string import ascii_letters
import os


print(os.listdir('data'))
# create dataframe
df = pd.read_csv('data/BRAZIL_CITIES.csv', sep=';', decimal=',')
# create backup copy
df_copy = df.copy(True)

''' select the columns you're interested in exploring. These are the one's I chose. 
if a model was being generated for some purpose, feature selection methods should be employed,
however, I will just being doing cleaning, vis and exploration '''

columns = ['CITY', 'STATE', 'CAPITAL', 'IBGE_RES_POP', 'IBGE_RES_POP_BRAS', 'IBGE_RES_POP_ESTR', 'IBGE_DU',
           'IBGE_DU_URBAN', 'IBGE_DU_RURAL', 'IBGE_POP','IBGE_1', 'IBGE_1-4', 'IBGE_5-9', 'IBGE_10-14',
           'IBGE_15-59', 'IBGE_60+', 'IBGE_PLANTED_AREA', 'IDHM', 'LONG', 'LAT', 'ALT', 'ESTIMATED_POP',
           'GDP_CAPITA', 'Cars','Motorcycles', 'UBER', 'MAC', 'WAL-MART', 'BEDS']
r_df = df[columns]
corr = r_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5,
            cbar_kws={'shrink': .5})
plt.show()
