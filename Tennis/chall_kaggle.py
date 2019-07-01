# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 09:45:51 2018

@author: DJEBALI
"""

#%% Libraries
import pandas as pd
import numpy as np
import os
import re as reg
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn import preprocessing
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

%matplotlib qt5
#os.chdir(r'C:\Users\DJEBALI\Documents\Python Scripts\challenge_kaggle'
os.chdir(r'C:\Users\user\Documents\Python Scripts\challenge_kaggle')
#%% Data preprocessing

df_atp = pd.read_csv('atp.csv',na_values=[' ','NR'])

df_atp.drop(df_atp.columns[0], axis=1, inplace = True)

## Il y a 54 variables et 52298 matchs joués

################# Format Date ###############

df_atp['Date'] = pd.to_datetime(df_atp['Date'])

### Remove columns with more than 60% NA ###

categ_features = list((df_atp.dtypes[df_atp.dtypes == 'object']).index.values)

nb_NA = df_atp.isna().sum()
nb_NA.sort_values(ascending=False,inplace=True)

NA_rate = (nb_NA/df_atp.shape[0])*100

dl = NA_rate.apply(lambda x: x>60).tolist()

dl = dl[0:(len(dl)-1)]

l = [i for i in range(len(dl)) if dl[i]]

feat_del = NA_rate.index[:len(l)].tolist()

df = df_atp.drop(feat_del, axis = 1)

########## Format quantitatives variables #############

l = ['EXW','L2','L3','LRank','Lsets','W2','W3', 'WRank']

tmp = set(categ_features+['Date'])-set(l)

quanti_features = list(set(df.columns)-tmp)

df[quanti_features].replace('nan',np.nan,inplace=True)

df.replace('`1','1',inplace=True)

df.replace('2.,3','2.3',inplace=True)

df[quanti_features] = df[quanti_features].apply(lambda x: pd.to_numeric(x,downcast='float'),axis=1)

#%% Number of players and Players encoding

players = list(set(df['Winner'].unique()).union(set(df['Loser'].unique())))

## Au total 1485 joueurs

df['J1'] = df['Winner'].apply(lambda x: players.index(x))

df['J2'] = df['Loser'].apply(lambda x: players.index(x))

#%% Categorical features encoding

categ_features = list((df.dtypes[df.dtypes == 'object']).index.values)

for x in categ_features:
    if not(x=='Loser') and not(x=='Winner'):
        tmp = list(df[x].unique())
        df[x+'_Enc'] = df[x].apply(lambda x: tmp.index(x))
        
#%% Date engineering

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df['Day'] = df['Date'].dt.day   

#%% NA rows dropout

df.dropna(inplace = True)
     
        
#%% ATP 2017

#import missingno as msno
df_atp2017 = df[df['Date'].dt.year== 2017].copy()

df_atp2017 = shuffle(df_atp2017)

X2017 = df_atp2017.drop(categ_features+['Date'],axis = 1)

Y2017 = df_atp2017['J1']

players2017 = list(set(df_atp2017['Winner'].unique()).union(set(df_atp2017['Loser'].unique())))

## Au total, il y a 307 joueurs en 2017

#%% Machine learning 

df_years = df.drop(df_atp2017.index)

df_years = shuffle(df_years)

X = df_years.drop(categ_features+['Date'],axis = 1).copy()

Y = df_years['J1']

############### Régression logistique #####################

modele_regression_logistique = linear_model. LogisticRegression ()

modele_regression_logistique.fit(X, Y)

classesPredites = modele_regression_logistique.predict(X2017)

probaClasses = modele_regression_logistique. predict_proba (X2017)

print(probaClasses)

#%% Descriptive Analysis

# Compute the correlation matrix
corr = df_atp.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})