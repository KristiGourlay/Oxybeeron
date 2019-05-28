import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle

##Bring in df from googlesheets and google form
user_df = pd.read_excel('data/final_user_dataframe_transposed.xlsx', index_col=0)

user_df.shape
user_df = user_df.replace(0, np.nan) #replacing all 0 values with NaN


num_of_users = len(user_df)
num_of_beers = len(user_df.columns)
print(f'Number of Users: {num_of_users}')
print(f'Number of Beers: {num_of_beers}')

user_df = user_df.dropna(axis=1, how='all') #Dropping all beers that do not have ratings
user_df.shape

avg_per_beer = np.mean(user_df) #Calculating averages
top_rated = avg_per_beer.sort_values(ascending=False).head(10)

user_averages = user_df.mean(axis=1) #Calculating average rating by user

#Because not all beers have the same number of ratings, its standard to calculate the Bayesian averages.
beer_stats = user_df.agg(['count', 'mean'])

beer_stats = beer_stats.T
beer_stats = beer_stats.reset_index()
beer_stats.columns =['beer', 'count', 'average']

C = beer_stats['count'].mean()
m = beer_stats['average'].mean()

def bayesian_avg(rating):
    bayesian_avg = (C*m+rating.sum())/(C+rating.count())
    return bayesian_avg


bayesian_avg_ratings = user_df.agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['beer', 'bayesian_avg']
beer_stats = beer_stats.merge(bayesian_avg_ratings, on='beer')
beer_stats = beer_stats.sort_values(by='bayesian_avg', ascending=False)




top_five_beers = beer_stats.head(5)
#top_five_beers: High Road Bronan and Cloud Piercer, Bellwoods Jelly King and Jelly King Pink Guava, and Ommegang Pale Sour

#These beers have the highest Bayesian average. Although this is clearly a bias from the fact that 12 of
#the 15 people who rated the entire beer list work at Trinity Common. I've only ever seen
#Cloud Piercer on tap at TC, and Bronan is very rare in Toronto (and exists at TC as a permanent tap).

lowest_ranking_beers = beer_stats.tail(5)
#lowest_ranking_beers: Labatt Brown Ale and Canadian Ale, Molson Carling, Sleeman Clear, and Ace Hill Pilsner
#Probably a bias of the unusually high hipster quotient.



user_df.head(2)

user_df = user_df.drop(columns=['Clifford Brewing Co.: Clifford Porter.1']) ##dropping duplicate

user_df = user_df.reset_index().rename(columns={'index': 'user'})
user_df.shape



## saving a copy before I melt it.
user_df.to_csv('processed_dataframe_not_melt.csv')

new_df = pd.melt(user_df, id_vars='user', var_name='beer', value_name='rating')

new_df = new_df.sort_values('user')

new_df = new_df.dropna()

new_df.shape
new_df.head(2)

## saving a a copy of the melted format
new_df.to_csv('processed_dataframe.csv')

new_df.head(2)
user_df.head(2)


##############Prepping the Styles dataframe ########

beers = pd.read_csv('data/processed_dataframe_not_melt.csv', index_col=0)

beers = beers.drop(columns='user')
beers.head(2)
beers = beers.T

beers = beers.reset_index()

beers.shape

beers.head(2)
beers = beers.rename(columns = {'index': 'beer'})

beers = beers[['beer', 0]]

beers.shape

beers.head(2)



og_beer = pd.read_csv('data/final_beer_df.csv', index_col=0)
og_beer['beer'] = og_beer['brewery'].map(str) + ": " + og_beer['name']
og_beer = og_beer[['beer', 'style_name', 'abv']]
og_beer.head(2)


og_beer['abv'] = pd.to_numeric(og_beer.abv, errors='coerce') #Fixing ABV incase I want to use it in the future
len(og_beer['style_name'].unique())




og_beer.shape
og_beer.style_name.unique()
og_beer.head(2)

#Fixing a wrong spelling
og_beer['beer'][14]
og_beer['beer'][14] = 'Sawdust City Brewing Co.: Gateway Kolsch'



beer_df = og_beer.merge(beers, on='beer') #merging the two dfs to create a 'styles df'.

beer_df = beer_df.drop(columns=['abv', 0,])
beer_df.shape
beer_df.head(2)

style_names = beer_df['style_name'].tolist()

style_names = [x.lower() for x in style_names]
beer_df['style_name'] = style_names


##Mapping the number of styles from 158 down to 31 in order to do content based analysis
beer_df.loc[beer_df.style_name.str.contains('kolsch|kölsch'), 'style_name'] = 'kolsch'
beer_df.loc[beer_df.style_name.str.contains('pilsner|pilsener'), 'style_name'] = 'pilsner'
beer_df.loc[beer_df.style_name.str.contains('cider'), 'style_name'] = 'cider'
beer_df.loc[beer_df.style_name.str.contains('imperial stout'), 'style_name'] = 'imperial st'
beer_df.loc[beer_df.style_name.str.contains('stout|american-style stout|milk stout|cream stout|oatmeal|oatmean|dry stout', case=False), 'style_name'] = 'stout'
beer_df.loc[beer_df.style_name.str.contains('lager'), 'style_name'] = 'lager'
beer_df.loc[beer_df.style_name.str.contains('wheat|wit|weiss|hefeweisen'), 'style_name'] = 'wheat'
beer_df.loc[beer_df.style_name.str.contains('sour|flanders|wild|gose'), 'style_name'] = 'sour'
beer_df.loc[beer_df.style_name.str.contains('ipa|hazy|hop'), 'style_name'] = 'ipa'
beer_df.loc[beer_df.style_name.str.contains('fruit'), 'style_name'] = 'fruit beer'
beer_df.loc[beer_df.style_name.str.contains('belgian'), 'style_name'] = 'belgian'
beer_df.loc[beer_df.style_name.str.contains('american-style stout|milk stout|cream stout'), 'style_name'] = 'stout'
beer_df.loc[beer_df.style_name.str.contains('belgian'), 'style_name'] = 'belgian'
beer_df.loc[beer_df.style_name.str.contains('mild|red|amber'), 'style_name'] = 'red, amber'
beer_df.loc[beer_df.style_name.str.contains('pale|apa'), 'style_name'] = 'pale ale'
beer_df.loc[beer_df.style_name.str.contains('bitter|brown|bock|porter'), 'style_name'] = 'browns'
beer_df.loc[beer_df.style_name.str.contains('black'), 'style_name'] = 'black ale'
beer_df.loc[beer_df.style_name.str.contains('scotch|rye|barrel'), 'style_name'] = 'barrelaged'
beer_df.loc[beer_df.style_name.str.contains('german|märzen'), 'style_name'] = 'germanic styles'

beer_df.shape
#Another duplicate. After removed, the dataset is now clean of duplicates.

beer_style_names_list = beer_df['style_name'].unique()

#Saving the final copy of the beer styles df
beer_df.to_csv('beer_style_names.csv')
