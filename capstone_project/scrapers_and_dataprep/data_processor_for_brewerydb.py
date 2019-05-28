import json
import pickle
import pandas as pd
# beers = pickle.load(open('new_beer.pkl', 'rb'))
beers = pickle.load(open('brewskis23.pkl', 'rb'))
num_posts = len(beers)

new_beer_dict = []

def parse_one_beer(beer):
    return {'id': beer.get('id', 'None'),
    'name': beer.get('name', 'None'),
    'abv': beer.get('abv', 'None'),
    'organic': beer.get('isOrganic', 'None'),
    'style_name': beer['style'].get('name', 'None'),
    'style_id': beer['style'].get('id', 'None'),
    'description': beer['style'].get('description', 'None'),
    'brewery': beer['breweries'][0].get('name', 'None'),
    'location': beer['breweries'][0]['locations'][0].get('region', 'None'),
    'type': beer['breweries'][0]['locations'][0].get('locationTypeDisplay', 'None')}

new_beer_dict = []
for beer in beers:
    parsed_beer = parse_one_beer(beer)
    new_beer_dict.append(parsed_beer)



new_beer_dataframe = pd.DataFrame(new_beer_dict) #place processed data into a dataframe



beer_dataframe = pd.read_csv('beer_dataframe.csv', index_col=0) #bring in previously processed data in a df

beer_dataframe = pd.concat([beer_dataframe, new_beer_dataframe], ignore_index=True) #add to existing processed data

beer_dataframe.tail(10) #check the added data

beer_dataframe.to_csv('beer_dataframe.csv') #export to csv
