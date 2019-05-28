import pandas as pd
import numpy as np

df = pd.read_csv('../data/beer_dataframe.csv', index_col=0) ##Brewerydb data
df2 = pd.read_csv('../data/beer_df_ratebeer.csv', index_col=0) #RateBeer data

ontario = df[df['location'] == 'Ontario']
quebec = df[df['location'] == 'Quebec']
macro = df[df['type'] == 'Macro Brewery']

ontario = df['location'] == 'Ontario'
quebec = df['location'] == 'Quebec'
québec = df['location'] == 'Québec'
macro = df['location'] == 'Macro Brewery'


four = ontario | quebec | macro | québec
df[four].shape


## Finding known breweries that were logged with no location
no_location = df['type'] == 'None'
no_location = df[no_location]

unique_breweries = df[four]['brewery'].unique()   ##Breweries in current dataframe

breweries_in_nope = no_location['brewery'].unique() ##Breweries listed with no location

both_lists = set(unique_breweries) & set(breweries_in_nope) ##overlap
both_lists #found 5 known breweries without a location.

##Adding those entries to the main dataframe
BB = df['brewery'] == 'Bellwoods Brewery'
GLB = df['brewery'] == 'Great Lakes Brewery'
BEAUS = df['brewery'] == 'Beaus All Natural Brewing Company'
DIABLE = df['brewery'] == 'Microbrasserie La Diable'

# nope = Nope[BB | GLB | BEAUS | DIABLE]
no_location_listed = BB | GLB | BEAUS | DIABLE



five = no_location_listed | ontario | quebec | macro | québec


df[five].shape

#### Checking for other beers easily found in Toronto that did not fall into the filtered sections above

cali = df['location'] == 'California'
ny = df['location'] == 'New York'
# df[ny]['brewery'].unique()

##Adding those beers
SFBC = df['brewery'] == 'San Francisco Brewing Co.'
RRBC = df['brewery'] == 'Russian River Brewing Company'
LAG = df['brewery'] == 'Lagunitas Brewing Company'
GI = df['brewery'] == 'Granville Island Brewing Company'
BB = df['brewery'] == 'Brooklyn Brewery'
BO = df['brewery'] == 'Brewery Ommegang'

extras = SFBC | RRBC | LAG | GI | BB | BO

six = no_location_listed | ontario | quebec | macro | québec | extras

final_df_from_db = df[six]
final_df_from_db.shape
    #515 useable beers from Brewerydb



### Bringing in beer dataframe from beeradvocate
df2.head(2)


df2 = df2.replace('I PA', 'IPA', regex=True) #Fixing misspell
df2.head(2)

brewery_names = df2['brewery'].unique().tolist()
df2['location'] = 'Ontario' #creating location information for breweries
df2['description'] = 'NaN' #empty columns.
df2['organic'] = 'NaN'
df2['id'] = 'NaN'
df2 = df2.rename(columns={'type': 'style_name'})

df2['type'] = 'Micro Brewery' #All these beers are MicroBreweries


df2.head(1)
df2.to_csv('beer_df2.csv')



#Comparing the two seperate dataframes side-by-side
df2.head(1)
final_df_from_db.head(1)

final_df_from_db = final_df_from_db.drop(['style_id'], axis=1)
final_df_from_db.head(1)
neworder = ['brewery', 'name', 'abv', 'style_name', 'location', 'description', 'type', 'organic', 'id']
final_df_from_db = final_df_from_db.reindex(columns=neworder)
df2 = df2.reindex(columns=neworder)

final_df_from_db.shape
df2.shape
final_df_from_db.head(1)
df2.head(1)

##Concatenating the two dataframes together
beer_df = pd.concat([final_df_from_db, df2,], ignore_index=True)
beer_df.shape #615 beers

beer_df = beer_df.sort_values('brewery')

beer_df.head(2)

beer_df.to_csv('final_beer_df.csv')

##The form was then uploaded to google spreadsheets. I also added 20 beers manually, to round out the beer choices.
##The form was then filled out by 15 'super users' and a beer form (a select 20 beers) was filled out by 86 people.
