import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import re
res = requests.get('https://www.beeradvocate.com/lists/ca/on/')

soup = bs(res.content, 'lxml')

##The website developers split the name of the beer and the rest of the beer info in seperate divs. So I decided
#to scrape the beer names seperately and concatenate after.

beer_section = soup.find('table')
tr_beers = []
for link in beer_section.find_all('span', {'class': 'muted'}):
    tr_dict = {}
    tr_dict['beers'] = link.text
    tr_beers.append(tr_dict)


beer_section = soup.find('table')
beer_name_list = []
for link in beer_section.find_all('a'):
    beer_name_list += [link.text]


beer_name_list = beer_name_list[1::3] #Every third beer was the beer name


#Now I will scrape the rest of the information

tr_beers = tr_beers[1:] #Every line contains the brewery, style, and abv


def add_spaces(dictionary):
    list_of_beers = []
    new_list_of_beers = []
    for b in dictionary:
        for k, v in b.items():
            list_of_beers.append(v)
    for beer in list_of_beers:
        beer = re.sub(r"(\w)([A-Z])", r"\1 \2", beer)
        new_list_of_beers.append(beer)

    return new_list_of_beers



fixed_beer_list = add_spaces(tr_beers) #This function used regex to fix the spacing.


def fixing(beer_list):
    new_fixed_beer_list = []
    for beer in fixed_beer_list:
        beer = beer.replace('Brewery', 'Brewery,')
        beer = beer.replace('Brewhouse', 'Brewhouse,')
        beer = beer.replace('Brewing', 'Brewing,')
        beer = beer.replace('|', ',')
        new_fixed_beer_list.append(beer)

    return new_fixed_beer_list

beer_list = fixing(fixed_beer_list) #This function fixed spacing and '|' in the lines.


def final_dict(list):
    final_beer_list = []
    for brew in list:
        beer_dict = {}
        beer_dict['brewery'] = brew.split(',')[0]
        beer_dict['type'] = brew.split(',')[1]
        beer_dict['abv'] = brew.split(',')[-1]
        final_beer_list.append(beer_dict)

    return final_beer_list

final = final_dict(beer_list) #This function placed information in a dictionary.


beer_df = pd.DataFrame(final)

beer_df['name'] = beer_name_list


beer_df.to_csv('beer_df_ratebeer.csv')


beer_df = pd.DataFrame(final_beer_list )























beer_df

beer_dict['brewery'] = beer.split(',')[1]













for b in fixed_beer_list:
    b = re.sub('(Brewery|Brewing', 'Brewery ', b)
    new_fixed_beer_list.append(b)










    new_fixed_beer_list.append(b.replace('Brewery' & 'Brewing', 'Brewery,'))
    new_fixed_beer_list.append(b.replace('Brewing', 'Brewing,'))




new_fixed_beer_list

beer_dict = {}
abv = []
new_bb = []
for beer in fixed_beer_list:
    abv.append(beer.split('|')[-1])
#
# for beer in fixed_beer_list:
#     new_bb.append(beer.split('Brewery', ""))

new_bb



beer_dict['abv'] = find_abv(fixed_beer_list)
beer_dict
fixed_beer_list


beer_dict = {}
beer_dict['beer_name'] = add_spaces(tr_beers)

beer_dict







type(tr_beers)
list_of_beers = []
for b in tr_beers:
    for k, v in b.items():
        list_of_beers.append(v)

# tr_beers = pd.DataFrame(tr_beers)
# type(beer_list)

# beer_list = tr_beers['beers'].tolist()
# len(beer_list)



def add_spaces(list):
    new_list_of_beers = []
    for beer in list_of_beers:
        beer = re.sub(r"(\w)([A-Z])", r"\1 \2", beer)
        new_list_of_beers.append(beer)

    return new_list_of_beers


add_spaces(list_of_beers)

new_list_of_beers











tr_beers

import re
regex = r'\b[Brewing: ]\b'
s = "NickelBrook BrewingIPA Company Butt"
print(re.sub(regex, ' \g<0> ', s))






tr_dict
abv = beer_section.find_all('span', {'class': 'muted'})[5].text.split('|')[-1]


poop = beer_section.find_all('span', {'class': 'muted'})[1].text


poop
