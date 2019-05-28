import requests
import os
import json
import time
import pandas as pd
from pandas.io.json import json_normalize
import pickle

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
BEER_TOKEN = os.environ['MY_KEY']
base_url = 'http://api.brewerydb.com/v2/'


import time
time.sleep(5)
print('done')

limit = 200  #only allowed 200 requests a day
offset = 0
total = 2
brewskis = []



url = 'http://api.brewerydb.com/v2/beers/?'
payload = {'key': BEER_TOKEN, 'withIngredients': 'Y', 'withBreweries': 'Y', 'countryIsoCode':'CA'}


while (offset < total):
    time.sleep(1)
    print('making a request from: ' + str(offset))
    resp = requests.get(url, + params=payload) + '&p=' + 'str(offset)'
    offset += 1
    brewskis += resp.json()['data']
    print(resp.json()['data'])

brewskis


# pickle.dump brewskis to open in 'data_processor_for_brewerydb.py'
