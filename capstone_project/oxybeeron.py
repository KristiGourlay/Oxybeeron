import pandas as pd
import numpy as np
import json
import pickle
import torch
import flask
from flask import Flask, request, render_template
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight import evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.sparse import csr_matrix
from collections import Counter



df = pd.read_csv('data/processed_dataframe.csv', index_col=0)
df2 = df.copy()
df3 = pd.read_csv('data/beer_style_names.csv', index_col=0)

df3 = df3.reset_index()  #Inspected dfs and saw that this one was missing row 89

#Quebec beer with french names were printing in flask improperly. Need to fix.
df['beer'] = df['beer'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df3['beer'] = df3['beer'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')


#Spotlight Model

beer_encoder = LabelEncoder()
beer_encoder.fit(df['beer'])
beer = beer_encoder.transform(df['beer'])

user_encoder = LabelEncoder()
user_encoder.fit(df['user'])
user = user_encoder.fit_transform(df['user'])

df['user'] = user_encoder.transform(df['user'])
df['beer'] = beer_encoder.transform(df['beer'])

user_ids = np.array(df['user'])
item_ids = np.array(df['beer'])
ratings = np.array(df['rating']).astype('float32')

explicit_interactions = Interactions(user_ids, item_ids, ratings)

train, test = random_train_test_split(explicit_interactions, random_state=np.random.RandomState(42))


explicit_model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=32,
                                   n_iter=10,
                                   batch_size=250,
                                   learning_rate=0.01)

explicit_model.fit(train)

pk, rk = evaluation.precision_recall_score(explicit_model, test, train=None, k=10)
np.mean(pk)




pipe = make_pipeline(ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=32,
                                   n_iter=10,
                                   batch_size=250,
                                   learning_rate=0.01))

pipe.fit(train)

user_df = pd.DataFrame(explicit_interactions.tocoo().todense())



#KNN Model

def create_X(new_df):
  
    '''
    Input: dataframe
    Output: sparse matrix and both beer and user mappers
    '''
    
    M = new_df['user'].nunique()
    N = new_df['beer'].nunique()

    user_mapper = dict(zip(np.unique(new_df["user"]), list(range(M))))
    beer_mapper = dict(zip(np.unique(new_df["beer"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(new_df["user"])))
    beer_inv_mapper = dict(zip(list(range(N)), np.unique(new_df["beer"])))

    user_index = [user_mapper[i] for i in new_df['user']]
    item_index = [beer_mapper[i] for i in new_df['beer']]

    X = csr_matrix((new_df['rating'], (user_index,item_index)), shape=(M,N))

    return X, user_mapper, beer_mapper, user_inv_mapper, beer_inv_mapper

X, user_mapper, beer_mapper, user_inv_mapper, beer_inv_mapper = create_X(df2)


#Normalize the beer data to deal with user-item bias
n_ratings_per_beer = X.getnnz(axis=0)
sum_ratings_per_beer = X.sum(axis=0)
mean_rating_per_beer = sum_ratings_per_beer/n_ratings_per_beer
X_mean_beer = np.tile(mean_rating_per_beer, (X.shape[0], 1))
X_mean_beer.shape
X_norm = X - csr_matrix(X_mean_beer)


def beer_finder2(beer):
    return df2[df2['beer'].str.contains(beer)]['beer'].tolist()


def find_similar_beers(beer_id, X=X_norm, beer_mapper=beer_mapper, beer_inv_mapper=beer_inv_mapper, k=10, metric='manhattan'):
    '''
    Input: beer_id
    Output: k nearest neighbour ids
    '''
    neighbour_ids = []
    title = beer_finder2(beer_id)[0]
    beer_ind = beer_mapper[title] #finding index number of beer
    beer_vec = X.T[beer_ind] #vector of beer chosen
    if isinstance(beer_vec, (np.ndarray)):
        beer_vec = beer_vec.reshape(1,-1)

    kNN = NearestNeighbors(n_neighbors = k + 1, algorithm="brute", metric='manhattan')
    kNN.fit(X.T) #fitting our knn model with the X matrix
    neighbour = kNN.kneighbors(beer_vec, return_distance=False) #finding nearest neighbours to chosen beer
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(beer_inv_mapper[n]) #creating the list with actual beer names
    neighbour_ids.pop(0) #removes the first element(the chosen beer)
    return neighbour_ids


#Cosine Similarity Model

styles = set(df3['style_name'].unique())
for s in styles:
    df3[s] = df3.style_name.transform(lambda x: int(s in x))


og_df3 = df3.copy()
df3 = df3.drop(columns=['beer', 'style_name'])


#Calculating the Cosine Similarity
cosine_sim = cosine_similarity(df3, df3)

beer_idx = dict(zip(og_df3['beer'], list(og_df3.index))) #indexing the beers

def beer_finder(beer):
    return og_df3[og_df3['beer'].str.contains(beer)]['beer'].tolist()


def return_beers(beer):
    '''
    Input: beer name
    Output: list of most similar beers
    '''
    title = beer_finder(beer)[0]
    n_recommendations = 10
    idx = beer_idx[title] #getting index number of beer
    sim_scores = list(enumerate(cosine_sim[idx])) #cosine similarity of each beer to given beer
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #sorting beers by cos sim
    sim_scores = sim_scores[1:(n_recommendations+1)] #finding recommendations
    similar_beers = [i[0] for i in sim_scores] #getting the index number for recommended beers
    return og_df3['beer'].iloc[similar_beers].tolist()


  
#Functions to tie all three models together

def new_user(inputs):
    '''
    Input: input from flask app
    Output: dataframe with new entry of flask user
    '''
    new_input = pd.DataFrame([0] * 337).T
    for i in inputs:
        if i == 'Corona':
            new_input[197] = 8.0
        if i == 'Jelly King':
            new_input[43] = 8.0
        if i == 'Lugtread':
            new_input[18] = 8.0
        if i == 'Heineken':
            new_input[198] = 8.0
        if i == 'Guiness':
            new_input[197] = 8.0
        if i == 'Coors':
            new_input[153] = 8.0
        if i == 'Woodhouse':
            new_input[333] = 8.0
        if i == 'Side Launch':
            new_input[302] = 8.0
        if i == 'La Chouffe':
            new_input[68] = 8.0
        if i == 'Octopus':
            new_input[191] = 8.0
        if i == 'Peche Mortel':
            new_input[69] = 8.0
        if i == 'Stella':
            new_input[124] = 8.0
        if i == 'Boneshaker':
            new_input[8] = 8.0
        if i == 'Blue Moon':
            new_input[67] = 8.0
        if i == 'Jutsu':
            new_input[45] = 8.0
        if i == 'Creemore Springs Premium Lager':
            new_input[157] = 8.0
        if i == 'Clifford':
            new_input[132] = 8.0
        else:
            pass

    return user_df.append(new_input)


def get_recos(inputs):
    '''
    Input: input from flask app
    Output: predictions based on Spotlight model
    '''
    df4 = new_user(inputs)
    app_user = pd.DataFrame(df4.iloc[-1]).T
    app_user_preds = pd.DataFrame({
        'beer': beer_encoder.classes_,
        'value': pipe.predict(np.array(app_user)), #needs to be passed as an array
        }).sort_values('value').head(10)
    new_preds_list = app_user_preds['beer'].tolist()
    return new_preds_list





def beer_collector(beer_choices):
    '''
    Input: input from flask app
    Output: Two lists of recommendations. One from KNN model and Spotlight model. One from content similarity.
    '''
    list_of_beers = []
    list_of_similar_beers = []
    df4 = user_df.append(new_user(beer_choices))
    for l in beer_choices:
            collecting = find_similar_beers(l)
            list_of_beers.extend(collecting)
            collecting2 = return_beers(l)
            list_of_similar_beers.extend(collecting2)
            collecting3 = get_recos(l)
            list_of_beers.extend(collecting3)


    return list_of_beers, list_of_similar_beers


def final_list_maker(list_of_beers, inputs):
    '''
    Input: Two lists created by beer_collector function
    Output: Final list of recommendations.
    '''
    final_recommendations = []
    list1 = list_of_beers[0]
    list2 = list_of_beers[1]
    list1 = set(x for x in list1 if list1.count(x) > 1) #Anything that exists in both Spotlight&KNN is kept
    final_recommendations.extend(list1)
    length_results = len(list1)
    x = abs(10 - length_results)
    if len(inputs) > 1:
        extras = list2[1::3] #If there is more than one flask input, sushi-roll over content recommendations to ensure variation
        final_recommendations.extend(extras)
    else:
        extras = list2[:x] #If only one input is given, fill the rest of the recos with the content recommender
        final_recommendations.extend(extras)

    return list(set(final_recommendations))[:9]

templates = ['beers2.html',
            "beers2A.html",
            'beers2B.html',
            'beers2C.html']


app = Flask(__name__)

@app.route('/page')
def page():
    random_template = np.random.choice(templates)
    return flask.render_template(random_template)

@app.route('/result', methods=['POST', 'GET'])
def contact():
    if flask.request.method == 'POST':
        inputs = request.form.getlist('check')
        results = beer_collector(inputs)
        results = final_list_maker(results, inputs)
    return flask.render_template('beer1.html', result = results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='4000', debug=False)
