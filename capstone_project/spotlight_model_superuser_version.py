import pandas as pd
import numpy as np
from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.explicit import ExplicitFactorizationModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('data/processed_dataframe.csv', index_col=0)
df2 = df.copy()
df3 = pd.read_csv('data/beer_style_names.csv', index_col=0)

##Encoding
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


#Explicit Factorization Model
explicit_interactions = Interactions(user_ids, item_ids, ratings)
explicit_interactions.tocoo().todense().shape


explicit_model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=32,
                                   n_iter=10,
                                   batch_size=250,
                                   learning_rate=0.01)

explicit_model.fit(explicit_interactions)


user_df = pd.DataFrame(explicit_interactions.tocoo().todense())



########SPOTLIGHT RECOMMENDATIONS###########

#This function uses the Spotlight model to make recommendations for a given user
def spotlight_predictions(user_id):
    app_user = pd.DataFrame(user_df.iloc[user_id]).T
    app_user_preds = pd.DataFrame({
        'beer': beer_encoder.classes_,
        'value': explicit_model.predict(np.array(app_user)), #needs to be passed as an array
        }).sort_values('value').tail(20)
    new_preds_list = app_user_preds['beer'].tolist()

    return new_preds_list

#This function creates a list of all the beer that the given user has rated/tried
def tried(user_id):
    tried_list = df['user'] == user_id
    tried_list = df[tried_list]['beer'].tolist()
    list_of_tried_beers = beer_encoder.inverse_transform(tried_list).tolist()

    return list_of_tried_beers


#This function finds the Spotlight recommendations that the given user has not tried
def advanced_recommendations(user_id):
    recommendations = []
    tried_beers = tried(user_id)
    reco_beers = spotlight_predictions(user_id)
    for item in reco_beers:
        if item in tried_beers:
            pass
        else:
            recommendations.append(item)

    return recommendations

advanced_recommendations(51)



#########KNN MODEL RECOMMENDATIONS#######
from scipy.sparse import csr_matrix

def create_X(new_df):

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

#Normalize the beer data.
n_ratings_per_beer = X.getnnz(axis=0)
sum_ratings_per_beer = X.sum(axis=0)
mean_rating_per_beer = sum_ratings_per_beer/n_ratings_per_beer
X_mean_beer = np.tile(mean_rating_per_beer, (X.shape[0], 1))
X_mean_beer.shape
X_norm = X - csr_matrix(X_mean_beer)


def beer_finder2(beer):
    return df2[df2['beer'].str.contains(beer)]['beer'].tolist()


#This function finds the beers closest to the target beer in a KNN model
def find_similar_beers(beer_id, X=X_norm, beer_mapper=beer_mapper, beer_inv_mapper=beer_inv_mapper, k=10, metric='manhattan'):

    neighbour_ids = []
    title = beer_finder2(beer_id)[0]
    beer_ind = beer_mapper[title]
    beer_vec = X.T[beer_ind]
    if isinstance(beer_vec, (np.ndarray)):
        beer_vec = beer_vec.reshape(1,-1)

    kNN = NearestNeighbors(n_neighbors = k + 1, algorithm="brute", metric='manhattan')
    kNN.fit(X.T)
    neighbour = kNN.kneighbors(beer_vec, return_distance=False)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(beer_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids



######COSINE SIMILARITY RECOMMENDATIONS######

styles = set(df3['style_name'].unique())
for s in styles:
    df3[s] = df3.style_name.transform(lambda x: int(s in x))


og_df3 = df3.copy()
df3 = df3.drop(columns=['beer', 'style_name'])


#Calculating the Cosine Similarity
cosine_sim = cosine_similarity(df3, df3)



beer_idx = dict(zip(og_df3['beer'], list(og_df3.index)))

def beer_finder(beer):
    return og_df3[og_df3['beer'].str.contains(beer)]['beer'].tolist()


#This function uses cosine similarity to find the most similar beers based on style to the target beer
def return_beers(beer):

    title = beer_finder(beer)[0]
    n_recommendations = 10
    idx = beer_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_beers = [i[0] for i in sim_scores]
    return og_df3['beer'].iloc[similar_beers].tolist()



#######FINAL FUNCTIONS########

#This function finds all the beer that the given user rated 8 or higher
def fetch_favourites(user_id):
    ratings_for_user = df['user'] == user_id
    favourites = df['rating'] >= 8
    users_preferences = df[ratings_for_user & favourites]['beer'].tolist()
    users_favourites = beer_encoder.inverse_transform(users_preferences)

    return list(users_favourites)


#This function uses the above the KNN and Cosine Similarity functions to find beers similar (both style
#and user-item interaction) to the beer that the given user score 8 or above
def sim_beers(list_of_beer):
    preferences = []
    for beer in list_of_beer:
        cosine_beers = return_beers(beer)
        preferences.extend(cosine_beers)
        knn_beers = find_similar_beers(beer)
        preferences.extend(knn_beers)

    return preferences


#This function combines the above functions to recommend the Spotlight recommendations that are similar to
#the beers that the given user rated 8 or above
def super_user_recs(user_id):
    final_recos = []
    list_of_beer = fetch_favourites(user_id)
    knn_cs_recs = sim_beers(list_of_beer)
    spotlight_recs = advanced_recommendations(user_id)
    for item in spotlight_recs:
        if item in knn_cs_recs:
            final_recos.append(item)
        else:
            pass

    return final_recos
