import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import polynomial_kernel
df = pd.read_csv('moviedata.csv')
features = ['keywords','cast','genres','director']
def combine_features(row):
    return row['keywords']+' '+row['cast']+' '+row['genres']+''+row['director']
for feature in features:
    df[feature] = df[feature].fillna('') #filling all NaNs with blank string
#applying combined_features() method over each rows of dataframe and storing the combined string in “combined_features” column
df['combined_features'] = df.apply(combine_features,axis=1) 
cv = CountVectorizer() #creating new CountVectorizer() object
count_matrix = cv.fit_transform(df['combined_features']) #feeding combined strings(movie contents) to CountVectorizer() object
cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]

def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]

#Users may enter their favorite movie and see the list of similar movies
movie_user_likes = 'Avatar'
movie_index = get_index_from_title(movie_user_likes)
#accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it
similar_movies = list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
i=0
print('Top 5 similar movies to '+movie_user_likes+' are:\n')
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>5:
        break
