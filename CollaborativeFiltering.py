#!/usr/bin/env python
# coding: utf-8

# # Collaborative Filtering Recommendation System

# ## Task 1: Import Modules

# In[3]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity


# ## Task 2: Import the Dataset

# In[4]:


import os
os.listdir('.')


# In[6]:


column_names = ['User_ID', 'User_Names', 'Movie_ID', 'Rating', 'Timestamp']
movie_data = pd.read_csv('Movie_data.csv', names=column_names)

titles_data = pd.read_csv('Movie_Id_Titles.csv')
titles_data.rename(columns = {'item_id' : 'Movie_ID', 'title' : 'Movie_Title'}, inplace=True)

movie_df = pd.merge(movie_data, titles_data, on='Movie_ID')


# ## Task 3: Explore the Dataset

# In[7]:


titles_data[0:5]


# In[8]:


movie_df.describe()


# In[9]:


n_users =movie_df.User_ID.unique().shape[0]
n_movies =movie_df.Movie_ID.unique().shape[0]


# In[10]:


n_users, n_movies


# In[11]:


movie_df[0:5]


# In[ ]:





# ## Task 4: Create an Interaction Matrix

# In[12]:


ratings = np.zeros((n_users, n_movies))

for i in range(len(movie_df)):
    ratings[movie_df['User_ID'][i], movie_df['Movie_ID'][i]-1] = movie_df['Rating'][i]
#     print(movie_df['User_ID'][i], movie_df['Movie_ID'][i], )


# In[13]:


print(ratings.shape)
print(ratings)


# ## Task 5: Explore the Interaction Matrix

# In[14]:


sparsity = len(np.where(ratings == 0)[0])*1.0/len(np.where(ratings > 0)[0])
# print(len(np.where(ratings == 0)), len(np.where(ratings >0)))
sparsity


# In[15]:


sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print(sparsity)


# ## Task 6 : Create a Similarity Matrix

# In[16]:


rating_cosine_similarity = cosine_similarity(ratings)
rating_cosine_similarity


# ## Task 7: Provide Recommendations

# In[17]:


def recommend(ratings, rating_cosine_similarity, user_id, k=10, top_n=10):
    
    similar_users = rating_cosine_similarity[user_id]
    
    sorted_users = np.argsort(-1*similar_users)
    
    top_k_users = sorted_users[0:k] ### ignoring 1 because that would be same user
#     print(top_k_users, similar_users.argpartition(-k)[-k:]) ## argpartition does the same thing as decided
    ratings_top_k = ratings[top_k_users]
    avg_ratings_top_k = ratings_top_k.mean(axis=0)
    
    sorted_movies = np.argsort(-1*avg_ratings_top_k).tolist()
    
    seen_movies = np.where(ratings[user_id] > 0)[0].tolist()
#     print(seen_movies)
    
    rec_movies = [x+1 for x in sorted_movies if x not in seen_movies]
#     print(rec_movies[0:top_n])
    
#     rec_movies_a=rec_movies.index.to_frame().reset_index(drop=True)
#     rec_movies_a.rename(columns={rec_movies_a.columns[0]: 'Movie_ID'}, inplace=True)

    ## also create a dataframe from rec_movies
    
    rec_movies_df = pd.DataFrame(rec_movies)
    rec_movies_df = rec_movies_df.head(top_n)
    
#     rec_movies_df=rec_movies_df.index.to_frame().reset_index(drop=True)
    rec_movies_df.rename(columns={rec_movies_df.columns[0]: 'Movie_ID'}, inplace=True)

    return rec_movies[0:top_n], rec_movies_df


# In[18]:


recommend(ratings, rating_cosine_similarity, 12)


# ## Task 8: View the Provided Recommendations 

# In[ ]:





# In[ ]:





# ## Task 9: Create Wrapper Function

# In[19]:


def movie_recommender_run(user_Name):
    #Get ID from Name
    user_ID=movie_df.loc[movie_df['User_Names'] == user_Name].User_ID.values[0]
    #Call the function
#     temp=movie_recommender(ratings, rating_cosine_similarity, user_ID)
    _, rec_movies_df = recommend(ratings, rating_cosine_similarity, user_ID)
    # Join with the movie_title_df to get the movie titles
    top_k_rec=rec_movies_df.merge(titles_data, how='inner')
    return top_k_rec


# In[20]:


movie_recommender_run('Shawn Wilson')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




