import json
import os
import pickle

import nltk
import numpy as np
import pandas as pd


anime_id = []
anime_title = []
anime_synopsis = []
anime_genres = []

# List all the data files
file_list = os.listdir('data')

# Read all the files
for filename in file_list:
    try:
        with open('data/{}'.format(filename)) as f:
            tmp = json.load(f)
        anime_id.append(tmp.get('id'))
        anime_title.append(tmp.get('title'))
        anime_synopsis.append(tmp.get('synopsis'))
        anime_genres.append(tmp.get('genres'))
    except:
        print('Error reading the file {}'.format(filename))

# Create a pandas dataframe for the data
data = pd.DataFrame({'id': anime_id,
                     'title': anime_title,
                     'synopsis': anime_synopsis,
                     'genres': anime_genres})

print(data.info())

# Since each 'id' is unique, there are no duplicates in the data
# We can that some entries do not have 'genres' populated
# We will be dropping these entries
data.dropna()

# Save the data as pickle for later
with open('data_combined.obj', 'wb') as f:
    pickle.dump(data, f)

# TODO (KJ):
# Clean the data using nltk/regex
    # I. Remove source from synopsis. Only some entries have source mentioned.
    # It occurs only at the end of the synopsis.
    # Type 1: (Source: .*)
    # (Source: AniDB), (Source: ANN), (Source: AnimeNfo), (Source: Wikipedia), etc.
    # Type 2: [Written by MAL Rewrite]
    # II. Stop words, White Spaces
    # III. Stemming, lemmitizing
# Create n-grams, tf-idf scores
# Create one-hot encoding for the genres
