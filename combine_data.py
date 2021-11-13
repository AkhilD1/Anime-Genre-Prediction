import json
import os
import pickle
import re

from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
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
    except Exception:
        raise Exception('Error reading the file {}'.format(filename))

# Create a pandas dataframe for the data
data = pd.DataFrame({'id': anime_id,
                     'title': anime_title,
                     'synopsis': anime_synopsis,
                     'genres': anime_genres})

print(data.info())

# Since each 'id' is unique, there are no duplicates in the data
# We can see that some entries do not have 'genres' populated
# We will be dropping these entries
data.dropna(inplace=True)

# Remove Source from synopsis
if any(data.synopsis.apply(lambda x: 'Source' in x)):
    data.synopsis = data.synopsis.apply(
        lambda x: re.sub(
            r'[\[\(\s]*Source[:\s]*.*[\]\)\s]*|\[[W]ritten.*\]', '', x))
if any(data.synopsis.apply(lambda x: 'Source' in x)):
    raise Exception('Error cleaning')
if any(data.synopsis.apply(lambda x: '[Written' in x)):
    raise Exception('Error cleaning')

# Convert data to lowercase
data.synopsis = data.synopsis.apply(lambda x: x.lower())

# Tokenize the data. Remove any punctuation and whitespaces
tokenizer = RegexpTokenizer(r'\w+')
data.synopsis = data.synopsis.apply(lambda x: tokenizer.tokenize(x))

# Remove stopwords
nltk.download('stopwords')
stops = set(stopwords.words("english"))
data.synopsis = data.synopsis.apply(
    lambda x: [word for word in x if word not in stops])

# Stemming
stemmer = PorterStemmer()
data.synopsis = data.synopsis.apply(
    lambda x: [stemmer.stem(word) for word in x])

# One-hot encoding
for index, row in data.iterrows():
    for genre in row.genres:
        if not genre['name'] in data:
            data[genre['name']] = 0
        data[genre['name']][index] = 1

# We can now drop the genres column
data.drop('genres', axis=1, inplace=True)

# Visualizing data distribution
genre_count = {}
for col in data:
    if col not in ['id', 'title', 'synopsis']:
        genre_count[col] = data[col].sum()

# Plotting the data distribution
fig, axs = plt.subplots(2, 1)
axs[0].bar(genre_count.keys(), genre_count.values())
axs[0].set(xlabel='genres', ylabel='count')
axs[0].set_xticklabels(genre_count.keys(), rotation='vertical')

# Log Scale plot
axs[1].bar(genre_count.keys(), genre_count.values(), log=True)
axs[1].set(xlabel='genres', ylabel='count (log scale)')
axs[1].set_xticklabels(genre_count.keys(), rotation='vertical')

fig.suptitle('Distribution of data across genres', y=0.98)
fig.set_size_inches(10, 7)
fig.tight_layout()
fig.savefig('anime_count.png')

# We see that the data is very imbalanced and genres
# have a very low number of entries. For the project
# we will keep only the genres which have more than
# 200 entries.

genre_list = list(genre_count.keys())
for genre in genre_list:
    if genre_count[genre] < 200:
        data.drop(genre, axis=1, inplace=True)
        del(genre_count[genre])

# Plotting the new data distribution
fig, axs = plt.subplots(1, 1)
axs.bar(genre_count.keys(), genre_count.values())
axs.set(xlabel='genres', ylabel='count')
axs.set_xticklabels(genre_count.keys(), rotation='vertical')
fig.tight_layout()
fig.savefig('new_anime_count.png')

# Drop the rows that only have zeros in remaining genres
row_del_list = []
for i, row in data.iterrows():
    if sum(row[3:]) == 0:
        row_del_list.append(i)

data.drop(row_del_list, inplace=True)

# Save the data as pickle for later
with open('data_combined.obj', 'wb') as f:
    pickle.dump(data, f)

# TODO (KJ):
# Create n-grams, tf-idf scores
