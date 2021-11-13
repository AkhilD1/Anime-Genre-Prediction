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
genre_dict = {}
genre_count = {}
for entry in data.genres:
    for x in entry:
        genre_count[x['id']] = genre_count.get(x['id'], 0) + 1
        genre_dict[x['id']] = x['name']

# Plotting the data distribution
plt.bar(genre_count.keys(), genre_count.values())
plt.xlabel('genres')
plt.ylabel('count')
plt.xticks(
    list(genre_dict.keys()), list(genre_dict.values()), rotation='vertical')
plt.savefig('anime_count.png')

# Log Scale plot
plt.bar(genre_count.keys(), genre_count.values(), log=True)
plt.xlabel('genres')
plt.ylabel('count (log scale)')
plt.xticks(
    list(genre_dict.keys()), list(genre_dict.values()), rotation='vertical')
plt.savefig('anime_count_log.png')

# Save the data as pickle for later
with open('data_combined.obj', 'wb') as f:
    pickle.dump(data, f)

# TODO (KJ):
# Create n-grams, tf-idf scores
# Create one-hot encoding for the genres
