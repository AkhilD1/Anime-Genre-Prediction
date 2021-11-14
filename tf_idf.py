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
from sklearn.feature_extraction.text import TfidfVectorizer


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

# Replace numbers
# We replace numeric words with NUM
# eg: 1st, 4th, 1,000, 8.12, 100,000th, 1920s etc.
data.synopsis = data.synopsis.apply(
    lambda x: re.sub(r'\d[\d.,]*(st|rd|th)?|\d[\d.,]*(s)?', ' NUM ', x))

# Stemming
# We use PorterStemmer from nltk for stemming
# TODO: Find a more efficient way
stemmer = PorterStemmer()
data.synopsis = data.synopsis.apply(
    lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Create a list of stop words
nltk.download('stopwords')
stops = set(stopwords.words("english"))

# Now let's find word frequencies
word_count = {}
for item in data.synopsis:
    item = item. split(" ")
    for index in item:
        if index in word_count:
            word_count[index] += 1
        else:
            word_count[index] = 1

print("Total number of unique words ", len(word_count))

number_of_words = []
for i in range(1, 11):
    count = 0
    for item in word_count:
        if word_count[item] == i:
            count += 1
    number_of_words.append(count)

plt.bar(range(1,11), number_of_words)

tokenizer = TfidfVectorizer().build_tokenizer()
regex_tokenizer = RegexpTokenizer(r'\w+')
# Use Unigrams,Bigrams as features
vectorizer = TfidfVectorizer(tokenizer=regex_tokenizer.tokenize,
                             stop_words=stops,
                             ngram_range=(1, 2),max_features=9000)

document_matrix = vectorizer.fit_transform(data.synopsis)

# We can now drop the synopsis column from the data
data.drop('synopsis', axis=1, inplace=True)

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
    if col not in ['id', 'title']:
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

# We can also drop id and title columns
data.drop(['id', 'title'], axis=1, inplace=True)

# Drop the rows that only have zeros in remaining genres
row_del_list = []
for i, row in data.iterrows():
    if row.sum() == 0:
        row_del_list.append(i)

data.drop(row_del_list, inplace=True)

# Save the data as pickle for later
if not os.path.exists('tfidf'):
    os.mkdir('tfidf')
with open('tfidf/tfidf_document_matrix.obj', 'wb') as f:
    pickle.dump(document_matrix, f)
with open('tfidf/labels.obj', 'wb') as f:
    pickle.dump(data, f)
with open('tfidf/tfidfvectorizer.obj', 'wb') as f:
    pickle.dump(vectorizer, f)