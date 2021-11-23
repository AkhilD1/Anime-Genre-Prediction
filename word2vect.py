import json
import os
import pickle
import re

from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import seaborn as sn
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


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
print(len(file_list))
# Create a pandas dataframe for the data
data = pd.DataFrame({'id': anime_id,
                     'title': anime_title,
                     'synopsis': anime_synopsis,
                     'genres': anime_genres})

print(data.info())

# Visualize synopsis length
synopsis_len = data.synopsis.str.len()
fig, ax = plt.subplots(1, 1)
synopsis_len.hist(
    bins=np.arange(min(synopsis_len) - 1, max(synopsis_len) + 1, 50), ax=ax)
ax.set_xlabel('Length of Synopsis')
ax.set_ylabel('Count')
fig.suptitle('Distribution of Synopsis Lengths')
fig.tight_layout()
fig.savefig('synopsis_length.png')

# Since each 'id' is unique, there are no duplicates in the data
# We can see that some entries do not have 'genres' populated
# We will be dropping these entries
data.dropna(inplace=True)

# We can also drop id and title columns as we will not be using them
data.drop(['id', 'title'], axis=1, inplace=True)

# We will generate One-hot encoding for the genres
for index, row in data.iterrows():
    for genre in row.genres:
        if not genre['name'] in data:
            data[genre['name']] = 0
        data[genre['name']][index] = 1

# We can now drop the genres column
data.drop('genres', axis=1, inplace=True)

# Visualizing data distribution

genre_count = data.sum(numeric_only=True)

# Plotting the data distribution
fig, axs = plt.subplots(2, 1)
axs[0].bar(genre_count.index, genre_count)
axs[0].set(xlabel='genres', ylabel='count')
axs[0].set_xticklabels(genre_count.index, rotation='vertical')

# Log Scale plot
axs[1].bar(genre_count.index, genre_count, log=True)
axs[1].set(xlabel='genres', ylabel='count (log scale)')
axs[1].set_xticklabels(genre_count.index, rotation='vertical')

fig.suptitle('Distribution of data across genres', y=0.98)
fig.set_size_inches(10, 7)
fig.tight_layout()
fig.savefig('anime_count.png')

# We see that the data is very imbalanced and genres have a very low number
# of entries. For the project we will keep only the genres which have more
# than 400 entries.
genre_count = genre_count[genre_count > 400]
data = data[data.columns.intersection(
    ['synopsis'] + genre_count.index.tolist())]

# Plotting the new data distribution
fig, axs = plt.subplots(1, 1)
axs.bar(genre_count.index, genre_count)
axs.set(xlabel='genres', ylabel='count')
axs.set_xticklabels(genre_count.index, rotation='vertical')
fig.tight_layout()
fig.savefig('new_anime_count.png')

# Drop the rows that only have zeros in remaining genres
data = data[data.sum(axis=1, numeric_only=True) != 0]

# Correlation between genres
fig, axs = plt.subplots(1, 1)
sn.heatmap(data[data.columns.intersection(genre_count.index)].corr(), ax=axs)
fig.show()
fig.suptitle('Correlation between different genres')
fig.tight_layout()
fig.savefig('genre_correlation.png')

# Preprocessing synopsis
# 1. Remove Source from synopsis
# Many entries have the source cited for the synopsis from various DBs
# eg. Source: Wikipedia, Source: ANN, (Source: ANN), Source AnimeDB, etc.
# Also, credit has been given to some who wrote the synopsis
# eg. [Written by XYZ]
if any(data.synopsis.apply(lambda x: 'Source' in x)):
    data.synopsis = data.synopsis.apply(
        lambda x: re.sub(
            r'[\[\(\s]*Source[:\s]*.*[\]\)\s]*|\[[W]ritten.*\]', '', x))
if any(data.synopsis.apply(lambda x: 'Source' in x)):
    raise Exception('Error cleaning')
if any(data.synopsis.apply(lambda x: '[Written' in x)):
    raise Exception('Error cleaning')

# 2. Replace numbers
# We replace numeric words with NUM
# eg: 1st, 4th, 1,000, 8.12, 100,000th, 1920s etc.
data.synopsis = data.synopsis.apply(
    lambda x: re.sub(r'\d[\d.,]*(st|rd|th)?|\d[\d.,]*(s)?', ' NUM ', x))

# Create a list of stop words
nltk.download('stopwords')
stops = set(stopwords.words("english"))
# Removing stop words from synopsis
data.synopsis = data.synopsis.apply(
    lambda x: " ".join([word.lower() for word in x.split(" ") if
                        word not in stops]))

# Tokenizing the synopsis into words
data.tokenized_synopsis = data.synopsis.apply(lambda x: x.split(" "))

# Implementing Word2Vec model
word2vec_model = Word2Vec(size=300, min_count=5)
word2vec_model.build_vocab(data.tokenized_synopsis)
word_dict = word2vec_model.wv.vocab


def word2vec_vectorizer(sentence):
    sentence = [word for word in sentence if word in word_dict]
    if len(sentence) == 0:
        print(sentence)
    sentence_vector = word2vec_model.wv[sentence].mean(axis=0)
    return sentence_vector


data.vectorized_synopsis = data.tokenized_synopsis.apply(
    lambda x: word2vec_vectorizer(x))


# Visualizing the Word2Vec using T-SNE
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    i = 0
    for word in model.wv.vocab:
        if i == 500:
            break
        i += 1
        tokens.append(model[word])
        labels.append(word)
    tsne_model = TSNE(perplexity=40, n_components=2,
                      init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title("Word Embeddings")
    plt.savefig("Word2_vec.png")
    

tsne_plot(word2vec_model)
data.drop('synopsis', axis=1, inplace=True)


# Save the data as pickle for later
if not os.path.exists('word2vec'):
    os.mkdir('word2vec')
with open('word2vec/vectorized_synopsis.obj', 'wb') as f:
    pickle.dump(data.vectorized_synopsis, f)
with open('word2vec/word2vec_vectorizer.obj', 'wb') as f:
    pickle.dump(word2vec_model, f)
