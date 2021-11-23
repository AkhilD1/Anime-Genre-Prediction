# Anime Genre Prediction

Project by Team 10: 

* Kartikeya Jain (kartikeyaj0)
* Yash Trivedi  (yashtrivedi2503)
* Akhil Reddy Dooliganti (AkhilD1)
 
 
In this project we try to predict the genre of an anime from its synopsis. We obtain the data from MyAnimeList API. MyAnimeList is a popular online anime and manga community and database. The API documentation can be found at https://myanimelist.net/apiconfig/references/api/v2. Please refer to the fetch_data.py script to look at how we use the API to obtain the data.

While we can query many details about each anime, we use only the following fields in this project:

* **id**: Unique Identifier
* **title**: Title of the anime
* **synopsis**: Short plot discription
* **genres**: List of genres of the anime

We will be using the **synopsis** from the above data to predict the genre. A complete list of genres along with their descriptions can be found at https://myanimelist.net/anime/genre/info.

As part of preprocessing, we remove the stopwords, use porter stemmer to stem the words in each synopsis and use TfidfVectorizer from scikit-learn to generate a document matrix consisting of the tokens and and their corresponding tf-idf scores. We also one-hot encode the genres. Please refer to tf_idf.py for more details.

Finally we compare the performance of the following classification techniques:

* Logistic Regression
* Linear SVC
* Random Forests

To measure the performance of our model, we check if any of the predicted labels / predicted genre with highest probability belongs to the original list of genres. We use K-Fold testing for the same. Please refer to the files logistic_regression.py, svc.py and random_forest.py.

We also include a Google Colab ipynb that combines the functionality from the above files, for ready reference. This also includes any visualizations relevant to the project. The following files contain the relevant visualizations from the ipynb file:

* anime_count.png
* genre_correlation.png
* new_anime_count.png
* synopsis_length.png

Please refer to the project report under doc/paper.pdf.

