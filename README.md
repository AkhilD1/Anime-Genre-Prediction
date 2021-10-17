# Anime Genre Prediction

Project by Team 10: 

* Kartikeya Jain (kartikeyaj0)
* Yash Trivedi  (yashtrivedi2503)
* Akhil Reddy Dooliganti (AkhilD1)
 
 
In this project we will try to predict the genre of anime from it's synopsis. The data we will be using for this project is obtained from MyAnimeList API https://myanimelist.net/apiconfig/references/api/v2.

A sample query (as shown in the documentation) to this api would be:

curl 'https://api.myanimelist.net/v2/anime/30230?fields=id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,pictures,background,related_anime,related_manga,recommendations,studios,statistics' \
-H 'Authorization: Bearer YOUR_TOKEN'


While we can get lots of details form the api, we plan on utilizing the following:

* **id**: Unique Identifier
* **title**: Title of the anime
* **alternate_titles**: Alternate titles of the anime
* **synopsis**: Short plot discription
* **genres**: Genre of the anime

We will be using the **synopsis** from this above data to predict the genre. A complete list of genres can be found at https://myanimelist.net/anime.php

We currently plan on using techniques like Naive Bayes, TF-IDF, n-grams and Word2Vec to achieve the task.

To measure the performance of our model on the test set, we will use metrics like accuracy, error rate, precision and recall, and F-scores.

