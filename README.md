# Seoul Bike Rental

Project by Team 10: 

* Kartikeya Jain (kartikeyaj0)
* Yash Trivedi  (yashtrivedi2503)
* Akhil Reddy Dooliganti (AkhilD1)
 
 
In this project we will try to predict the number of bikes that will be rented every hour from a bike rental service in Seoul. This data set we will be using for this project is available on UCI ML repository https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand. The original source of this data is http://data.seoul.go.kr/

Rental bikes are important for enhancing mobility comfort in urban areas. It is therefore important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. This can be ensured by providing the city with a stable supply of rental bikes. Therefore, it is crucial to accurately predict the demand at each hour.

The dataset contains the following attributes:

* **Date**: year-month-day
* **Rented Bike count**: Count of bikes rented at each hour
* **Hour**: Hour of the day
* **Temperature**: Celsius
* **Humidity**: %
* **Wind Speed**: m/s
* **Visibility**: 10m
* **Dew point temperature**: Celsius
* **Solar radiation**: MJ/m2
* **Rainfall**: mm
* **Snowfall**: cm
* **Seasons**: Winter, Spring, Summer, Autumn
* **Holiday**: Holiday/No holiday
* **Functional Day**: NoFunc(Non Functional Hours), Fun(Functional hours)


Using this data we will try to predict the demand for rental bikes at any given time of day.

To achieve this task, we currently plan on using regression techniques like Linear Regression and Support Vector Regression. We plan on experimenting more with the dataset, and explore different techniques in the later stages.

To measure the performance of our model on the test set, we will use metrics like accuracy, error rate, and F-score.
