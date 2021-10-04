# seoul-bike-rental-project

Project Proposal for Team-10: Kartikeya Raj ( kartikeyaj0)
                              Yash Trivedi  ( yashtrivedi2503)
                              Akhil Reddy Dooliganti (AkhilD1)
 
 
**Q1) Project Title

--> "Seoul Bike Sharing Demand"

**Q2 & Q3) What data you’ll use and where you’ll get it? Description of the problem you’ll solve or the question you’ll investigate.

ANS) In this project we will try to predict the number of bikes that will be rented every hour from a bike rental service in Seoul. This data set we will be using for this project is available on UCI ML repository https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand 
The original source of this data is http://data.seoul.go.kr/
Rental bikes are important for enhancing mobility comfort in urban areas. It is therefore important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. This can be ensured by providing the city with a stable supply of rental bikes. Therefore, it is crucial to accurately predict the demand at each hour.

The dataset contains the following attributes:
Date: year-month-day
Rented Bike count: Count of bikes rented at each hour
Hour: Hour of the day
Temperature: Celsius
Humidity:  %
Wind Speed: m/s
Visibility: 10m
Dew point temperature: Celsius
Solar radiation: MJ/m2
Rainfall: mm
Snowfall: cm
Seasons: Winter, Spring, Summer, Autumn
Holiday: Holiday/No holiday
Functional Day: NoFunc(Non Functional Hours), Fun(Functional hours)

**Q4 Potential methods you will consider apply (these can change as you play with the data.

Ans-->Using this data we will try to predict the demand for rental bikes at any given time of day.  To achieve this, we will use regression techniques like Linear Regression and Support Vector Regression. And we also plan on experimenting more with the available dataset as we play around more with the data in later stages.

**Q5 How will you measure success?

Ans--> We will measure the accuracy of our model on the test set using metrics like accuracy, error rate, and F-score.

