# Data Science Projects (Readme is a work in progress right now)
This repository contains *Data Science* projects that I worked on while learning and understanding data science & machine learning models.

## Table of contents
1. [Simple Linear Regression](#slr)
2. [Multiple Linear Regression](#mlr)
3. [Logistic Regression](#lr)
4. [Cluster Analysis](#ca)
5. [K Nearest Neighbour](#knn)
6. [Decision Tree](#dt)
7. [Random Forest](#rf)
8. [Support Vector Machine](#svm)
9. [Principal Component Analysis](#rf)
10. [Naive Bayes](#nb)
11. [Recommendation system](#rs)
12. [Time Series](#ts)
13. [Neural Networks](#nn)
14. [Text Mining](#tm)




### ‣ [Simple Linear Regression](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Simple%20linear%20Regression) <a name="slr"></a>
  * [Delivery Time Prediction](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Simple%20linear%20Regression/Delivery%20Time%20SLR%20model.ipynb): This model has just 1 feature i.e. Sorting Time. The target was to predict the Delivery time using the 'Sorting Time' variable. The given data was transformed to find the best suited model. 

    * *Metrics used for checking accuracy: R-squared, RMSE (root mean squared error)*
    * *Tools used: pandas, numpy, statsmodels, matplotlib, seaborn*

  * [Salary Prediction](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Simple%20linear%20Regression/Salary%20SLR%20model.ipynb): Used the x variable 'YearsExperience' to predict the y variable 'Salary'. The data had a very linear correlation and therefore no transformations were required. This was proved by making different transformation and checking the metrics.

    * *Metrics used for checking accuracy: R-squared, RMSE (root mean squared error)*
    * *Tools used: pandas, numpy, statsmodels, matplotlib, seaborn*

### ‣ [Multiple Linear Regression](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Multiple%20linear%20regression) <a name="mlr"></a>
  * [Prediciton of Profit on 50_Startups data](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Multiple%20linear%20regression/MLR_50_Startups.ipynb): This model made on the famous 50_Startups dataset has four x variables and one y variable. This data had a categorical variable, which had to be One Hot Encoded to get the 'States' variable in binary form. In this model, I used the *statsmodels.formula.api.ols* algorithm to make the Multiple Linear Regression, but this algorithm does not remove the influencing datas and variables by itself. Therefore I have shown different methods to make the model better in my code.

    * *Metrics used for checking accuracy: Adjusted R-squared, AIC, P-values, RMSE (root mean squared error)*
    * *Tools used: pandas, numpy, statsmodels, matplotlib, seaborn*

  * [Prediction of Price in ToyotaCorrolla data](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Simple%20linear%20Regression/Salary%20SLR%20model.ipynb): Predicting the price variable with the help of the age, km, hp, cc, doors, gear, quarterly tax & weight variables. Here I used the *statsmodels.formula.api.ols* algorithm which does not optimize the model by itself. Therefore we have used different methods to clean the data and make the model better.

    * *Metrics used for checking accuracy: Adjusted R-squared, AIC, P-values, RMSE (root mean squared error)*
    * *Tools used: pandas, numpy, statsmodels, matplotlib, seaborn*
    
### ‣ [Logistic Regression](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Logistic%20Regression) <a name="lr"></a>
  * [Client term deposit prediciton on bank_data](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Simple%20linear%20Regression/Salary%20SLR%20model.ipynb): This model predicts the 'y' variable using logistic regression. The 'y' variable tells us whether the client has subscribed to a term deposit or not. The prediction is based on the different x variables given in the dataset. most of the independant variables were categorical & therefore had to be encoded. I encoded some categories to a binary version and others to One Hot Encoding. After the categorical data was dealt with, I was left with a lot of features, so I used the mutual info classif method to get the best features and built the model from there. I also made the **logit model** to see how it differs from the **logistic regression** model.

    * *Metrics used for checking accuracy: Recall, Precision, Accuracy using confusion matrix*
    * *Tools used: pandas, numpy, statsmodels, sklearn, matplotlib, seaborn, pandas_profiling (for EDA)*

### ‣ [Cluster Analysis](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Clustering) <a name="ca"></a>
  * [Clustering groups in EastWestAirlines data](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Clustering/Airline_customer_miles.ipynb): I used three clustering models to get different groups of customers in the data. The **first** model used is the *hierarchial clustering* model which uses the *Euclidean* distance to pair / group the data. We have to give the number of clusters so that the algorithm can make a cut in the hierarchy to give us the specified number of clusters. The **second** model used is the *K-Means clustering* model which selects centeroids in the data and use Euclidean distances to pair the data in a group. The K-Means algorithm requires the value of K, which I found out using elbow method. The **third** model is the *DBSCAN* model which again pairs the group of data using distance measures, usually the Euclidean distance. DBSCAN tries to find the dense regions based on the epsilon value given by us.

    * *Tools used: pandas, numpy, statsmodels, sklearn, scipy, matplotlib*

  * [Clustering groups of states in USA accoriding to crime](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Clustering/USA_crime_data_clustering.ipynb): I used three clustering models to group different states in the USA according to the crimes. The **first** model used is the *hierarchial clustering* model which uses the *Euclidean* distance to group the data. We have to give the number of clusters so that the algorithm can make a cut in the hierarchy to give us the specified number of clusters. The **second** model used is the *K-Means clustering* model which selects centeroids in the data and use euclidean distances to pair the data in a group. The K-Means algorithm requires the value of K, which I found out using elbow method. The **third** model is the *DBSCAN* model which again pairs the group of data using distance measures, usually the Euclidean distance. DBSCAN tries to find the dense regions based on the epsilon value given by us.

    * *Tools used: pandas, numpy, statsmodels, sklearn, scipy, matplotlib*



### ‣ [K Nearest Neighbour](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/K%20Nearest%20neighbour) <a name="knn"></a>
### ‣ [Decision Tree](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Decision%20Tree) <a name="dt"></a>
### ‣ [Random Forest](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Random%20Forest) <a name="rf"></a>
### ‣ [Support Vector Machine](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Support%20Vector%20Machine) <a name="svm"></a>
### ‣ [Principal Component Analysis](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/PCA) <a name="pca"></a>
### ‣ [Naive Bayes](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Naive%20Bayes) <a name="nb"></a>
### ‣ [Recommendation system](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Recommendation%20system) <a name="rs"></a>
### ‣ [Time Series](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Time%20Series) <a name="ts"></a>
### ‣ [Neural Networks](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Neural%20Networks) <a name="nn"></a>
### ‣ [Text Mining](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Text%20mining) <a name="tm"></a>
