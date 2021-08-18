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
<a title="Jsmura, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Dispersion-con-regresion.png"><img width="350" alt="Dispersion-con-regresion" src="https://upload.wikimedia.org/wikipedia/commons/d/de/Dispersion-con-regresion.png"></a>
 
 Simple linear regression model fits the best line in the given data. This best line actually tries to find the best relation between the x variable and y variable. This is basically just a simple line, for which the formula in mathematics is y = mx + c, where m is the slope, c is the intercept, x is the input (independant) feature and y is the output (dependant) feature. Accuracy of the model can be checked using the R-squared value, which tells us how much variance on both sides of the line is being captured by the line fitted by the model. Higher the R-squared value, better the model. I have used 2 methods to make the model. First is using the *statsmodels.fomrmula.api.ols* method and second is the *scipy.stats.linregress* method.
 
  * [Delivery Time Prediction](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Simple%20linear%20Regression/Delivery%20Time%20SLR%20model.ipynb): This dataset has just 1 feature as x variable i.e. Sorting Time. The target was to predict the Delivery time using the 'Sorting Time' variable. The given data was transformed to find the best suited model. 

    * *Metrics used for checking accuracy: R-squared, RMSE (root mean squared error)*
    * *Tools used: pandas, numpy, statsmodels, matplotlib, seaborn*

  * [Salary Prediction](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Simple%20linear%20Regression/Salary%20SLR%20model.ipynb): Used the x variable 'YearsExperience' to predict the y variable 'Salary'. The data had a very linear correlation and therefore no transformations were required. This was proved by making different transformation and checking the metrics.

    * *Metrics used for checking accuracy: R-squared, RMSE (root mean squared error)*
    * *Tools used: pandas, numpy, statsmodels, matplotlib, seaborn*

* [Salary prediction using scipy.stats.linregress](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Simple%20linear%20Regression/linear_regression_using_scipy.ipynb): This was a very straightforward method where I used scipy.stats.linregress to get the best fitted line in the same Salary data as above.

    * *Metrics used for checking accuracy: R-squared*
    * *Tools used: pandas, numpy, scipy, matplotlib*


### ‣ [Multiple Linear Regression](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Multiple%20linear%20regression) <a name="mlr"></a>
  * [Prediciton of Profit on 50_Startups data](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Multiple%20linear%20regression/MLR_50_Startups.ipynb): This model made on the famous 50_Startups dataset has four x variables and one y variable. This data had a categorical variable, which had to be One Hot Encoded to get the 'States' variable in binary form. In this model, I used the *statsmodels.formula.api.ols* algorithm to make the Multiple Linear Regression, but this algorithm does not remove the influencing datas and variables by itself. Therefore I have shown different methods to make the model better in my code.

    * *Metrics used for checking accuracy: Adjusted R-squared, AIC, P-values, RMSE (root mean squared error)*
    * *Tools used: pandas, numpy, statsmodels, matplotlib, seaborn*

  * [Prediction of Price in ToyotaCorrolla data](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Multiple%20linear%20regression/toyota_corolla_MLR.ipynb): Predicting the price variable with the help of the age, km, hp, cc, doors, gear, quarterly tax & weight variables. Here I used the *statsmodels.formula.api.ols* algorithm which does not optimize the model by itself. Therefore we have used different methods to clean the data and make the model better.

    * *Metrics used for checking accuracy: Adjusted R-squared, AIC, P-values, RMSE (root mean squared error)*
    * *Tools used: pandas, numpy, statsmodels, matplotlib, seaborn*
    
### ‣ [Logistic Regression](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Logistic%20Regression) <a name="lr"></a>
  * [Client term deposit prediciton on bank_data](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Logistic%20Regression/bank_data_logistic_regression.ipynb): This model predicts the 'y' variable using logistic regression. The 'y' variable tells us whether the client has subscribed to a term deposit or not. The prediction is based on the different x variables given in the dataset. most of the independant variables were categorical & therefore had to be encoded. I encoded some categories to a binary version and others to One Hot Encoding. After the categorical data was dealt with, I was left with a lot of features, so I used the mutual info classif method to get the best features and built the model from there. I also made the **logit model** to see how it differs from the **logistic regression** model.

    * *Metrics used for checking accuracy: Recall, Precision, Accuracy using confusion matrix*
    * *Tools used: pandas, numpy, statsmodels, sklearn, matplotlib, seaborn, pandas_profiling (for EDA)*

### ‣ [Cluster Analysis](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Clustering) <a name="ca"></a>
<a title="Chire, CC BY-SA 3.0 &lt;https://creativecommons.org/licenses/by-sa/3.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:EM-Gaussian-data.svg"><img width="256" alt="EM-Gaussian-data" src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/EM-Gaussian-data.svg/256px-EM-Gaussian-data.svg.png" ></a>

I used three clustering models to get different groups in the data. The **first** model used is the *hierarchial clustering* model which uses the *Euclidean* distance to pair / group the data. We have to give the number of clusters so that the algorithm can make a cut in the hierarchy to give us the specified number of clusters. The **second** model used is the *K-Means clustering* model which selects centeroids in the data and use Euclidean distances to pair the data in a group. The K-Means algorithm requires the value of K, which I found out using elbow method. The **third** model is the *DBSCAN* model which again pairs the group of data using distance measures, usually the Euclidean distance. DBSCAN tries to find the dense regions (data points near each other) based on the epsilon value given by us.

  * [Clustering groups in EastWestAirlines data](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Clustering/Airline_customer_miles.ipynb): Used all three models mentioned above to classify the type of customer in the airline data based on the different features given.

    * *Tools used: pandas, numpy, statsmodels, sklearn, scipy, matplotlib*

  * [Clustering groups of states in USA accoriding to crime](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/Clustering/USA_crime_data_clustering.ipynb): different states in USA can be classified into groups based on the crime-rates. For this, I used all three models mentioned above to classify the states in different groups.

    * *Tools used: pandas, numpy, statsmodels, sklearn, scipy, matplotlib*

### ‣ [K Nearest Neighbour](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/K%20Nearest%20neighbour) <a name="knn"></a>
<a title="Agor153, CC BY-SA 3.0 &lt;https://creativecommons.org/licenses/by-sa/3.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Map1NNReducedDataSet.png"><img width="512" alt="Map1NNReducedDataSet" src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Map1NNReducedDataSet.png"></a>

K Nearest Neighbour (KNN) algorithm is a supervised algorithm that can help solving both classification and regression problems. KNN needs us to specify the value of k which tells it how many neighbours of a data point to consider as nearest neighbours and include them in a group. The value of k can be optimized by trying different values. KNN calculates the distances between the data points using different distance metrics like *"manhattan","euclidean","minkowski"* etc. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html" target="_blank">(find more info on distances here)</a>.

  * [Classifying glass type using KNN](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/K%20Nearest%20neighbour/KNN_Glass.ipynb): This model uses KNN algorithm to differentiate the type of glasses using the values of it's elements. Grid search was used to find the best value of k. Also, I used the *Euclidean distance* as the measure for distance.

    * *Metrics used for checking accuracy: Accuracy using confusion matrix*
    * *Parameter tuning technique: GridSearchCV (sklearn.model_selection)*
    * *Tools used: pandas, numpy, sklearn, matplotlib, seaborn*

  * [Classifying animal type in a zoo using KNN](https://github.com/FloydFernandes/data-science-portfolio/blob/main/projects/K%20Nearest%20neighbour/KNN_Zoo.ipynb): Using information about animals like, whether it has hairs, feathers, teeth, tail and other features like whether it gives eggs, milk or how many legs it has and some more information, our KNN model tries to classify or group the similar animals together. GridSearchCV helped us find the optimum value for k which in this case was 5. *Euclidean distance* was used as the parameter for distance.

    * *Metrics used for checking accuracy: Accuracy using confusion matrix*
    * *Parameter tuning technique: GridSearchCV (sklearn.model_selection)
    * *Tools used: pandas, numpy, sklearn, matplotlib, seaborn*

### ‣ [Decision Tree](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Decision%20Tree) <a name="dt"></a>
<a title="Pkuwangyan06, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Decision_Trees.png"><img width="400" alt="Decision Trees" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Decision_Trees.png/512px-Decision_Trees.png"></a>



### ‣ [Random Forest](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Random%20Forest) <a name="rf"></a>
### ‣ [Support Vector Machine](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Support%20Vector%20Machine) <a name="svm"></a>
<a title="Qluong2016, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Support_vector_machine.jpg"><img width="450" alt="Support vector machine" src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Support_vector_machine.jpg"></a>
### ‣ [Principal Component Analysis](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/PCA) <a name="pca"></a>
### ‣ [Naive Bayes](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Naive%20Bayes) <a name="nb"></a>
### ‣ [Recommendation system](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Recommendation%20system) <a name="rs"></a>
### ‣ [Time Series](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Time%20Series) <a name="ts"></a>
<a title="Cike, CC BY-SA 3.0 &lt;https://creativecommons.org/licenses/by-sa/3.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:TimeSerie-3.png"><img width="512" alt="TimeSerie-3" src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/TimeSerie-3.png/512px-TimeSerie-3.png"></a>
### ‣ [Neural Networks](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Neural%20Networks) <a name="nn"></a>
<a title="Akritasa, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Two-layer_feedforward_artificial_neural_network.png"><img width="512" alt="Two-layer feedforward artificial neural network" src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Two-layer_feedforward_artificial_neural_network.png/512px-Two-layer_feedforward_artificial_neural_network.png"></a>
### ‣ [Text Mining](https://github.com/FloydFernandes/data-science-portfolio/tree/main/projects/Text%20mining) <a name="tm"></a>
