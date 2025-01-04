## **Abstract**

This project analyses and predicts US airline flight fares (1993–2024) using a dataset of
245,955 instances and 23 features. Key factors like market competition, distance, seasonal
variations, and airline-specific pricing strategies are explored. Statistical analysis and machine
learning models, including Linear Regression, Random Forest, Gradient Boosting, and
XGBoost, are employed to identify fare patterns and forecast prices. The project involves data
preprocessing, exploratory analysis, feature engineering, and model development.
Performance metrics like R-squared, MAE, and RMSE guide model evaluation. Insights from
feature importance analysis reveal critical variables impacting fare predictions.

## **Introduction**

Air travel plays a crucial role in modern transportation, with airline fares influenced by factors
such as route distance, competition, seasonal demand, and airline-specific strategies.
Understanding these pricing patterns is essential for consumers seeking affordable travel
options and for airlines aiming to optimize revenue strategies. This study examines US airline
flight fares from 1993 to 2024 to uncover trends and build predictive tools using machine
learning.

## **Data Preprocessing**

### **Handling Missing Values:**

Imputing Missing Values Using Grouped Data:
Missing numerical values (e.g., fare_lg, fare_low) were filled using the mean of grouped data
based on city1 and city2. Categorical values (e.g., carrier_lg, carrier_low) were imputed using
the mode within each group. Missing geocoded city data (Geocoded_City1, Geocoded_City2)
was replaced with corresponding city names for consistency. This method ensured a robust
imputation strategy tailored to data characteristics.


### **Outlier Detection and removal:**

**Data Exploration and Understanding:**

Loaded and explored the dataset to identify numeric columns, such as fare,
passengers, and nsmiles, which were key for analyzing fare trends and identifying
outliers.

**Box Plot Visualization:**

Created box plots for each numeric column to visualize the spread of the data and
detect potential outliers, which is essential for understanding the data distribution.

**Outlier Detection Using IQR:**

The Interquartile Range (IQR) method was applied to each numeric column to detect
outliers. Outliers were defined as values falling below Q1−1.5×IQRQ1 - 1.5 \times
IQRQ1−1.5×IQR or above Q3+1.5×IQRQ3 + 1.5 \times IQRQ3+1.5×IQR.


**Outlier Removal Process:**

Filtered out data points identified as outliers using the IQR method, cleaning the
dataset by removing extreme values that could skew the analysis.
Prepared Cleaned Data for Analysis:

Finalized a cleaned dataset post-outlier removal, ensuring it was ready for subsequent
steps like exploratory data analysis and model building with minimized noise and more
accurate data distribution.


### **Encoding categorical variables:**
Before using feature selection, we have to encode the categorical variables to numeric.
LabelEncoder() is used since the variables are nominal.

### **Feature selection:**
There are 20 variables in the dataset for predicting the variable ‘fare’. Every variable is not
correlated to the ‘fare’ variable. So we have to find which variables are important for prediction.
Hence feature selection is used and it also helps in improving the accuracy of the machine
learning models.


#### **Correlation Matrix Analysis**

Correlation coefficient is used to find the relationship between the two variables. The values range
from -1 to 1.

- Strong relationship: r <= -0.8 and r>=0.8
- Weak relationship: -0.8 < r < -0.5 and 0.5 < r < 0.8
- No relationship: -0.5 <= r >= 0.5
   
Selected feature names : ['fare', 'fare_lg', 'fare_low', 'nsmiles', 'lf_ms', 'large_ms', 'Year',
'passengers', 'carrier_lg', 'city1']

#### **Lasso Regularisation**

Lasso Regularization (Least Absolute Shrinkage and Selection Operator) is a regularization
technique used in linear regression and other machine learning models to prevent overfitting
and perform feature selection.
Lasso () function hyperparameter alpha of 0.00001 is used after splitting the dataset into train
and test set. Lasso1.coef_ gives the coefficient of all the variables and it can be used to find
the feature importance.
On plotting the graph with the coefficients of variables, it can be observed that quarter,
large_ms, fare_lg, carrier_low, if_ms and fare_low are the important features for prediction.

#### **Bidirectional feature selection**

Bidirectional feature selection is a hybrid approach to selecting features for machine learning
models. It combines both forward and backward selection strategies to optimize the selection
of relevant features from the dataset.
SequentialFeatureSelector from the mlxtend package is used for bidirectional feature
selection. Hyperparameters - The function is set as Linear Regression, and the forward and
floating parameters are set as True, enabling bidirectional feature selection.

#### **Forward feature selection**

Forward Feature Selection is a stepwise approach to selecting a subset of relevant features
for building a predictive model. It's a greedy algorithm that begins with an empty set of features
and adds one feature at a time, selecting the feature that improves the model's performance
the most at each step.
SequentialFeatureSelector from the mlxtend package is used for forward feature selection.
Hyperparameters - The function is set as Linear Regression, and the forward is set as True
and floating parameters are set as False, enabling bidirectional forward selection.

### **Linear Regression:**

#### **Objective:**

Linear Regression was used to predict airline fares by modeling the linear relationship between a dependent variable (fare) and independent features.
 
#### **Data Preparation:**

The dataset was split into training and testing sets, and MinMaxScaler was applied to normalize feature scales due to Linear Regression's sensitivity to data scaling.

#### **Model Training:**

The SGDRegressor from the sklearn library was employed for training. GridSearchCV optimized hyperparameters by testing combinations to minimize overfitting and improve RMSE.

#### **Evaluation Metrics:**

-Model 1: All variables included, RMSE: 17.318
-Model 2: Selected variables used, RMSE: 17.264
-Model 3: Selected variables with GridSearchCV, RMSE: 17.263


#### **Conclusion:**

Linear Regression achieved a RMSE of 17.263, demonstrating its effectiveness as a baseline model.


### **Support Vector Regression (SVM)**

#### **Objective:**

SVM was utilized to predict airline fares by modeling the relationship between features and the target variable, using a regression approach.

#### **Data Preparation:**

The dataset was split into training and testing sets, and MinMaxScaler was applied to
normalize feature scales due to SVM's sensitivity to scaling.

#### **Model Training:**

SVR (Support Vector Regression) from sklearn was employed for training. GridSearchCV was used to optimize hyperparameters, enhancing performance.

#### **Evaluation Metrics:**

- Model 1: All variables included, RMSE: 17.728
- Model 2: Selected variables used, RMSE: 17.737
- Model 3: Selected variables with GridSearchCV, RMSE: 17.737

#### **Conclusion:**

While SVM performed reliably, it resulted in a higher RMSE (17.737) compared to Linear Regression, making it less optimal for this dataset.


### **K-Nearest Neighbours (KNN)**

#### **Objective:**

KNN was applied to predict airline fares by averaging the values of the nearest data points in feature space.

#### **Data Preparation:**
The dataset was split into training and testing sets, with MinMaxScaler used to normalize feature scales due to KNN's sensitivity to scaling.

#### **Model Training:**
KNeighborsRegressor from sklearn was utilized for training. GridSearchCV optimized hyperparameters, improving model performance.

#### **Evaluation Metrics:**

- Model 1: All variables included, RMSE: 22.545
- Model 2: Selected variables used, RMSE: 14.84
- Model 3: Selected variables with GridSearchCV, RMSE: 14.674

#### **Conclusion:**
KNN proved to be an efficient model for this dataset, achieving a lower RMSE of 14.674 compared to Linear Regression and SVM, demonstrating its effectiveness.

**Gradient Boosting**

**Objective:**

Gradient Boosting was employed to predict average flight fares between U.S. city pairs based on historical data (1993–2024). The goal was to leverage route-specific attributes (distance, passenger volume, market share, etc.) and optimize model performance through hyperparameter tuning.

**Data Preprocessing:**

- Missing values were dropped for key features.
- Scaling: Numerical features were standardized using StandardScaler for consistent scaling.

**Model Training:**

- Model: Gradient Boosting Regressor from sklearn.ensemble for handling non-linear relationships and feature interactions.
- Hyperparameter Tuning: GridSearchCV optimized:
○ n_estimators (number of stages), learning_rate (step size), and max_depth
(tree depth).
- Training: 80-20 train-test split with 5-fold cross-validation ensured robust
performance.

**Evaluation Metrics:**

- MSE: Lower error (373.93) indicated better predictions.
- R²: A value of 19.34 highlighted effective data relationship capture.

**Model Insights:**

Feature Importance:
- Top predictors: nsmiles (distance) and fare_low (lowest fare offered).
- Temporal features (Year, quarter) moderately influenced fare variations.


**Visualizations:**

1. Feature Importance:
Horizontal bar chart highlighting nsmiles, fare_low, and passengers as key
predictors.
2. Actual vs. Predicted Fares:
Scatter plot showing alignment of predicted fares with actual values, indicating model accuracy.
3. Error Distribution:
Histogram displaying a symmetric, narrow error spread around zero, confirming well- calibrated predictions.

**Conclusion:**

The Gradient Boosting model effectively predicts flight fares, demonstrating strong R² and low error metrics. Key predictors like route distance and competitive pricing underline its ability to
capture fare dynamics. This model benefits airlines and passengers by offering actionable insights into pricing trends and planning budgets efficiently.

