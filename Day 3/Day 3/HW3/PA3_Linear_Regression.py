#!/usr/bin/env python
# coding: utf-8

# # DSE 230: Programming Assignment 3 - Linear Regression
# 
# #### Tasks: 
# 
# - Linear Regression on the Boston Housing dataset.  
#   
# - Submission on Gradescope:
#   - Submit this Jupyter Notebook as a PDF to "PA3 Notebook"
#   - Convert this Notebook to a .py file and submit that to "PA3"
# 
# #### Due date: Friday 5/14/2021 at 11:59 PM PST
# 
# ---
# 
# Remember: when in doubt, read the documentation first. It's always helpful to search for the class that you're trying to work with, e.g. pyspark.sql.DataFrame. 
# 
# PySpark API Documentation: https://spark.apache.org/docs/latest/api/python/index.html
# 
# Spark DataFrame Guide:  https://spark.apache.org/docs/latest/sql-programming-guide.html
# 
# Spark MLlib Guide: https://spark.apache.org/docs/latest/ml-guide.html

# ### Import libraries/functions

# In[1]:


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np


# ### Initialize Spark
# * Initialize Spark with 2 cores

# In[2]:


conf = pyspark.SparkConf().setAll([('spark.master', 'local[2]'),
                                   ('spark.app.name', 'Python Spark SQL Demo')])
spark = SparkSession.builder.config(conf=conf).getOrCreate()


# ### Read the data from Boston_Housing.csv file
# * Print the number of rows in the dataframe

# In[3]:


# Create DataFrame based on contents of a JSON file
df = spark.read.csv("file:///home/work/Boston_Housing.csv", header = True, inferSchema = True)
print(df.count())


# ### Column names in file and their description
# 
# CRIM — per capita crime rate by town.
# 
# ZN — proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# INDUS — proportion of non-retail business acres per town.
# 
# CHAS — Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 
# NOX — nitrogen oxides concentration (parts per 10 million).
# 
# RM — average number of rooms per dwelling.
# 
# AGE — proportion of owner-occupied units built prior to 1940.
# 
# DIS — weighted mean of distances to five Boston employment centres.
# 
# RAD — index of accessibility to radial highways.
# 
# TAX — full-value property-tax rate per $10,000.
# 
# PTRATIO — pupil-teacher ratio by town.
# 
# BLACK — 1000(Bk — 0.63)² where Bk is the proportion of blacks by town.
# 
# LSTAT — lower status of the population (percent).
# 
# MV — median value of owner-occupied homes in $1000s. This is the target variable.

# ### See one row of the dataframe

# In[4]:


df.show(1)


# ### Helper function for filling columns using mean or median strategy

# In[5]:


from pyspark.ml.feature import Imputer

def fill_na(df, strategy):    
    imputer = Imputer(
        strategy=strategy,
        inputCols=df.columns, 
        outputCols=["{}_imputed".format(c) for c in df.columns]
    )
    
    new_df = imputer.fit(df).transform(df)
    
    # Select the newly created columns with all filled values
    new_df = new_df.select([c for c in new_df.columns if "imputed" in c])
    
    for col in new_df.columns:
        new_df = new_df.withColumnRenamed(col, col.split("_imputed")[0])
        
    return new_df


# ### Feature selection
# * Print schema to verify

# In[6]:


# These are the column names in the csv file as described above.
col_names = ['CRIM' , 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'BLACK', 'LSTAT', 'MV']

df.printSchema()


# ### Drop NA's in the target variable `MV`
# * Print the number of remaining rows

# In[7]:


df = df.dropna(subset = ['MV'])
print(df.count())


# ### Fill the NA's for remaining columns using a mean strategy
# * Use the `fill_na` function provided above

# In[8]:


fill_na(df, 'mean')


# ### Create feature vector using VectorAssembler
# 
# * Create a vector column composed of _all_ the features
# * Don't include the label "MV" here since label isn't a feature

# In[9]:


vectorAssembler = VectorAssembler(inputCols = ['CRIM' , 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'BLACK', 'LSTAT'], outputCol = 'features')
vhouse_df = vectorAssembler.transform(df)
vhouse_df = vhouse_df.select(['features', 'MV'])
vhouse_df.show(2)


# ### Print first 5 rows of the created dataframe

# In[10]:


print(vhouse_df.show(5))


# ### Rename the column `MV` to `Label`

# In[11]:


vhouse_df = vhouse_df.withColumnRenamed('MV', 'Label')


# ### Split the dataframe using the randomSplit() function 
#  * Train dataframe and test dataframe with a 75:25 split between them
#  * Use seed=42 as one the parameters of the randomSplit() function to maintain consistency among all submissions.
#  * Print the number of rows in train and test dataframes

# In[12]:


train_df, test_df = vhouse_df.randomSplit([0.75, 0.25], seed = 42)
print('number of rows in train dataframe:', train_df.count())
print('number of rows in test dataframe:',test_df.count())


# ### Use the StandardScaler to standardize your data.
# * **IMPORTANT** - Use only the training data for scaling
# * Standardize values to have zero mean and unit standard deviation

# In[13]:


scaler = StandardScaler(inputCol = "features", outputCol = "features_scaled", withStd = True, withMean = False)
scalerModel = scaler.fit(train_df)


# ### Scale your training and test data with the same mean and std that you'll get from the scaler.

# In[14]:


train_df_s = scalerModel.transform(train_df)
test_df_s  = scalerModel.transform(test_df)
train_df_s.show(2), test_df_s.show(2)


# ### Use `scaler_model.mean`, `scaler_model.std` to see the mean and std for each feature

# In[15]:


print(scalerModel.mean, '\n')
print(scalerModel.std)


# ### Select only the `features` and `label` columns from both train and test dataset

# In[16]:


lr_train_df = train_df_s.drop("features")
lr_test_df = test_df_s.drop("features")


# ### Show the first 5 rows of the resulting train dataframe

# In[17]:


lr_train_df.show(5)
lr_test_df.show(5)


# ### Use LinearRegression for training a regression model.
# * Use maxIter = 100.
# * Use the following values for regParam and elasticNetParam and see which one works better.
#   1. regParam = 0, elasticNetParam = 0
#   2. regParam = 0.3, elasticNetParam = 0.5
# 
# Look into the [API](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.ml.regression.LinearRegression.html) specification to get more details.

# In[18]:


lr_1 = LinearRegression(featuresCol = 'features_scaled', labelCol = 'Label', maxIter = 100, regParam = 0, elasticNetParam = 0)
lr_2 = LinearRegression(featuresCol = 'features_scaled', labelCol = 'Label', maxIter = 100, regParam = 0.3, elasticNetParam = 0.5)
lr_model_1 = lr_1.fit(lr_train_df)
lr_model_2 = lr_2.fit(lr_train_df)


# ### Print the coefficients and intercept of the linear regression model

# In[19]:


print("Coefficients: " + str(lr_model_1.coefficients))
print("Intercept: " + str(lr_model_1.intercept))

print("\nCoefficients: " + str(lr_model_2.coefficients))
print("Intercept: " + str(lr_model_2.intercept))


# ### Print the training results
# * Print the root mean squared error(RMSE) of the training
# * Print the coefficient of determination(r2) of the training

# In[20]:


trainingSummary_1 = lr_model_1.summary
print("RMSE: %f" % trainingSummary_1.rootMeanSquaredError)
print("r2: %f" % trainingSummary_1.r2)

trainingSummary_2 = lr_model_2.summary
print("\nRMSE: %f" % trainingSummary_2.rootMeanSquaredError)
print("r2: %f" % trainingSummary_2.r2)


# In[21]:


# RMSE (Root Mean Squared Error) is the error rate by the square root of MSE.
# R-squared (Coefficient of determination) represents the coefficient of how well the values fit compared to the original values. The value from 0 to 1 interpreted as percentages. The higher the value is, the better the model is.
print('1. regParam = 0, elasticNetParam = 0 works better')


# ### Test the model on test data
# * Print the RMSE and r2 on test data
# * Hint - Refer to [`RegressionEvaluator`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RegressionEvaluator.html)

# In[22]:


lr_predictions = lr_model_2.transform(lr_test_df)
lr_predictions.select("prediction", "Label", "features_scaled")

lr_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "Label", metricName = "r2")

test_result = lr_model_2.evaluate(lr_test_df)

print("RMSE: %f" % test_result.rootMeanSquaredError)
print("r2: %f" % lr_evaluator.evaluate(lr_predictions))


# ### Plot results on test data(using matplotlib)
# 
#  * In the test data, you have labels, and you also have predictions for each of the test data.
#  * Plot a scatter plot of the labels(in blue) and predictions(in red) on a single plot so that you can visualize how the predictions look as compared to the ground truth.
# 

# In[23]:


figure(figsize = (10, 10), dpi = 80)

plt.title('Results on test data')
plt.xlabel('row number')
plt.ylabel('labels and predictions')

x = np.arange(lr_predictions.count())
y1 = lr_predictions.select('Label').toPandas()
y2 = lr_predictions.select('prediction').toPandas()

plt.scatter(x, y1, color = 'blue', marker = '*', label = 'labels', alpha = 0.9, s = 60)
plt.scatter(x, y2, color = 'red', marker = 'o', label = 'predictions', alpha = 0.9, s = 60)

plt.xlim(-10, 110)
plt.ylim(0, 60)

plt.legend()
plt.grid()
plt.show()


# ### Add regularization to model
# * Try different values of regularization parameters `regParam` and `elasticNetParam` to see how performance changes.
# * Look into the API specification for [regParam](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html#pyspark.ml.regression.LinearRegression.regParam) and [elasticNetParam](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html#pyspark.ml.regression.LinearRegression.elasticNetParam) to get more details.

# In[24]:


# Fixed maxIter = 100 and elasticNetParam = 0.0
for i in [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
    lr_turn = LinearRegression(featuresCol = 'features_scaled', labelCol = 'Label', maxIter = 100, regParam = i, elasticNetParam = 0.0)
    lr_model_trun = lr_turn.fit(lr_train_df)
    trainingSummary_turn = lr_model_trun.summary
    print("regParam:", i)
    print("RMSE:", trainingSummary_turn.rootMeanSquaredError)
    print("r2:", trainingSummary_turn.r2, '\n')


# In[25]:


# Fixed maxIter = 100 and regParam = 0.0
for i in [0.0, 0.25, 0.5, 0.75, 1.0]:
    lr_turn_2 = LinearRegression(featuresCol = 'features_scaled', labelCol = 'Label', maxIter = 100, regParam = 0.5 , elasticNetParam = i)
    lr_model_trun_2 = lr_turn_2.fit(lr_train_df)
    trainingSummary_turn_2 = lr_model_trun_2.summary
    print("elasticNetParam:", i)
    print("RMSE:", trainingSummary_turn_2.rootMeanSquaredError)
    print("r2:", trainingSummary_turn_2.r2, '\n')


# In[26]:


print('regParam is bigger, the error rate is bigger, vice versa.')
print('elasticNetParam is bigger, the error rate is bigger, vice versa.')


# In[27]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

lr_3 = LinearRegression(featuresCol = 'features_scaled', labelCol = 'Label')

# Create ParamGrid for Cross Validation
lrparamGrid = (ParamGridBuilder()
             .addGrid(lr_3.regParam, [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0])
             .addGrid(lr_3.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])
             .addGrid(lr_3.maxIter, [1, 5, 10, 20, 50, 100, 200])
             .build())

lrevaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "Label", metricName = "rmse")

# Create 5-fold CrossValidator
lrcv = CrossValidator(estimator = lr_3, estimatorParamMaps = lrparamGrid, evaluator = lrevaluator, numFolds = 5)

# Run cross validations
lrcvModel = lrcv.fit(lr_test_df)
best_model = lrcvModel.bestModel


# In[28]:


best_reg_param = best_model._java_obj.getRegParam()
best_elasticnet_param = best_model._java_obj.getElasticNetParam()
best_max_Iter = best_model._java_obj.getMaxIter()
print('best regParam:', best_reg_param, '\nbest elasticNetParam:', best_elasticnet_param, '\nbest max iter:', best_max_Iter)


# In[29]:


Summary = best_model.summary
print("\nRMSE: %f" % Summary.rootMeanSquaredError)
print("r2: %f" % Summary.r2)


# ### Stop the spark session

# In[30]:


spark.stop()

