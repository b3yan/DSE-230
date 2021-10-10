#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Load-Libraries" data-toc-modified-id="Load-Libraries-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load Libraries</a></span></li><li><span><a href="#Initialize-pyspark-framework" data-toc-modified-id="Initialize-pyspark-framework-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Initialize pyspark framework</a></span></li><li><span><a href="#Load-data" data-toc-modified-id="Load-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#Create-functions" data-toc-modified-id="Create-functions-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Create functions</a></span></li><li><span><a href="#Data-preparation-process" data-toc-modified-id="Data-preparation-process-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Data preparation process</a></span><ul class="toc-item"><li><span><a href="#Merge-datasets" data-toc-modified-id="Merge-datasets-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Merge datasets</a></span></li><li><span><a href="#Data-cleaning" data-toc-modified-id="Data-cleaning-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Data cleaning</a></span><ul class="toc-item"><li><span><a href="#Drop-unused-features" data-toc-modified-id="Drop-unused-features-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Drop unused features</a></span></li><li><span><a href="#Drop-duplicated-values" data-toc-modified-id="Drop-duplicated-values-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>Drop duplicated values</a></span></li><li><span><a href="#Drop-nullable-values" data-toc-modified-id="Drop-nullable-values-5.2.3"><span class="toc-item-num">5.2.3&nbsp;&nbsp;</span>Drop nullable values</a></span></li><li><span><a href="#Convert-numeric-to-double" data-toc-modified-id="Convert-numeric-to-double-5.2.4"><span class="toc-item-num">5.2.4&nbsp;&nbsp;</span>Convert numeric to double</a></span></li><li><span><a href="#Convert-datetime-to-date" data-toc-modified-id="Convert-datetime-to-date-5.2.5"><span class="toc-item-num">5.2.5&nbsp;&nbsp;</span>Convert datetime to date</a></span></li><li><span><a href="#Convert-date-to-quarter" data-toc-modified-id="Convert-date-to-quarter-5.2.6"><span class="toc-item-num">5.2.6&nbsp;&nbsp;</span>Convert date to quarter</a></span></li><li><span><a href="#Add-id-column" data-toc-modified-id="Add-id-column-5.2.7"><span class="toc-item-num">5.2.7&nbsp;&nbsp;</span>Add id column</a></span></li></ul></li><li><span><a href="#Feature-engineering" data-toc-modified-id="Feature-engineering-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Feature engineering</a></span></li><li><span><a href="#Feature-visualization" data-toc-modified-id="Feature-visualization-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Feature visualization</a></span></li><li><span><a href="#Create-feature-vector" data-toc-modified-id="Create-feature-vector-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Create feature vector</a></span></li><li><span><a href="#Split-datasets" data-toc-modified-id="Split-datasets-5.6"><span class="toc-item-num">5.6&nbsp;&nbsp;</span>Split datasets</a></span></li><li><span><a href="#Standardize-data" data-toc-modified-id="Standardize-data-5.7"><span class="toc-item-num">5.7&nbsp;&nbsp;</span>Standardize data</a></span></li></ul></li><li><span><a href="#Modeling" data-toc-modified-id="Modeling-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modeling</a></span><ul class="toc-item"><li><span><a href="#Ridge-regression" data-toc-modified-id="Ridge-regression-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Ridge regression</a></span></li><li><span><a href="#Lasso-regression" data-toc-modified-id="Lasso-regression-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Lasso regression</a></span></li><li><span><a href="#Hyperparameter-tuning" data-toc-modified-id="Hyperparameter-tuning-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Hyperparameter tuning</a></span></li><li><span><a href="#Gradient-boosted-tree-regression" data-toc-modified-id="Gradient-boosted-tree-regression-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Gradient-boosted tree regression</a></span></li><li><span><a href="#Decision-tree-regression" data-toc-modified-id="Decision-tree-regression-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Decision tree regression</a></span></li><li><span><a href="#Random-forest-regression" data-toc-modified-id="Random-forest-regression-6.6"><span class="toc-item-num">6.6&nbsp;&nbsp;</span>Random forest regression</a></span></li></ul></li><li><span><a href="#Results" data-toc-modified-id="Results-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Results</a></span></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Conclusion</a></span></li><li><span><a href="#Stop-the-spark-session" data-toc-modified-id="Stop-the-spark-session-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Stop the spark session</a></span></li></ul></div>

# # Load Libraries

# In[1]:


# Import PySpark related modules
import pyspark
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions
from pyspark.sql.functions import lit, desc, col, size, array_contains, isnan, udf, hour, array_min, array_max, countDistinct, quarter
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import Imputer
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import *

# Import other modules not related to PySpark
import os
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
import math
from IPython.core.interactiveshell import InteractiveShell
from datetime import *
import statistics as stats

# This helps auto print out the items without explixitly using 'print'
InteractiveShell.ast_node_interactivity = "all" 

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns


# # Initialize pyspark framework

# In[2]:


conf = pyspark.SparkConf().setAll([('spark.master', 'local[4]'),
                                   ('spark.app.name', 'Python Spark SQL Demo')])
spark = SparkSession.builder.config(conf=conf).getOrCreate()


# # Load data

# In[3]:


get_ipython().system('pwd')


# In[4]:


get_ipython().system('ls')


# In[5]:


get_ipython().system('hadoop fs -mkdir /data')


# In[6]:


get_ipython().system('hadoop fs -copyFromLocal products_dataset.csv /data')


# In[7]:


get_ipython().system('hadoop fs -copyFromLocal product_category_name_translation.csv /data')


# In[8]:


get_ipython().system('hadoop fs -copyFromLocal customers_dataset.csv /data')


# In[9]:


get_ipython().system('hadoop fs -copyFromLocal sellers_dataset.csv /data')


# In[10]:


get_ipython().system('hadoop fs -copyFromLocal orders_dataset.csv /data')


# In[11]:


get_ipython().system('hadoop fs -copyFromLocal order_payments_dataset.csv /data')


# In[12]:


get_ipython().system('hadoop fs -copyFromLocal order_items_dataset.csv /data')


# In[13]:


get_ipython().system('hadoop fs -copyFromLocal geolocation_dataset.csv /data')


# In[14]:


get_ipython().system('hadoop fs -copyFromLocal customer_reviews_dataset.csv /data')


# In[15]:


DATA_PATH="hdfs:///data/"
products_dataset = spark.read.csv(DATA_PATH+"products_dataset.csv", header=True, inferSchema = True)
product_category_name_translation = spark.read.csv(DATA_PATH+"product_category_name_translation.csv", header=True, inferSchema = True)
customers_dataset = spark.read.csv(DATA_PATH+"customers_dataset.csv", header=True, inferSchema = True)
sellers_dataset = spark.read.csv(DATA_PATH+"sellers_dataset.csv", header=True, inferSchema = True)
orders_dataset = spark.read.csv(DATA_PATH+"orders_dataset.csv", header=True, inferSchema = True)
order_payments_dataset = spark.read.csv(DATA_PATH+"order_payments_dataset.csv", header=True, inferSchema = True)
order_items_dataset = spark.read.csv(DATA_PATH+"order_items_dataset.csv", header=True, inferSchema = True)
geolocation_dataset = spark.read.csv(DATA_PATH+"geolocation_dataset.csv", header=True, inferSchema = True)
customer_reviews_dataset = spark.read.csv(DATA_PATH+"customer_reviews_dataset.csv", header=True, inferSchema = True)


# In[16]:


# Show sample data in each dataset
products_dataset.show(3)
product_category_name_translation.show(3)
customers_dataset.show(3)
sellers_dataset.show(3)
orders_dataset.show(3)
order_payments_dataset.show(3)
order_items_dataset.show(3)
geolocation_dataset.show(3)
customer_reviews_dataset.show(3)


# # Create functions

# In[17]:


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

def plot_function(predictions):
    figure(figsize = (20, 20), dpi = 80)

    plt.title('Results on test data')
    plt.xlabel('row number')
    plt.ylabel('labels and predictions')

    num = 290
    lr_p_1 = predictions.select('label').toPandas()
    lr_p_2 = predictions.select('prediction').toPandas()

    x = np.arange(num)
    y_list_1 = []
    y_list_2 = []
    y_list_1 = lr_p_1[:290]
    y_list_2 = lr_p_2[:290]

    plt.scatter(x, y_list_1, color = 'blue', marker = '*', label = 'labels', alpha = 0.9, s = 60)
    plt.scatter(x, y_list_2, color = 'red', marker = 'o', label = 'predictions', alpha = 0.9, s = 60)

    plt.xlim(-10, 300)
    plt.ylim(-10, 1000)

    plt.legend()
    plt.grid()
    plt.show()
    
def fit_predict_plot_function(featureIndexer, model, trainingData, testData):

    # Chain indexer and GBT in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, model])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    model_= model.stages[1]
    print(model_)  # summary only

    plot_function(predictions)
    
    return rmse


# # Data preparation process

# ## Merge datasets

# In[18]:


# Merge these four dataframes together: orders_dataset, order_payments_dataset, order_items_dataset, and customer_reviews_dataset.
df_merge_1 = orders_dataset.join(order_payments_dataset, on=['order_id'], how='inner')
df_merge_2 = df_merge_1.join(order_items_dataset, on=['order_id'], how='inner')
df_merge_3 = df_merge_2.join(customer_reviews_dataset, on=['order_id'], how='inner')

# Select 2500 data to speed up running the notebook. You can remove this line for running the whole dataset. 
# But that will consume quite a lot of time and may cause kernel crashes.
df_merge_3 = df_merge_3.limit(2500)
df_merge_3.printSchema()
df_merge_3.show(3)

print("Data count = {}".format(df_merge_3.count()))


# ## Data cleaning

# ### Drop unused features

# In[19]:


df_merge = df_merge_3.drop('customer_id', 'order_status',                            'order_approved_at', 'order_carrier_delivery_date', 'order_customer_delivery_date',                           'order_estimated_delivery_date', 'payment_sequential', 'payment_type', 'payment_installments',                           'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date',                           'review_id', 'survey_review_title', 'survey_review_content', 'survey_send_date', 'survey_completion_date')
df_merge.printSchema()
df_merge.show(3)

print("Data count = {}".format(df_merge.count()))


# ### Drop duplicated values

# In[20]:


df_merge = df_merge.drop_duplicates(['order_id'])

df_merge.printSchema()
df_merge.show(3)

print("Data count = {}".format(df_merge.count()))


# ### Drop nullable values

# In[21]:


df_merge = df_merge.dropna()

df_merge.printSchema()
df_merge.show(3)

print("Data count = {}".format(df_merge.count()))


# ### Convert numeric to double

# In[22]:


df_merge = df_merge.select('order_id', 'order_purchase_timestamp', 'price', 'freight_value', 'payment_value', col("survey_score").cast('double')).orderBy('order_id')

df_merge.printSchema()
df_merge.show(3)

print("Data count = {}".format(df_merge.count()))


# ### Convert datetime to date

# In[23]:


df_date = df_merge.select(date_format(col('order_purchase_timestamp'),"yyyy-MM-dd").alias('order_purchase_date').cast("date"))

df_date.printSchema()
df_date.show(3)

print("Data count = {}".format(df_date.count()))


# ### Convert date to quarter

# In[24]:


df_quarter = df_date.select(quarter('order_purchase_date').alias('quarter'))

df_quarter.printSchema()
df_quarter.show(3)

print("Data count = {}".format(df_quarter.count()))


# ### Add id column

# In[25]:


df_quarter = df_quarter.withColumn("id", monotonically_increasing_id())
df_merge = df_merge.withColumn("id", monotonically_increasing_id())

df_quarter.printSchema()
df_quarter.show(3)

print("Data count = {}".format(df_quarter.count()))

df_merge.printSchema()
df_merge.show(3)

print("Data count = {}".format(df_merge.count()))


# In[26]:


# Merge df_merge and quarter_df together and drop unused id column and order_purchase_timestamp column
df = df_merge.join(df_quarter, on=["id"], how="left").drop("id", "order_purchase_timestamp").orderBy('order_id')

# Drop nullable values
df = df.dropna()

df.printSchema()
df.show(3)

print("Data count = {}".format(df.count()))


# ## Feature engineering

# In[27]:


# Select features for ml models.
df = df.select('price', 'freight_value', 'payment_value', 'survey_score', 'quarter')

df.printSchema()
df.show(10)

print("Data count = {}".format(df.count()))


# ## Feature visualization

# In[28]:


plot = df.toPandas().boxplot(figsize = (25,10))


# In[29]:


df.toPandas().hist(figsize = (25,15), bins = 80)


# In[30]:


sns.pairplot(df.toPandas())
plt.show()


# In[31]:


display(df.toPandas().describe())


# ## Create feature vector

# In[32]:


vectorAssembler = VectorAssembler(inputCols = ['quarter', 'price','freight_value', 'survey_score'], outputCol = 'features')
v_df = vectorAssembler.transform(df)
v_df = v_df.select(['features', 'payment_value'])
v_df = v_df.withColumnRenamed('payment_value', 'label')

v_df.printSchema()
v_df.show(10, False)

print("Data count = {}".format(v_df.count()))


# ## Split datasets

# In[33]:


train_df, test_df = v_df.randomSplit([0.7, 0.3], seed = 42)

train_df.show(10, False)
print('number of rows in train dataframe:', train_df.count())

test_df.show(10, False)
print('number of rows in test dataframe:',test_df.count())


# ## Standardize data

# In[34]:


scaler = StandardScaler(inputCol = "features", outputCol = "features_scaled", withStd = True, withMean = False)
scalerModel = scaler.fit(train_df)


# In[35]:


train_df_s = scalerModel.transform(train_df)
test_df_s  = scalerModel.transform(test_df)
train_df_s.show(10, False)
test_df_s.show(10, False)


# In[36]:


print(scalerModel.mean, '\n')
print(scalerModel.std)


# In[37]:


lr_train_df = train_df_s.drop("features")
lr_test_df = test_df_s.drop("features")


# # Modeling

# ## Ridge regression

# In[38]:


rmse_list = []


# In[39]:


lr_1 = LinearRegression(featuresCol = 'features_scaled', labelCol = 'label', maxIter = 5, regParam = 1, elasticNetParam = 0)
lr_model_1 = lr_1.fit(lr_train_df)
lr_predictions = lr_model_1.transform(lr_test_df)
lr_predictions.select("prediction", "label", "features_scaled")
lr_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "label", metricName = "rmse")
test_result = lr_model_1.evaluate(lr_test_df)
rmse = test_result.rootMeanSquaredError

print("RMSE: %f" % rmse)

plot_function(lr_predictions)

rmse_list.append(rmse)


# ## Lasso regression

# In[40]:


lr_train_df = train_df_s.drop("features")
lr_test_df = test_df_s.drop("features")

lr_2 = LinearRegression(featuresCol = 'features_scaled', labelCol = 'label', maxIter = 5, regParam = 0.3, elasticNetParam = 1) 
lr_model_2 = lr_2.fit(lr_train_df)
lr_predictions = lr_model_2.transform(lr_test_df)
lr_predictions.select("prediction", "label", "features_scaled")
lr_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "label", metricName = "rmse")
test_result = lr_model_2.evaluate(lr_test_df)
rmse = test_result.rootMeanSquaredError

print("RMSE: %f" % rmse)

plot_function(lr_predictions)

rmse_list.append(rmse)


# ## Hyperparameter tuning

# In[41]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

lr_3 = LinearRegression(featuresCol = 'features_scaled', labelCol = 'label')

# Create ParamGrid for Cross Validation
lrparamGrid = (ParamGridBuilder()
             .addGrid(lr_3.regParam, [0.1, 0.5, 1.0, 2.0])
             .addGrid(lr_3.elasticNetParam, [0.0, 0.5, 0.7, 1.0])
             .addGrid(lr_3.maxIter, [1, 5, 10, 15])
             .build())

lrevaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "label", metricName = "rmse")

# Create 5-fold CrossValidator
lrcv = CrossValidator(estimator = lr_3, estimatorParamMaps = lrparamGrid, evaluator = lrevaluator, numFolds = 5)

# Run cross validations
lrcvModel = lrcv.fit(lr_test_df)
best_model = lrcvModel.bestModel


# In[42]:


best_reg_param = best_model._java_obj.getRegParam()
best_elasticnet_param = best_model._java_obj.getElasticNetParam()
best_max_Iter = best_model._java_obj.getMaxIter()
print('best regParam:', best_reg_param, '\nbest elasticNetParam:', best_elasticnet_param, '\nbest max iter:', best_max_Iter)


# In[43]:


lr_3 = LinearRegression(featuresCol = 'features_scaled', labelCol = 'label', maxIter = best_max_Iter, regParam = best_reg_param, elasticNetParam = best_elasticnet_param) 
lr_model_3 = lr_3.fit(lr_train_df)
lr_predictions = lr_model_3.transform(lr_test_df)
lr_predictions.select("prediction", "label", "features_scaled")
lr_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "label", metricName = "rmse")
test_result = lr_model_3.evaluate(lr_test_df)
rmse = test_result.rootMeanSquaredError

print("RMSE: %f" % rmse)

plot_function(lr_predictions)

rmse_list.append(rmse)


# ## Gradient-boosted tree regression

# In[44]:


# Identify categorical features, and index them.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(v_df)

# Train a GBT model.
gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

rmse = fit_predict_plot_function(featureIndexer, gbt, train_df, test_df)

rmse_list.append(rmse)


# ## Decision tree regression

# In[45]:


# Identify categorical features, and index them.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(v_df)

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

rmse = fit_predict_plot_function(featureIndexer,dt , train_df, test_df)

rmse_list.append(rmse)


# ## Random forest regression

# In[46]:


# Identify categorical features, and index them.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(v_df)

# Train a DecisionTree model.
rf = RandomForestRegressor(featuresCol="indexedFeatures")

rmse = fit_predict_plot_function(featureIndexer, rf, train_df, test_df)

rmse_list.append(rmse)


# # Results

# In[47]:


# Show the RMSE results
rmse_list


# In[48]:


figure(figsize = (18, 10), dpi = 80)

plt.title('Compare results')
plt.ylabel('RMSE')
x = ['Ridge regression', 'Lasso regression', 'Hyperparameter Tuning']
y = rmse_list[:3]

plt.plot(x, y, color = 'blue', marker = '*', label = 'RMSE', alpha = 0.9)

y_max = np.max(y)+0.1
y_min = np.min(y)-0.1

plt.xlim(-0.5, 2.5)
plt.ylim(y_min, y_max)

plt.legend()
plt.grid()
plt.show()


# In[49]:


figure(figsize = (18, 10), dpi = 80)

plt.title('Compare results')
plt.ylabel('RMSE')
x = ['Gradient-boosted tree regression', 'Decision tree regression', 'Random forest regression']
y = rmse_list[3:]

plt.plot(x, y, color = 'blue', marker = '*', label = 'RMSE', alpha = 0.9)

y_max = np.max(y)+0.1
y_min = np.min(y)-0.1

plt.xlim(-0.5, 2.5)
plt.ylim(y_min, y_max)

plt.legend()
plt.grid()
plt.show()


# # Conclusion

# As we know, RMSE is a vary good measure of how accurately different models predict the future, and it is the most important criterion for fit if the main purpose of the model is prediction. Since lower values of RMSE indicate better fit, we can see from the above models that the results for Linear regression after hyperparameter tuning is much better than Gradient-boosted tree regression, Decision tree regression, and Random forest regression. Hence, for sales forecast of e-commerce data, linear regression is preferred.

# # Stop the spark session

# In[50]:


spark.stop()


# In[ ]:




