#!/usr/bin/env python
# coding: utf-8

# In[1]:


# initialize Spark

import pyspark
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, asc,desc
import os
from pyspark.sql.functions import *
import time
conf = pyspark.SparkConf().setAll([('spark.master', 'local[4]'),
                                   ('spark.app.name', 'PySpark DataFrame Demo')])
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
print (spark.version, pyspark.version.__version__)


# In[2]:


import pprint
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as sklKMeans
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
from pyspark.sql.types import IntegerType,FloatType
from pyspark.sql.functions import unix_timestamp, from_unixtime
from pyspark.sql.functions import  asin, acos, sin, sqrt, cos
from pyspark.sql.functions import pow, col
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import DateType
from pyspark.sql.functions import radians
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

pp = pprint.PrettyPrinter(indent=4)


# In[3]:


def get_columns(df_list, counts=False):
    df_dict = {}
    for df in df_list:
        df_dict[namestr(df)] = {}
        if(counts):
            df_dict[namestr(df)]['count'] = df.count()
        else:
            df_dict[namestr(df)]['columns'] = df.schema.names
            df_dict[namestr(df)]['count'] = df.count()
    
    return df_dict

def namestr(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj][0]


# In[4]:


DATA_PATH="hdfs:///data/"
df_customers = spark.read.csv(DATA_PATH+"customers_dataset.csv", header=True)
df_customer_reviews = spark.read.csv(DATA_PATH+"customer_reviews_dataset.csv", header=True)
df_geolocation = spark.read.csv(DATA_PATH+"geolocation_dataset.csv", header=True)
df_orders = spark.read.csv(DATA_PATH+"orders_dataset.csv", header=True)
df_order_items = spark.read.csv(DATA_PATH+"order_items_dataset.csv", header=True)
df_order_payments = spark.read.csv(DATA_PATH+"order_payments_dataset.csv", header=True)
df_sellers = spark.read.csv(DATA_PATH+"sellers_dataset.csv", header=True)
df_products = spark.read.csv(DATA_PATH+"products_dataset.csv", header=True)
df_product_category_name_translation = spark.read.csv(DATA_PATH+"product_category_name_translation.csv", header=True)

# This will restrict the datasets and prevent kernel crashes while running 
N_TRAIN_DATA = 30000
N_TEST_DATA = 20000


# In[5]:


df_list = [df_customers,
           df_customer_reviews,
           df_geolocation,
           df_orders,
           df_order_items,
           df_order_payments,
           df_sellers,
           df_products,
           df_product_category_name_translation]

pp.pprint(get_columns(df_list))


# ## Data cleaning
#   * ### Drop NA

# In[6]:


for df in df_list:
    df = df.dropna()

pp.pprint(get_columns(df_list, True))


#   * ### Filter out only intergers for the customer reviews

# In[7]:


df_customer_reviews = df_customer_reviews[
                        (df_customer_reviews['survey_score']=='0')|
                        (df_customer_reviews['survey_score']=='1')|
                        (df_customer_reviews['survey_score']=='2')|
                        (df_customer_reviews['survey_score']=='3')|
                        (df_customer_reviews['survey_score']=='4')|
                        (df_customer_reviews['survey_score']=='5')
                        ]
# do the type conversion of the text score to integer
df_customer_reviews = df_customer_reviews.withColumn('survey_score', df_customer_reviews['survey_score'].cast(IntegerType()))


#   * ### Remove duplicate geolocations

# In[8]:


print("Raw data count = {}".format(df_geolocation.count()))
df_geolocation = df_geolocation.dropDuplicates(['geo_zip_code_prefix'])
print("Data count after dropping duplicates = {}".format(df_geolocation.count()))


# ## Merge order_items, product_category into products 

# In[9]:


df_grp_product_cat = df_order_items.join(df_products, on=['product_id'], how='inner')
df_grp_product_cat = df_grp_product_cat.dropna(subset=["product_category_name"])
pp.pprint(get_columns([df_grp_product_cat]))
df_grp_product_cat = df_grp_product_cat.dropna()
df_order_merged = df_grp_product_cat.join(df_orders, on=['order_id'], how='inner')
df_order_merged = df_order_merged.dropna()
pp.pprint(get_columns([df_order_merged]))

# df_grp_product_cat.dropna().count()


# ## Data Conversion

# In[10]:


# Convert to integer Type
name_type = ['shipping_limit_date']
int_col = ['product_photos_qty',
           'product_height_cm', 
           'product_length_cm', 
           'product_weight_g', 
           'product_width_cm', 
           'product_name_lenght', 
           'product_description_lenght']
float_col = ['price', 
             'freight_value'] 


for k in range(len(name_type)):
    df_order_merged = df_order_merged.drop(name_type[k])
    
for k in range(len(int_col)):
    df_order_merged = df_order_merged.withColumn(int_col[k], df_order_merged[int_col[k]].cast(IntegerType()))

for k in range(len(float_col)):
    df_order_merged = df_order_merged.withColumn(float_col[k], df_order_merged[float_col[k]].cast(FloatType()))    

display(df_order_merged.schema)


# ## Product dimesnsions 
#   - ### multiply l x w x h and get rid of the l,w,h columns, name the resulting volume calculation as \_dim 

# In[11]:


df_order_merged = df_order_merged.withColumn('product_dim_cm', df_order_merged['product_length_cm']*df_order_merged['product_height_cm']*df_order_merged['product_width_cm'])

df_order_merged  = df_order_merged.drop('product_length_cm', 'product_height_cm', 'product_width_cm')

df_order_merged.schema.names


#   - ### Effect of product dimensions on the orders 

# In[12]:


product_dims = df_order_merged.select('product_dim_cm').toPandas()
fig = plt.figure(figsize=(25, 10))
ax = product_dims['product_dim_cm'].plot.hist(bins=1000, alpha=0.5)
ax.set_ylabel("count")
plt.title(f"Product dimensions\n")
plt.show()
product_dims.boxplot(column=['product_dim_cm'])
display(product_dims.describe())


# ## Geolocation

#   - ### Get the lat,lng for sellers and customers  

# In[13]:


df_geolocation.schema.names
#df_order_merged.join(df_order_merged) 
df_order_merged = df_order_merged.join(df_sellers.drop('seller_city','seller_state'), on=['seller_id'], how='inner')
df_order_merged = df_order_merged.join(df_customers.drop('customer_unique_id', 'customer_city', 'customer_state'), on=['customer_id'], how='inner')
df_order_merged = df_order_merged.join(df_geolocation.selectExpr(" geo_zip_code_prefix as seller_zip_code_prefix", "geo_lat as seller_lat", "geo_lng as seller_lng"), on=['seller_zip_code_prefix'], how='inner')
df_order_merged = df_order_merged.join(df_geolocation.selectExpr(" geo_zip_code_prefix as customer_zip_code_prefix", "geo_lat as customer_lat", "geo_lng as customer_lng"), on=['customer_zip_code_prefix'], how='inner')
df_order_merged = df_order_merged.dropna()
display(get_columns([df_order_merged]))


#   - ### Haversine calculation for distance
#     - **convert all the lat,lng into radians and drop original lat,lng**

# In[14]:



df_order_merged = df_order_merged.withColumn('seller_lat_rad',radians(df_order_merged['seller_lat']))
df_order_merged = df_order_merged.withColumn('seller_lng_rad',radians(df_order_merged['seller_lng']))
df_order_merged = df_order_merged.withColumn('customer_lat_rad',radians(df_order_merged['customer_lat']))
df_order_merged = df_order_merged.withColumn('customer_lng_rad',radians(df_order_merged['customer_lng']))

df_order_merged = df_order_merged.drop('seller_lat')
df_order_merged = df_order_merged.drop('seller_lng')
df_order_merged = df_order_merged.drop('customer_lat')
df_order_merged = df_order_merged.drop('customer_lng')

display(get_columns([df_order_merged]))


#    - **Get the difference between lat, lng seller and customer**

# In[15]:


df_order_merged = df_order_merged.withColumn('dlng',(df_order_merged['seller_lng_rad'] - df_order_merged['customer_lng_rad'])/2)
df_order_merged = df_order_merged.withColumn('dlat',(df_order_merged['seller_lat_rad'] - df_order_merged['customer_lat_rad'])/2)
display(get_columns([df_order_merged]))


#    - **Calculations for haversine equation**

# In[16]:



df_order_merged = df_order_merged.withColumn('dlng_sin', sin(df_order_merged['dlng']))
df_order_merged = df_order_merged.withColumn('dlng_sin_square', pow(df_order_merged['dlng_sin'],2))

df_order_merged = df_order_merged.withColumn('dlat_sin', sin(df_order_merged['dlat']))
df_order_merged = df_order_merged.withColumn('dlat_sin_square', pow(df_order_merged['dlat_sin'],2))

df_order_merged = df_order_merged.withColumn('seller_lat_rad_cos', cos(df_order_merged['seller_lat_rad']))
df_order_merged = df_order_merged.withColumn('customer_lat_rad_cos', cos(df_order_merged['customer_lat_rad']))


df_order_merged = df_order_merged.withColumn('A',  df_order_merged['dlat_sin_square'] + df_order_merged['customer_lat_rad_cos']*df_order_merged['seller_lat_rad_cos']*df_order_merged['dlng_sin_square'])

df_order_merged = df_order_merged.withColumn('A_sqrt', sqrt(df_order_merged['A']))
df_order_merged = df_order_merged.withColumn('distance', 7912*asin(df_order_merged['A_sqrt']))


cal_list = [
            'dlng',
            'dlat',
            'dlng_sin',
            'dlat_sin',
            'dlng_sin_square',
            'dlat_sin_square',
            'seller_lat_rad_cos',
            'customer_lat_rad_cos',
            'seller_lat_rad',
            'customer_lat_rad',
            'seller_lng_rad',
            'seller_lat_rad',
            'customer_lng_rad',
            'customer_lat_rad',
            'A',
            'A_sqrt'
           ]
## Drop the temporary rows
for drop_col in cal_list:
    df_order_merged = df_order_merged.drop(drop_col)


#    - **Display the first 10 rows in the dataframe**

# In[17]:


df_order_merged.limit(10)


# ## Effect of distance on orders

# In[18]:


df_dist = df_order_merged.select('distance').toPandas()
df_dist.plot.hist(bins=1500, ylim=(0,2000), grid=True, figsize=[10,8])
display(df_dist.describe())


# ## Merge the customer reviews

# In[19]:


df_order_merged = df_order_merged.join(df_customer_reviews.select('order_id', 'survey_score'), on=['order_id'], how='inner')
display(df_order_merged.limit(10))


# ## Distribution plot of customer review

# In[20]:


df_order_merged.select('survey_score').toPandas().hist(bins=5)


# ## Timestamps

#    - **Merge the month**

# In[21]:


get_month =  udf (lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month, IntegerType())
df_order_merged = df_order_merged.withColumn('month', get_month(col('order_purchase_timestamp')))

display(df_order_merged.limit(10))


# display(df_order_merged.count())
# display(df_order_merged.count())
# display(df_order_merged.count())
# display(df_order_merged.count())
# display(df_order_merged.count())
# display(df_order_merged.count())
#    - **Plot of orders per month**

# In[22]:


df_pd_timestamps = df_order_merged.select('order_id','order_purchase_timestamp').toPandas()
df_pd_timestamps.groupby(df_pd_timestamps['order_purchase_timestamp'].astype('datetime64[ns]').dt.month)    .agg({"order_id": "nunique"})    .plot(figsize=(15,10), kind="bar",
          title="Orders per month",
          ylabel="Num of orders",
          xlabel="Month",
          legend=False)
plt.xticks(np.arange(0,12), ['Jan','Feb','Mar','Apr','May','Jun',
                             'Jul','Aug','Sept','Oct','Nov','Dec'], 
           rotation='horizontal')
plt.show()


# In[23]:


pp.pprint(get_columns([df_order_merged]))


# ## Create a incrementing id for product_catergory

# In[24]:


df_product_category_name_translation = df_product_category_name_translation.withColumn("cat_id", F.monotonically_increasing_id())
df_order_merged = df_order_merged.join(df_product_category_name_translation.select('product_category_name','cat_id'), on=['product_category_name'], how='inner')
df_order_merged = df_order_merged.withColumn('cat_id', df_order_merged['cat_id'].cast(IntegerType()))


# In[25]:


df_order_merged.schema


# In[26]:


df_order_merged.limit(10)


# ## Using a smaller dataframe for the final K-Means model.
#   - ### Using a dataset split of 60%, 10%, 30% for the train, validation and test datasets
#   - ### To prevent the kernel crashes the train, test data sets have been restricted to 30,000 and 20,000 data points

# In[27]:


features = ['distance', 'survey_score', 'month', 'price', 'freight_value' , 'product_dim_cm', 'product_weight_g', 'product_photos_qty', 'cat_id']
features_ = ['distance', 'survey_score', 'month', 'price', 'freight_value' , 'product_dim_cm', 'product_weight_g', 'product_photos_qty', 'cat_id','product_id','customer_id']

df_order_merged_short = df_order_merged.select(features_)
#df_order_merged = df_order_merged.select(features)
VA = VectorAssembler(inputCols=features, outputCol='features')
transformed_data = VA.transform(df_order_merged_short)

train_df,val_df,  test_df = transformed_data.randomSplit([0.6, 0.1,0.3], seed = 42)


# ## Check the datatypes of the columns before fitting it to the model

# In[28]:


display(df_order_merged_short.schema)


# ## Scale the train, validation, test datasets

# In[29]:


print("Fit the trained data and create a scaler model")
scale = StandardScaler(inputCol='features',outputCol='scaled')

train_df = train_df.limit(N_TRAIN_DATA)
train_scaled_data = scale.fit(train_df)
train_scaled_data_output = train_scaled_data.transform(train_df)
train_scaled_data_output.show(2)

print("Fit the test data to the scaler model")
test_df = test_df.limit(N_TEST_DATA)
test_scaled_data_output = train_scaled_data.transform(test_df)
test_scaled_data_output.show(2)

print("Fit the validation data to the scaler model")
val_scaled_data_output = train_scaled_data.transform(val_df)
val_scaled_data_output.show(2)


print("Number of rows in train_df = {}, val_df = {}, test_df = {}".format(N_TRAIN_DATA,val_df.count(), N_TEST_DATA))


# ## Results

#   - ### Find the optimal 'k' for clustering

# In[30]:


silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='scaled',                                 metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2,15):    
    KMeans_algo=KMeans(featuresCol='scaled', k=i)
    KMeans_fit=KMeans_algo.fit(train_scaled_data_output)    
    output=KMeans_fit.transform(train_scaled_data_output)
    score=evaluator.evaluate(output)
    silhouette_score.append(score)
    print("Silhouette Score for k = {} : {}".format(i,score))


#   - ### Plot of Cost against 'k'

# In[31]:


(x,y) = (range(2,15),silhouette_score)
x_new = np.linspace(1, 14, 200)
a_BSpline = make_interp_spline(x, y)
y_new = a_BSpline(x_new)

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(x_new,y_new)
ax.set_title("Cost vs 'k'")
ax.set_xlabel('k')
ax.set_ylabel('cost')


# ## It can be seen from the above plot that a optimal value of k = 4

# In[32]:


KMeans_algo=KMeans(featuresCol='scaled', k=4)
KMeans_fit=KMeans_algo.fit(train_scaled_data_output)    
train_output=KMeans_fit.transform(train_scaled_data_output)
score=evaluator.evaluate(train_output)
silhouette_score.append(score)
print("Silhouette Score for training data and k = {}: {}".format(4,score))


#   - ### Plot the clusters

# In[33]:


train_df = train_output.toPandas()
# unpack the dense scaled vectors
train_df_scaled = train_df['scaled'].apply(lambda x: pd.Series(x.toArray()))
# use PCA to reduce the dimension to 2
pca = PCA(2)
data_pca = pca.fit_transform(train_df_scaled)
kmeans_model = sklKMeans(n_clusters= 4)
label = kmeans_model.fit_predict(data_pca)
unique_labels = np.unique(label)
centroids = kmeans_model.cluster_centers_

fig, ax = plt.subplots(figsize=(15, 10))

for i in unique_labels:
    ax.scatter(data_pca[label == i , 0] , data_pca[label == i , 1] , label = i, s=10)

ax.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'black', marker='X')
ax.set_title("Plot of the clusters in 2D")
plt.legend()
plt.show()


#   - ### Score for the validation data

# In[34]:


val_output=KMeans_fit.transform(val_scaled_data_output)
score=evaluator.evaluate(val_output)
print("Silhouette Score for validation data:",score)


#   - ### Score for the test data

# In[35]:


test_output=KMeans_fit.transform(test_scaled_data_output)
score=evaluator.evaluate(test_output)
print("Silhouette Score for test data:",score)


# ## Combine all the outputs together

# In[36]:


Final_output = train_output.union(train_output.union(val_output)).dropna()


# In[37]:


Final_output[Final_output['prediction']==3].sample(fraction=0.5)


# ## Product Recommendations
#   - ### Assume there were 5 purchases and we generated the test outputs
#   - ### Use the test output prediction labels and filter the Final_output dataframe
#   - ### Randomly choose 3 products from the cluster for each purchase

# In[38]:


sample_test = test_output.limit(5)


# In[39]:


purchase_list = sample_test.select('product_id', 'customer_id', 'prediction').toPandas().values.tolist()


# In[40]:


for item in purchase_list:
    print('Product : {} purchased by the customer :{}'.format(item[0],item[1]))
    print('Top 3 recommended products for the specific customer')
    display(Final_output[Final_output['prediction']==item[2]].sample(fraction=0.5).limit(3))


# ## Stop Spark session

# In[42]:


spark.stop()


# In[ ]:




